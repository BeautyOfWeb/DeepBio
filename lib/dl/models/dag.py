import collections
import functools

import numpy as np
import pandas

import torch
import torch.nn as nn

from .transformer import StackedEncoder
from ..utils.gen_conv_params import join_dict, get_itemset


class EmbedBigraph(nn.Module):
  r"""Map two sets of nodes from a bipartite graph
    An nn.Conv1d is used to map its corresponding subset of source nodes for each target node
    This is hard to evaluate the impact of a source node to its target node
    A better way to do it might be to use attention based mapping, but it requires much more computation
  
  Args:
    bigraph: a list, eg: [[0,1,3], [1,2]], input node ids are from 0 to length-1
    in_channels: int
    out_channels: int
    use_layer_norm: if True, apply nn.LayerNorm to each instance
    bias: whether or not to use bias in nn.Conv1d
    
  Shape:
    - Input: N * in_channels * M, where M = the number of input nodes
    - Output: N * out_channels * L, where L = len(bigraph), the number of output nodes
    
  Attributes:
    a series (length = len(bigraph)) of weights (and biases) of nn.Conv1d
    
  Examples::
  
    >>> bigraph = [[0,1,3], [1,2,3]]
    >>> x = torch.randn(1, 3, 4)
    >>> model = EmbedBigraph(bigraph, 3, 10)
    >>> model(x).shape
  """
  def __init__(self, bigraph, in_channels, out_channels, use_layer_norm=True, bias=True, nonlinearity=None):
    super(EmbedBigraph, self).__init__()
    self.bigraph = bigraph
    self.use_layer_norm = use_layer_norm
    self.nonlinearity = nonlinearity
    self.maps = nn.ModuleList(
      [nn.Conv1d(in_channels, out_channels, kernel_size=len(v), bias=bias) for v in self.bigraph]
    )
    if self.use_layer_norm:
      self.layer_norms = nn.ModuleList(
        [nn.LayerNorm([in_channels, len(v)], eps=1e-5, elementwise_affine=False) for v in self.bigraph]
      )
    
  def forward(self, x):
    out = []
    for i, v in enumerate(self.bigraph):
      z = x[:, :, v]
      if self.use_layer_norm:
        z = self.layer_norms[i](z)
      out.append(self.maps[i](z))
    out = torch.cat(out, dim=-1)
    if isinstance(self.nonlinearity, nn.Module):
      out = self.nonlinearity(out)
    return out


class DAGLayer(nn.Module):
  r"""Build a computatinal graph from DAG
  
  Args:
    dag: dictionary, e.g.: {2:[0,1], 3:[0,1,2]}, node ids are topological order for computation.
    in_channels: int, embedding dimension
    use_layer_norm: if True, for each target node, apply nn.LayerNorm to its corresponding subset of source nodes 
    gibbs_sampling: only used when use_layer_norm is True; 
      if True, in-place change the source node representations 
        while calculating target node representations (expanding the node representations);
      if False, once the representation of a node is generated, it won't be changed
    nonlinearity: if not None, apply it to the output before return
    bias: default True; whether or not to use bias in nn.Conv1d
    
  Shape:
    - Input: N * in_channels * L, where L = the number of leaf nodes in dag
    - Output: N * in_channels * M, where M = the number of all nodes in dag
      if self.nonlinearity=None, and either self.gibbs_sampling and self.use_layer_norm is False, 
        the output will include the original input;
        otherwise, the original input will be modified in-place
    
  Attributes:
    a series (length = len(dag)) of weights (and biases) of nn.Conv1d
    
  Examples::
  
  >>> dag = {2:[0,1], 3:[0,1,2], 4:[1,2,3], 5:[0,2,3]}
  >>> model = DAGLayer(dag, 10)
  >>> x = torch.randn(1, 10, 2, device=device)
  >>> model(x).shape
  """
  def __init__(self, dag, in_channels, use_layer_norm=True, gibbs_sampling=True, nonlinearity=nn.ReLU(), bias=True):
    super(DAGLayer, self).__init__()
    self.dag = dag
    self.in_channels = in_channels
    self.use_layer_norm = use_layer_norm
    self.gibbs_sampling = gibbs_sampling
    self.nonlinearity = nonlinearity
    self.embed = nn.ModuleList(
      [nn.Conv1d(in_channels, in_channels, kernel_size=len(v), bias=bias) 
       for k, v in sorted(self.dag.items())]
    )
    if self.use_layer_norm:
      self.layer_norms = nn.ModuleList(
        [nn.LayerNorm([in_channels, len(v)], eps=1e-5, elementwise_affine=False) 
          for k, v in sorted(self.dag.items())]
      )
    self.num_leaf_nodes = min(self.dag)
    self.num_all_nodes = max(dag)+1
    nonleaf_nodes = set(self.dag)
    all_nodes = set(functools.reduce(lambda x, y: x + y, self.dag.values())).union(nonleaf_nodes)
    leaf_nodes = all_nodes.difference(nonleaf_nodes)
    assert self.num_leaf_nodes == len(leaf_nodes) and self.num_all_nodes == len(all_nodes)
    
  def forward(self, x):
    x_dim = x.dim()
    if x_dim == 2 and self.in_channels == 1:
      # handle a special case when input only have 1 feature plane; this reduce to a linear mapping
      x = x.unsqueeze(1)
    if x.size(-1) == self.num_leaf_nodes:
      isolated_input = None
    elif x.size(-1) <= self.num_all_nodes and x.size(-1) > self.num_leaf_nodes:
      if x.size(-1) < self.num_all_nodes:
        print(f'Warning: size mismatch: input x.size(-1)={x.size(-1)}, num_leaf_nodes={self.num_leaf_nodes}, ' 
          f'num_all_nodes={self.num_all_nodes}')
      isolated_input = None
      x = x[..., :self.num_leaf_nodes]
    elif x.size(-1) > self.num_all_nodes:
      # assume the isolated nodes are appended in the end
      # print(f'Warning: there are {x.size(-1)-self.num_all_nodes} isolated nodes in input: '
      #   f'input x.size(-1)={x.size(-1)}, num_leaf_nodes={self.num_leaf_nodes}, num_all_nodes={self.num_all_nodes}')
      isolated_input = x[..., self.num_all_nodes:]
      x = x[..., :self.num_leaf_nodes]
    else:
      raise ValueError(f'Size mismatch: input x.size(-1)={x.size(-1)}, num_leaf_nodes={self.num_leaf_nodes}, ' 
        f'num_all_nodes={self.num_all_nodes}')
    for i, (k, v) in enumerate(sorted(self.dag.items())):
      z = x[:, :, v]
      if self.use_layer_norm:
        if self.gibbs_sampling:
          x[:, :, v] = self.layer_norms[i](z)
          z = x[:, :, v]
        else:
          z = self.layer_norms[i](z)
      y = self.embed[i](z)
      x = torch.cat([x, y], dim=-1)
    if isinstance(self.nonlinearity, nn.Module):
      x = self.nonlinearity(x)
    if isolated_input is not None:
      x = torch.cat([x, isolated_input], dim=-1)
    if x_dim == 2 and self.in_channels == 1:
      # handle a special case when input is a 2-D tensor; make it back to 2-D tensor
      x = x.squeeze(1)
    return x


class StackedDAGLayers(nn.Module):
  r"""Stack multiple DAG layers
  
  Args:
    dag: dictionary, e.g.: {2:[0,1], 3:[0,1,2]}, node ids are topological order for computation.
    in_channels_list: a list of int
    residual: if True, use residual connections between two consecutive layers;
      if residual is False, only the last DAG layer is actually used; so set residual=True for almost all cases
    duplicated_dag: all stacked DAG layers share the same parameters
    use_layer_norm, gibbs_sampling, nonlinearity are passed to DAGLayer
    bias: default True, passed to nn.Conv1d
    
  Shape:
    - Input: N * in_channels_list[0] * L, where L = the number of leaf nodes in dag
    - Output: N * in_channels_list[-1] * M, where M = the number of all nodes in dag
    
  Attributes:
    a series (length = len(in_channels_list)) of series (length = len(dag)) 
      of weights (and biases) for a ModuleList of nn.Conv1d
    
  Examples::
  
    >>> dag = {2:[0,1], 3:[0,1,2], 4:[1,2,3], 5:[0,2,3]}
    >>> model = StackedDAGLayers(dag, [10, 10], residual=True, duplicate_dag=True, 
        use_layer_norm=True, gibbs_sampling=True, nonlinearity=nn.ReLU(), bias=True)
    >>> x = torch.randn(3, 10, 2, device=device)
    >>> model(x).shape
  """
  def __init__(self, dag, in_channels_list, residual=True, duplicate_dag=True, use_layer_norm=True, 
    gibbs_sampling=True, nonlinearity=nn.ReLU(), bias=True):
    super(StackedDAGLayers, self).__init__()
    self.dag = dag
    self.in_channels_list = in_channels_list
    self.num_layers = len(self.in_channels_list)
    self.residual = residual
    self.duplicate_dag = duplicate_dag
    self.use_layer_norm = use_layer_norm
    self.nonlinearity = nonlinearity
    self.bias = bias
    if self.duplicate_dag:
      for n in in_channels_list:
        assert n == in_channels_list[0]
      self.layers = nn.ModuleList(
        [DAGLayer(dag, in_channels_list[0], use_layer_norm=use_layer_norm, gibbs_sampling=gibbs_sampling,
                  nonlinearity=nonlinearity, bias=bias)] * self.num_layers
      )
    else:
      self.layers = nn.ModuleList(
        [DAGLayer(dag, in_channels, use_layer_norm=use_layer_norm, gibbs_sampling=True, 
                  nonlinearity=nonlinearity, bias=bias) for in_channels in in_channels_list]
      )
      self.bottlenecks = nn.ModuleList(
        [nn.Conv1d(in_channels_list[i-1], in_channels_list[i], kernel_size=1, bias=bias)
          for i in range(1, self.num_layers)]
      )
  
  def forward(self, x):
    x_dim = x.dim()
    if x_dim == 2 and self.in_channels_list[0] == 1:
      # handle a special case when input only have 1 feature plane; this reduce to a linear mapping
      x = x.unsqueeze(1)
    for i in range(self.num_layers):
      out = self.layers[i](x) # the input x contains only the representations for leaf nodes
      if i>0 and self.residual: 
        out = out + x
      x = out
      if i < self.num_layers-1 and not self.duplicate_dag:
        x = self.bottlenecks[i](x)
    if x_dim == 2 and self.in_channels_list[-1] == 1:
      # handle a special case when input is a 2-D tensor; make it back to 2-D tensor
      x = x.squeeze(1)
    return x


class GraphAttention(nn.Module):
  r"""Refine graph embedding using its network structure
    can be trained with stochastic depth
    Fixed graph attention matrices generated from deterministic/random walk on the graph
  
  Args:
    num_features: int
    in_dim: int
    out_dim: int (for a single layer or recurrent layers) or a list of ints (for multiple non-recurrent layers)
    recurrent: if True, perform self-attention recurrently; assert in_dim==out_dim
    residual: only used for non-recurrent multiple layers, each with a different weight
      requires all out_dim be the same as in_dim to facilitate skip connections without additional transformations
    use_layer_norm: if True, apply nn.LayerNorm to the output from each layer
  
  Shape:
    Input: x is torch.Tensor of size (N, in_dim, num_features)
            Let M be the adjacency matrix (symmetric, source and target are the same)
              attention_mats = [M.T, (M*M).T, (M*M*M).T, ...]
              attention_mats stores transition mat with 1,2,... steps
              These transition mats are transposed
    Output: depending return_layers in forward function; 
      history = a list of all outputs from all layers including the input layer;
      the output of each layer will have the shape (N, dim, num_features), where dim is from in_dim or out_dim
  
  Examples:
  
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    attention_mats, id_to_name = adj_list_to_attention_mats(adj_list, num_steps=10, name_to_id=None, 
                                                  bipartite=False)
    num_features = len(id_to_name)
    in_dim = 11
    out_dim = [11]*10

    model = GraphAttention(num_features, in_dim, out_dim, recurrent=True, residual=True, 
                          use_layer_norm=True)
    x = torch.randn(3, in_dim, num_features)
    model(x, attention_mats, max_num_layers=11, min_num_layers=2, return_layers='last').shape
  
  """
  def __init__(self, num_features, in_dim, out_dim, recurrent=True, residual=True, use_layer_norm=True):
    super(GraphAttention, self).__init__()
    self.recurrent = recurrent
    self.residual = residual
    if isinstance(out_dim, int):
      out_dim = [out_dim]
    if self.recurrent:
      for d in out_dim:
        assert in_dim == d
      self.weight = nn.Parameter(torch.randn(in_dim, num_features, num_features, in_dim), 
                                            requires_grad=True)
    else:
      if residual and len(out_dim) > 1:
        # in this case, each layer will have its own weight; 
        # they all have the same shape so that skip connections are possible without additional transformations
        for d in out_dim:
          assert d == in_dim
      self.weights = nn.ParameterList([nn.Parameter(torch.randn(in_dim if i==0 else out_dim[i-1], num_features,
        num_features, out), requires_grad=True) for i, out in enumerate(out_dim)])                             
    self.use_layer_norm = use_layer_norm
  
  def forward(self, x, attention_mats, max_num_layers=2, min_num_layers=2, 
              return_layers='last'):
    # stochastic depth
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    history = [x] # the first layer is the original input
    for i in range(1, num_layers):
      if self.recurrent or self.residual:
        y = [] # this stores the new representations of ith layer from all previous layers
        for j in range(i):
          if self.recurrent:
            weight = self.weight
          else:
            weight = self.weights[j]
          if isinstance(attention_mats, (list, tuple)):
            weight = weight * attention_mats[i-j-1].unsqueeze(-1)
          new_y = (history[j].unsqueeze(-1).unsqueeze(-1) * weight).sum(dim=1).sum(dim=1).transpose(1,2)
          y.append(new_y)
        y = torch.stack(y, dim=0).mean(dim=0)
      else:
        # the ith layer is computed from the i-1 layer only
        # assert len(self.weight) >= num_layers-1
        weight = self.weights[i-1] 
        if isinstance(attention_mats, (list, tuple)):
          weight = weight * attention_mats[i-1].unsqueeze(-1)
        elif isinstance(attention_mats, torch.Tensor):
          weight = weight * attention_mats.unsqueeze(-1)
        y = (history[i-1].unsqueeze(-1).unsqueeze(-1) * weight).sum(dim=1).sum(dim=1).transpose(1,2)
      if self.use_layer_norm:
        y = torch.nn.functional.layer_norm(y.transpose(1,2), normalized_shape=[y.size(1)], weight=None, 
          bias=None, eps=1e-05).transpose(1,2)
      history.append(y)
    if return_layers == 'last':
      return history[-1]
    elif return_layers == 'last-two':
      return history[-2:]
    elif return_layers == 'all-but-input':
      return history[1:]
    elif isinstance(return_layers, list):
      return [history[i] for i in return_layers]
    elif return_layers == 'all':
      return history


class Conv1d2Score(nn.Module):
  r"""Calculate a N*out_dim tensor from N*in_dim*seq_len using nn.Conv1d
  Essentially it is a linear layer
  
  Args:
    in_dim: int
    out_dim: int, usually number of classes
    seq_len: int
    
  Shape:
    - Input: N*in_dim*seq_len
    - Output: N*out_dim
    
  Attributes:
    weight (Tensor): the learnable weights of the module of shape 
      out_channels (out_dim) * in_channels (in_dim) * kernel_size (seq_len)
    bias (Tensor): shape: out_channels (out_dim)
    
  Examples::
  
    >>> x = torch.randn(2, 3, 4, device=device)
    >>> model = Conv1d2Score(3, 5, 4)
    >>> model(x).shape
  """
  def __init__(self, in_dim, out_dim, seq_len, bias=True):
    super(Conv1d2Score, self).__init__()
    self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=seq_len, bias=bias)
  
  def forward(self, x):
    out = self.conv(x).squeeze(-1)
    return out

class DAGEncoder(nn.Module):
  r"""A customized model chaining nn.Embedding, EmbedBigraph, StackedDAGLayers, and StackedEncoder

  Args:
    num_features: int
    embedding_dim: int
    in_channels_list: a list of ints; pass it to StackedDAGLayers; 
      when duplicate_dag is True, in_channels_list must contain only one unique int
    bigraph: passed to EmbedBigraph
    dag: passed to StackedDAGLayers
    key_dim, value_dim, fc_dim, num_cls, dim_per_cls, num_heads,num_attention, knn, and duplicated_attention 
      are passed to StackedEncoder
    feature_max_norm: default 1, passed to nn.Embedding
    bias, nonlinearity, use_layer_norm are passed to EmbedBigraph, StackedDAGLayers, and StackedEncoder
    residual is passed to  StackedDAGLayers and StackedEncoder
    gibbs_sampling: passed to StackedDAGLayers
    graph_encoder, graph_weight_encoder, graph_decoder, graph_weight_decoder are passed to StackedEncoder
      if used, graph_encoder should have size that is consistent with dag, i.e., (max(dag.keys())+1, max(dag)+1)
      if used, graph_decoder should have size (num_cls, max(dag.keys()) + 1)

  Shape:
    Input: (N, num_features)
    Output: 
      if return_attention is False:
        return class scores y of size (N, num_cls)
      else:
        return y and attention_mats of size (num_heads, N, num_cls, num_features)

  Examples:
    bigraph = [[0,1,3], [1,2,3]]
    dag = {2:[0,1], 3:[0,1,2], 4:[1,2,3], 5:[0,2,3]}
    x = torch.randn(5, 4)
    graph_encoder = torch.randn(6, 6)
    graph_decoder = torch.randn(2, 6)
    model = DAGEncoder(num_features=4, embedding_dim=10, in_channels_list=[10,10], 
                      bigraph=bigraph, dag=dag, key_dim=7, value_dim=9, fc_dim=11, 
                      num_cls=2, dim_per_cls=77, feature_max_norm=1, use_layer_norm=True, bias=True, 
                      nonlinearity=nn.ReLU(), residual=True, duplicate_dag=True, 
                      gibbs_sampling=True, num_heads=1, num_attention=1, knn=2, 
                      duplicated_attention=True, graph_encoder=graph_encoder, 
                      graph_decoder=graph_decoder)
    model(x)
  """
  def __init__(self, num_features, embedding_dim, in_channels_list, bigraph, dag, key_dim, value_dim, fc_dim, 
    num_cls, dim_per_cls=2, feature_max_norm=1, use_layer_norm=True, bias=True, nonlinearity=nn.ReLU(), residual=True, 
    duplicate_dag=True, gibbs_sampling=True, num_heads=1, num_attention=1, knn=None, duplicated_attention=True,
    graph_encoder=None, graph_weight_encoder=0.5, graph_decoder=None, graph_weight_decoder=0.5, use_encoders=True):
    super(DAGEncoder, self).__init__()
    self.use_encoders = use_encoders
    self.feature_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim, padding_idx=None, 
      max_norm=feature_max_norm, norm_type=2, scale_grad_by_freq=True, sparse=False, _weight=None)
    self.bigraph = EmbedBigraph(bigraph, embedding_dim, out_channels=in_channels_list[0], 
      use_layer_norm=use_layer_norm, bias=bias, nonlinearity=nonlinearity)
    self.dag_layers = StackedDAGLayers(dag, in_channels_list, residual=residual, duplicate_dag=duplicate_dag, 
      use_layer_norm=use_layer_norm, gibbs_sampling=gibbs_sampling, nonlinearity=nonlinearity, bias=bias)
    if self.use_encoders:
      self.encoders = StackedEncoder(in_channels_list[-1], key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
        num_cls=num_cls, dim_per_cls=dim_per_cls, num_heads=num_heads, num_attention=num_attention, knn=knn, 
        residual=residual, use_layer_norm=use_layer_norm, nonlinearity=nonlinearity, 
        duplicated_attention=duplicated_attention, graph_encoder=graph_encoder, graph_weight_encoder=graph_weight_encoder, 
        graph_decoder=graph_decoder, graph_weight_decoder=graph_weight_decoder)
      self.classifier = nn.Linear(dim_per_cls, 1, bias=bias)
    else:
      self.classifier = Conv1d2Score(in_dim=in_channels_list[-1], out_dim=num_cls, seq_len=len(bigraph)+len(dag), 
        bias=bias)

  def forward(self, x, return_attention=False, graph_encoder=None, graph_decoder=None, 
    encoder_stochastic_depth=False):
    x = x.unsqueeze(-1) * self.feature_embedding.weight
    out = self.bigraph(x.transpose(-1,-2))
    out = self.dag_layers(out)
    if self.use_encoders:
      out = out.transpose(-1,-2)
      if return_attention:
        out, attention_mats = self.encoders(out, return_attention=True, graph_encoder=graph_encoder, 
          graph_decoder=graph_decoder, stochastic_depth=encoder_stochastic_depth)
      else:
        out = self.encoders(out, return_attention=False, graph_encoder=graph_encoder, graph_decoder=graph_decoder,
          stochastic_depth=encoder_stochastic_depth)
      y = self.classifier(out).squeeze(-1)
    else:
      y = self.classifier(out)
    if self.use_encoders and return_attention:
      return y, attention_mats
    else:
      return y


def get_upper_closure(graph, vertices):
  r"""Given a subset of nodes of a graph, return its upperward closure

  Args:
    graph: np.array of shape (N, 2) with two columns being parent <-- child
    vertices: assume vertices are only from child nodes
  
  Returns:
    a subgraph that can be reached from given vertices

  """

  subgraphs = [np.array([p for p in graph if p[1] in vertices])]
  vertices_seen = set(vertices)
  vertices_unseen = set(subgraphs[-1][:, 0]).difference(vertices_seen)
  while len(vertices_unseen) > 0:
    # print('seen:{}, unseen:{}'.format(len(vertices_seen), len(vertices_unseen)))
    subgraph = np.array([p for p in graph if p[1] in vertices_unseen])
    if subgraph.shape[0] > 0:
      subgraphs.append(subgraph)
      vertices_seen = vertices_seen.union(vertices_unseen)
      vertices_unseen = set(subgraph[:,0]).difference(vertices_seen)
    else:
      break
  return np.concatenate(subgraphs, axis=0)


def get_topological_order(adj_list, edge_direction='left<-right'):
  """Get topological order based on adjacency list
    This function will only generate one possible topological order; there can be many;
    If it is DAG, then output name_to_id (dictionary mapping node name to int from 0) 
    and chain_graph (a list of list); 
      chain_graph[0] is a list of nodes whose IDs can be all set 0, and so on;
      partents will have lower ids than their children; 
        for example, edge left<-right implies ID(left)>ID(right)
  
  Args:
    adj_list: a np.array of shape (N, 2), 
      each row corresponds to one edge with its direction determined by argument edge_direction
    edge_direction: default 'left<-right', this is not a natural option, keep it for backward compatibility
      if it is anything not the default value, then assume edge_direction == 'left->right' in adj_list

  Examples::
    >>> adj_list = np.array([[1, 2], [3, 2], [1, 3], [2, 4], [5,3], [1, 5], [2, 6], [5,2]])
    >>> get_topological_order(adj_list)
  """
  if edge_direction != 'left<-right':
    # assume edge_direction == 'left->right' in adj_list
    adj_list = adj_list[:, [1,0]]
  # leaf nodes without children
  nodes = sorted(set(adj_list[:, 1]).difference(adj_list[:, 0]))
  chain_graph = [nodes]
  name_to_id = {n: i for i, n in enumerate(nodes)}
  # the remaining subgraph after removing leaf nodes
  subgraph = np.array([s for s in adj_list if s[1] not in nodes])
  while len(subgraph) > 0:
    nodes = sorted(set(subgraph[:, 1]).difference(subgraph[:, 0]))
    if len(nodes) == 0:
      # Hypothesis: if all children are also parents, there must be a cycle; 
      # I forgot how to prove it
      print('There are cycles!')
      return subgraph, name_to_id
    chain_graph.append(nodes)
    cur_size = len(name_to_id)
    for i, n in enumerate(nodes):
      name_to_id[n] = i + cur_size
    subgraph = np.array([s for s in subgraph if s[1] not in nodes])
  cur_size = len(name_to_id)
  nodes = sorted(set(adj_list[:,0]).difference(name_to_id))
  if len(nodes) > 0:
    chain_graph.append(nodes)
    for i, n in enumerate(nodes):
      name_to_id[n] = i + cur_size
  return name_to_id, chain_graph


# a helper function; 
# d is a dictionary with its values being lists;
# print the ordered length of its values
len_cnt = lambda d: sorted(collections.Counter(len(v) for v in d.values()).items())

def get_bigraph_dag(feature_ids, go_leaf_feature, go_edges, min_num_features=5):
  """Generate bigraph, dag, selected_feature_ids, name_to_id_feature, name_to_id_go
  
  Args:
    feature_ids: a list or 1-d np.array of feature_ids;
      even though we call it feature_ids, they are in fact feature names (i.e., gene symbols)
    go_leaf_feature: a dictionary with keys GO IDs and values lists of feature IDs
    go_edges: np.array of shape (N, 3) with columns being child, parent, key (is-a, etc.), child-->key-->parent
      all the keys (GO IDs) of go_leaf_feature must be the leaf nodes of go_edges
    min_num_features: used to filter out GO leaf nodes with less feature IDs
    
  Returns:
    bigraph: a list of list of ints
      each item (a list of int) of bigraph corresponds to one GO leaf node, 
        with name space name_to_id_go
      each int in an item corresponds to a feature ID, with name space name_to_id_feature
    dag: a dictionary with keys ints and values lists of ints
      all the ints correspond to GO IDs, with name space name_to_id_go
    name_to_id_feature: a dictionary, with keys feature IDs (names) and values int (0,1,2,...)
    name_to_id_go: same as name_to_id_feature, except keys are GO IDs  
    go_relation: a np.array of size (N, 2) with columns being parent <-- child;
      name_to_id_go and chain_graph_go had been generated from go_relation; return them for convenience
    chain_graph_go: a Partial DAG as chain graph;
      a list of lists; lower levels are decendants; level 0 are leaf nodes
    selected_feature_ids: a list of feature_ids, 
      with their IDs starting from 0 to len(name_to_id_feature)-1
      this is a redundant representation of name_to_id_feature; keep it for convenience
  """
  # only keep GO terms that have at least min_num_feature children from feature_ids
  # min_num_features = 5
  go_leaf_feature_selected = join_dict({v: [v] for v in feature_ids}, go_leaf_feature)
  go_leaf_feature_selected = {k: v for k, v in go_leaf_feature_selected.items() 
                              if len(v) >= min_num_features}
  # Prune the go_edges by only including the nodes in go_leaf_feature_selected;
  # Caveat: the argument graph of get_upper_closure has two columns being parent and child;
  # but go_edges has [child, parent, key]
  # now go_relation has two columns being parent and child, respectively
  go_relation = get_upper_closure(graph=go_edges[:, [1,0]], vertices=go_leaf_feature_selected)
  # make sure that go_relation have no duplicate rows
  go_relation = pandas.DataFrame(go_relation).drop_duplicates().values

  # prepare dag for GO
  name_to_id_go, chain_graph_go = get_topological_order(go_relation)
  # name_to_id_go follows the order to compute GO nodes;
  # therefore the leaf nodes must have the lowest ids
  assert (sorted([name_to_id_go[k] for k in go_leaf_feature_selected]) 
          == [i for i in range(len(go_leaf_feature_selected))])

  dag = collections.defaultdict(list)
  for s in go_relation:
    parent_id = name_to_id_go[s[0]]
    child_id = name_to_id_go[s[1]]
    dag[parent_id].append(child_id)
  dag = {k: sorted(set(v)) for k, v in dag.items()}

  # # I considered further pruning dag so that high-level GO terms have at least min_num_go children;
  # # but I then need to re-calculate go_relation and dag again;
  # # this is very complicated and buggy;
  # # this has to be done iteratively; 
  # # the following code only works partially because it is not iterative
  # min_num_go = 2
  # id_to_name_go = {v: k for k, v in name_to_id_go.items()}
  # selected_go_parents = {id_to_name_go[k] for k, v in dag.items() if len(v) >= min_num_go}
  # go_relation_filtered = np.array([s for s in go_relation if s[0] in selected_go_parents])
  # name_to_id_go, chain_graph_go = get_topological_order(go_relation_filtered)
  # go_leaf_feature_selected2 = {k: v for k, v in go_leaf_feature_selected.items() 
  #                              if k in chain_graph_go[0]}
  # go_relation_filtered = get_upper_closure(graph=go_relation_filtered, 
  #                                          vertices=go_leaf_feature_selected2)
  # name_to_id_go, chain_graph_go = get_topological_order(go_relation_filtered)
  # # name_to_id_go follows the order to compute GO nodes;
  # # therefore the leaf nodes must have the lowest ids
  # assert (sorted([name_to_id_go[k] for k in go_leaf_feature_selected2]) 
  #         == [i for i in range(len(go_leaf_feature_selected2))])
  # dag = collections.defaultdict(list)
  # for s in go_relation_filtered:
  #   parent_id = name_to_id_go[s[0]]
  #   child_id = name_to_id_go[s[1]]
  #   dag[parent_id].append(child_id)
  # dag = {k: sorted(v) for k, v in dag.items()}

  selected_feature_ids = get_itemset(keys=go_leaf_feature_selected, dic=go_leaf_feature_selected)
  name_to_id_feature = {v: i for i, v in enumerate(selected_feature_ids)}
  bigraph = {}
  for k, v in go_leaf_feature_selected.items():
    # this makes sure bigraph and dag are aligned well through using name_to_id_go
    bigraph[name_to_id_go[k]] = sorted([name_to_id_feature[j] for j in v])
  bigraph = [v for k, v in sorted(bigraph.items())]
  assert (sorted(collections.Counter(len(v) for v in bigraph).items()) 
          == len_cnt(go_leaf_feature_selected))
  
  return bigraph, dag, name_to_id_feature, name_to_id_go, go_relation, chain_graph_go, selected_feature_ids