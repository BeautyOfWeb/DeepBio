import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dag import StackedDAGLayers


class Factor1d(nn.Module):
  """Similar to masked attention
  
  """
  def __init__(self, in_features, in_dim, out_features, out_dim, adj_mat=None, bias=True):
    super(Factor1d, self).__init__()
    self.linear1 = nn.Linear(in_dim, out_dim, bias) # based on intuition, not justified
    self.linear2 = nn.Linear(out_dim, out_dim, bias)
    self.linear3 = nn.Linear(in_features, out_features, bias)
    self.linear4 = nn.Linear(out_features, out_features, bias)
    self.adj_mat = adj_mat

  def forward(self, x):
    out = F.relu(self.linear2(F.relu(self.linear1(x))).transpose(1, 2)) # (NxDxC -> NxCxD)
    if self.adj_mat is None:
      return self.linear4(F.relu(self.linear3(out))).transpose(1, 2)
    else:
      return self.linear4(F.relu(
        F.linear(out, self.linear3.weight*self.adj_mat.float(), self.linear3.bias))).transpose(1, 2)


class EmbedCell(nn.Module):
  r"""This is a bottleneck layer(s) using 1-D convolution layer(s) with kernel_size = 1
    The goal is to transform vectors in R^in_channels to R^out_channels 
    An nn.Conv1d is used to map its corresponding subset of source nodes for each target node
    It is essentially to a linear transformation; 
      1-D convolution with kernel_size=1 enables parameter sharing
  
  Args:
    in_channels: int
    out_channels: int for a single layer or a list/tuple of ints for multiple layers
    use_layer_norm: if True, apply nn.LayerNorm to each instance
    bias: whether or not to use bias in nn.Conv1d
    residual: only used for multiple layers; if True, add skip connections
    duplicate_cell: only used for multiple layers; 
      if True, all layers share the same parameters like recurrent neural networks
    nonlinearlity: None, nn.ReLU() or other nonlinearity; apply to output in the middle
      I have NOT figured out how to arrange the LayerNorm and nonlinearity and residual connections
    
  Shape:
    - Input: N * in_channels * M, where M = the number of input nodes
    - Output: N * out * M, where out = out_channels or out_channels[-1] (multiple layers)
    
  Attributes:
    weights (and biases) for a nn.Conv1d or a list of nn.Conv1d
    
  Examples::
  
    >>> x = torch.randn(2, 3, 5)
    >>> model = EmbedCell(3, [3,3], use_layer_norm=True, bias=True, 
               residual=True, duplicate_cell=True, nonlinearity=nn.ReLU())
    >>> y = model(x)
    >>> y.shape, y.mean(1), y.std(1, unbiased=False)
  """
  def __init__(self, in_channels, out_channels, use_layer_norm=True, bias=True, 
               residual=True, duplicate_cell=True, nonlinearity=None):
    super(EmbedCell, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.use_layer_norm = use_layer_norm
    self.bias = bias
    self.residual = residual
    self.duplicate_cell = duplicate_cell
    self.nonlinearity = nonlinearity
    if isinstance(out_channels, int):
      out_channels = [out_channels]
      self.out_channels = out_channels
    if isinstance(out_channels, (list, tuple)):
      if len(out_channels)>1 and (duplicate_cell or residual):
        for out in out_channels:
          assert out == in_channels
      if duplicate_cell:
        self.maps = nn.ModuleList([nn.Conv1d(in_channels, out_channels[0], kernel_size=1, bias=bias)] 
                                  * len(out_channels))
      else:
        self.maps = nn.ModuleList([nn.Conv1d(in_channels if i==0 else out_channels[i-1], 
                                             out, kernel_size=1, bias=bias)
                                  for i, out in enumerate(out_channels)])
      if self.use_layer_norm:
        # we can directly use torch.nn.functional.layer_norm in forward function without parameters
        self.layer_norms = nn.ModuleList(
          [nn.LayerNorm(out, eps=1e-5, elementwise_affine=False) 
           for out in out_channels]
        )
    else:
      raise ValueError(f'out_channels must have type int, list or tuple, but is {type(out_channels)}')
    
  def forward(self, x):
    for i in range(len(self.out_channels)):
      out = self.maps[i](x)
      # Should I put nonlinearity before layer_norm?
      if isinstance(self.nonlinearity, nn.Module):
        out = self.nonlinearity(out)
      if self.use_layer_norm:
        out = self.layer_norms[i](out.transpose(-1,-2)).transpose(-1,-2)
      if self.residual and i<len(self.out_channels)-1: # no residual in the last layer
        out += x
      x = out
    return out


class GraphConvolution1d(nn.Module):
  r"""Implement modified Graph Convolutional Neural Network
    Provide with options ResNet-like model with stochastic depth
    Fixed graph attention matrices generated from deterministic/random walk on the bipartite graph 
    We can use BipartiteGraph1d to implement much of this; but for clarity, write a separate class here
  
  Args:
    num_features: int
    num_layers: int
    duplicate_layers: if True, all layers will share the same parameters
    dense: if True, connect all previous layers to the current layer
    residual: if True, use skip connections; only used when dense is False
    use_bias: default, False
    use_layer_norm: if True, apply nn.LayerNorm to the output from each layer
    num_cls: if num_cls>=1, then add a classification/regression head on top of the last target layer 
      and return the final output
  
  Shape:
    Input:x is torch.Tensor of size (N, num_features)
          attention_mats can store a list of normalized adjacency matrices from current layers 
            to the nodes in previous layers; 
            in Graph Convolution Network paper, it only have one fixed first-order adjacency matrix;
            here it is enabled for using multi-scale reception field;
              Let M be the adjacency matrix from source to target (itself)
                attention_mats = [M.T, (M*M).T, (M*M*M).T, ...]
                these transition mats are normalized and transposed    
    Output: depending on return_layers: e.g., if return_layers=='all', then return torch.stack(history, dim=-1)
  
  Examples:
  
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    adj_mat, _ = adj_list_to_mat(adj_list, bipartite=False)
    in_features, out_features = adj_mat.shape
    attention_mats, _ = adj_list_to_attention_mats(adj_list, num_steps=10, bipartite=False)
    model = GraphConvolution1d(num_features=in_features, num_layers=5, duplicate_layers=False, 
      dense=False, residual=False, use_bias=True, use_layer_norm=False, nonlinearity=nn.ReLU(), 
      num_cls=2, classifier_bias=True)
    x = torch.randn(5, in_features)
    y = model(x, attention_mats, max_num_layers=10, min_num_layers=10, 
              return_layers='last-layer')
    y.shape

  """
  def __init__(self, num_features, num_layers, duplicate_layers=False, dense=False, residual=False, 
    use_bias=False, use_layer_norm=False, nonlinearity=nn.ReLU(), num_cls=0, classifier_bias=True):
    super(GraphConvolution1d, self).__init__()
    self.num_features = num_features
    self.num_layers = num_layers
    self.duplicate_layers = duplicate_layers
    self.dense = dense
    self.residual = residual
    self.use_bias = use_bias
    self.use_layer_norm = use_layer_norm
    self.nonlinearity = nonlinearity
    self.num_cls = num_cls
    if self.duplicate_layers:
      self.weights = nn.ParameterList([nn.Parameter(torch.randn(num_features, num_features), 
        requires_grad=True)]*self.num_layers)
      if self.use_bias:
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(num_features), 
          requires_grad=True)]*self.num_layers)
    else:
      self.weights = nn.ParameterList([nn.Parameter(torch.randn(num_features, num_features), 
        requires_grad=True) for _ in range(self.num_layers)])
      if self.use_bias:
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(num_features), 
          requires_grad=True) for _ in range(self.num_layers)])
    if self.use_layer_norm:
      self.layer_norm = nn.LayerNorm(num_features, eps=1e-05, elementwise_affine=False)
    if self.num_cls >= 1:
      self.classifier = nn.Linear(num_features, num_cls, bias=classifier_bias)
    
  def forward(self, x, attention_mats, max_num_layers=2, min_num_layers=2, return_layers='last-layer'):
    """
    Args:
      x: 2-D tensor with shape (N, num_features)
      attention_mats: normalized attention matrix with shape (num_features, num_features); 
        or a list of attention matrices
    """
    # stochastic depth; num_layers can even be larger than self.num_layers
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    history = [x] # the first layer is the original input
    for i in range(1, num_layers):
      if self.dense:
        y = [] # this is for i th layer; if self.dense is True, then connect all previous layers to current layer
        for j in range(i):
          if isinstance(attention_mats, list):
            adj = attention_mats[(i-j-1) % len(attention_mats)]
          else:
            adj = attention_mats
          # if num_layers > len(self.weights), we can reuse the weight by using j % len(self.weights)
          cur_y = torch.mm(history[j], self.weights[j % len(self.weights)] * adj)
          if self.use_bias:
            cur_y = cur_y + self.biases[j % len(self.biases)]
          y.append(cur_y)
        cur_y = torch.stack(y, dim=0).mean(dim=0)
      else:
        if isinstance(attention_mats, list):
          adj = attention_mats[0]
        else:
          adj = attention_mats
        cur_y = torch.mm(history[i-1], self.weights[(i-1) % len(self.weights)] * adj)
        if self.use_bias:
          cur_y = cur_y + self.biases[(i-1) % len(self.biases)]
      if isinstance(self.nonlinearity, nn.Module):
        cur_y = self.nonlinearity(cur_y)
      if self.residual:
        cur_y += history[i-1]
      if self.use_layer_norm:
        cur_y = self.layer_norm(cur_y)
      history.append(cur_y)

    if self.num_cls >= 1:
      return self.classifier(history[-1])
    if return_layers == 'last-layer':
      return history[-1]
    elif return_layers == 'all-but-first':
      # excluding the original input
      return torch.stack(history[1:], dim=-1)
    elif return_layers == 'all':
      return torch.stack(history, dim=-1)


class BipartiteGraph1d(nn.Module):
  r"""Encode a bipartite graph into the model architecture;
    ResNet-like model with stochastic depth
    Fixed graph attention matrices generated from deterministic/random walk on the bipartite graph 
  
  Args:
    in_features: int
    out_features: int
    use_layer_norm: if True, apply nn.LayerNorm to the output from each layer
    num_cls: if num_cls>=1, then add a classification/regression head on top of the last target layer 
      and return the final output
  
  Shape:
    Input: x is torch.Tensor of size (N, in_features)
            attention_mats = [source_attention_mats, target_attention_mats];
              source_attention_mats stores attention mats from source to the nodes in previous layers;
              target_attention_mats stores attention mats from target to the nodes in previous layers;
              Let Ms be the adjacency matrix from source to target, and Mt from target to source
                source_attention_mats = [Ms.T, (Ms*Mt).T, (Ms*Mt*Ms).T, ...],
                target_attention_mats = [Mt.T, (Mt*Ms).T, (Mt*Ms*Mt).T, ...];
                source_attention_mats stores transition mat from source with 1,2,... steps,
                target_attention_mats stores transition mat from target with 1,2,... steps,
                these transition mats are normalized and transposed    
  
  Examples:
  
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    adj_mat, _ = adj_list_to_mat(adj_list, bipartite=True)
    in_features, out_features = adj_mat.shape
    attention_mats, _ = adj_list_to_attention_mats(adj_list, num_steps=10, bipartite=True)
    model = BipartiteGraph1d(in_features=in_features, out_features=out_features, 
                             use_layer_norm=True)
    x = torch.randn(5, in_features)
    y = model(x, attention_mats, max_num_layers=10, min_num_layers=10, 
              return_layers='last-two')
    y[0].shape, y[1].shape

  """
  def __init__(self, in_features, out_features, nonlinearity=nn.ReLU(), use_layer_norm=True, num_cls=0,
    classifier_bias=True):
    super(BipartiteGraph1d, self).__init__()
    self.source_to_target = nn.Parameter(torch.randn(in_features, out_features), 
                                          requires_grad=True)
    self.target_to_source = nn.Parameter(torch.randn(out_features, in_features),
                                          requires_grad=True)
    self.nonlinearity = nonlinearity
    self.use_layer_norm = use_layer_norm
    if self.use_layer_norm:
      self.layer_norm_source = nn.LayerNorm(in_features, eps=1e-05, elementwise_affine=False)
      self.layer_norm_target = nn.LayerNorm(out_features, eps=1e-05, elementwise_affine=False)
    self.num_cls = num_cls
    if self.num_cls >= 1:
      self.classifier = nn.Linear(out_features, num_cls, bias=classifier_bias)
    
  def forward(self, x, attention_mats, max_num_layers=2, min_num_layers=2, return_layers='last-two'):
    # stochastic depth
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    if num_layers % 2 != 0:
      # make sure num_layers is even so that the last two layers are source and target
      num_layers += 1
    history = [x] # the first layer is the original input (source)
    for i in range(1, num_layers):
      y = [] # this is for i th layer; if i is even, then it is source; otherwise target
      for j in range(i):
        if i%2 == 0 and j%2 == 0: # both i and j are sources
          # # if attention_mats is too big, we may only store two of them, 
          # # thus disabling multi-scale long-range interaction; 
          # # this is why (i-j-1) % len(attention_mats[0]) instead of i-j-1 is used
          # # to avoid size mismatch:
          # assert len(attention_mats[0]) % 2 == 0 and len(attention_mats[1]) % 2 == 0
          y.append(torch.mm(history[j], attention_mats[0][(i-j-1) % len(attention_mats[0])]))
        elif i%2 == 0 and j%2 != 0: # j is target, i is source
          y.append(torch.mm(history[j], 
            self.target_to_source * attention_mats[0][(i-j-1) % len(attention_mats[0])]))  
        elif i%2 != 0 and j%2 == 0: # j is source, i is target
          y.append(torch.mm(history[j], 
            self.source_to_target * attention_mats[1][(i-j-1) % len(attention_mats[1])]))
        else: # both i and j are targets
          y.append(torch.mm(history[j], attention_mats[1][(i-j-1) % len(attention_mats[1])]))
      y = torch.stack(y, dim=0).mean(dim=0)
      if isinstance(self.nonlinearity, nn.Module):
        y = self.nonlinearity(y)
      if self.use_layer_norm:
        if i%2 == 0: # even numbers are source
          y = self.layer_norm_source(y)
        else: # odd numbers are target
          y = self.layer_norm_target(y)
      history.append(y)
    
    if self.num_cls >= 1:
      return self.classifier(history[-1])
    if return_layers == 'last-target':
      return history[-1]
    elif return_layers == 'last-two':
      return history[-2:]
    elif return_layers == 'all-source-target':
      # source.size() = (N, in_features, num_layers/2)
      source = torch.stack([history[i] for i in range(num_layers) if i%2==0], dim=-1)
      # target.size() = (N, out_features, num_layers/2)
      target = torch.stack([history[i] for i in range(num_layers) if i%2!=0], dim=-1)
      return source, target
    elif return_layers == 'all':
      return history


class BipartiteGraph(nn.Module):
  r"""Encode a bipartite graph into the model architecture;
    ResNet-like model with stochastic depth
    Fixed graph attention matrices generated from deterministic/random walk on the bipartite graph 
  
  Args:
    in_features: int
    out_features: int
    in_dim: int
    out_dim: int
    use_layer_norm: if True, apply nn.LayerNorm to the output from each layer
  
  Shape:
    Input: x is torch.Tensor of size (N, in_dim, in_features)
            attention_mats = [source_attention_mats, target_attention_mats];
              source_attention_mats stores attention mats from source to the nodes in previous layers;
              target_attention_mats stores attention mats from target to the nodes in previous layers;
              Let Mt be the adjacency matrix from source to target, and Ms from target to source;
              in the obsolete version:
                source_attention_mats = [Ms, Mt*Ms, Ms*Mt*Ms, ...],
                target_attention_mats = [Mt, Ms*Mt, Mt*Ms*Mt, ...];
                source_attention_mats are to reach source with 1,2,... steps,
                target_attention_mats are to reach target
              in the CURRENT version:
                source_attention_mats = [Mt.T, (Mt*Ms).T, (Mt*Ms*Mt).T, ...],
                target_attention_mats = [Ms.T, (Ms*Mt).T, (Ms*Mt*Ms).T, ...];
                source_attention_mats stores transition mat from source with 1,2,... steps,
                target_attention_mats stores transition mat from target with 1,2,... steps,
                these transition mats are transposed
  
  Examples:
  
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    attention_mats, _ = adj_list_to_attention_mats(adj_list, num_steps=10, bipartite=True,
                                                  use_transition_matrix=True)                               
    model = BipartiteGraph(in_features, out_features, in_dim=5, out_dim=11, 
                          use_layer_norm=True)
    x = torch.randn(7, 5, 3)
    y = model(x, attention_mats, max_num_layers=10, min_num_layers=8, 
              return_layers='last-two')
    y[0].shape, y[1].shape
  
  """
  def __init__(self, in_features, out_features, in_dim, out_dim, use_layer_norm=True):
    super(BipartiteGraph, self).__init__()
    self.source_to_target = nn.Parameter(torch.randn(in_dim, in_features, out_features, out_dim), 
                                          requires_grad=True)
    self.target_to_source = nn.Parameter(torch.randn(out_dim, out_features, in_features, in_dim),
                                          requires_grad=True)
    self.use_layer_norm = use_layer_norm
    if self.use_layer_norm:
      self.layer_norm_source = nn.LayerNorm([in_dim, in_features], eps=1e-05, elementwise_affine=False)
      self.layer_norm_target = nn.LayerNorm([out_dim, out_features], eps=1e-05, elementwise_affine=False)
  
  def forward(self, x, attention_mats, max_num_layers=2, min_num_layers=2, 
              return_layers='last-two'):
    # stochastic depth
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    if num_layers % 2 != 0:
      # make sure num_layers is even so that the last two layers are source and target
      num_layers += 1
    history = [x] # the first layer is the original input (source)
    for i in range(1, num_layers):
      y = [] # this is for i th layer; if i is even, then it is source; otherwise target
      for j in range(i):
        if i%2 == 0 and j%2 == 0: # both i and j are sources
          # # if attention_mats is too big, we may only store two of them, or four, six, eight, ..., of them,
          # # thus disabling multi-scale long-range interaction and saves memeory;
          # # this is why (i-j-1) % len(attention_mats[0]) instead of i-j-1 is used
          # # to avoid size mismatch:
          # assert len(attention_mats[0]) % 2 == 0 and len(attention_mats[1]) % 2 == 0
          y.append(torch.matmul(history[j], attention_mats[0][(i-j-1) % len(attention_mats[0])]))
        elif i%2 == 0 and j%2 != 0: # j is target, i is source
          weight = self.target_to_source * attention_mats[0][(i-j-1) % len(attention_mats[0])].unsqueeze(-1)
          new_y = (history[j].unsqueeze(-1).unsqueeze(-1) * weight).sum(dim=1).sum(dim=1).transpose(1,2)
          y.append(new_y)  
        elif i%2 != 0 and j%2 == 0: # j is source, i is target
          weight = self.source_to_target * attention_mats[1][(i-j-1) % len(attention_mats[1])].unsqueeze(-1)
          new_y = (history[j].unsqueeze(-1).unsqueeze(-1) * weight).sum(dim=1).sum(dim=1).transpose(1,2)
          y.append(new_y)
        else: # both i and j are targets
          y.append(torch.matmul(history[j], attention_mats[1][(i-j-1) % len(attention_mats[1])]))
      y = torch.stack(y, dim=0).mean(dim=0)
      if self.use_layer_norm:
        if i%2 == 0: # even numbers are source
          y = self.layer_norm_source(y)
        else: # odd numbers are target
          y = self.layer_norm_target(y)
      history.append(y)
    if return_layers == 'last-target':
      return history[-1]
    elif return_layers == 'last-two':
      return history[-2:]
    elif return_layers == 'all-source-target':
      # source.size() = (N, in_features, num_layers/2)
      source = torch.stack([history[i] for i in range(num_layers) if i%2==0], dim=-1)
      # target.size() = (N, out_features, num_layers/2)
      target = torch.stack([history[i] for i in range(num_layers) if i%2!=0], dim=-1)
      return source, target
    elif return_layers == 'all':
      return history


class GeneNet(nn.Module):
  r"""Gene-Pathway(GO) network: gene0->gene1->pathway0->pathway1->gene0->...

  Args:
    num_genes: int
    num_pathways: int
    attention_mats: if provided, it should be a dictionary with keys:
      'gene1->gene0': a list of the attention mats from genes to genes; 
        the computation is from gene0->gene1
      'pathway0->gene1': a list of the attention mats from pathways to genes;
        the computation is from gene1->pathway0
      'pathway1->pathway0': a list of the attention mats from pathways to pathways;
        the computation is from pathway0->pathway1
      'gene0->pathway1': a list of the attention mats from genes to pathways;
        the computation is from pathway1->gene0
    dense: if True, add skip connections from all previous layers to current layer
    nonlinearity: if provided as nn.Module, then apply it to output
    use_layer_norm: if True, apply layer_norm to output
      Currently, I put nonlinearity before layer norm;
        Should I put nonlinearity before layer norm or otherwise?
    num_cls: if num_cls>=1, then add an classifier or regression head using the pathway1-last-layer output as input;
      otherwise do nothing

  Shape:
    Input: x: (N, num_genes)
      attention_mats: see class doc
    Output: if return_layers=='all'
      return a dictionary with four keys: 'gene0', 'gene1', 'pathway0', 'pathway1', 
        the values have shape (N, num_genes/pathways, num_layers)

  Examples:

    attention_mats = {}
    num_steps = 10
    num_genes = 23
    num_pathways = 11
    name_to_id_gene = {i: i for i in range(num_genes)}
    p = 0.4
    gene_gene_mat = np.random.uniform(0, 1, (num_genes, num_genes))
    gene_gene_list = np.array(np.where(gene_gene_mat < p)).T
    # adj_list_to_mat(gene_gene_list, name_to_id=name_to_id_gene, add_self_loop=True, symmetric=True,
    #                 bipartite=False)
    attention_mats['gene1->gene0'], id_to_name_gene = adj_list_to_attention_mats(
      gene_gene_list, num_steps=num_steps, name_to_id=name_to_id_gene, bipartite=False, 
      add_self_loop=True, symmetric=True, target_to_source=None, use_transition_matrix=True, 
      softmax_normalization=False, min_value=-100, device=torch.device('cpu'))

    pathway_pathway_list = np.array([[1, 2], [3, 2], [1, 3], [2, 4], [5,3], [1, 5], [2, 6], [5,2]])
    name_to_id_pathway, _ = get_topological_order(pathway_pathway_list, 
                                                  edge_direction='left->right')
    for i in range(num_pathways):
      if i not in name_to_id_pathway:
        name_to_id_pathway[i] = len(name_to_id_pathway)
    dag = collections.defaultdict(list)
    for s in pathway_pathway_list:
      left = name_to_id_pathway[s[0]]
      right = name_to_id_pathway[s[1]]
      dag[right].append(left)
    dag = {k: sorted(set(v)) for k, v in dag.items()}
    attention_mats['pathway1->pathway0'], id_to_name_pathway = adj_list_to_attention_mats(
      pathway_pathway_list, num_steps=num_steps, name_to_id=name_to_id_pathway, bipartite=False, 
      add_self_loop=False, symmetric=False, target_to_source=None, use_transition_matrix=True, 
      softmax_normalization=False, min_value=-100, device=torch.device('cpu'))

    gene_pathway_mat = np.random.uniform(0, 1, (num_genes, num_pathways))
    gene_pathway_list = np.array(np.where(gene_pathway_mat < p)).T
    # adj_list_to_mat(gene_pathway_list, name_to_id=[name_to_id_gene, name_to_id_pathway], 
    #                 bipartite=True)
    mats, _ = adj_list_to_attention_mats(
      gene_pathway_list, num_steps=num_steps*2, name_to_id=[name_to_id_gene, name_to_id_pathway], 
      bipartite=True, add_self_loop=False, symmetric=False, target_to_source=None, 
      use_transition_matrix=True, softmax_normalization=False, min_value=-100, 
      device=torch.device('cpu'))
    # this is very tricky: 
    # the even positions are all gene->pathway in mats[0], while odd ones gene->gene
    attention_mats['gene0->pathway1'] = [m for i, m in enumerate(mats[0]) if i%2==0]
    attention_mats['pathway0->gene1'] = [m for i, m in enumerate(mats[1]) if i%2==0]

    model = GeneNet(num_genes, num_pathways, attention_mats=None, dense=True, use_dag_layer=True,
                    dag=dag, dag_in_channel_list=[1,1,1], 
                    dag_kwargs={'residual':True, 'duplicate_dag':True}, nonlinearity=nn.ReLU(), 
                    use_layer_norm=True)

    x = torch.randn(5, num_genes)
    y = model(x, attention_mats=attention_mats, max_num_layers=num_steps, min_num_layers=num_steps, 
              return_layers='all')
    y[0].shape, y[1].shape, y[2].shape, y[3].shape
  """

  def __init__(self, num_genes, num_pathways, attention_mats=None, dense=True, use_dag_layer=False, 
    dag=None, dag_in_channel_list=[1], dag_kwargs={}, nonlinearity=nn.ReLU(), use_layer_norm=True, 
    num_cls=0, classifier_bias=True):
    super(GeneNet, self).__init__()
    self.weights = nn.ParameterDict()
    self.weights['gene0->gene1'] = nn.Parameter(torch.randn(num_genes, num_genes))
    self.weights['gene1->pathway0'] = nn.Parameter(torch.randn(num_genes, num_pathways))
    self.weights['pathway0->pathway1'] = nn.Parameter(torch.randn(num_pathways, num_pathways))
    self.weights['pathway1->gene0'] = nn.Parameter(torch.randn(num_pathways, num_genes))
    self.dense = dense
    self.nonlinearity = nonlinearity
    self.use_layer_norm = use_layer_norm
    self.attention_mats = attention_mats
    self.use_dag_layer = use_dag_layer and dag is not None
    if self.use_dag_layer:
      self.dag_layers = StackedDAGLayers(dag=dag, in_channels_list=dag_in_channel_list, **dag_kwargs)
    self.num_cls = num_cls
    if num_cls>=1:
      self.classifier = nn.Linear(num_pathways, num_cls, bias=classifier_bias)

  def forward_one_layer(self, history, attention_mats, in_name, out_name, i, j):
    # print(in_name, out_name, i, j)
    ## use j%len_attention_mats instead of j so that we can forward more steps beyond the range of attention_mats
    len_attention_mats = len(attention_mats[f'{out_name}->{in_name}'])
    x = torch.mm(history[in_name][i], 
      self.weights[f'{in_name}->{out_name}'] * attention_mats[f'{out_name}->{in_name}'][j%len_attention_mats])
    if isinstance(self.nonlinearity, nn.Module):
      x = self.nonlinearity(x)
    if self.use_layer_norm:
      x = nn.functional.layer_norm(x, (x.size(-1),), weight=None, bias=None, eps=1e-5)
    return x

  def forward(self, x, attention_mats=None, max_num_layers=2, min_num_layers=2, 
    return_layers='pathway1-last-layer'):
    """
    Args:
      x: (N, num_genes)
      attention_mats: see class doc; if provided here use this instead of self.attention_mats
      max_num_layers: int
      min_num_layers: int
      return_layers: only used when self.num_cls <= 1;
        when self.num_cls > 1, then return classification score matrix instead
    """
    if attention_mats is None:
      assert self.attention_mats is not None
      attention_mats = self.attention_mats
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    history = {'gene0': [x], 'gene1': [], 'pathway0': [], 'pathway1': []}
    for l in range(num_layers):
      gene0 = []
      gene1 = []
      pathway0 = []
      pathway1 = []
      if self.dense:
        start = 0
      else:
        start = l
      for j in range(start, l+1):
        x = self.forward_one_layer(history, attention_mats, 'gene0', 'gene1', j, l-j)
        gene1.append(x)
      if self.dense:
        history['gene1'].append(torch.stack(gene1, dim=-1).mean(dim=-1))
      else:
        history['gene1'].append(gene1[-1])
        
      for j in range(start, l+1):
        x = self.forward_one_layer(history, attention_mats, 'gene1', 'pathway0', j, l-j)
        pathway0.append(x)
      if self.dense:
        history['pathway0'].append(torch.stack(pathway0, dim=-1).mean(dim=-1))
      else:
        history['pathway0'].append(pathway0[-1])

      for j in range(start, l+1):  
        x = self.forward_one_layer(history, attention_mats, 'pathway0', 'pathway1', j, l-j)
        if self.use_dag_layer:
          x = self.dag_layers(x)
        pathway1.append(x)
      if self.dense:
        history['pathway1'].append(torch.stack(pathway1, dim=-1).mean(dim=-1))
      else:
        history['pathway1'].append(pathway1[-1])

      if l < num_layers-1:
        for j in range(start, l+1):  
          x = self.forward_one_layer(history, attention_mats, 'pathway1', 'gene0', j, l-j)
          gene0.append(x)
        if self.dense:
          history['gene0'].append(torch.stack(gene0, dim=-1).mean(dim=-1))
        else:
          history['gene0'].append(gene0[-1])
    if self.num_cls >= 1:
      cls_score = self.classifier(history['pathway1'][-1])
      return cls_score
    if return_layers=='all':
      return (torch.stack(history['gene0'], dim=-1), torch.stack(history['gene1'], dim=-1), 
        torch.stack(history['pathway0'], dim=-1), torch.stack(history['pathway1'], dim=-1))
    if return_layers=='pathway1-all':
      return torch.stack(history['pathway1'], dim=-1)
    if return_layers=='pathway1-last-layer':
      return history['pathway1'][-1]
    if return_layers=='gene1-all':
      return torch.stack(history['gene1'], dim=-1)
    if return_layers=='gene1-last-layer':
      return history['gene1'][-1]


class PathNet(nn.Module):
  r"""Gene-Pathway(GO) network: gene->pathway0->pathway1->gene->...
  The only difference between PathNet and GeneNet is PathNet do not have gene-gene interaction data available;
  so gene0->gene1 was replaced by gene in PathNet

  Args:
    num_genes: int
    num_pathways: int
    attention_mats: if provided, it should be a dictionary with keys:
      'pathway0->gene': a list of the attention mats from pathways to genes;
        the computation is from gene->pathway0
      'pathway1->pathway0': a list of the attention mats from pathways to pathways;
        the computation is from pathway0->pathway1
      'gene->pathway1': a list of the attention mats from genes to pathways;
        the computation is from pathway1->gene
    dense: if True, add skip connections from all previous layers to current layer
    nonlinearity: if provided as nn.Module, then apply it to output
    use_layer_norm: if True, apply layer_norm to output
      Currently, I put nonlinearity before layer norm;
        Should I put nonlinearity before layer norm or otherwise?
    num_cls: if num_cls>=1, then add an classifier or regression head using the pathway1-last-layer output as input;
      otherwise do nothing

  Shape:
    Input: x: (N, num_genes)
      attention_mats: see class doc
    Output: if return_layers=='all'
      return a dictionary with three keys: 'gene', 'pathway0', 'pathway1', 
        the values have shape (N, num_genes/pathways, num_layers)

  Examples:

    attention_mats = {}
    num_steps = 10
    num_genes = 23
    num_pathways = 11
    name_to_id_gene = {i: i for i in range(num_genes)}
    pathway_pathway_list = np.array([[1, 2], [3, 2], [1, 3], [2, 4], [5,3], [1, 5], [2, 6], [5,2]])
    name_to_id_pathway, _ = get_topological_order(pathway_pathway_list, 
                                                  edge_direction='left->right')
    for i in range(num_pathways):
      if i not in name_to_id_pathway:
        name_to_id_pathway[i] = len(name_to_id_pathway)
    dag = collections.defaultdict(list)
    for s in pathway_pathway_list:
      left = name_to_id_pathway[s[0]]
      right = name_to_id_pathway[s[1]]
      dag[right].append(left)
    dag = {k: sorted(set(v)) for k, v in dag.items()}
    attention_mats['pathway1->pathway0'], id_to_name_pathway = adj_list_to_attention_mats(
      pathway_pathway_list, num_steps=num_steps, name_to_id=name_to_id_pathway, bipartite=False, 
      add_self_loop=False, symmetric=False, target_to_source=None, use_transition_matrix=True, 
      softmax_normalization=False, min_value=-100, device=torch.device('cpu'))
    
    p = 0.4
    gene_pathway_mat = np.random.uniform(0, 1, (num_genes, num_pathways))
    gene_pathway_list = np.array(np.where(gene_pathway_mat < p)).T
    # adj_list_to_mat(gene_pathway_list, name_to_id=[name_to_id_gene, name_to_id_pathway], 
    #                 bipartite=True)
    mats, _ = adj_list_to_attention_mats(
      gene_pathway_list, num_steps=num_steps*2, name_to_id=[name_to_id_gene, name_to_id_pathway], 
      bipartite=True, add_self_loop=False, symmetric=False, target_to_source=None, 
      use_transition_matrix=True, softmax_normalization=False, min_value=-100, 
      device=torch.device('cpu'))
    # this is very tricky: 
    # the even positions are all gene->pathway in mats[0], while odd ones gene->gene
    attention_mats['gene->pathway1'] = [m for i, m in enumerate(mats[0]) if i%2==0]
    attention_mats['pathway0->gene'] = [m for i, m in enumerate(mats[1]) if i%2==0]

    model = PathNet(num_genes, num_pathways, attention_mats=None, dense=True, use_dag_layer=True,
                    dag=dag, dag_in_channel_list=[1,1,1], 
                    dag_kwargs={'residual':True, 'duplicate_dag':True}, nonlinearity=nn.ReLU(), 
                    use_layer_norm=True, num_cls=0)

    x = torch.randn(5, num_genes)
    y = model(x, attention_mats=attention_mats, max_num_layers=num_steps, min_num_layers=num_steps, 
              return_layers='all')
    y[0].shape, y[1].shape, y[2].shape
  """

  def __init__(self, num_genes, num_pathways, attention_mats=None, dense=True, use_dag_layer=False, 
    dag=None, dag_in_channel_list=[1], dag_kwargs={}, nonlinearity=nn.ReLU(), use_layer_norm=True, 
    num_cls=0, classifier_bias=True):
    super(PathNet, self).__init__()
    self.weights = nn.ParameterDict()
    self.weights['gene->pathway0'] = nn.Parameter(torch.randn(num_genes, num_pathways))
    self.weights['pathway0->pathway1'] = nn.Parameter(torch.randn(num_pathways, num_pathways))
    self.weights['pathway1->gene'] = nn.Parameter(torch.randn(num_pathways, num_genes))
    self.dense = dense
    self.nonlinearity = nonlinearity
    self.use_layer_norm = use_layer_norm
    self.attention_mats = attention_mats
    self.use_dag_layer = use_dag_layer and dag is not None
    if self.use_dag_layer:
      self.dag_layers = StackedDAGLayers(dag=dag, in_channels_list=dag_in_channel_list, **dag_kwargs)
    self.num_cls = num_cls
    if num_cls>=1:
      self.classifier = nn.Linear(num_pathways, num_cls, bias=classifier_bias)

  def forward_one_layer(self, history, attention_mats, in_name, out_name, i, j):
    # print(in_name, out_name, i, j)
    ## use j%len_attention_mats instead of j so that we can forward more steps beyond the range of attention_mats
    len_attention_mats = len(attention_mats[f'{out_name}->{in_name}'])
    x = torch.mm(history[in_name][i], 
      self.weights[f'{in_name}->{out_name}'] * attention_mats[f'{out_name}->{in_name}'][j%len_attention_mats])
    if isinstance(self.nonlinearity, nn.Module):
      x = self.nonlinearity(x)
    if self.use_layer_norm:
      x = nn.functional.layer_norm(x, (x.size(-1),), weight=None, bias=None, eps=1e-5)
    return x

  def forward(self, x, attention_mats=None, max_num_layers=2, min_num_layers=2, 
    return_layers='pathway1-last-layer'):
    """
    Args:
      x: (N, num_genes)
      attention_mats: see class doc; if provided here use this instead of self.attention_mats
      max_num_layers: int
      min_num_layers: int
      return_layers: only used when self.num_cls <= 1;
        when self.num_cls > 1, then return classification score matrix instead
    """
    if attention_mats is None:
      assert self.attention_mats is not None
      attention_mats = self.attention_mats
    num_layers = np.random.randint(min_num_layers, max_num_layers+1)
    history = {'gene': [x], 'pathway0': [], 'pathway1': []}
    for l in range(num_layers):
      gene = []
      pathway0 = []
      pathway1 = []
      if self.dense:
        start = 0
      else:
        start = l

      for j in range(start, l+1):
        x = self.forward_one_layer(history, attention_mats, 'gene', 'pathway0', j, l-j)
        pathway0.append(x)
      if self.dense:
        history['pathway0'].append(torch.stack(pathway0, dim=-1).mean(dim=-1))
      else:
        history['pathway0'].append(pathway0[-1])

      for j in range(start, l+1):  
        x = self.forward_one_layer(history, attention_mats, 'pathway0', 'pathway1', j, l-j)
        if self.use_dag_layer:
          x = self.dag_layers(x)
        pathway1.append(x)
      if self.dense:
        history['pathway1'].append(torch.stack(pathway1, dim=-1).mean(dim=-1))
      else:
        history['pathway1'].append(pathway1[-1])

      if l < num_layers-1:
        for j in range(start, l+1):  
          x = self.forward_one_layer(history, attention_mats, 'pathway1', 'gene', j, l-j)
          gene.append(x)
        if self.dense:
          history['gene'].append(torch.stack(gene, dim=-1).mean(dim=-1))
        else:
          history['gene'].append(gene[-1])
    if self.num_cls >= 1:
      cls_score = self.classifier(history['pathway1'][-1])
      return cls_score
    if return_layers=='all':
      return (torch.stack(history['gene'], dim=-1), 
        torch.stack(history['pathway0'], dim=-1), torch.stack(history['pathway1'], dim=-1))
    if return_layers=='pathway1-all':
      return torch.stack(history['pathway1'], dim=-1)
    if return_layers=='pathway1-last-layer':
      return history['pathway1'][-1]
    if return_layers=='gene-all':
      return torch.stack(history['gene'], dim=-1)
    if return_layers=='gene-last-layer':
      return history['gene'][-1]