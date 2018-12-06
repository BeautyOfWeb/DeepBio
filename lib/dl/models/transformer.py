import numpy as np

import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
  r"""Key-value multi-head attention

  Args:
    in_dim: int, input feature dimension, i.e., x.size(-1)
    out_dim: int, output feature dimension, i.e., out.size(-1)
    key_dim: int, the feature dimension for keys, i.e., key.size(-1)
    value_dim: int, the feature dimension for the output of each head, i.e., value.size(-1)
    num_heads: int
    mask: if True, mask the right side of the information when applying self-attention
    query_in_dim: default None, not used; 
      if given as an int (only used when query is not None when calling forward()),
        then self.query_keys will be used for calculating keys for query provided in forward();
        thus query_in_dim == query.size(-1)
    knn: default None, not used; if given as an int, then apply k-nearest-neighbor attention pooling
    graph: a 2-d array of size (input_seq_length, input_seq_length); 
      graph_ij (ith row, jth column) is the unnormalized attention from i to j;
      this graph can be derived from variable structure;
      e.g., if the input is a sequence of GO terms, then this graph can be derived from Gene Ontology graph;
      whether or not to mask the right side can be encoded in this graph as well
    graph_weight: a float in [0, 1]; only used when graph is given; 
      calculate a weighted attention combining provided graph attention and computed key-value attention

  Shape:
    Input: (N, input_seq_length, in_dim)
    Output: 
      if return_attention is False: 
        return y of size (N, query_seq_length, out_dim)
      else:
        return y and attention_mats of size (num_heads, N, query_seq_length, input_seq_length)
      for self-attention, query_seq_length = input_seq_length = x.size(1)

  Examples:
    x = torch.randn(2,3,5)
    y = torch.randn(2,4,4)
    graph = torch.randn(4, 3)
    model = MultiheadAttention(in_dim=5, out_dim=7, key_dim=9, value_dim=11, num_heads=2, mask=True, 
                  query_in_dim=4, knn=11, graph=None, graph_weight=1)
    model(x, query=y, return_attention=True, graph=graph)[1].sum(dim=-1)
  """

  def __init__(self, in_dim, out_dim, key_dim, value_dim, num_heads=1, mask=False, 
               query_in_dim=None, knn=None, graph=None, graph_weight=0.5):
    super(MultiheadAttention, self).__init__()
    self.key_dim = key_dim
    self.keys = nn.ModuleList([nn.Linear(in_dim, key_dim) for i in range(num_heads)])
    if query_in_dim is not None:
      self.query_keys = nn.ModuleList([nn.Linear(query_in_dim, key_dim) 
                                       for i in range(num_heads)])
    self.values = nn.ModuleList([nn.Linear(in_dim, value_dim) for i in range(num_heads)])
    self.out = nn.Linear(value_dim*num_heads, out_dim)
    self.mask = mask
    self.knn = knn
    self.graph = graph
    if self.graph is not None:
      self.graph = nn.Parameter(data=torch.tensor(self.graph).float(), requires_grad=False)
      # make sure the weights in each row sum to 1 as the node attention
      self.graph.data = nn.functional.softmax(self.graph, dim=-1)
    self.graph_weight = graph_weight
    assert self.graph_weight >= 0 and self.graph_weight <= 1
    
  def forward(self, x, query=None, return_attention=False, graph=None):
    """Forward pass
    Args:
      x: in almost all cases, assume x.size() = (N, seq_len, in_dim); 
        otherwise there might be bugs in the code; 
        I did not handle this in order to save computation for the most commonly used case
      query: if None, then apply self-attention;
      return_attention: if True, return adjacency matrices of normalized attention
      graph: if graph is not None and self.knn is not None, then overwrite self.graph and prepare att_graph
    """
    if self.knn is not None and graph is not None:
      # this graph will overwrite the graph provided when initializing the model
      if isinstance(graph, torch.Tensor):
        att_graph = graph # assume the provided graph already normalized 
      else:
        att_graph = x.new_tensor(graph)
        # make sure the weights in each row sum to 1 as the node attention
        att_graph = nn.functional.softmax(att_graph, dim=-1)
    else:
      att_graph = self.graph
    y = []
    if return_attention:
      attention_mats = []
    self_attention = True
    if query is not None:
      #assert self.knn is None or self.knn <= x.size(-2) # found bug here, not clear why yet
      size_x = x.size()
      size_q = query.size()
      x = x.contiguous().view(size_x[0], -1, size_x[-1])
      input_query = query.contiguous().view(size_q[0], -1, size_q[-1])
      self_attention = False
      self.mask = False # self.mask is only used for self-attention
    for i, (K, V) in enumerate(zip(self.keys, self.values)):
      key = K(x)
      value = V(x)
      if self_attention:
        query = key
      else:
        if hasattr(self, 'query_keys'):
          query = self.query_keys[i](input_query)
        else:
          # assert self.in_dim == input_query.size(-1)
          query = K(input_query)
      # this is based on the paper "Attention is all you need";
      # assert query.dim()==3 and key.dim()==3 # otherwise it is probably wrong
      att_unnorm = (query.unsqueeze(-2) * key.unsqueeze(-3)).sum(-1) / np.sqrt(self.key_dim)
      if self.mask: # mask right side; useful for decoder with sequential output
        seq_len = att_unnorm.size(-2)
        if att_unnorm.dim() == 3:
          for i in range(seq_len-1):
            att_unnorm[:, i, (i+1):] = float('-inf')
        elif att_unnorm.dim() == 4:
          # rarely used
          for i in range(seq_len-1):
            att_unnorm[:, :, i, (i+1):] = float('-inf')
        else:
          raise ValueError('Expect x.dim() <= 4, but x.dim() = {0}'.format(x.dim()))
      if isinstance(self.knn, int):
        self.knn = min(self.knn, att_unnorm.size(-1))
        att_topk, idx = att_unnorm.topk(self.knn, dim=-1)
        att_ = att_unnorm.new_zeros(att_unnorm.size()).fill_(float('-inf'))
        att_.scatter_(-1, idx, att_topk)
        att_unnorm = att_
      att = nn.functional.softmax(att_unnorm, dim=-1)
      if self.knn is not None and att_graph is not None:
        att = self.graph_weight*att_graph + (1-self.graph_weight)*att
      if return_attention:
        attention_mats.append(att)
      # tricky
      cur_y = (att.unsqueeze(-1) * value.unsqueeze(-3)).sum(-2)
      if not self_attention:
        cur_y = cur_y.contiguous().view(*size_q[:-1], cur_y.size(-1))
      y.append(cur_y)   
    y = torch.cat(y, -1)
    y = self.out(y)
    if return_attention:
      attention_mats = torch.stack(attention_mats, dim=0)
      return y, attention_mats
    return y

  
class EncoderAttention(nn.Module):
  r"""MultiheadAttention layer plus fully connected layer (and normalization layer)
  
  Args:
    most arguments are passed to MultiheadAttention
    fc_dim: Following MultiheadAttention layer is a two-layer perceptron with hidden dim as fc_dim;
      this provides a nonlinear transformation
    residual: if True, then assert in_dim==out_dim; follow the transformer model
    use_layer_norm: default True; if True, then apply nn.LayerNorm to the output before return;
      this follows the transformer model

  Shape:
    Input: (N, input_seq_length, in_dim)
    Output: 
      if return_attention is False:
        return y of size (N, query_seq_length, out_dim)
      else:
        return y and attention_mats of size (num_heads, N, query_seq_length, input_seq_length)
      for self-attention, query_seq_length=input_seq_length=x.size(1)

  Examples:
    x = torch.randn(2,3,5)
    model = EncoderAttention(in_dim=5, out_dim=5, key_dim=9, value_dim=11, fc_dim=13, num_heads=3, residual=True, 
      use_layer_norm=True, nonlinearity=nn.ReLU(), mask=True, query_in_dim=None, knn=2)
    model(x, query=None, return_attention=True)[1].sum(-1)
  """
  def __init__(self, in_dim, out_dim, key_dim, value_dim, fc_dim, num_heads=1, residual=True,
              use_layer_norm=True, nonlinearity=nn.ReLU(), mask=False, query_in_dim=None, knn=None,
              graph=None, graph_weight=0.5):
    super(EncoderAttention, self).__init__()
    self.attention = MultiheadAttention(in_dim, out_dim, key_dim, value_dim, num_heads=num_heads, mask=mask, 
                                        query_in_dim=query_in_dim, knn=knn, graph=graph, graph_weight=graph_weight)
    self.residual = residual
    self.use_layer_norm = use_layer_norm
    self.fc = nn.Sequential(nn.Linear(out_dim, fc_dim),
                           nonlinearity,
                           nn.Linear(fc_dim, out_dim))
    if self.residual:
      assert in_dim == out_dim
    if self.use_layer_norm:
      self.layer_norm = nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=False)

  def forward(self, x, query=None, return_attention=False, graph=None):
    if return_attention:
      out, attention_mats = self.attention(x, query=query, return_attention=True, graph=graph)
    else:
      out = self.attention(x, query=query, return_attention=False, graph=graph)
    # this closely follows the transformer model from the paper "attention is all you need"
    if self.residual:
      # assert x.size() == out.size() # omit this to save computatation time
      out += x
    if self.use_layer_norm:
      out = self.layer_norm(out)
    x = self.fc(out)
    if self.residual:
      x += out
    if self.use_layer_norm:
      x = self.layer_norm(x)
    if return_attention:
      return x, attention_mats
    else:
      return x
  

class DecoderAttention(nn.Module):
  """Similar to EncoderAttention but with two MultiheadAttenion modules

  Args:
    Similar to EncoderAttention, most of the arguments are passed to MultiheadAttention
    residual: if True, use skip connections following the transformer model
    use_layer_norm: default True; if True, use nn.LayerNorm following the transformer model
    fc_dim: the hidden units for the two-layer preceptron
    nonlinearity: nn.Module nonlinearity, used for the two layer preceptron
    query_key: if True, use two different linear transformations for encoder and decoder query key respectively
      in self.attention_encoder
    graph_encoder, graph_weight_encoder are passed to MultiheadAttention that combines encoder output and 
      possibly decoder self-attention; 
      if y (in forwrd function) is given, graph_encoder should have size (y.size(1), z.size(1)),
        otherwise (z.size(1), z.size(1))
    graph_decoder, graph_weight_decoder are passed to decoder self MultiheadAttention
      only used when y in forward function is given; 
      if not None, graph_decoder should have size (y.size(1), y.size(1))

  Shape:
    Input: (N, input_seq_length, in_dim)
    Output:
      if return_attention is False:
        return y of size (N, query_seq_length, out_dim)
      else:
        return y and attention_mats of size (num_heads, N, query_seq_length, input_seq_length)
      for self-attention, query_seq_length=input_seq_length=x.size(1)

  Examples:
    x = torch.randn(2,3,5)
    y = torch.randn(2,3,5)
    graph = torch.randn(3,3)
    model = DecoderAttention(in_dim=5, out_dim=5, key_dim=9, value_dim=11, fc_dim=13, num_heads=3, 
      residual=True, use_layer_norm=True, nonlinearity=nn.ReLU(), mask=False, 
      query_key=True, knn=2, graph_encoder=graph, graph_decoder=graph)
    model(x, y=y, return_attention=True, graph_encoder=None, graph_decoder=None)[1].sum(-1)

  """
  def __init__(self, in_dim, out_dim, key_dim, value_dim, fc_dim, num_heads=1, residual=True,
              use_layer_norm=True, nonlinearity=nn.ReLU(), mask=True, query_key=False, knn=None,
              graph_encoder=None, graph_weight_encoder=0.5, graph_decoder=None, graph_weight_decoder=0.5):
    super(DecoderAttention, self).__init__()
    # this is for decoder input self-attention
    self.attention_decoder = MultiheadAttention(in_dim, out_dim, key_dim, value_dim, num_heads=num_heads, 
      mask=mask, query_in_dim=None, knn=knn, graph=graph_decoder, graph_weight=graph_weight_decoder)
    # combine encoder output and decoder self-attention
    # assume both have the same dimension: out_dim
    self.attention_encoder = MultiheadAttention(out_dim, out_dim, key_dim, value_dim, num_heads=num_heads, 
                                            mask=False, query_in_dim=out_dim if query_key else None, knn=knn,
                                            graph=graph_encoder, graph_weight=graph_weight_encoder)
    self.residual = residual
    self.use_layer_norm = use_layer_norm
    self.fc = nn.Sequential(nn.Linear(out_dim, fc_dim),
                           nonlinearity,
                           nn.Linear(fc_dim, out_dim))
    if self.residual:
      assert in_dim == out_dim
    if self.use_layer_norm:
      self.layer_norm = nn.LayerNorm(out_dim, eps=1e-05, elementwise_affine=False)
    
  def forward(self, z, y=None, return_attention=False, graph_decoder=None, graph_encoder=None):
    """Decoder forward pass

    Args:
      z: encoder output as decoder input
      y: decoder input; 
        default None, then return the self-attention of encoder ouput z; 
          in this case, encoder and decoder must have the same length
        as decoder output may have a different length, y should be provided
      return_attention: only used when y is provided; if True, then return the multi-head self-attention matrices

    Examples:
      x = torch.randn(2,3,5)
      y = torch.randn(2,3,5)
      model = DecoderAttention(in_dim=5, out_dim=5, key_dim=9, value_dim=11, fc_dim=13, num_heads=3, 
        residual=True, use_layer_norm=True, nonlinearity=nn.ReLU(), mask=False, 
        query_key=True, knn=2)
      model(x, y, return_attention=True)[1].sum(-1)
    """
    assert not (y is None and return_attention)
    if y is not None:
      # apply self-attention to decoder input
      if return_attention:
        out, attention_mats = self.attention_decoder(y, query=None, return_attention=True, graph=graph_decoder)
      else:
        out = self.attention_decoder(y, query=None, return_attention=False, graph=graph_decoder)
      if self.residual:
        out = out + y
      if self.use_layer_norm:
        out = self.layer_norm(out)
      y = out
    x = self.attention_encoder(z, query=y, return_attention=False, graph=graph_encoder)
    if self.residual and y is None: # if y is not None, x and z may have different size
      x = x + z
    if self.use_layer_norm:
      x = self.layer_norm(x)
    out = self.fc(x)
    if self.residual:
      out = out + x
    if self.use_layer_norm:
      out = self.layer_norm(out)
    if return_attention:
      return out, attention_mats
    else:
      return out
  

def get_uniq_topk(rank, history=None):
  """Based on rank and history, select the top ranked that is not in history 

  Args:
    rank: (N, seq_len) torch.LongTensor, from torch.sort(x)[1]
    history: either None, or a torch.LongTensor of size (N, history_len); history_len <= seq_len

  Returns:
    res: torch.LongTensor of size (N,1)
    history: torch.LongTensor of size (N, history_len); if initially history=None, then history_len = 1;
      otherwise the new history will have res appended as the last column
  """
  res = []
  if history is None:
    res = rank[:, :1]
    history = res
  else:
    for r, h in zip(rank, history):
      for i in r:
        if i in h:
          continue
        else:
          res.append(i)
          break
    res =  torch.stack(res, dim=0).unsqueeze(-1)
    history = torch.cat([history, res], dim=-1) # in fact, dim=1
  return res, history


class Transformer(nn.Module):
  """Modified google transformer model; 
    this is deprecated as the design may have flaws

  Args:
    in_dim: int, the embedding dimension for input and output; 
      model_dim in the paper "Attention is all you need"
    key_dim: int, key dimension for both encoders and decoders
    value_dim: int, value dimension for both encoders and decoders
    fc_dim: int, num of hidden units for the two-layer preceptrons used in encoders and decoders
    linear_dim: int
    in_voc_size: int, input vocabulary size
    out_voc_size: int, output vocabulary size
    in_seq_len: int, input sequence length; fixed in_seq_len, because we use not using recurrent network
    out_seq_len: int, output sequence length
    encode_input_position: if True, use position encoding for input
    encode_output_position: if True, use position encoding for output
    num_heads: int, num of heads for each MultiheadAttention
    num_attention: int, num of encoders and decoders
    residual: default True, pass it to encoders and decoders
    use_layer_norm: default True; pass it to encoders and decoders
    nonlinearity: the nonlinearity used in the two-layer perceptrons in encoders and decoders
    duplicated_attention: if True, all encoders (decoders) share the same parameters
    mask: pass it to decoders
    knn: pass it to encoders and decoders
    graph_encoder, graph_weight_encoder are passed to encoders
    graph_decoder, graph_weight_decoder are passed to decoders (set argument graph_encoder is None in decoders)
  
  Shape:
    Input: torch.LongTensor of size (N, in_seq_len)
    Output: class scores, torch.FloatTensor of (N, out_seq_len, out_voc_size+1)

  Examples:
    x = torch.randint(0, 3, (2,6)).long()
    graph_encoder = torch.randn(6, 6)
    graph_decoder = torch.randn(4, 4)
    model = Transformer(in_dim=5, key_dim=7, value_dim=9, fc_dim=11, linear_dim=13, in_voc_size=3,
                  out_voc_size=4, in_seq_len=6, out_seq_len=4, encode_input_position=True, 
                  encode_output_position=True, num_heads=2, num_attention=2, residual=True, 
                  use_layer_norm=True, nonlinearity=nn.ReLU(), duplicated_attention=True, 
                  mask=False, knn=2, graph_encoder=graph_encoder, graph_weight_encoder=0.5, 
                  graph_decoder=graph_decoder, graph_weight_decoder=0.5)
    model(x, sequential=True, unique_output=True, last_output_only=True)
  """
  def __init__(self, in_dim, key_dim, value_dim, fc_dim, linear_dim, in_voc_size,
               out_voc_size, in_seq_len, out_seq_len, encode_input_position=True, 
               encode_output_position=False, num_heads=1, num_attention=1, residual=True, 
               use_layer_norm=True, nonlinearity=nn.ReLU(), duplicated_attention=True, mask=True,
               knn=None, graph_encoder=None, graph_weight_encoder=0.5, graph_decoder=None, graph_weight_decoder=0.5):
    super(Transformer, self).__init__()
    self.in_dim = in_dim
    assert out_seq_len <= out_voc_size
    self.out_seq_len = out_seq_len
    self.out_voc_size = out_voc_size
    # Maybe I should add UNKNOWN token to the input vocabulary; current not added
    self.in_embed = nn.Embedding(in_voc_size, in_dim, padding_idx=None, max_norm=1, norm_type=2, 
      scale_grad_by_freq=True, sparse=False, _weight=None)
    # add two tokens to output vocabulary: NULL (decoder output): out_voc_size; START (decoder input): out_voc_size+1
    self.out_embed = nn.Embedding(out_voc_size+2, in_dim, padding_idx=None, max_norm=1, norm_type=2, 
      scale_grad_by_freq=True, sparse=False, _weight=None)
    self.encode_input_position = encode_input_position
    if self.encode_input_position:
      self.input_pos_weight = nn.Parameter(torch.ones(2)) 
      # follow the paper "attention is all you need" 
      wavelength_range = 10000
      self.input_pos_vec = torch.tensor([
        [np.sin(i / wavelength_range**(j/in_dim)) if j%2==0 else np.cos(i / wavelength_range**((j-1)/in_dim)) 
          for j in range(in_dim)] 
        for i in range(in_seq_len)]).float()
    self.encode_output_position = encode_output_position
    if self.encode_output_position:
      self.output_pos_weight = nn.Parameter(torch.ones(2))
      # follow the paper "attention is all you need" 
      wavelength_range = 10000
      self.output_pos_vec = torch.tensor([
        [np.sin(i / wavelength_range**(j/in_dim)) if j%2==0 else np.cos(i / wavelength_range**((j-1)/in_dim)) 
          for j in range(in_dim)] 
        for i in range(out_seq_len)]).float()
    if duplicated_attention:
      self.encoders = nn.ModuleList(
        [EncoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
        num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, nonlinearity=nonlinearity, mask=False, 
        query_in_dim=None, knn=knn, graph=graph_encoder, graph_weight=graph_weight_encoder)] * num_attention
      )    
      self.decoders = nn.ModuleList(
        [DecoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
          num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, 
          nonlinearity=nonlinearity, mask=mask, query_key=False, knn=knn, 
          graph_encoder=None, graph_weight_encoder=0.5, 
          graph_decoder=graph_decoder, graph_weight_decoder=graph_weight_decoder)] * num_attention
      )
    else:
      self.encoders = nn.ModuleList(
          [EncoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
            num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, nonlinearity=nonlinearity, 
            mask=False, query_in_dim=None, knn=knn, graph=graph_encoder, graph_weight=graph_weight_encoder) 
          for i in range(num_attention)]
      )
      self.decoders = nn.ModuleList(
          [DecoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
          num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, 
          nonlinearity=nonlinearity, mask=mask, query_key=False, knn=knn, 
          graph_encoder=None, graph_weight_encoder=0.5, 
          graph_decoder=graph_decoder, graph_weight_decoder=graph_weight_decoder) for i in range(num_attention)]
      )
    self.linear = nn.Linear(in_dim, out_voc_size+1)
    
  def forward(self, x, sequential=False, unique_output=False, last_output_only=False):
    """Forward pass of transformer model; now deprecated
    Args:
      x: input, torch.LongTensor of size (N, in_seq_len); I had not taken care of x with more than two dimensions
      sequential: if True, produce one item at a time
      unique_output: only used when sequential is True; if True, make sure each generated token is unique
      last_output_only: only used when sequential is True; 
        if True, use the last generated token as decoder input; otherwise use all the generated tokens as input 
    """
    x = self.in_embed(x)
    if self.encode_input_position:
      pos_weight = nn.functional.softmax(self.input_pos_weight, dim=0)
      x = x*pos_weight[0] + self.input_pos_vec*pos_weight[1]  
    for encoder in self.encoders:
      x = encoder(x)
    if sequential:
      y = []
      # for each instance, the initial decoder input is a START token
      out = self.out_embed(x.new_tensor([self.out_voc_size + 1] * x.size(0)).unsqueeze(-1).long())
      for i in range(self.out_seq_len):
        # sequentially generate one token (more precisely, the token class scores for each position) at a time
        for decoder in self.decoders:
          out = decoder(x, out)
        # cur_y.size() = (N, self.out_voc_size+1)
        cur_y = self.linear(out)[:, -1]
        y.append(cur_y)
        if unique_output:
          if i == 0:
            seq_generated = None
          rank = cur_y.topk(self.out_seq_len, dim=-1)[1]
          idx, seq_generated = get_uniq_topk(rank, seq_generated)
        else:
          idx = cur_y.topk(1, dim=-1)[1]
        next_out = self.out_embed(idx)
        if self.encode_output_position:
          pos_weight = nn.functional.softmax(self.output_pos_weight, dim=0)
          next_out = next_out*pos_weight[0] + self.output_pos_vec[i]*pos_weight[1]
        if last_output_only:
          out = next_out
        else:
          out = torch.cat([out, next_out], dim=-2)
      y = torch.stack(y, dim=-2) # dim=1 in fact
    else:
      # This may not work well, but should be much faster than sequentially generating tokens
      # The initial decoder input is a list of START tokens
      out = self.out_embed(x.new_tensor([[self.out_voc_size + 1] * self.out_seq_len] * x.size(0)).long())
      if self.encode_output_position:
        pos_weight = nn.functional.softmax(self.output_pos_weight, dim=0)
        out = out*pos_weight[0] + self.output_pos_vec*pos_weight[1]
      for decoder in self.decoders:
        out = decoder(x, out)
      y = self.linear(out)
    return y


class StackedEncoder(nn.Module):
  """Stacked Encoder with self-attention for multi-label classification
    use the same in_dim and out_dim for num_attention stacked encoders;
    the output of stacked encoders will be fed to a decoder, 
      which will produce binary classification scores for each class

  Args:
    most parameters are past to EncoderAttention
    num_cls: int, num of classes
    dim_per_cls: int, the dimension for each class; default 2 for multi-label classification
    duplicated_attention: if True, all the encoders share the same parameters
    graph_encoder and graph_weight_encoder are passed to encoders
      if used, graph_encoder should have size (input_seq_length, input_seq_length)
    graph_decoder and graph_weight_decoder are passed to the decoder (still an EncoderAttention module)
      if used, graph_decoder should have size (num_cls, input_seq_length)

  Shape:
    Input: (N, input_seq_length, in_dim)
    Output: 
      if return_attention is False:
        return y of size (N, num_cls, dim_per_cls)
      else:
        return y and attention_mats of size (num_heads, N, num_cls, input_seq_length)
  
  Examples:
  
    x = torch.randn(2, 3, 5)
    graph_encoder = torch.randn(3, 3)
    graph_decoder = torch.randn(13, 3)
    model = StackedEncoder(in_dim=5, key_dim=7, value_dim=9, fc_dim=11, num_cls=13, dim_per_cls=2, num_heads=3, 
                          num_attention=2, residual=True, use_layer_norm=True, nonlinearity=nn.ReLU(), 
                          duplicated_attention=True, knn=10, graph_encoder=graph_encoder,
                          graph_decoder=graph_decoder)
    model(x, return_attention=True)[1].sum(-1)
  """
  def __init__(self, in_dim, key_dim, value_dim, fc_dim, num_cls, dim_per_cls=2, num_heads=1, num_attention=1, 
    knn=None, residual=True, use_layer_norm=True, nonlinearity=nn.ReLU(), duplicated_attention=False,
    graph_encoder=None, graph_weight_encoder=0.5, graph_decoder=None, graph_weight_decoder=0.5):
    super(StackedEncoder, self).__init__()
    self.in_dim = in_dim
    self.num_cls = num_cls
    self.num_attention = num_attention
    if duplicated_attention:
      self.encoders = nn.ModuleList(
        [EncoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
        num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, nonlinearity=nonlinearity, mask=False, 
        query_in_dim=None, knn=knn, graph=graph_encoder, graph_weight=graph_weight_encoder)] * num_attention
      )  
    else:
      self.encoders = nn.ModuleList(
          [EncoderAttention(in_dim, out_dim=in_dim, key_dim=key_dim, value_dim=value_dim, fc_dim=fc_dim, 
            num_heads=num_heads, residual=residual, use_layer_norm=use_layer_norm, nonlinearity=nonlinearity, 
            mask=False, query_in_dim=None, knn=knn,
            graph=graph_encoder, graph_weight=graph_weight_encoder) 
          for i in range(num_attention)]
      )
    # out_dim = 2 corresponds to the binary classification scores for each class;
    # self.decoder actually uses EncoderAttention
    self.decoder = EncoderAttention(in_dim, out_dim=dim_per_cls, key_dim=key_dim, value_dim=value_dim, 
      fc_dim=fc_dim, num_heads=num_heads, residual=False, use_layer_norm=False, nonlinearity=nonlinearity, 
      mask=False, query_in_dim=in_dim, knn=knn, graph=graph_decoder, graph_weight=graph_weight_decoder)
    # utilize max_norm and scale_grad_by_freq in nn.Embedding 
    self.cls_embed = nn.Embedding(num_cls, in_dim, padding_idx=None, max_norm=1, norm_type=2, 
      scale_grad_by_freq=True, sparse=False, _weight=None)

  def forward(self, x, return_attention=False, graph_encoder=None, graph_decoder=None, stochastic_depth=False):
    """Forward pass

    Args:
      x: (N, input_seq_length, in_dim)
      return_attention: if True, return attention between each class and input x, 
        size = (num_heads, num_cls, input_seq_length)
      graph_encoder is passed to encoders for forward pass; 
        it can be a 2-d array/tensor, 
          or a list of 2-d arrays/tensors corresponding to different attention matrices for different encoder layers
      graph_decoder is passed to decoder for forward pass
      stochastic_depth: if True, use a randomly chosen number of encoder layers
    """
    if stochastic_depth:
      num_layers = np.random.randint(1, self.num_attention+1)
    else:
      num_layers = self.num_attention
    for i, encoder in enumerate(self.encoders):
      if i < num_layers:
        x = encoder(x, query=None, return_attention=False, 
          graph=graph_encoder[i%len(graph_encoder)] if isinstance(graph_encoder, list) else graph_encoder)
    # self.cls_embed.weight.size() == (num_cls, in_dim); broadcast it to (N, num_cls, in_dim)
    out = self.decoder(x, self.cls_embed.weight.expand(x.size(0), self.num_cls, self.in_dim), 
      return_attention=return_attention, graph=graph_decoder)
    return out


def get_target(prediction, target):
  """For multi-label prediction such as predicting mesh terms, the order of the predicted items does not matter;
    align target so that we can calculate proper loss;
    this is no longer used; there are much better ways such as using binary cross-entropy loss
  
  Args:
    prediction: a torch.LongTensor of (N, seq_len_pred)
    target: a torch.LongTensor of (N, seq_len_target)
  
  Returns:
    aligned_target: a torch.LongTensor of (N, seq_len_pred)
  """
  aligned_target = []
  for pred, tar in zip(prediction, target):
    # it is desirable to have set operators in pytorch
    pred = pred.cpu().detach().numpy().tolist()
    tar = tar.cpu().detach().numpy().tolist()
    missed_tar = list(set(tar).difference(pred))
    aligned_target.append([i if i in tar else np.random.choice(missed_tar) for i in pred])
  return prediction.new_tensor(aligned_target).long()