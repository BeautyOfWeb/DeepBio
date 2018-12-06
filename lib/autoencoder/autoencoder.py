import sys
import os
import collections
import copy

import numpy as np

lib_path = 'I:/code'
if not os.path.exists(lib_path):
  lib_path = '/media/6T/.tianle/.lib'
if os.path.exists(lib_path) and lib_path not in sys.path:
  sys.path.append(lib_path)

import torch
import torch.nn as nn
from torch.nn import functional as F

from dl.models.basic_models import DenseLinear, get_list, get_attr
from dl.utils.train import cosine_similarity, adjust_learning_rate

class AutoEncoder(nn.Module):
  r"""Factorization autoencoder
  
  Args:
  
  Shape:
  
  Attributes:
  
  Examples::
  
  
  """
  def __init__(self, in_dim, hidden_dims, num_classes, dense=True, residual=False, residual_layers='all',
    decoder_norm=False, decoder_norm_dim=0, uniform_decoder_norm=False, nonlinearity=nn.ReLU(), 
    last_nonlinearity=True, bias=True):
    super(AutoEncoder, self).__init__()
    self.encoder = DenseLinear(in_dim, hidden_dims, nonlinearity=nonlinearity, last_nonlinearity=last_nonlinearity, 
      dense=dense, residual=residual, residual_layers=residual_layers, forward_input=False, return_all=False, 
      return_layers=None, bias=bias)
    self.decoder_norm = decoder_norm
    self.uniform_decoder_norm = uniform_decoder_norm
    if self.decoder_norm:
      self.decoder = nn.utils.weight_norm(nn.Linear(hidden_dims[-1], in_dim), 'weight', dim=decoder_norm_dim)
      if self.uniform_decoder_norm:
        self.decoder.weight_g.data = self.decoder.weight_g.new_ones(1) # This changed the tensor shape, but it's ok
        self.decoder.weight_g.requires_grad_(False)
    else:
      self.decoder = nn.Linear(hidden_dims[-1], in_dim)
    self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
  def forward(self, x):
    out = self.encoder(x)
    return self.classifier(out), self.decoder(out)


class MultiviewAE(nn.Module):
  r"""Multiview autoencoder. 

  Args:
    in_dims: a list (or iterable) of integers
    hidden_dims: a list of ints if every view has the same hidden_dims; otherwise a list of lists of ints
    out_dim: for classification, out_dim = num_cls
    fuse_type: default 'sum', add up the outputs of all encoders; require all ouputs has the same dimensions
      if 'cat', concatenate the outputs of all encoders
    dense, residual, residual_layers, nonlinearity, last_nonlinearity, bias are passed to DenseLinear
    decoder_norm: if True, add forward prehook torch.nn.utils.weight_norm  to decoder (a nn.Linear module)
    decoder_norm_dim: default 0; pass to torch.nn.utils.weight_norm
    uniform_decoder_norm: if True, ensure that decoder weight norm is 1 for dim=decoder_norm_dim

  Shape:
    Input: can be a list of tensors or a single tensor which will be splitted into a list
    Output: two heads: score matrix of shape (N, out_dim), concatenated decoder output: (N, sum(in_dims))

  Attributes:
    A list of DenseLinear modules as encoders and decoders
    An nn.Linear as output layer (e.g., class score matrix)

  Examples:
    >>> x = torch.randn(10, 5)
    >>> model = MultiviewAE([2,3], [5, 5], 7)
    >>> y = model(x)
    >>> y[0].shape, y[1].shape

  """
  def __init__(self, in_dims, hidden_dims, out_dim, fuse_type='sum', dense=False, residual=True, 
    residual_layers='all', decoder_norm=False, decoder_norm_dim=0, uniform_decoder_norm=False, 
    nonlinearity=nn.ReLU(), last_nonlinearity=True, bias=True):
    super(MultiviewAE, self).__init__()
    self.num_views = len(in_dims)
    self.in_dims = in_dims
    self.out_dim = out_dim
    self.fuse_type = fuse_type
    if not isinstance(hidden_dims[0], collections.Iterable):
      # hidden_dims is a list of ints, which means all views have the same hidden dims
      hidden_dims = [hidden_dims] * self.num_views
    self.hidden_dims = hidden_dims
    assert len(self.hidden_dims) == self.num_views and isinstance(self.hidden_dims[0], collections.Iterable)
    self.encoders = nn.ModuleList()
    self.decoders = nn.ModuleList()
    for in_dim, hidden_dim in zip(in_dims, hidden_dims):
      self.encoders.append(DenseLinear(in_dim, hidden_dim, nonlinearity=nonlinearity, 
        last_nonlinearity=last_nonlinearity, dense=dense, forward_input=False, return_all=False, 
        return_layers=None, bias=bias, residual=residual, residual_layers=residual_layers))
      decoder = nn.Linear(hidden_dim[-1], in_dim)
      if decoder_norm:
        torch.nn.utils.weight_norm(decoder, 'weight', dim=decoder_norm_dim)
        if uniform_decoder_norm:
          decoder.weight_g.data = decoder.weight_g.new_ones(decoder.weight_g.size())
          decoder.weight_g.requires_grad_(False)
      self.decoders.append(decoder)
    self.fuse_dims = [hidden_dim[-1] for hidden_dim in self.hidden_dims]
    if self.fuse_type == 'sum':
      fuse_dim = self.fuse_dims[0]
      for d in self.fuse_dims:
        assert d == fuse_dim
    elif self.fuse_type == 'cat':
      fuse_dim = sum(self.fuse_dims)
    else:
      raise ValueError(f"fuse_type should be 'sum' or 'cat', but is {fuse_type}")
    self.output = nn.Linear(fuse_dim, out_dim)

  def forward(self, xs):
    if isinstance(xs, torch.Tensor):
      xs = xs.split(self.in_dims, dim=1)
    # assert len(xs) == self.num_views
    encoder_out = []
    decoder_out = []
    for i, x in enumerate(xs):
      out = self.encoders[i](x)
      encoder_out.append(out)
      decoder_out.append(self.decoders[i](out))
    if self.fuse_type == 'sum':
      out = torch.stack(encoder_out, dim=-1).mean(dim=-1)
    else:
      out = torch.cat(encoder_out, dim=-1)
    out = self.output(out)
    return out, torch.cat(decoder_out, dim=-1), torch.cat(encoder_out, dim=-1)


def get_interaction_loss(interaction_mat, w, loss_type='graph_laplacian', normalize=True):
  """Calculate loss on the inconsistency between feature representations w (N*D) 
  and feature interaction network interaction_mat (N*N)
  A trivial solution is all features (row vectors of w) have cosine similarity = 1 or distance = 0
  
  Args:
    interaction_mat: non-negative symmetric torch.Tensor with shape (N, N)
    w: feature representation tensor with shape (N, D)
    normalize: if True, call w = w / w.norm(p=2, dim=1, keepdim=True) /np.sqrt(w.size(0)) 
      for loss_type = 'graph_laplacian' or 'dot_product',
        this makes sure w.norm() = 1 and the row vectors of w have the same norm: len(torch.unique(w.norm(dim=1)))==1
      call loss = loss / w.size(0) for loss_type = 'cosine_similarity'; 
      By doing this we ensure the number of features is factored out; 
      this is useful for combining losses from multi-views.

  See Loss_feature_interaction for more documentation

  """
  if loss_type == 'cosine_similarity':
    # -(|cos(w,w)| * interaction_mat).sum()
    cos = cosine_similarity(w).abs() # get the absolute value of cosine simiarity
    loss = -(cos * interaction_mat).sum()
    if normalize:
      loss = loss / w.size(0)
  elif loss_type == 'graph_laplacian':
    # trace(w' * L * w)
    if normalize:
      w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
      interaction_mat = interaction_mat / interaction_mat.norm() # this will ensure interaction_mat is normalized
    diag = torch.diag(interaction_mat.sum(dim=1))
    L_interaction_mat = diag - interaction_mat
    loss = torch.diagonal(torch.mm(torch.mm(w.t(), L_interaction_mat), w)).sum()
  elif loss_type == 'dot_product':
    # pairwise distance mat * interaction mat
    if normalize:
      w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
    d = torch.sum(w*w, dim=1) # if normalize is True, then d is a vector of the same element 1/w.size(0)
    dist = d.unsqueeze(1) + d - 2*torch.mm(w, w.t())
    loss = (dist * interaction_mat).sum()
    # loss = (dist / dist.norm() * interaction_mat).sum() # This is an alternative to 'normalize' loss
  else:
    raise ValueError(f"loss_type can only be 'cosine_similarity', "
                     f"graph_laplacian' or 'dot_product', but is {loss_type}")
  return loss


class Loss_feature_interaction(nn.Module):
  r"""A customized loss function for a graph Laplacian constraint on the feature interaction network
    For factorization autoencoder model, the decoder weights can be seen as feature representations;
    This loss measures the inconsistency between learned feature representations and their interaction network.
    A trivial solution is all features have cosine similarity = 1 or distance = 0

  Args:
    interaction_mat: torch.Tensor of shape (N, N), a non-negative (symmetric) matrix; 
      or a list of matrices; each is an interaction mat; 
      To control the magnitude of the loss, it is preferred to have argument interaction_mat.norm() = 1
    loss_type: if loss_type == 'cosine_similarity', calculate -(cos(m, m).abs() * interaction_mat).sum()
               if loss_type == 'graph_laplacian' (faster), calculate trace(m' * L * m)
               if loss_type == 'dot_product', calculate dist(m) * interaction_mat 
                 where dist(m) is the pairwise distance matrix of features; the name 'dot_product' is misleading
              If all features have norm 1, all three types are equivalent in a sense
              cosine_similarity is preferred because the magnitude of features are implicitly ignored, 
               while the other two will be affected by the magnitude of features.
    weight_path: default ['decoder', 'weight'], with the goal to get w = model.decoder.weight
    normalize: pass it to get_interaction_loss; 
      if True, call w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
        for loss_type 'graph_laplacian' or 'dot_product',
          this makes sure each row vector of w has the same norm, and w.norm() = 1
        call loss = loss / w.size(0) for loss_type = 'cosine_similarity'; 
      By doing this we ensure the number of features is factored out; 
      this is useful for combining losses from multi-views.
  
  Inputs:
    model: the above defined AutoEnoder model or other model
    or given weight matrix w
    if interaction_mat has shape (N,N), then w has shape (N, D)

  Returns:
    loss: torch.Tensor that can call loss.backward()
  """

  def __init__(self, interaction_mat, loss_type='graph_laplacian', weight_path=['decoder', 'weight'], 
    normalize=True):
    super(Loss_feature_interaction, self).__init__()
    self.loss_type = loss_type
    self.weight_path = weight_path
    self.normalize = normalize
    # If interaction_mat is a list, self.sections will be the used for splitting the weight matrix
    self.sections = None # when interaction_mat is a single matrix, self.sections is None
    if isinstance(interaction_mat, (list, tuple)):
      if normalize: # ensure interaction_mat is normalized
        interaction_mat = [m/m.norm() for m in interaction_mat]
      self.sections = [m.shape[0] for m in interaction_mat]
    else:
      if normalize: # ensure interaction_mat is normalized
        interaction_mat = interaction_mat / interaction_mat.norm()
    if self.loss_type == 'graph_laplacian':
      # precalculate self.L_interaction_mat save some compute for each forward pass
      if self.sections is None:
        diag = torch.diag(interaction_mat.sum(dim=1))
        self.L_interaction_mat = diag - interaction_mat # Graph Laplacian; should I normalize it?
      else:
        self.L_interaction_mat = []
        for mat in interaction_mat:
          diag = torch.diag(mat.sum(dim=1))
          self.L_interaction_mat.append(diag - mat)
    else: # we don't need to store interaction_mat for loss_type=='graph_laplacian'
      self.interaction_mat = interaction_mat
  
  def forward(self, model=None, w=None):
    if w is None:
      w = get_attr(model, self.weight_path)
    if self.sections is None:
      # There is only one interaction matrix; self.interaction_mat is a torch.Tensor
      if self.loss_type == 'graph_laplacian':
        # Used precalculated L_interaction_mat to save some time
        if self.normalize:
          # interaction_mat had already been normalized during initialization
          w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
        return torch.diagonal(torch.mm(torch.mm(w.t(), self.L_interaction_mat), w)).sum()
      else:
        return get_interaction_loss(self.interaction_mat, w, loss_type=self.loss_type, normalize=self.normalize)
    else:
      # self.interaction_mat is a list of torch.Tensors
      if isinstance(w, torch.Tensor):
        w = w.split(self.sections, dim=0)
      if self.loss_type == 'graph_laplacian': # handle 'graph_laplacian' differently to save time during training
        loss = 0
        for w_, L in zip(w, self.L_interaction_mat):
          if self.normalize: # make sure w_.norm() = 1 and each row vector of w_ has the same norm
            w_ = w_ / w_.norm(p=2, dim=1, keepdim=True) / np.sqrt(w_.size(0))
          loss += torch.diagonal(torch.mm(torch.mm(w_.t(), L), w_)).sum()  
        return loss
      # for the case 'cosine_similarity' and 'dot_product'
      return sum([get_interaction_loss(mat, w_, loss_type=self.loss_type, normalize=self.normalize) 
                  for mat, w_ in zip(self.interaction_mat, w)])


class Loss_view_similarity(nn.Module):
  r"""The input is a multi-view representation of the same set of patients, 
      i.e., a set of matrices with shape (num_samples, feature_dim). feature_dim can be different for each view
    This loss will penalize the inconsistency among different views.
    This is somewhat limited, because different views should have both shared and complementary information
      This loss only encourages the shared information across views, 
      which may or may not be good for certain applications.
    A trivial solution for this is multi-view representation are all the same; then loss -> -1
    The two loss_types 'circle' and 'hub' can be quite different and unstable.
      'circle' tries to make all feature representations across views have high cosine similarity,
      while 'hub' only tries to make feature representations within each view have high cosine similarity;
      by multiplying 'mean-feature' target with 'hub' loss_type, it might 'magically' capture both within-view and 
        cross-view similarity; set as default choice; but my limited experimental results do not validate this;
        instead, 'circle' and 'hub' are dominant, while explicit_target and cal_target do not make a big difference 
    Cosine similarity are used here; To do: other similarity metrics

  Args:
    sections: a list of integers (or an int); this is used to split the input matrix into chunks;
      each chunk corresponds to one view representation.
      If input xs is not a torch.Tensor, this will not be used; assume xs to be a list of torch.Tensors
      sections being an int implies all feature dim are the same, set sections = feature_dim, NOT num_sections!
    loss_type: supose there are three views x1, x2, x3; let s_ij = cos(x_i,x_j), s_i = cos(x_i,x_i)
      if loss_type=='cicle', similarity = s12*s23*target if fusion_type=='multiply'; s12+s23 if fusion_type=='sum'                   
        This is fastest but requires x1, x2, x3 have the same shape
      if loss_type=='hub', similarity=s1*s2*s3*target if fusion_type=='multiply'; 
        similarity=|s1|+|s2|+|s3|+|target| if fusion_type=='sum'
        Implicitly, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') if explicit_target is False
        if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)
    explicit_target: if False, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') implicitly
      if True, use given target or calculate it from xs
      # to do handle the case when we only use the explicitly given target
    cal_target: if 'mean-similarity', target = (cos(x1,x1) + cos(x2,x2) + cos(x3,x3))/3
                if 'mean-feature', x = (x1+x2+x3)/3; target = cos(x,x); this requires x1,x2,x3 have the same shape
    target: default None; only used when explicit_target is True
      This saves computation if target is provided in advance or passed as input
    fusion_type: if 'multiply', similarity=product(similarities); if 'sum', similarity=sum(|similarities|);
      work with loss_type
    graph_laplacian:  if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)

  Inputs:
    xs: a set of torch.Tensor matrices of (num_samples, feature_dim), 
      or a single matrix with self.sections being specified
    target: the target cosine similarity matrix; default None; 
      if not given, first check if self.targets is given; 
        if self.targets is None, then calulate it according to cal_target;
      only used when self.explicit_target is True

  Output:
    loss = -similarity.abs().mean() if graph_laplacian is False # Is this the right way to do it?
      = sum_i(x_i * L_s * x_i^T) if graph_laplacian is True # call get_interaction_loss()
    
  """
  def __init__(self, sections=None, loss_type='hub', explicit_target=False, 
    cal_target='mean-feature', target=None, fusion_type='multiply', graph_laplacian=False):
    super(Loss_view_similarity, self).__init__()
    self.sections = sections
    if self.sections is not None:
      if not isinstance(self.sections, int):
        assert len(self.sections) >= 2  
    self.loss_type = loss_type
    assert self.loss_type in ['circle', 'hub']
    self.explicit_target = explicit_target
    self.cal_target = cal_target
    self.target = target
    self.fusion_type = fusion_type
    self.graph_laplacian = graph_laplacian
    # I got nan losses easily for whenever graph_laplacian is True, especially the following case; did not know why
    # probably I need normalize similarity during every forward?
    assert not (fusion_type=='multiply' and graph_laplacian) and not (loss_type=='circle' and graph_laplacian)

  def forward(self, xs, target=None):
    if isinstance(xs, torch.Tensor):
      # make sure xs is a list of tensors corresponding to multiple views
      # this requires self.sections to valid
      xs = xs.split(self.sections, dim=1) 
    # assert len(xs) >= 2 # comment this to save time for many forward passes
    similarity = 1
    if self.loss_type == 'circle':
      # assert xs[i-1].shape == xs[i].shape
      # this saves computation
      similarity_mats = [cosine_similarity(xs[i-1], xs[i]) for i in range(1, len(xs))]
      similarity_mats = [(m+m.t())/2 for m in similarity_mats] # make it symmetric
    elif self.loss_type == 'hub':
      similarity_mats = [cosine_similarity(x) for x in xs]
    if self.fusion_type=='multiply':
      for m in similarity_mats:
        similarity = similarity * m # element multiplication ensures the larget value to be 1
    elif self.fusion_type=='sum':
      similarity = sum(similarity_mats) / len(similarity_mats) # calculate mean to ensure the largest value to be 1

    if self.explicit_target:
      if target is None:
        if self.target is None:
          if self.cal_target == 'mean-similarity':
            target = torch.stack(similarity_mats, dim=0).mean(0)
          elif self.cal_target == 'mean-feature':
            x = torch.stack(xs, -1).mean(-1) # the list of view matrices must have the same dimension
            target = cosine_similarity(x)
          else:
            raise ValueError(f'cal_target should be mean-similarity or mean-feature, but is {self.cal_target}')
        else:
          target = self.target
      if self.fusion_type=='multiply':
        similarity = similarity * target
      elif self.fusion_type=='sum':
        similarity = (len(similarity_mats)*similarity + target) / (len(similarity_mats) + 1) # Moving average
    similarity = similarity.abs() # ensure similarity to be non-negative
    if self.graph_laplacian:
      # Easily get nan loss when it is True; do not know why
      return sum([get_interaction_loss(similarity, w, loss_type='graph_laplacian', normalize=True) for w in xs]) / len(xs)
    else:
      return -similarity.mean() # to ensure the loss is within range [-1, 0]

    
class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden_dim=400, latent_size=20, num_cls=0):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_size)
        self.fc22 = nn.Linear(hidden_dim, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, in_dim)
        self.num_cls = num_cls
        if self.num_cls>0:
          self.classifier = nn.Linear(latent_size, num_cls)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, cls_score_only=False):
        if x.dim()!=2:
          x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.num_cls > 0:
          cls_score = self.classifier(mu)
          if cls_score_only:
            return cls_score
          else:
            return cls_score, self.decode(z), mu, logvar
        return self.decode(z), mu, logvar

def loss_vae(cls_score, recon_x, mu, logvar, x, y=None):
    BCE = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = BCE + KLD
    if cls_score is not None and y is not None:
      CE = F.cross_entropy(cls_score, y, reduction='sum')
      loss += CE
    return  loss

def run_one_epoch_vae(model, x, y, num_cls=2, train=True, optimizer=None, batch_size=None, return_loss=True, 
  epoch=0, print_every=10, verbose=True, forward_kwargs={}):
  """Run one epoch for VAE model
  Almost the same as run_one_epoch_single_loss

  Args:
    num_cls: as long as it is bigger than 1, perform classification
    all other arguments are the same as run_one_epoch_single_loss
  """
  is_grad_enabled = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  if batch_size is None:
    batch_size = len(x)
  total_loss = 0
  acc = 0
  for batch_idx, i in enumerate(range(0, len(x), batch_size)):
    x_batch = x[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    y_pred = model(x_batch, **forward_kwargs)
    if num_cls > 1:
      loss = loss_vae(*y_pred, x_batch, y_batch)
      cls_score = y_pred[0]
      acc_batch = (cls_score.topk(1)[1]==y_batch.unsqueeze(1)).float().mean().item()
      acc += acc_batch
    else:
      loss = loss_vae(None, *y_pred, x_batch, y_batch)
    total_loss += loss.item()
    if verbose:
      msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x_batch), len(x),
          100. * batch_idx / (len(x_batch)+(batch_size-1))//batch_size,
          loss.item() / len(x_batch))
      if num_cls > 1:
        msg += f' Acc={acc_batch:.2f}'
      print(msg)
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  torch.set_grad_enabled(is_grad_enabled)

  total_loss /=  len(x) # now total loss is average loss
  acc /= (len(x)+(batch_size-1))//batch_size
  if epoch % print_every == 0:
    print('Epoch{} {}: loss={:.3e}, acc={:.2f}'.format(
          epoch, 'Train' if train else ' Test', total_loss, acc))
  if return_loss:
    return total_loss, acc
  
def train_vae(model, x_train, y_train, x_val=[], y_val=[], x_test=[], y_test=[], 
  lr=1e-2, weight_decay=1e-4, amsgrad=True, batch_size=None, num_epochs=1, 
  reduce_every=200, eval_every=1, print_every=1, verbose=False, 
  loss_train_his=[], loss_val_his=[], loss_test_his=[], 
  acc_train_his=[], acc_val_his=[], acc_test_his=[], return_best_val=True, 
  forward_kwargs_train={}, forward_kwargs_val={}, forward_kwargs_test={}):
  """Train VAE for classification tasks

  Args:
    Most arguments are passed to run_one_epoch_vae
    lr, weight_decay, amsgrad are passed to torch.optim.Adam
    reduce_every: call adjust_learning_rate if cur_epoch % reduce_every == 0
    eval_every: call run_one_epoch_single_loss on validation and test sets if cur_epoch % eval_every == 0
    print_every: print epoch loss if cur_epoch % print_every == 0
    verbose: if True, print batch loss
    return_best_val: if True, return the best model on validation set for classification task 
    forward_kwargs_train: default {}, passed to run_one_epoch_single_loss for model(x, **forward_kwargs)
    forward_kwargs_train, forward_kwargs_val and forward_kwargs_test 
      are passed to train, val, and test set, respectively;
      if they are different if they are sample-related, 
        and in this cases, batch_size should be None, otherwise there can be size mismatch
      if they are not sample-related, then they are the same for almost all cases
  """
  best_val_acc = -1 # best_val_acc >=0 after the first epoch
  for i in range(num_epochs):   
    if i == 0:
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    # Should I create a new torch.optim.Adam instance every time I adjust learning rate? 
    adjust_learning_rate(optimizer, lr, i, reduce_every=reduce_every)

    train_loss, train_acc = run_one_epoch_vae(model, x_train, y_train, num_cls=2, train=True, optimizer=optimizer, 
      batch_size=batch_size, return_loss=True, epoch=i, print_every=print_every, verbose=verbose, 
      forward_kwargs=forward_kwargs_train)
    loss_train_his.append(train_loss)
    acc_train_his.append(train_acc)
    if i % eval_every == 0:
      if len(x_val)>0 and len(y_val)>0:
        val_loss, val_acc = run_one_epoch_vae(model, x_val, y_val, num_cls=2, train=False, optimizer=None, 
          batch_size=batch_size, return_loss=True, epoch=i, print_every=print_every, verbose=verbose, 
          forward_kwargs=forward_kwargs_val)
        loss_val_his.append(val_loss)
        acc_val_his.append(val_acc)

        if acc_val_his[-1] > best_val_acc:
          best_val_acc = acc_val_his[-1]
          best_model = copy.deepcopy(model)
          best_epoch = i
          print('epoch {}, best_val_acc={:.2f}, train_acc={:.2f}'.format(
            best_epoch, best_val_acc, acc_train_his[-1]))
      if len(x_test)>0 and len(y_test)>0:
        test_loss, test_acc = run_one_epoch_vae(model, x_test, y_test, num_cls=2, train=False, optimizer=None, 
          batch_size=batch_size, return_loss=True, epoch=i, print_every=print_every, verbose=verbose, 
          forward_kwargs=forward_kwargs_test)
        loss_test_his.append(test_loss)
        acc_test_his.append(test_acc) # Set train to be False

  if return_best_val and len(x_val)>0 and len(y_val)>0:
    return best_model, best_val_acc, best_epoch
  else:
    return model, acc_train_his[-1], i


class AdversarialAE(nn.Module):
  r"""Implement a supervised Adversarial AutoEncoder
    For the encoder, there are two heads: one for classification, one for variation;
      the two are concatenated as input for the decoder;
    During training, there are two phases: 
      reconstruction and classification: minimize reconstruction and classification losses
      adversarial training: matching the variation head to a prior (e.g., Gaussian)
      
  Args:
    in_dim: int
    hidden_dims: int or a list of int; used for encoder and decoder;
      add layers between the input and the encoder output based on hidden_dims;
      add exactly the same layers but in a reversed way for the decoder
    num_cls: int
    latent_dim: int, dimension for the encoder variation head
    discriminator_dims: a list of int for the discriminator; 
      in many cases we set the last dimension to be 1 
        as the (log) probability of the sample being real of fake
    nonlinearity: default nn.LeakyReLU()
    
  Returns:
    cls_score: classification score, the same as for any other classifiers
    x_bar: the reconstructed input
    critic_data: pass the variation head output of the encoder to the disciminator 
      and get critic_data for the real data,
        used to compare with the critic_prior for the sample from the target prior distribution
    critic_prior: pass the sample generated from torch.randn(z.size()) to the discriminator
      and get critic_prior;
      critic_data and critic_prior are used for calculate Wasserstein GAN loss
        
  Examples:
    model = AdversarialAE(in_dim=784, hidden_dims=[100], num_cls=10, latent_dim=20, 
                          discriminator_dims=[20, 1])
    x = torch.randn(3, 784)
    cls_score, x_bar, critic_data, critic_prior = model(x)
    cls_score.shape, x_bar.shape, critic_data.shape, critic_prior.shape
    
  """
  def __init__(self, in_dim, hidden_dims, num_cls, latent_dim, discriminator_dims, 
               nonlinearity=nn.LeakyReLU(), bias=True):
    super(AdversarialAE, self).__init__()
    self.in_dim = in_dim
    self.hidden_dims = hidden_dims
    self.num_cls = num_cls
    self.latent_dim = latent_dim
    self.discriminator_dims = discriminator_dims
    self.nonlinearity = nonlinearity
    self.encoder = nn.Sequential()
    self.decoder = nn.Sequential()
    if isinstance(self.hidden_dims, int):
      # this is a special case; it should be the same as self.hidden_dims = [int]
      self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dims, bias=bias),
                                   nonlinearity,
                                  nn.Linear(hidden_dims, num_cls+latent_dim, bias=bias))
      self.decoder = nn.Sequential(nn.Linear(num_cls+latent_dim, hidden_dims, bias=bias),
                                  nonlinearity,
                                  nn.Linear(hidden_dims, in_dim, bias=bias))
    else:
      for i, h in enumerate(self.hidden_dims):
        if i == 0:
          in_dim_encoder = in_dim
          in_dim_decoder = num_cls+latent_dim
        else:
          in_dim_encoder = self.hidden_dims[i-1]
          in_dim_decoder = self.hidden_dims[-i]
        self.encoder.add_module(f'layer{i}', nn.Linear(in_dim_encoder, h))
        self.encoder.add_module(f'activation{i}', nonlinearity)
        self.decoder.add_module(f'layer{i}', nn.Linear(in_dim_decoder, self.hidden_dims[-i-1]))
        self.decoder.add_module(f'activation{i}', nonlinearity)
      # add the last output layer for encoder and decoder
      self.encoder.add_module(f'layer{len(self.hidden_dims)}', 
                              nn.Linear(self.hidden_dims[-1], num_cls+latent_dim))
      self.decoder.add_module(f'layer{len(self.hidden_dims)}', 
                              nn.Linear(self.hidden_dims[0], in_dim))
    self.discriminator = nn.Sequential()
    for i, h in enumerate(self.discriminator_dims):
      in_dim_discriminator = latent_dim if i==0 else self.discriminator_dims[i-1]
      self.discriminator.add_module(f'layer{i}', nn.Linear(in_dim_discriminator, h, bias=bias))
      if i < len(self.discriminator_dims)-1:
        self.discriminator.add_module(f'activation{i}', nonlinearity)
      
  def forward(self, x):
    out = self.encoder(x)
    cls_score = out[:, :self.num_cls]
    z = out[:, self.num_cls:]
    x_bar = self.decoder(out)
    critic_data = self.discriminator(z)
    prior_sample = z.new_tensor(torch.randn(z.size()))
    critic_prior = self.discriminator(prior_sample)
    return cls_score, x_bar, critic_data, critic_prior

def run_one_epoch_aae(model, x, y, num_critic=1, clip_value=0.01, train=True, optimizer=None, 
                      batch_size=None, return_loss=True, loss_weight=[1., 1., 1.],
                      loss_fn_cls=nn.CrossEntropyLoss(), loss_fn_reg=nn.MSELoss(),
                      loss_fn_critic=nn.L1Loss(),
  epoch=0, print_every=1, verbose=True, forward_kwargs={}):
  """Run one epoch for Adversarial AutoEncoder (AAE) model using modified Wasserstein GAN loss
    Note this implementation is based on AAE and Wasserstein GAN but have been modified
    Provide the same interface as run_one_epoch_single_loss

  Args:
    num_critic: how often do we need to update 
    loss_weight: default [1., 1., 1.], corresponding to the losses 
      for classification, reconstruction, and discriminator losses
    all other arguments are the same as run_one_epoch_single_loss
  """
  is_grad_enabled = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  if batch_size is None:
    batch_size = len(x)
  if loss_weight is None:
    loss_weight = [1., 1., 1.]
  total_loss = 0
  acc = 0
  loss_batches = []
  for batch_idx, i in enumerate(range(0, len(x), batch_size)):
    x_batch = x[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    cls_score, x_bar, critic_data, critic_prior = model(x_batch, **forward_kwargs)
    loss_cls = loss_fn_cls(cls_score, y_batch)
    loss_reg = loss_fn_reg(x_bar, x_batch) # reconstruction loss
    # This is different Wasserstein GAN; I used sigmoid_() to control the loss scale
    # I used nn.L1Loss() but Wasserstein GAN does not use the absolute value
    loss_discriminator = loss_fn_critic(critic_prior.sigmoid_(), critic_data.sigmoid_())
    loss = loss_cls*loss_weight[0] + loss_reg*loss_weight[1] + loss_discriminator*loss_weight[2]
    loss_batch = [loss_cls.item(), loss_reg.item(), loss_discriminator.item()]
    loss_batches.append(loss_batch)
    total_loss += loss.item()
    acc_batch = (cls_score.topk(1)[1]==y_batch.unsqueeze(1)).float().mean().item()
    acc += acc_batch
    if verbose:
      msg = 'Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x_batch), len(x),
          100. * batch_idx / (len(x_batch)+(batch_size-1))//batch_size,
          loss.item() / len(x_batch))
      msg += f' Acc={acc_batch:.2f}'
      print(msg)
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      torch.nn.utils.clip_grad_value_(model.discriminator.parameters(), clip_value)
    if epoch % num_critic == 0:
      # Note this is different from Wasserstein GAN and AAE
      critic_prior = model.discriminator(
        critic_prior.new_tensor(torch.randn(batch_size, model.latent_dim))).sigmoid_()
      loss_discriminator = -critic_prior.mean()
      if verbose:
        print(f'Epoch {epoch}\tloss_discriminator={loss_discriminator.item()}')
      if train:
        optimizer.zero_grad()
        loss_discriminator.backward()
        optimizer.step()
  torch.set_grad_enabled(is_grad_enabled)

  total_loss /=  len(x) # now total loss is average loss
  acc /= (len(x)+(batch_size-1))//batch_size
  if epoch % print_every == 0:
    print('Epoch{} {}: loss={:.3e}, acc={:.2f}'.format(
          epoch, 'Train' if train else ' Test', total_loss, acc))
  if return_loss:
    return total_loss, acc
  
def train_aae(model, x_train, y_train, x_val=[], y_val=[], x_test=[], y_test=[], 
  num_critic=1, clip_value=0.01, loss_weight=[1., 1., 1.], 
  loss_fn_cls=nn.CrossEntropyLoss(), loss_fn_reg=nn.MSELoss(), loss_fn_critic=nn.L1Loss(),
  lr=1e-2, weight_decay=1e-4, 
  amsgrad=True, batch_size=None, num_epochs=1, 
  reduce_every=200, eval_every=1, print_every=1, verbose=False, 
  loss_train_his=[], loss_val_his=[], loss_test_his=[], 
  acc_train_his=[], acc_val_his=[], acc_test_his=[], return_best_val=True, 
  forward_kwargs_train={}, forward_kwargs_val={}, forward_kwargs_test={}):
  """Train Adversarial AutoEncoder (AAE) for classification tasks

  Args:
    Most arguments are passed to run_one_epoch_aae
    lr, weight_decay, amsgrad are passed to torch.optim.Adam
    reduce_every: call adjust_learning_rate if cur_epoch % reduce_every == 0
    eval_every: call run_one_epoch_single_loss on validation and test sets if cur_epoch % eval_every == 0
    print_every: print epoch loss if cur_epoch % print_every == 0
    verbose: if True, print batch loss
    return_best_val: if True, return the best model on validation set for classification task 
    forward_kwargs_train: default {}, passed to run_one_epoch_single_loss for model(x, **forward_kwargs)
    forward_kwargs_train, forward_kwargs_val and forward_kwargs_test 
      are passed to train, val, and test set, respectively;
      if they are different if they are sample-related, 
        and in this cases, batch_size should be None, otherwise there can be size mismatch
      if they are not sample-related, then they are the same for almost all cases
  """
  best_val_acc = -1 # best_val_acc >=0 after the first epoch
  for i in range(num_epochs):   
    if i == 0:
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    # Should I create a new torch.optim.Adam instance every time I adjust learning rate? 
    adjust_learning_rate(optimizer, lr, i, reduce_every=reduce_every)

    train_loss, train_acc = run_one_epoch_aae(model, x_train, y_train, num_critic=num_critic, 
                clip_value=clip_value, train=True, optimizer=optimizer, 
                batch_size=batch_size, return_loss=True, loss_weight=loss_weight,
                loss_fn_cls=loss_fn_cls, loss_fn_reg=loss_fn_reg, loss_fn_critic=loss_fn_critic,
                epoch=i, print_every=print_every, verbose=verbose, 
                forward_kwargs=forward_kwargs_train)
    loss_train_his.append(train_loss)
    acc_train_his.append(train_acc)
    if i % eval_every == 0:
      if len(x_val)>0 and len(y_val)>0:
        val_loss, val_acc = run_one_epoch_aae(model, x_val, y_val, num_critic=num_critic, 
            clip_value=clip_value, train=False, optimizer=None, 
            batch_size=batch_size, return_loss=True, loss_weight=loss_weight,
            loss_fn_cls=loss_fn_cls, loss_fn_reg=loss_fn_reg, loss_fn_critic=loss_fn_critic,
            epoch=i, print_every=print_every, verbose=verbose, 
            forward_kwargs=forward_kwargs_val)
        loss_val_his.append(val_loss)
        acc_val_his.append(val_acc)

        if acc_val_his[-1] > best_val_acc:
          best_val_acc = acc_val_his[-1]
          best_model = copy.deepcopy(model)
          best_epoch = i
          print('epoch {}, best_val_acc={:.2f}, train_acc={:.2f}'.format(
            best_epoch, best_val_acc, acc_train_his[-1]))
      if len(x_test)>0 and len(y_test)>0:
        test_loss, test_acc = run_one_epoch_aae(model, x_test, y_test, num_critic=num_critic, 
                clip_value=clip_value, train=False, optimizer=None, 
                batch_size=batch_size, return_loss=True, loss_weight=loss_weight,
                loss_fn_cls=loss_fn_cls, loss_fn_reg=loss_fn_reg, loss_fn_critic=loss_fn_critic,
                epoch=i, print_every=print_every, verbose=verbose, 
                forward_kwargs=forward_kwargs_test)
        loss_test_his.append(test_loss)
        acc_test_his.append(test_acc) # Set train to be False

  if return_best_val and len(x_val)>0 and len(y_val)>0:
    return best_model, best_val_acc, best_epoch
  else:
    return model, acc_train_his[-1], i