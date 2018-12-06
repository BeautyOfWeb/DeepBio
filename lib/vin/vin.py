import sys
import os
import collections
import re
import copy

import numpy as np

lib_path = 'I:/code'
if not os.path.exists(lib_path):
  lib_path = '/media/6T/.tianle/.lib'
if not os.path.exists(lib_path):
  lib_path = '/projects/academic/azhang/tianlema/lib'
if os.path.exists(lib_path) and lib_path not in sys.path:
  sys.path.append(lib_path)

import torch
import torch.nn as nn

from dl.models.basic_models import DenseLinear, get_list, get_attr
from dl.utils.train import cosine_similarity, eval_classification, adjust_learning_rate
from dl.utils.utils import merge_inner_list, append_dict, tensor_to_num


class VIN(nn.Module):
  r"""Variable Input Network (VIN)
  Handle multiple inputs (views) with missing views; targets can be multiple, too.
  
  Args:
    input_list: a list of dictionaries with keys including in_dim, in_type, hidden_dim, and other args for DenseLinear;
      for discrete variables, keys also include embedding_dim, padding_idx, max_norm, scale_grad_by_freq for nn.Embedding
    output_info: for one target, output_info is a dictionary;
      for multiple targets,  output_info is a list of dictionaries 
        with keys including out_dim, hidden_dim and other args for DenseLinear
    fusion_lists: a list of lists of dictionaries with keys including 'fusion_type';
      len(fusion_lists)==num_levels, len(fusion_lists[level_i])=num_outputs_level_i
      fusion_type is:
        'repr-cat': concatenate reprs from multiviews; the new repr will not be saved
        'repr-cat_repr': same as 'repr-cat', but save the new repr
        'repr-avg': average reprs from multiviews; the new repr will not be saved; 
        'repr-avg_repr': same as 'repr-avg', but save the averaged repr
        'repr-weighted-avg': weighted average with learable weights self.weights[f'fusion{level}[_target{t}]_view_weight']
        'repr-weighted-avg_repr': same as 'repr-weighted-avg', but save the weighted avg repr
        'repr-loss-avg': weighted avg with weights same as self.weights[f'fusion{level}[_target{t}]_loss_weight']
        'repr-loss-avg_repr': same as 'repr-loss-avg' but pass the weighted avg repr
        'out-avg': avg outputs from the previous level
        'out-weighted-avg': weighted avg outputs from the previous level 
          with weights being self.weights[f'fusion{level}[_target{t}]_out_weight']
        'out-loss-avg': same as 'out-weighted-avg' 
          but with weights same as self.weights[f'fusion{level}[_target{t}]_loss_weight']
        
  Attributes:
    self.weights nn.ParameterDict() with keys f'fusion{level}[_target{t}]_[loss|out|view]_weight'
    self.layers nn.ModuleDict() with elements DenseLinear
    self.input_embeddings nn.ModuleDict() with elements nn.Embeddings for discrete variable input

  Shape:
  
  
  Examples::
  
  
  """
  def __init__(self, input_list, output_info, fusion_lists, nonlinearity=nn.ReLU()):
    super(VIN, self).__init__()
    self.input_list = input_list
    # This output_info is only for the level0 outputs, i.e., outputs from each input type separately
    self.output_info = output_info
    self.fusion_lists = fusion_lists
    
    self.num_inputs = len(input_list)
    self.num_levels = len(fusion_lists)
    if isinstance(output_info, dict):
      self.num_targets = 1
    elif isinstance(output_info, list):
      self.num_targets = len(output_info)
      assert self.num_targets > 1
    else:
      raise ValueError(f'output_info must be either a dict (one target) or a list (multiple targets), '
                       f'but is {type(output_info)}')

    self.weights = nn.ParameterDict()
    self.layers = nn.ModuleDict()
    # Embed discrete variables with nn.Embedding
    self.input_embeddings = nn.ModuleDict()
    # self.repr_dims stores the dimensions of the learned representations from each level;
    # it will be used when combining all the learned representations
    self.repr_dims = [[]]
    # self.repr_locations is only used for fusing repr_list with fusion_type='repr-loss-avg'
    # because there can be more outputs than latent representations in a level (each output is associated a loss),
    # len(self.weights['level{i}[_target{t}]_loss_weight']) can be bigger than len(repr_list)
    # use self.repr_locations to specify the correspondance between loss weight and view weight (through subscript)
    self.repr_locations = [[]]
    
    # provide default parameters for in_dict
    default_dict = {'padding_idx':0, 'max_norm':1, 'norm_type':2, 'scale_grad_by_freq':True,
      'last_nonlinearity':False, 'bias':False, 'dense':True, 'residual':False, 'residual_layers':'all'}
    for i, in_dict in enumerate(input_list):
      # This is used to produce the learned vector representations from all the input data types individually
      in_dim = in_dict['in_dim']
      in_type = in_dict['in_type']
      hidden_dim = in_dict['hidden_dim'] # hidden_dim is a list
      self.repr_dims[0].append(hidden_dim[-1])
      self.repr_locations[0].append(i)
      # in case in_dict does not contain all required keys, append them from default values
      append_dict(in_dict, default_dict)
      if in_type == 'discrete':
        # If padding_idx=0 (for missing values), the index for a discrete variable should start from 1
        self.input_embeddings[str(i)] = torch.nn.Embedding(
          num_embeddings=in_dim if in_dict['padding_idx'] is None else in_dim+1, 
          embedding_dim=in_dict['embedding_dim'], padding_idx=in_dict['padding_idx'], max_norm=in_dict['max_norm'], 
          norm_type=in_dict['norm_type'], scale_grad_by_freq=in_dict['scale_grad_by_freq'], 
          sparse=False, _weight=None)
        in_dim = in_dict['embedding_dim']
      else:
        assert in_type == 'continuous', (f'Currently only handle discrete or continous input type, '
                                         f'but {i}th in_type is {in_type}!')
      self.layers[f'input{i}_hidden_layers'] = DenseLinear(in_dim=in_dim, hidden_dim=hidden_dim, 
        nonlinearity=nonlinearity, last_nonlinearity=in_dict['last_nonlinearity'], bias=in_dict['bias'], 
        dense=in_dict['dense'], residual=in_dict['residual'], residual_layers=in_dict['residual_layers'], 
        forward_input=False, return_all=False, return_layers=None, return_list=False)
    
    # provide default parameters for output_info and fusion_lists
    default_dict = {'last_nonlinearity':False, 'bias':False, 'dense':True, 'residual':False, 
      'residual_layers':'all-but-last'}
    if self.num_targets==1:
      # Generate level0 outputs from each input using their high-level representations with DenseLinear model;
      # For code simplicity, make output_layers from all views have the same hidden_dim
      # output_info is a dictionary
      hidden_dim = output_info['hidden_dim']
      self.out_dim = hidden_dim[-1]
      append_dict(output_info, default_dict) # provide default values in case they are missing in output_info
      for i, in_dim in enumerate(self.repr_dims[0]):
        # For coding simplicity, the output layers from all views will have the same hidden_dim
        self.layers[f'input{i}_output_layers'] = DenseLinear(in_dim=in_dim, hidden_dim=hidden_dim, 
          nonlinearity=nonlinearity, last_nonlinearity=output_info['last_nonlinearity'], bias=output_info['bias'], 
          dense=output_info['dense'], residual=output_info['residual'], residual_layers=output_info['residual_layers'], 
          forward_input=False, return_all=False, return_layers=None, return_list=False)
    else: # self.num_targets > 1
      # For each target, generate level0 outputs from each input 
      # using their high-level representations with DenseLinear model;
      self.out_dims = []
      # output_info is a list of dictionaries
      # self.num_targets == len(output_info)
      for j, out_dict in enumerate(output_info):
        hidden_dim = out_dict['hidden_dim'] # it is a list
        self.out_dims.append(hidden_dim[-1])
        append_dict(out_dict, default_dict) # provide default values in case they are missing in out_dict
        for i, in_dim in enumerate(self.repr_dims[0]):
          # For each target, compute an output from each input type
          self.layers[f'input{i}_target{j}_output_layers'] = DenseLinear(in_dim=in_dim, hidden_dim=hidden_dim, 
            nonlinearity=nonlinearity, last_nonlinearity=out_dict['last_nonlinearity'], bias=out_dict['bias'], 
            dense=out_dict['dense'], residual=out_dict['residual'], residual_layers=out_dict['residual_layers'], 
            forward_input=False, return_all=False, return_layers=None, return_list=False)
    
    for level, fusion_list in enumerate(fusion_lists):
      num_outputs = self.num_inputs if level==0 else len(fusion_lists[level-1])
      # loss weight at each level
      if self.num_targets==1:
        self.weights[f'fusion{level}_loss_weight'] = nn.Parameter(torch.empty(num_outputs), requires_grad=True)
        nn.init.constant_(self.weights[f'fusion{level}_loss_weight'], 1.)
      else:
        for t in range(self.num_targets):
          self.weights[f'fusion{level}_target{t}_loss_weight'] = nn.Parameter(torch.empty(num_outputs), requires_grad=True)
          nn.init.constant_(self.weights[f'fusion{level}_target{t}_loss_weight'], 1.)
      new_repr_dim = []
      new_repr_location = []
      for i, fusion_dict in enumerate(fusion_list):
        fusion_type = fusion_dict['fusion_type']
        append_dict(fusion_dict, default_dict) # provide default values in case they are missing in fusion_dict
        if fusion_type.startswith('repr'):
          # learn a new hidden representations from fused representations
          # prepare in_dim
          if re.search('avg', fusion_type):
            for d in self.repr_dims[-1]:
              assert d==self.repr_dims[-1][0]
            in_dim = self.repr_dims[-1][0]
          elif re.search('cat', fusion_type):
            in_dim = sum(self.repr_dims[-1])
          elif re.search('repr[0-9]', fusion_type): 
            in_dim = self.repr_dims[-1][int(fusion_type[4:])] # here, in most cases, fusion_type='repr0'
          else:
            raise ValueError(f'fusion_type={fusion_type} starting with repr must be repr0 '
                             f'or contain either avg or cat')
          hidden_dim = fusion_dict['hidden_dim'] # a list of ints
          if re.search('_repr', fusion_type):
            new_repr_dim.append(hidden_dim[-1])
            new_repr_location.append(i)
          self.layers[f'fusion{level}-{i}_hidden_layers'] = DenseLinear(in_dim, hidden_dim, nonlinearity=nonlinearity, 
            last_nonlinearity=fusion_dict['last_nonlinearity'], bias=fusion_dict['bias'], 
            dense=fusion_dict['dense'], residual=fusion_dict['residual'], 
            residual_layers=fusion_dict['residual_layers'], forward_input=False, return_all=False, 
            return_layers=None, return_list=False)

          # output_info is a dictionary if self.num_targets==1 else a list of dictionaries
          output_info = fusion_dict['output_info']
          if self.num_targets==1:
            append_dict(output_info, default_dict) # provide default values in case they are missing in output_info
            # initialize view weights
            if fusion_type.startswith('repr-weighted-avg'):
              self.weights[f'fusion{level}_view_weight'] = nn.Parameter(torch.empty(len(self.repr_dims[-1])), requires_grad=True)
              nn.init.constant_(self.weights[f'fusion{level}_view_weight'], 1.)
            self.layers[f'fusion{level}-{i}_output_layers'] = DenseLinear(in_dim=hidden_dim[-1], 
              hidden_dim=output_info['hidden_dim'], nonlinearity=nonlinearity, 
              last_nonlinearity=output_info['last_nonlinearity'], bias=output_info['bias'], 
              dense=output_info['dense'], residual=output_info['residual'], 
              residual_layers=output_info['residual_layers'], 
              forward_input=False, return_all=False, return_layers=None, return_list=False)
          else:
            if fusion_type.startswith('repr-weighted-avg'):
              for t in range(self.num_targets):
                self.weights[f'fusion{level}_target{t}_view_weight'] = nn.Parameter(torch.empty(len(self.repr_dims[-1])), requires_grad=True)
                nn.init.constant_(self.weights[f'fusion{level}_target{t}_view_weight'], 1.)
            for t, out_dict in enumerate(output_info):
              append_dict(out_dict, default_dict) # provide default values in case they are missing in out_dict
              self.layers[f'fusion{level}-{i}_target{t}_output_layers'] = DenseLinear(in_dim=hidden_dim[-1], 
                hidden_dim=out_dict['hidden_dim'], nonlinearity=nonlinearity, 
                last_nonlinearity=out_dict['last_nonlinearity'], bias=out_dict['bias'], 
                dense=out_dict['dense'], residual=out_dict['residual'], residual_layers=out_dict['residual_layers'], 
                forward_input=False, return_all=False, return_layers=None, return_list=False)
        elif fusion_type.startswith('out'):
          if self.num_targets==1:
            if fusion_type == 'out-weighted-avg':
              self.weights[f'fusion{level}_out_weight'] = nn.Parameter(torch.empty(num_outputs), requires_grad=True)
              nn.init.constant_(self.weights[f'fusion{level}_out_weight'], 1.)
          else:
            if fusion_type.startswith('out-weighted-avg'):
              for t in range(self.num_targets):
                self.weights[f'fusion{level}_target{t}_out_weight'] = nn.Parameter(torch.empty(num_outputs), requires_grad=True)
                nn.init.constant_(self.weights[f'fusion{level}_target{t}_out_weight'], 1.)
        else:
          raise ValueError(f'fusion_type must start with repr or out, but is {fusion_type}')
      self.repr_dims.append(new_repr_dim)
      self.repr_locations.append(new_repr_location)
    # the loss weight for the last level; not used in almost all the cases; mainly used to avoid error in get_vin_loss
    if self.num_targets==1:
      self.weights[f'fusion{self.num_levels}_loss_weight'] = nn.Parameter(torch.empty(len(fusion_lists[-1])), requires_grad=True)
      nn.init.constant_(self.weights[f'fusion{self.num_levels}_loss_weight'], 1.)
    else:
      for t in range(self.num_targets):
        self.weights[f'fusion{self.num_levels}_target{t}_loss_weight'] = nn.Parameter(torch.empty(len(fusion_lists[-1])), requires_grad=True)
        nn.init.constant_(self.weights[f'fusion{self.num_levels}_target{t}_loss_weight'], 1.)
  
  def fusion_output(self, repr_list, output_lists, fusion_list, level=0, valid_loc=None, 
    subset_repr=None, subset_output=None):
    r"""Compute the fusion results from the results of the previous level

    Args:
      repr_list: a list of latent representations; the size is like [(N, h1), (N, h2), ...]; I usually set h1=h2=...
      output_lists: if self.num_targets==1, output_lists is a list of output tensors; the size is like [(N,d), (N,d), ...]
                    if self.num_targets>1, output_lists is a list of lists of tensors for each target; it is like
                      [target1, target2, ...] where target1 is like [(N, d1), (N, d1), ...]
      fusion_list: a list of dictionaries with keys including fusion_type and others (e.g., for DenseLinear model)
      level: int; this is used to construct the keys to access the modules in the self.layers
      valid_loc: a 2-d tensor usually just used for the raw input; the size is like (N, v), 
        where N is the batch size and v is the number of data types; 
        if a data type is missing, then the corresponding element in valid_loc is 0, otherwise 1;
        default None, assume no missing values
      subset_repr: a list of ints, indicating which latent representations are valid to use
      subset_output: similar to subset_repr, indictating which outputs are valid to use
    
    Returns:
      new_repr_list, new_output_list
    """
    new_repr_list = []
    new_output_lists = []
    if valid_loc is not None:
      # I mainly use valid_loc for handling missing values
      # for simplicity, assume when valid_loc is given, then subset_repr and subset_output are None
      assert subset_repr is None and subset_output is None
      # the last dimension sums to 1
      valid_loc_weight = (valid_loc / valid_loc.sum(-1, keepdim=True)).unsqueeze(-2)
    if subset_repr is not None:
        repr_list_subset = [h for i, h in enumerate(repr_list) if i in subset_repr]
    for i, fusion_dict in enumerate(fusion_list):
      fusion_type = fusion_dict['fusion_type']
      if fusion_type.startswith('repr'):
        if fusion_type.startswith('repr-avg'):
          if subset_repr is not None:
            h = torch.stack(repr_list_subset, dim=-1).mean(-1)
          elif valid_loc is not None:
            h = torch.stack(repr_list, dim=-1)
            h = (h * valid_loc_weight).sum(-1)
          else:
            h = torch.stack(repr_list, dim=-1).mean(-1)
        elif fusion_type.startswith('repr-cat'):
          h = torch.cat(repr_list, dim=-1)
        elif re.search('repr[0-9]', fusion_type):
          h = repr_list[int(fusion_type[4:])]
        else:
          # these two cases are handled slightly differently
          assert fusion_type.startswith('repr-weighted-avg') or fusion_type.startswith('repr-loss-avg')
        
        if self.num_targets==1:
          if fusion_type.startswith('repr-weighted-avg') or fusion_type.startswith('repr-loss-avg'):
            if fusion_type.startswith('repr-weighted-avg'):
              if subset_repr is not None: # only for the input level0 input
                h = torch.stack(repr_list_subset, dim=-1)
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_view_weight'][subset_repr], dim=0)
              else:
                h = torch.stack(repr_list, dim=-1)
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_view_weight'], dim=0)
                if valid_loc is not None:
                  fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                  fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
            else:
              # Using loss weight as view weight, assuming the previous level produce the same number of reprs and ouputs
              if subset_repr is not None: # only for the input level0 input
                h = torch.stack(repr_list_subset, dim=-1)
                # I did not use [self.repr_locations[level]] as in the else case 
                # because I assume subset_repr is only used in the input level0, 
                # where len(self.repr_locations[0]) = self.num_inputs, i.e., all inputs will have both reprs and outputs
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_loss_weight'][subset_repr], dim=0)
              else:
                h = torch.stack(repr_list, dim=-1)
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_loss_weight'][self.repr_locations[level]], dim=0)
                if valid_loc is not None:
                  fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                  fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
            h = (h * fusion_weight).sum(-1)
          h = self.layers[f'fusion{level}-{i}_hidden_layers'](h)
          if fusion_type.startswith('repr-cat'):
            if subset_repr is not None:
              # scale h so that they are on the same scale even with different number of missing views (input data types) 
              h = h*(len(repr_list) / len(subset_repr))
            elif valid_loc is not None:
              h = h / valid_loc.mean(-1, keepdim=True)
          new_output_lists.append(self.layers[f'fusion{level}-{i}_output_layers'](h))
        else:
          new_outputs = []
          # for the following three cases
          # we have to save the shared latent representations instead of overwriting it when calculating the first target
          # this is not required for a single target
          if fusion_type.startswith('repr-avg') or fusion_type.startswith('repr-cat') or re.search('repr[0-9]', fusion_type):
            h_shared = h
          for t in range(self.num_targets):
            if fusion_type.startswith('repr-weighted-avg') or fusion_type.startswith('repr-loss-avg'):
              if fusion_type.startswith('repr-weighted-avg'):
                if subset_repr is not None:
                  h = torch.stack(repr_list_subset, dim=-1)
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_view_weight'][subset_repr], dim=0)
                else:
                  h = torch.stack(repr_list, dim=-1)
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_view_weight'], dim=0)
                  if valid_loc is not None:
                    fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                    fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
              else: 
                # fusion_type.startswith('repr-loss-avg')
                # see more documentation in the case where self.num_targets==1
                if subset_repr is not None:
                  h = torch.stack(repr_list_subset, dim=-1)
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_loss_weight'][subset_repr], dim=0)
                else:
                  h = torch.stack(repr_list, dim=-1)
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_loss_weight'][self.repr_locations[level]], dim=0)
                  if valid_loc is not None:
                    fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                    fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
              h_shared = (h*fusion_weight).sum(-1)
            # for the case fusion_type.startswith('repr-avg') and fusion_type.startswith('repr-cat'),
            # this will be duplicatedly calculated self.num_targets times, 
            # but for the other two cases, h_share is different for each target
            h = self.layers[f'fusion{level}-{i}_hidden_layers'](h_shared)
            if fusion_type.startswith('repr-cat'):
              if subset_repr is not None:
                h = h*(len(repr_list) / len(subset_repr))
              elif valid_loc is not None:
                h = h / valid_loc.mean(dim=-1, keepdim=True)
            new_outputs.append(self.layers[f'fusion{level}-{i}_target{t}_output_layers'](h))
          new_output_lists.append(new_outputs)
        if re.search('_repr', fusion_type):
          new_repr_list.append(h)
      elif fusion_type.startswith('out'):
        if fusion_type.startswith('out-avg'):
          if self.num_targets==1:
            if subset_output is not None:
              output_lists_subset = [out for i, out in enumerate(output_lists) if i in subset_output]
              new_output_lists.append(torch.stack(output_lists_subset, dim=-1).mean(-1))
            else:
              new_outputs = torch.stack(output_lists, dim=-1)
              if valid_loc is not None:
                new_output_lists.append((new_outputs * valid_loc_weight).sum(-1))
              else:
                new_output_lists.append(new_outputs.mean(-1))
          else:
            if subset_output is not None:
              new_output_lists.append([torch.stack([out for i, out in enumerate(outputs) if i in subset_output], dim=-1).mean(-1) 
                for outputs in output_lists])
            else:
              new_outputs = []
              for outputs in output_lists:
                out = torch.stack(outputs, dim=-1)
                if valid_loc is not None:
                  new_outputs.append((out * valid_loc_weight).sum(-1))
                else:
                  new_outputs.append(out.mean(-1))
              new_output_lists.append(new_outputs)
        elif fusion_type.startswith('out-weighted-avg') or fusion_type.startswith('out-loss-avg'):
          if self.num_targets==1:
            if fusion_type.startswith('out-weighted-avg'):
              if subset_output is not None:
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_out_weight'][subset_output], dim=0)
              else:
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_out_weight'], dim=0)
                if valid_loc is not None:
                  fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                  fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
            else:  # fusion_type.startswith('out-loss-avg')
              if subset_output is not None:
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_loss_weight'][subset_output], dim=0)
              else:
                fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_loss_weight'], dim=0)
                if valid_loc is not None:
                  fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                  fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
            if subset_output is not None:
              output_lists_subset = [out for i, out in enumerate(output_lists) if i in subset_output]
              new_output_lists.append((torch.stack(output_lists_subset, dim=-1) * fusion_weight).sum(-1))
            else:
              new_output_lists.append((torch.stack(output_lists, dim=-1) * fusion_weight).sum(-1))
          else:
            new_outputs = []
            for t, outputs in enumerate(output_lists):
              if fusion_type.startswith('out-weighted-avg'):
                if subset_output is not None:
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_out_weight'][subset_output], dim=0)
                else:
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_out_weight'], dim=0)
                  if valid_loc is not None:
                    fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                    fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
              else:
                if subset_output is not None:
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_loss_weight'][subset_output], dim=0)
                else:
                  fusion_weight = torch.nn.functional.softmax(self.weights[f'fusion{level}_target{t}_loss_weight'], dim=0)
                  if valid_loc is not None:
                    fusion_weight = fusion_weight * valid_loc.unsqueeze(-2)
                    fusion_weight = fusion_weight / fusion_weight.sum(-1, keepdim=True)
              if subset_output is not None:
                outputs_subset = [out for i, out in enumerate(outputs) if i in subset_output]
                new_outputs.append((torch.stack(outputs_subset, dim=-1) * fusion_weight).sum(-1))
              else:
                new_outputs.append((torch.stack(outputs, dim=-1) * fusion_weight).sum(-1))
            new_output_lists.append(new_outputs)
        else:
          raise ValueError(f'fusion_type={fusion_type} is not handled')
      else:
        raise ValueError(f'{fusion_type} is not handled')
    if self.num_targets>1:
      # reorder the nested list so that len(new_output_lists)=self.num_targets
      new_output_lists = list(map(list, zip(*new_output_lists)))
      # new_output_lists = [[out[t] for out in new_output_lists] for t in range(self.num_targets)]
    return new_repr_list, new_output_lists

  def forward(self, xs, valid_loc=None, subset_repr=None, subset_output=None, return_repr=False):
    repr_list = []
    if return_repr:
      repr_all = []
    output_lists = []
    output_all = []
    for i, x in enumerate(xs):
      if str(i) in self.input_embeddings:
        x = self.input_embeddings[str(i)](x)
      h = self.layers[f'input{i}_hidden_layers'](x)
      repr_list.append(h)
      if self.num_targets==1:
        out = self.layers[f'input{i}_output_layers'](h)
      else:
        out = [self.layers[f'input{i}_target{t}_output_layers'](h) for t in range(self.num_targets)]
      output_lists.append(out)
    if self.num_targets>1:
      # output_lists = [[out[t] for out in output_lists] for t in range(self.num_targets)]
      output_lists = list(map(list, zip(*output_lists)))
    output_all.append(output_lists)
    if return_repr:
      repr_all.append(repr_list)
    if len(self.fusion_lists) >= 1:
      # only use valid_loc, subset_repr, subset_output in the first level
      repr_list, output_lists = self.fusion_output(repr_list, output_lists, self.fusion_lists[0], level=0, 
        valid_loc=valid_loc, subset_repr=subset_repr, subset_output=subset_output)
      output_all.append(output_lists)
      if return_repr:
        repr_all.append(repr_list)
      if len(self.fusion_lists) > 1:
        for level, fusion_list in enumerate(self.fusion_lists[1:]):
          repr_list, output_lists = self.fusion_output(repr_list, output_lists, fusion_list, level+1,
            valid_loc=None, subset_repr=None, subset_output=None)
          output_all.append(output_lists)
          if return_repr:
            repr_all.append(repr_list)
    if self.num_targets>1:
      output_all = list(map(list, zip(*output_all)))
    if return_repr:
      return repr_all, output_all
    return output_all


def get_vin_loss(pred, target, loss_fn, model, valid_loc=None, target_id=None, level_weight=None, train=True):
  """Compute VIN model losses
  
  Args:
    pred: a list (correspond to levels) of lists (correspond to outputs in each level) of tensors for one target; 
      or a list (each element corresponds to a target) of lists of lists of tensors for multiple targets
    target: a tensor for one target, or a list of tensors for multiple targets
    loss_fn: a loss function for one target, or a list of loss functions for multiple targets
    model: VIN model, used to access model.weights[f'fusion{level}[_target{t}]_loss_weight'] 
      for calculating weighted loss at each level
    valid_loc: a 2-d tensor, correspond to the level0 of pred; only used for level0 loss
    target_id: for both single target and multiple targets, set target_id to be None; handled internally;
      not used for one target; 
      for multiple targets, internally loop through all targets,
        and use target_id (int) to access model.weights[f'fusion{level}_target{t}_loss_weight']
    level_weight: a list of floats, a 1-d np.array, or a 1-d tensor; the weight for losses from each level
      if None, then the weight is 1 for all levels

  Returns:
    loss_overall, loss_lists
      for one target:
        loss_overall is the overall loss combining all levels
        loss_lists is a list of [losses_level_i, loss_level_i]; len(loss_lists)==model.num_levels;
          losses_level_i is a list of singleton tensors len(losses_level_i)==num_outputs in that level;
          loss_level_i is the weighted sum of losses_level_i, a singleton tensor
      for multiple targets:
        return a list of results each corresponding to one target as a tuple (loss_overall, loss_lists)
  """
  if isinstance(target, list):
    return [get_vin_loss(p, t, l, model, valid_loc=valid_loc, target_id=i, level_weight=level_weight, train=train) 
      for i, (p, t, l) in enumerate(zip(pred, target, loss_fn))]
  elif isinstance(target, torch.Tensor):
    is_grad_enabled = torch.is_grad_enabled()
    if train:
      model.train()
      torch.set_grad_enabled(True)
    else:
      model.eval()
      torch.set_grad_enabled(False)
    loss_overall = 0
    loss_lists = []
    if level_weight is None:
      level_weight = [1.] * len(pred)
    for level_i, ys in enumerate(pred):
      if target_id is not None:
        weight = torch.nn.functional.softmax(model.weights[f'fusion{level_i}_target{target_id}_loss_weight'],dim=0)
      else:
        weight = torch.nn.functional.softmax(model.weights[f'fusion{level_i}_loss_weight'],dim=0)
      if level_i==0 and valid_loc is not None:
        # By (limited) design, valid_loc is only used for level_0
        tmp = loss_fn.reduction
        loss_fn.reduction = 'none'
        losses_level_i = [torch.sum(loss_fn(y, target) * valid_loc[:,i]) / valid_loc[:,i].sum() 
          for i, y in enumerate(ys)]
        loss_fn.reduction = tmp
      else:
        losses_level_i = [loss_fn(y, target) for y in ys]
      # # The following line is a detrimental bug; torch.tensor will create a new tensor whose gradient history is empty
      # loss_level_i = torch.sum(torch.tensor(losses_level_i)*weight)
      loss_level_i = sum([loss*weight[i] for i, loss in enumerate(losses_level_i)])
      loss_lists.append([losses_level_i, loss_level_i])
      loss_overall = loss_overall + level_weight[level_i] * loss_level_i
    torch.set_grad_enabled(is_grad_enabled)
    return loss_overall, loss_lists


def vin_loss_reduce_fn(losses, loss_weight):
  """For multiple targets, get_vin_loss will return a list of losses; 
    calculate the aggregated loss from all the losses
  Args:
    losses: the output of get_vin_loss for multiple targets
    loss_weights: a list of floats as loss weights for all the targets
  """
  return sum(losses[j][0]*loss_weight[j] for j in range(len(losses)))


def get_vin_acc(pred, target, target_idx=None, level=-1, loc=0, acc_only=True, verbose=True):
  """Calculate accuracy or call eval_classification
  When writing this function, I experienced the problem with default parameters in python functions
  
  Args:
    pred: the output from VIN model; a nested list of tensors
    target: a tensor or a list of tensors (multiple targets);
      for multiple targets, len(pred)=len(target)
    target_idx: a list of ints, specify which classification targets are used for calculating accuracy;
      only used for multiple targets; 
      default None, will set target_idx based on if target is torch.LongTensor; 
        if there is only one torch.LongTensor in the target list, then return one accuracy;
          otherwise, return a list
    level: specify which level of pred is used for calculating accuracy
    loc: specify the loc of the chosen level for calculating accuracy;
      for single target, pred[level][loc]
    acc_only: if True, only calculate accuracy, otherwise call eval_classification
    verbose: only used when acc_only is False; passed to eval_classification
    
  Returns:
    for single target:
      if acc_only is True:
        return acc
      else:
        return (np.array([acc, precision, recall, f1_score, adjusted_mutual_info, auc, 
          average_precision]), confusion_mat)
    for multiple targets:
      return a list of results of single targets
    if there is no classification target, return None
  """
  if isinstance(target, torch.Tensor) and isinstance(target.cpu(), torch.LongTensor):
    pred = pred[level][loc]
    if acc_only:
      acc = (pred.topk(1, dim=1)[1]==target.unsqueeze(-1)).sum().item()/len(pred)
      return acc
    else:
      return eval_classification(target, pred, verbose=verbose)
  elif isinstance(target, (list, tuple)):
    if target_idx is None:
      target_idx = [i for i, t in enumerate(target) if isinstance(t.cpu(), torch.LongTensor)]
    if isinstance(target_idx, int) or (isinstance(target_idx, list) and len(target_idx)==1):
      if isinstance(target_idx, int):
        i = target_idx
      else:
        i = target_idx[0]
      return get_vin_acc(pred[i], target[i], level=level, loc=loc, acc_only=acc_only, verbose=verbose)
    elif len(target_idx)>1:
      return [get_vin_acc(pred[i], target[i], level=level, loc=loc, acc_only=acc_only, verbose=verbose) 
              for i in target_idx]


def predict_func(model, xs, batch_size=None, train=True, valid_loc=None, subset_repr=None, 
                 subset_output=None, return_repr=False, return_tensor=True, target_idx=None, 
                level=-1, loc=0):
  """Customized predict_func for VIN model
  
  Args:
    model: VIN model
    xs: a list of torch.Tensor as inputs
    batch_size: None or int; for a large model and input, to save memory, predict in chunks and then combine the results
    train: if False, torch.set_grad_enabled(False) to save time; 
      execute model.eval() if train is False otherwise model.train()
    valid_loc: None or a 2-d tensor of size (N, num_inputs); 
      if there is a missing input in a record, then the corresponding element is 0
    subset_repr, subset_output, return_repr are all passed to model.forward()
      return_repr must be False when batch_size is not None
    return_tensor: if True, then use target_idx, level, loc to return a tensor; if False, return a nested list of tensors
    target_idx, level, loc are used only when return_tensor is True
      target_idx: None for single target; for multiple targets, target_idx is an int
      level: int; specify which level of output is used
      loc: int; specify which output in the selected level is used

  Returns:
    a torch.Tensor if return_tensor is True
    a nested list of tensors otherwise
    
  """
  if return_repr:
    assert batch_size is None, 'when batch_size is not None, must set return_repr to be False'
  is_grad_enabled = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  if batch_size is None:
    pred = model(xs, valid_loc=valid_loc, subset_repr=subset_repr, subset_output=subset_output, 
                 return_repr=return_repr)
  else:
    pred = []
    for i in range(0, len(xs[0]), batch_size):
      pred.append(model([x[i:i+batch_size] for x in xs], 
                        valid_loc=None if valid_loc is None else valid_loc[i:i+batch_size], 
                        subset_repr=subset_repr, subset_output=subset_output, 
                        return_repr=return_repr))
    pred = merge_inner_list(pred, dim=0)                
  torch.set_grad_enabled(is_grad_enabled)
  if return_tensor:
    if target_idx is not None:
      pred = pred[target_idx]
    pred = pred[level][loc]
  return pred


def run_one_epoch_vin(model, xs, targets, loss_fn=nn.CrossEntropyLoss(), loss_history=[], acc_history=[], 
  level_weight=None, target_loss_weight=None, cls_target_idx=None, acc_level=-1, acc_loc=0, acc_only=True, 
  train=True, optimizer=None, batch_size=None, valid_loc=None, subset_repr=None, subset_output=None,
  epoch=0, print_every=10, verbose=True):
  """Run one epoch with VIN model

  Args:
    model: a VIN model
    xs: a list of tensors, the input for model
    targets: a torch.Tensor for a single target, a list of tensors for multiple targets
    loss_fn: 1-1 correspond to targets
    loss_history: a (mutable) list storing the losses returned by get_vin_loss
    acc_history: a (mutable) list storing accuracy returned by get_vin_acc;
      if there is no classification target to calcuate acc, then append None
    level_weight: loss weight for each level, passed to get_vin_loss
    target_loss_weight: not used for a single target; 
      for multiple targets, if None, then set it to 1 for each target, otherwise it should be an collections.Iterable
    cls_target_idx: default None; 
      set to be an int when targets is a list and we only want to calculate accuracy for one target
    acc_level: the output of which level is used to calculate accuracy
    acc_loc: the location index of the chosen level used to calculate accuracy
    acc_only: if True, only return acc; otherwise call eval_classification and return more metrics
    train: if True, call model.train() and enable grad; otherwise disable grad to save time
    optimizer: not used when train is False; must be provided when train is True
    epoch and print_every are used for print loss and accuracy
    verbose: passed to get_vin_acc, used when eval_classification is called (acc_only is False);
      if it is False, it won't print loss and acc
  """
  pred = predict_func(model, xs, batch_size=batch_size, train=train, valid_loc=valid_loc, 
                      subset_repr=subset_repr, subset_output=subset_output, return_repr=False, 
                      return_tensor=False)
  losses = get_vin_loss(pred, targets, loss_fn, model, valid_loc=valid_loc, target_id=None, 
                        level_weight=level_weight, train=train)
  if isinstance(targets, (list, tuple)) and target_loss_weight is None:
    target_loss_weight = [1. for i in range(len(targets))]
  if target_loss_weight is not None: # there are multiple targets
    loss = vin_loss_reduce_fn(losses, target_loss_weight)
  else:
    loss = losses[0]
  loss_history.append(tensor_to_num(losses))
  acc = get_vin_acc(pred, targets, target_idx=cls_target_idx, level=acc_level, loc=acc_loc, acc_only=acc_only, 
    verbose=verbose)
  acc_history.append(acc)
  if epoch % print_every == 0 and verbose:
    msg = 'Train ' if train else 'Test  '
    msg += f'{epoch}: loss={loss.item():.2e}'
    if isinstance(acc, list):
      if len(acc)>0:
        msg += ', acc=' + ', '.join(map(lambda s: f'{s:.2f}', acc))
    elif isinstance(acc, float):
      msg += f', acc={acc:.2f}'
    print(msg)
  if train:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  

def train_vin(model, x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None, 
    loss_fn=nn.CrossEntropyLoss(), level_weight=None, target_loss_weight=None, cls_target_idx=0, acc_level=-1, 
    acc_loc=0, acc_only=True, valid_loc=None, subset_repr=None, subset_output=None, lr=1e-2, 
    weight_decay=1e-4, amsgrad=True, batch_size=None, num_epochs=1, 
    reduce_every=200, eval_every=1, print_every=1, verbose=False, 
    loss_train_his=[], loss_val_his=[], loss_test_his=[], 
    acc_train_his=[], acc_val_his=[], acc_test_his=[], return_best_val=True):
  """Run a number of epochs to backpropagate

  Args:
    Most arguments are passed to run_one_epoch_vin
    cls_target_idx: default 0; Suppose the main objective is (binary) classification. Put that target in the first place
    lr, weight_decay, amsgrad are passed to torch.optim.Adam
    reduce_every: call adjust_learning_rate if cur_epoch % reduce_every == 0
    eval_every: call run_one_epoch_single_loss on validation and test sets if cur_epoch % eval_every == 0
    print_every: print epoch loss if cur_epoch % print_every == 0
    verbose: if True, print batch loss
    return_best_val: if True, return the best model on validation set for classification task 
  """
  best_val_acc = -1 # best_val_acc >=0 after the first epoch for classification tasks
  for i in range(num_epochs):   
    if i == 0:
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    # Should I create a new torch.optim.Adam instance every time I adjust learning rate? 
    adjust_learning_rate(optimizer, lr, i, reduce_every=reduce_every)

    run_one_epoch_vin(model, x_train, y_train, loss_fn=loss_fn, loss_history=loss_train_his, 
      acc_history=acc_train_his, level_weight=level_weight, target_loss_weight=target_loss_weight, 
      cls_target_idx=cls_target_idx, acc_level=acc_level, acc_loc=acc_loc, acc_only=acc_only, train=True, 
      optimizer=optimizer, batch_size=batch_size, valid_loc=valid_loc, subset_repr=subset_repr, 
      subset_output=subset_output, epoch=i, print_every=print_every, verbose=verbose)
    if i % eval_every == 0:
      if x_val is not None and y_val is not None:
        run_one_epoch_vin(model, x_val, y_val, loss_fn=loss_fn, loss_history=loss_val_his, 
          acc_history=acc_val_his, level_weight=level_weight, target_loss_weight=target_loss_weight, 
          cls_target_idx=cls_target_idx, acc_level=acc_level, acc_loc=acc_loc, acc_only=acc_only, train=False, 
          optimizer=optimizer, batch_size=batch_size, valid_loc=valid_loc, subset_repr=subset_repr, 
          subset_output=subset_output, epoch=i, print_every=print_every, verbose=verbose) # Set train to be False is crucial!
        if acc_val_his[-1] is not None: # this means the there is a classification target
          if acc_val_his[-1] > best_val_acc:
            best_val_acc = acc_val_his[-1]
            best_model = copy.deepcopy(model)
            best_epoch = i
            print('epoch {}, best_val_acc={:.2f}, train_acc={:.2f}'.format(
              best_epoch, best_val_acc, acc_train_his[-1]))
      if x_test is not None and y_test is not None:
        run_one_epoch_vin(model, x_test, y_test, loss_fn=loss_fn, loss_history=loss_test_his, 
          acc_history=acc_test_his, level_weight=level_weight, target_loss_weight=target_loss_weight, 
          cls_target_idx=cls_target_idx, acc_level=acc_level, acc_loc=acc_loc, acc_only=acc_only, train=False, 
          optimizer=optimizer, batch_size=batch_size, valid_loc=valid_loc, subset_repr=subset_repr, 
          subset_output=subset_output, epoch=i, print_every=print_every, verbose=verbose) # Set train to be False is crucial!

  if acc_train_his[-1] is not None: # for classification tasks
    if return_best_val and x_val is not None and y_val is not None:
      return best_model, best_val_acc, best_epoch
    else:
      return model, acc_train_his[-1], i