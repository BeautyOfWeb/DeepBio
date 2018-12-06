import os
import sys

import torch
import torch.nn as nn

lib_path = 'I:/code'
if not os.path.exists(lib_path):
  lib_path = '/media/6T/.tianle/.lib'
if not os.path.exists(lib_path):
  lib_path = '/projects/academic/azhang/tianlema/lib'
if os.path.exists(lib_path) and lib_path not in sys.path:
  sys.path.append(lib_path)
  
from dl.models.basic_models import *
from vin.vin import *


in_dict = {}
in_dict['in_dim'] = 2
in_dict['in_type'] = 'continuous'
in_dict['hidden_dim'] = [3, 5]
in_dict['embedding_dim'] = 5
in_dict['max_norm'] = False
in_dict['scale_grad_by_freq'] = False
in_dict['last_nonlinearity'] = True
in_dict['bias'] = False
in_dict['dense'] = False
in_dict['residual'] = False
in_dict['residual_layers'] = 'all'
input_list = [in_dict]

in_dict = {}
in_dict['in_dim'] = 3
in_dict['in_type'] = 'discrete'
in_dict['hidden_dim'] = [3, 5]
in_dict['embedding_dim'] = 5
in_dict['max_norm'] = None
in_dict['scale_grad_by_freq'] = False
in_dict['last_nonlinearity'] = True
in_dict['bias'] = False
in_dict['dense'] = False
in_dict['residual'] = False
in_dict['residual_layers'] = 'all'
input_list += [in_dict]

output_info = {}
output_info['hidden_dim'] = [7,11]
output_info['last_nonlinearity'] = False
output_info['bias'] = False
output_info['dense'] = False 
output_info['residual'] = False 
output_info['residual_layers'] = 'all-but-last'   
output_info = [output_info, output_info, output_info]

fusion_dict = {}
fusion_dict['type'] = 'repr-weighted-avg_repr'
fusion_dict['hidden_dim'] = [13, 17]
fusion_dict['output_info'] = output_info
fusion_dict['last_nonlinearity'] = True
fusion_dict['bias'] = False
fusion_dict['dense'] = False 
fusion_dict['residual'] = False
fusion_dict['residual_layers'] = 'all-but-last'
fusion_list = [fusion_dict]*2
fusion_lists = [fusion_list]*4

model = VIN(input_list, output_info, fusion_lists, 
            nonlinearity=nn.LeakyReLU(negative_slope=0.01, inplace=True))

for n, p in model.named_parameters():
  print(n, p.size())

x0 = torch.randn(3,2)
x1 = torch.tensor([1,2,1]).long()

xs = [x0, x1]
valid_loc = torch.tensor([[0,1],[1,0],[1,1]]).float()
model(xs, valid_loc, subset_repr=None, subset_output=None, return_repr=False)