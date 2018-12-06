import os
import functools
import itertools
import collections
import numpy as np
import pandas
from PIL import Image
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from .outlier import normalization
from .train import get_label_prob


def count_model_parameters(model):
  cnt = 0
  for p in model.parameters():
    cnt += p.numel()
  return cnt


def combine_dict(dict0, dict1, update_func=lambda a, b: b):
  """Combine two dictionaries
  
  Args:
    dict0: dict
    dict1: dict
    update_func: the function to combine two values from the same key;
      default function uses the value from dict1
      
  Returns:
    dict2: combined dict
    
  Examples:
    dict0 = {'a': [1], 'b': [2], 'c': [3]}
    dict1 = {'a': [11], 'd': [4]}
    combine_dict(dict0, dict1)
  """
  dict2 = {k: dict0[k] for k in set(dict0).difference(dict1)} # the unique part from dict0
  dict2.update({k: dict1[k] for k in set(dict1).difference(dict0)}) # the unique part from dict1
  for k in set(dict0).intersection(dict1):
    dict2[k] = update_func(dict0[k], dict1[k])
  return dict2


def compare_dict(dict0, dict1):
  """Compare two dictionary to see if they are equal

  Args:
    dict0: dictionary
    dict1: dictionary

  Returns:
    True if dict0 is the same as dict1 else False
  
  Examples:
    compare_dict({2:{2:np.array([1,2])}}, {2:{2:np.array([1,2])}}) # True
    
  """
  # if the keys are not equal, then return False
  if sorted(dict0) != sorted(dict1):
    return False
  for k, v0 in dict0.items():
    v1 = dict1[k]
    if isinstance(v0, dict):
      # the elements are dictionaries; call itself to compare
      if not isinstance(v1, dict):
        return False
      else:
        if not compare_dict(v0, v1):
          return False
    elif isinstance(v0, np.ndarray):
      # the elements are np.ndarray; call np.array_equal
      if not isinstance(v1, np.ndarray):
        return False
      else:
        if not np.array_equal(v0, v1):
          return False
    else:
      # otherwise directly use v0 != v1; it works for simple lists; but it may fail for some complex data
      if v0 != v1:
        return False
  return True


def append_dict(d, default):
  """Add content to dictionary d with default values from dictionary default
  
  Args:
    d: dictionary to be updated
    default: dictionary
    
  Implicitly returns updated d
  """
  for k, v in default.items():
    if k not in d:
      d[k] = v


def discrete_to_id(targets, start=0, sort=True, complex_object=False):
  """Change discrete variable targets to numeric values

  Args:
    targets: 1-d torch.Tensor or np.array, or a list
    start: the starting index for the first elements
    sort: sort the unique value, so that the 'smaller' values have smaller indices
    complex_object: input is not numeric, but complex objects, e.g., tuple

  Returns:
    target_ids: torch.Tensor or np.array with integer elements starting from start(=0 default)
    cls_id_dict: a dictionary mapping variables to their numeric ids

  """
  if complex_object:
    unique_targets = sorted(collections.Counter(targets))
  else:
    if isinstance(targets, torch.Tensor):
      targets = targets.cpu().detach().numpy()
    else:
      targets = np.array(targets) # if targets is already an np.array, then it does nothing
    unique_targets = np.unique(targets)
    if sort:
      unique_targets = np.sort(unique_targets)
  cls_id_dict = {v: i+start for i, v in enumerate(unique_targets)}
  target_ids = np.array([cls_id_dict[v] for v in targets])
  if isinstance(targets, torch.Tensor):
    target_ids = targets.new_tensor(target_ids)
  return target_ids, cls_id_dict
  

def get_f1_score(m, average='weighted', verbose=False):
  """Given a confusion matrix for binary classification, 
    calculate accuracy, precision, recall, F1 measure
    
  Args:
    m: confusion mat for binary classification
    average: if 'weighted': calculate metrics for each label, then get weighted average (weights are supports)
      if 'average': calculate average metrics for each label
      see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    verbose: if True, print result
  """
  def cal_f1(precision, recall):
    if precision + recall == 0:
      print('Both precision and recall are zero')
      return 0
    return 2*precision*recall / (precision+recall)
  m = np.array(m)
  t0 = m[0,0] + m[0,1]
  t1 = m[1,0] + m[1,1]
  p0 = m[0,0] + m[1,0]
  p1 = m[0,1] + m[1,1]
  prec0 = m[0,0] / p0
  prec1 = m[1,1] / p1
  recall0 = m[0,0] / t0
  recall1 = m[1,1] / t1
  f1_0 = cal_f1(prec0, recall0)
  f1_1 = cal_f1(prec1, recall1)
  if average == 'macro':
    w0 = 0.5
    w1 = 0.5
  elif average == 'weighted':
    w0 = t0 / (t0+t1)
    w1 = t1 / (t0+t1)
  prec = prec0*w0 + prec1*w1
  recall = recall0*w0 + recall1*w1
  f1 = f1_0*w0 + f1_1*w1
  acc = (m[0,0] + m[1,1]) / (t0+t1)
  if verbose:
    print(f'prec0={prec0}, recall0={recall0}, f1_0={f1_0}\n'
         f'prec1={prec1}, recall1={recall1}, f1_1={f1_1}')
  return acc, prec, recall, f1


def get_split(total_num, split_portion=[0.5, 0.5], split_size=None):
  """A helper function to split total_num into a number of chunks specified by split_portion or 
  split_size
  
  Args:
    split_portion: only used when split_size is None; if used, must be a list;
      every element must be positive; will be normalized internally
    split_size: if provided, must be a list; every element must be positive integers
      if sum(split_size) < total_num, add another chunk
      
  Returns:
    split_size: a list of positive integers; sum(split_size)==total_num
    
  """
  if split_size is not None:
    assert isinstance(split_size, list)
    if sum(split_size) < total_num:
      split_size.append(total_num - sum(split_size))
      print('Warning: added one split')
    if sum(split_size) > total_num:
      raise ValueError(f'sum(split_size)={sum(split_size)} > {total_num} (total_num)')
  else:
    split_size = []
    for s in split_portion[:-1]:
      assert s>0
      split_size.append(int(s/sum(split_portion)*total_num))
    split_size.append(total_num - sum(split_size))
  for s in split_size:
    assert s > 0 and isinstance(s, int)
  return split_size


def dist(params1, params2=None, dist_fn=torch.norm): #pylint disable=no-member
    """Calculate the norm of params1 or the distance between params1 and params2; 
        Common usage calculate the distance between two model state_dicts.
    Args:
        params1: dictionary; with each item a torch.Tensor
        params2: if not None, should have the same structure (data types and dimensions) as params1
    """
    if params2 is None:
        return dist_fn(torch.Tensor([dist_fn(params1[k]) for k in params1]))
    d = torch.Tensor([dist_fn(params1[k] - params2[k]) for k in params1])
    return dist_fn(d)
    
class AverageMeter(object):
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def pil_loader(path, format = 'RGB'):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(format)


class ImageFolder(data.Dataset):
    def __init__(self, root, imgs, transform = None, target_transform = None, 
                 loader = pil_loader, is_test = False):
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.is_test = is_test
    
    def __getitem__(self, idx):
        if self.is_test:
            img = self.imgs[idx]
        else:
            img, target = self.imgs[idx]
        img = self.loader(os.path.join(self.root, img))
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test and self.target_transform is not None:
            target = self.target_transform(target)
        if self.is_test:
            return img
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs)

        
def check_acc(output, target, topk=(1,)):
    if isinstance(output, tuple):
        output = output[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1)
    res = []
    for k in topk:
        acc = (pred.eq(target.contiguous().view(-1,1).expand(pred.size()))[:, :k]
               .float().contiguous().view(-1).sum(0))
        acc.mul_(100 / target.size(0))
        res.append(acc)
    return res


### Mainly developed for TCGA data analysis
def select_samples(mat, aliquot_ids, feature_ids, patient_clinical=None, clinical_variable='PFI', 
                   sample_type='01', drop_duplicates=True, remove_na=True):
  """Select samples with given sample_type ('01');
     if drop_duplicates is True (by default), remove technical duplicates; 
     and if remove_na is True (default), remove features that have NA;
     If patient_clinical is not None, further filter out samples with clinical_variable being NA
  """
  mat = pandas.DataFrame(mat, columns=feature_ids) # Use pandas to drop NA
  # Select samples with sample_type(='01')
  idx = np.array([[i,s[:12]] for i, s in enumerate(aliquot_ids) if s[13:15]==sample_type])
  # Remove technical duplicate
  if drop_duplicates:
    idx = pandas.DataFrame(idx).drop_duplicates(subset=[1]).values
    mat = mat.iloc[idx[:,0].astype(int)]
  aliquot_ids = aliquot_ids[idx[:,0].astype(int)]
  if remove_na:
  # Remove features that have NA values
    mat = mat.dropna(axis=1)
    feature_ids = mat.columns.values
  mat = mat.values
  if patient_clinical is not None:
    idx = [s[:12] in patient_clinical and not np.isnan(patient_clinical[s[:12]][clinical_variable]) 
           for s in aliquot_ids]
    mat = mat[idx]
    aliquot_ids = aliquot_ids[idx]
  return mat, aliquot_ids, feature_ids


def get_feature_feature_mat(feature_ids, gene_ids, feature_gene_adj, gene_gene_adj, 
                            max_score=1000):
  """Calculate feature-feature interaction matrix based on their mapping to genes 
    and gene-gene interactions:
    feature_feature = feature_gene * gene_gene * feature_gene^T (transpose)
  
  Args:
    feature_ids: np.array([feature_names]), dict {id: feature_name}, or {feature_name: id}
    gene_ids: np.array([gene_names]), dict {id: gene_name}, or {gene_name: id}
    feature_gene_adj: np.array([[feature_name, gene_name, score]]) 
      with rows corresponding to features and columns genes; 
      or (Deprecated) a list (gene) of lists of feature_ids. 
        Note this is different from np.array input; len(feature_gene_adj) = len(gene_ids)
    gene_gene_adj: an np.array. Each row is (gene_name1, gene_name2, score)
    max_score: default 1000. Normalize confidence scores in gene_gene_adj to be in [0, 1]
    
  Returns:
    feature_feature_mat: np.array of shape (len(feature_ids), len(feature_ids))
    
  """
  def check_input_ids(ids):
    if isinstance(ids, np.ndarray) or isinstance(ids, list):
      ids = {v: i for i, v in enumerate(ids)} # Map feature names to indices starting from 0
    elif isinstance(ids, dict):
      if sorted(ids) == list(range(len(ids))):
        # make sure it follows format {feature_name: id}
        ids = {v: k for k, v in ids.items()}
    else:
      raise ValueError(f'The input ids should be a list/np.ndarray/dictionary, '
                       'but is {type(feature_ids)}')
    return ids
  feature_ids = check_input_ids(feature_ids)
  gene_ids = check_input_ids(gene_ids)
  
  idx = []
  if isinstance(feature_gene_adj, list): # Assume feature_gene_adj is a list; this is deprecated
    for i, v in enumerate(feature_gene_adj):
      for j in v:
        idx.append([j, i, 1])
  elif isinstance(feature_gene_adj, np.ndarray) and feature_gene_adj.shape[1] == 3:
    for v in feature_gene_adj: 
      if v[0] in feature_ids and v[1] in gene_ids:
        idx.append([feature_ids[v[0]], gene_ids[v[1]], float(v[2])])
  else:
    raise ValueError('feature_gene_adj should be an np.ndarray of shape (N, 3) '
                     'or a list of lists (deprecated).')
  idx = np.array(idx).T
  feature_gene_mat = torch.sparse.FloatTensor(torch.tensor(idx[:2]).long(), 
                                              torch.tensor(idx[2]).float(), 
                                              (len(feature_ids), len(gene_ids)))
  # Extract a subnetwork from gene_gene_adj
  # Assume there is no self-loop in gene_gene_adj 
  # and it contains two records for each undirected edge
  idx = []
  for v in gene_gene_adj: 
    if v[0] in gene_ids and v[1] in gene_ids:
      idx.append([gene_ids[v[0]], gene_ids[v[1]], v[2]/max_score])
  # Add self-loops
  for i in range(len(gene_ids)):
    idx.append([i, i, 1.])
  idx = np.array(idx).T
  gene_gene_mat = torch.sparse.FloatTensor(torch.tensor(idx[:2]).long(),
                                          torch.tensor(idx[2]).float(),
                                          (len(gene_ids), len(gene_ids)))
  feature_feature_mat = feature_gene_mat.mm(gene_gene_mat.mm(feature_gene_mat.to_dense().t()))
  return feature_feature_mat.numpy()


def get_overlap_samples(sample_lists, common_list=None, start=0, end=12, return_common_list=False):
  """Given a list of aliquot_id lists, find the common sample ids
  
  Args:
    sample_lists: a iterable of sample (aliquot) id lists
    common_list: if None (default), find the interaction of sample_lists; 
      if provided, it should not be a set, because iterating over a set can be different from different runs
    start: default 0; assume sample ids are strings; 
      when finding overlapping samples, only consider a specific range [start, end)
    end: default 12, for TCGA BCR barcode
    return_common_list: if True, return a set containing common list for backward compatiablity,
      returns a sorted common list is a better option
  
  Returns:
    np.array of shape (len(sample_lists), len(common_list))
  """ 
  sample_lists = [[s_id[start:end] for s_id in sample_list] for sample_list in sample_lists]
  if common_list is None:
    common_list = functools.reduce(lambda x,y: set(x).intersection(y), sample_lists)
    if return_common_list:
      return common_list
    common_list = sorted(common_list) # iterate over set can vary from different runs
  for s in sample_lists: # make sure every list in sample_lists contains all elements in common_list
    assert len(set(common_list).difference(s)) == 0 
  idx_lists = np.array([[sample_list.index(s_id) for s_id in common_list] 
                        for sample_list in sample_lists])
  return idx_lists


# Select samples that have target variable(s) is in clinical file
def filter_clinical_dict(target_variable, target_variable_type, target_variable_range, 
                         clinical_dict):
  """Select patients with given target variable, its type and range in clinical data
  To save computation time, I assume all target variable(s) names are in clinical_dict without verification;
  
  Args:
    target_variable: str or a list of strings
    target_variable_type: 'discrete' or 'continuous' or a list of 'discrete' or 'continuous'
    target_variable_range: a list of values for 'continous' type, it is [lower_bound, upper_bound]
      or a list of list; target_variable, target_variable_type, target_variable_range must match
    clinical_dict: a dictionary of dictinaries; 
      first-level keys: patient ids, second-level keys: variable names
  
  Returns:
    clinical_dict: newly constructed clinical_dict with all patients having target_variables
    
  Examples:
    target_variable = ['PFI', 'OS.time'] 
    target_variable_type = ['discrete', 'continuous']
    target_variable_range = [[0, 1], [0, float('Inf')]]
    clinical_dict = filter_clinical_dict(target_variable, target_variable_type, target_variable_range, 
                            patient_clinical)
    assert sorted([k for k, v in patient_clinical.items() if v['PFI'] in [0,1] and not np.isnan(v['OS.time'])]) == 
      sorted(clinical_dict.keys())

  """
  if isinstance(target_variable, str):
    if target_variable_type == 'discrete':
      clinical_dict = {p:v for p, v in clinical_dict.items() 
                       if v[target_variable] in target_variable_range}
    elif target_variable_type == 'continuous':
      clinical_dict = {p:v for p, v in clinical_dict.items() 
                       if v[target_variable] >= target_variable_range[0] 
                       and v[target_variable] <= target_variable_range[1]}
  
  elif isinstance(target_variable, (list, tuple)):
    # Brilliant recursion
    for tar_var, tar_var_type, tar_var_range in zip(target_variable, target_variable_type, target_variable_range):
      clinical_dict = filter_clinical_dict(tar_var, tar_var_type, tar_var_range, clinical_dict)
      
  return clinical_dict


def get_target_variable(target_variable, clinical_dict, sel_patient_ids):
  """Extract target_variable from clinical_dict for sel_patient_ids
  If target_variable is a single str, it is only one line of code
  If target_variable is a list, recursively call itself and return a list of target variables
  
  Assume all sel_patient_ids have target_variable in clinical_dict
  
  """
  if isinstance(target_variable, str):
    return [clinical_dict[s][target_variable] for s in sel_patient_ids]
  elif isinstance(target_variable, (list, str)):
    return [[clinical_dict[s][tar_var] for s in sel_patient_ids] for tar_var in target_variable]


def normalize_continuous_variable(y_targets, target_variable_type, transform=True, forced=False, 
                        threshold=10, rm_outlier=True, whis=1.5, only_positive=True, max_val=1):
  """Normalize continuous variable(s)
    If a variable is 'continuous', then call normalization() in outlier.py
  
  Args:
    y_targets: a np.array or a list of np.array
    target_variable_type: can be a string: 'continous' or 'discrete' (do nothing but return the input)
      or a list of strings
    transform, forced, threshold, rm_outlier, whis, only_positive, max_val are all passed to normalization

  """
  if isinstance(target_variable_type, str):
    if target_variable_type=='continuous':
      y_targets = normalization(y_targets, transform=transform, forced=forced, threshold=threshold, 
                                rm_outlier=rm_outlier, whis=whis, only_positive=only_positive, 
                                max_val=max_val, diagonal=False, symmetric=False)
    return y_targets
  elif isinstance(target_variable_type, list):
    return [normalize_continuous_variable(y, var_type, transform=transform, forced=forced, 
            threshold=threshold, rm_outlier=rm_outlier, whis=whis, only_positive=only_positive, 
            max_val=max_val) for y, var_type in zip(y_targets, target_variable_type)]
  else:
    raise ValueError(f'target_variable_type should be a str or list of strs, but is {target_variable_type}')


def get_label_distribution(ys, check_num_cls=True):
  """Get label distributions for a list of labels
  
  Args:
    ys: an iterable (e.g., list) of labels (1-d numpy.array or torch.Tensor);
      the most common usage is get_label_distribution([y_train, y_val, y_test])
    check_num_cls: only if it is True, ensure that each list of labels will have the same number of classes 
      and also print out the message
    
  Returns:
    label_prob: a list of label distributions (multinomial);
    
  """
  num_cls = 0
  label_probs = []
  for i, y in enumerate(ys):
    if len(y)>0:
      label_prob = get_label_prob(y, verbose=False)
      label_probs.append(label_prob)
      if check_num_cls:
        if num_cls > 0:
          assert num_cls == len(label_probs[-1]), f'{i}: {num_cls} != {len(label_probs[-1])}'
        else:
          num_cls = len(label_probs[-1])
    else:
      label_probs.append([])
  if check_num_cls:
    if isinstance(label_probs, torch.Tensor):
      print('label distribution:\n', torch.stack(label_probs, dim=1))
    else:
      print('label distribution:\n', np.stack(label_probs, axis=1))
  return label_probs


def get_shuffled_data(sel_patient_ids, clinical_dict, cv_type, instance_portions, group_sizes,
                     group_variable_name, seed=None, verbose=True):
  """Shuffle sel_patient_ids and split them into multiple splits, 
    in most cases, train, val and test sets; 
  
  Args:
    sel_patient_ids: a list of object (patient) ids
    clinical_dict: a dictionary of dictionaries; 
      first-level keys: object ids; second-level keys: attribute names;
    cv_type: either 'group-shuffle' or 'instance-shuffle'; in most cases:
      if 'group-shuffle', split groups into train, val and test set according to group_sizes or
      implicitly instance_portions;
      if 'instance-shuffle': split based on instance_portions
    instance_portions: a list of floats; the proportions of samples in each split; 
      when cv_type=='group-shuffle' and group_sizes is given, then instance_portions is not used
    group_sizes: the number of groups in each split; only used when cv_type=='group-shuffle'
    group_variable_name: the attribute name for group information
    
  Returns:
    sel_patient_ids: shuffled object ids
    idx_splits: a list of indices, e.g., [train_idx, val_idx, test_idx]
      sel_patient_ids[train_idx] will get patient ids for training
      
  """
  np.random.seed(seed)
  sel_patient_ids = np.random.permutation(sel_patient_ids)
  num_samples = len(sel_patient_ids)
  idx_splits = []
  if cv_type == 'group-shuffle':
    # for my TCGA project, I used disease types as groups; thus the variable name is named 'disease_types'
    disease_types = sorted({clinical_dict[s][group_variable_name] for s in sel_patient_ids})
    num_disease_types = len(disease_types)
    np.random.shuffle(disease_types)
    type_splits = []
    cnt = 0
    for i in range(len(group_sizes)-1):
      if group_sizes[i] < 0: 
        # use instance_portion as group portions
        assert sum(instance_portions) == 1
        group_sizes[i] = round(instance_portions[i] * num_disease_types)
      type_splits.append(disease_types[cnt:cnt+group_sizes[i]])
      cnt = cnt+group_sizes[i]
      # do not use i to enumerate sel_patient_ids because i is used
      idx_splits.append([j for j, s in enumerate(sel_patient_ids) 
                         if clinical_dict[s][group_variable_name] in type_splits[i]])
    # process the last split
    if group_sizes[-1] >=0: # for most of time, set group_sizes[-1] = num_test_types = -1
      # almost never set group_sizes[-1] = 0, which will be useless
      assert group_sizes[-1] == num_disease_types - sum(group_sizes[:-1])
    if cnt == len(disease_types):
      print('The last group is empty, thus not included')
    else:
      type_splits.append(disease_types[cnt:]) 
      idx_splits.append([i for i, s in enumerate(sel_patient_ids) 
                          if clinical_dict[s][group_variable_name] in type_splits[-1]])
  elif cv_type == 'instance-shuffle':
    # because sel_patient_ids has already been shuffled, we do not need to shuffle indices
    cnt = 0
    assert sum(instance_portions) == 1
    for i in range(len(instance_portions)-1):
      n = round(instance_portions[i]*num_samples)
      idx_splits.append(list(range(cnt, cnt+n)))
      cnt = cnt + n
    # process the last split
    if cnt == num_samples:
      # this can rarely happen
      print('The last split is empty, thus not included')
    else:
      idx_splits.append(list(range(cnt, num_samples)))
  
  def get_type_cnt_msg(p_ids):
    """For a list p_ids, prepare group statistics for printing
    """
    cnt_dict = dict(collections.Counter([clinical_dict[p_id][group_variable_name] 
                                       for p_id in p_ids]))
    return f'{len(cnt_dict)} groups: {cnt_dict}'

  if verbose:
    msg = f'{cv_type}: \n'
    msg += '\n'.join([f'split {i}: {len(v)} samples ({len(v)/num_samples:.2f}), '
                      f'{get_type_cnt_msg(sel_patient_ids[v])}'
                      for i, v in enumerate(idx_splits)])
    print(msg)
  return sel_patient_ids, idx_splits


def target_to_numpy(y_targets, target_variable_type, target_variable_range):
  """y_targets is a list or a list of lists; transform it to numpy array
  For a discrete variable, generate numerical class labels from 0;
  for a continous variable, simply call np.array(y_targets);
  use recusion to handle a list of target variables
  
  Args:
    y_targets: a list of objects (strings/numbers, must be comparable) or lists
    target_variable_type: a string or a list of string ('discrete' or 'continous')
    target_variable_range: only used for sanity check for discrete variables
    
  Returns:
    y_true: a numpy array or a list of numpy arrays of type either float or int
    
  """
  
  if isinstance(target_variable_type, str):
    y_true = np.array(y_targets)
    if target_variable_type == 'discrete':
      unique_cls = np.unique(y_true)
      num_cls = len(unique_cls)
      if sorted(unique_cls) != sorted(target_variable_range):
        print(f'unique_cls: {unique_cls} !=\ntarget_variable_range {target_variable_range}')
      cls_idx_dict = {p.item(): i for i, p in enumerate(sorted(unique_cls))}
      y_true = np.array([cls_idx_dict[i.item()] for i in y_true])
      print(f'Changed class labels for the model: {cls_idx_dict}')
  elif isinstance(target_variable_type, (list, tuple)):
    y_true = [target_to_numpy(y, tar_var_type, tar_var_range) 
              for y, tar_var_type, tar_var_range in 
              zip(y_targets, target_variable_type, target_variable_range)]
  else:
    raise ValueError(f'target_variable_type must be str, list or tuple, '
                     f'but is {type(target_variable_type)}')
  return y_true


def get_mi_acc(xs, y_true, var_names, var_name_length=35):
  """Get mutual information (MI), adjusted MI, the maximal acc from Bayes classifier 
  for a list of discrete predictors xs and target y_true
  For all combinations of xs calculate MI, Adj_MI, and Bayes_ACC

  Args:
    xs: a list of tensors or numpy arrays
    y_true: a tensor or numpy array

  Returns:
    a list of dictionaries with key being the variable name
  """
  if isinstance(xs[0], torch.Tensor):
    xs = [x.cpu().detach().numpy() for x in xs]
  if isinstance(y_true, torch.Tensor):
    y_true = y_true.cpu().detach().numpy()
  result = []
  print('{:^{var_name_length}}\t{:^5}\t{:^6}\t{:^9}'.format('Variable', 'MI', 'Adj_MI', 'Bayes_ACC', 
    var_name_length=var_name_length))
  for i, l in enumerate(itertools.chain.from_iterable(itertools.combinations(range(len(xs)), r) 
                                     for r in range(1, 1+len(xs)))):
    if len(l) == 1:
      new_x = xs[l[0]]
      msg = f'{var_names[i]:^{var_name_length}}\t'
    else: # len(l) > 1
      new_x = [tuple([v.item() for v in s]) for s in zip(*[xs[j] for j in l])]
      new_x = discrete_to_id(new_x, complex_object=True)[0]
      msg = f'{"-".join(map(str, l)):^{var_name_length}}\t'
    mi = sklearn.metrics.mutual_info_score(y_true, new_x)
    adj_mi = sklearn.metrics.adjusted_mutual_info_score(y_true, new_x)
    bayes_acc = (sklearn.metrics.confusion_matrix(y_true, new_x).max(axis=0).sum() / len(y_true))
    result.append({msg: [mi, adj_mi, bayes_acc]})
    msg += f'{mi:^5.3f}\t{adj_mi:^6.3f}\t{bayes_acc:^9.3f}'
    print(msg)
  return result
  # p1 = sklearn.metrics.confusion_matrix(y_true.numpy(), new_x)[:2].reshape(-1)
  # p2 = (np.bincount(y_true.numpy())[:,None] * np.bincount(new_x)).reshape(-1)
  # p = torch.distributions.categorical.Categorical(torch.tensor(p1, dtype=torch.float))
  # q = torch.distributions.categorical.Categorical(torch.tensor(p2, dtype=torch.float))
  # torch.distributions.kl.kl_divergence(p,q)


def merge_inner_list(ys, dim=0):
  """Concatenate tensors in the inner list; 
    this is mainly used for construct the entire output from batch outputs;
    this is used in pred_func in ...vin.vin.py
    for ease of description, use integers to represent tensors, 
      and tuples to represent concatenated tensors in the following toy examples
      ys = [1,2,3] --> (1,2,3)
      ys = [[1,2,3], [4,5,6]] --> [(1,4), (2,5), (3,6)]
      ys = [[[1,2,3], [4,5]], [[6,7,8], [9,10]]] --> [[(1,6),(2,7),(3,8)], [(4,9),(5,10)]]

  Args:
    ys: a list of tensors, or a nested list of tensors
    dim: default 0 (for almost all the cases), passed for torch.cat

  Examples:
    ys = [[[torch.tensor([1]), torch.tensor([2]), torch.tensor([3])], 
            [torch.tensor([4]), torch.tensor([5])]], 
          [[torch.tensor([6]), torch.tensor([7]), torch.tensor([8])], 
            [torch.tensor([9]), torch.tensor([10])]]]
    merge_inner_list(ys)
    
  """
  if isinstance(ys[0], torch.Tensor):
    return torch.cat(ys, dim=dim)
  elif isinstance(ys[0], (list, tuple)):
    merged = list(map(list, zip(*ys)))
    return [merge_inner_list(s, dim=dim) for s in merged]


def tensor_to_num(tensors):
  """Given a (nested) list/tuple of singlton tensors, get their corresponding numerical values by call tensor.item()
    For example: if input tensors = [(torch.tensor(1), torch.tensor(2))]
      then returns [(1, 2)];
      note the nested list and tuple structure are kept
    This is another example of recursive function
  """
  if isinstance(tensors, torch.Tensor):
    return tensors.item()
  elif isinstance(tensors, tuple):
    return tuple([tensor_to_num(tensor) for tensor in tensors])
  elif isinstance(tensors, list):
    return [tensor_to_num(tensor) for tensor in tensors]


def adj_list_to_mat(adj_list, name_to_id=None, bipartite=False, add_self_loop=False, symmetric=False, 
  return_list=False):
  """Generate adjacency matrix from adjacency list, which is directed, from left to right;
    for bipartite graph, the source and target have their own name spaces,
    otherwise both source and target share the same name space
  
  Args:
    adj_list: a 2-d array (either a list of lists or a 2-d np.array) of size (N, 2);
      the EDGE DIRECTION is LEFT --> RIGHT (LEFT should have a lower id)
    name_to_id: default None; generate one from adj_list
      if given, 
        if bipartite is False, it must be a dictionary with values from 0 to len(name_to_id)-1;
        otherwise it is a list [source_name_to_id, target_name_to_id]
    bipartite: if True, treat as a bipartite graph, with left column being source and right being target
    add_self_loop: only used when bipartite is False and return_list is False; 
      if True, then add a self loop for each node
    symmetric: only used when bipartite is False and return_list is False; 
      if True, make the adj_mat to be symmetric
    return_list: if True, instead of adjacency matrix, return adjacency list with new integer ids;
      this is especially for large sparse matrix
      
  Returns:
    adj_mat: a 2-d np.array; the elements are either 0 or 1;
      with element locating at (i, j) = 1 having the meaning there is an directed edge from i to j
    id_to_name: if bipartite is False, it is a dictionary, reverse of name_to_id (i.e., switching the keys and values)
      if bipartite is True, id_to_name = [source_id_to_name, target_id_to_name]

  Examples:
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    adj_list_to_mat(adj_list, name_to_id=None, bipartite=False)
  """
  # in case it is a list
  adj_list = np.array(adj_list)
  if name_to_id is None:
    print('Generating name_to_id')
    if bipartite:
      source_names = np.unique(adj_list[:, 0])
      target_names = np.unique(adj_list[:, 1])
      source_name_to_id = {n: i for i, n in enumerate(sorted(source_names))}
      target_name_to_id = {n: i for i, n in enumerate(sorted(target_names))}
      name_to_id = [source_name_to_id, target_name_to_id]
    else:
      names = np.unique(adj_list[:,:2]) # when adj_list have more than two columns, only use the first two
      name_to_id = {n: i for i, n in enumerate(sorted(names))}
  if bipartite:
    # in case name_to_id is given
    source_name_to_id, target_name_to_id = name_to_id
    # # name_to_id can contain nodes that are not in adj_list; so these two assertions should not be used any more
    # assert set(source_name_to_id) == set(np.unique(adj_list[:, 0]))
    # assert set(target_name_to_id) == set(np.unique(adj_list[:, 1]))
    source_id_to_name = {i: n for n, i in source_name_to_id.items()}
    target_id_to_name = {i: n for n, i in target_name_to_id.items()}
    id_to_name = [source_id_to_name, target_id_to_name]
  else:
    # # name_to_id can contain nodes that are not in adj_list; so these two assertions should not be used any more
    # assert set(name_to_id) == set(np.unique(adj_list[:,:2]))
    id_to_name = {i: n for n, i in name_to_id.items()}
    # to provide the same interface as for bipartite graph
    source_name_to_id = name_to_id
    target_name_to_id = name_to_id
  if return_list:
    adj_mat = np.zeros_like(adj_list)
    for i, s in enumerate(adj_list):
      left = source_name_to_id[s[0]]
      right = target_name_to_id[s[1]]
      adj_mat[i, 0] = left
      adj_mat[i, 1] = right
  else:
    adj_mat = np.zeros((len(source_name_to_id), len(target_name_to_id)))
    for s in adj_list:
      left = source_name_to_id[s[0]]
      right = target_name_to_id[s[1]]
      adj_mat[left, right] = 1
      if not bipartite and symmetric:
        adj_mat[right, left] = 1
    if not bipartite and add_self_loop:
      adj_mat[range(len(source_name_to_id)), range(len(target_name_to_id))] = 1
      print('Added self-loop')
  return adj_mat, id_to_name


def adj_list_to_attention_mats(adj_list, num_steps=10, name_to_id=None, bipartite=True, add_self_loop=False,
                               symmetric=False, target_to_source=None, use_transition_matrix=True, 
                               softmax_normalization=False, min_value=-100, Ms=None, Mt=None, 
                               device=torch.device('cpu')):
  """Use adj_list to generate attention matrices
  
  Args:
    adj_list: a 2-d array as an adjacency list, which is directed, from left to right
    num_steps: int, number of steps (layers) of walk on the graph
    name_to_id: if given, 
      for non-bipartite graph, it is a dictionary mapping node names to their integer ids;
      for bipartite graph, it is a list of two dictionaries
      default None, infer it from adj_list
    bipartite: if True, treat left side and right side as source and target, with two name spaces;
      if target_to_source is None, then assume the bipartite graph is undirected
    add_self_loop and symmetric are passed to adj_list_to_mat
    target_to_source: only used when for bipartite graph; 
      if not None, it should be a 2-d array like adj_list;
      while adj_list is source --> target, target_to_source is target --> source
    use_transition_matrix: if True, normalize Ms to be a transition matrix, 
      and then perform random walk, no need normalization
      if False, multiply adjacency matrix a few times, and then normalize
    softmax_normalization: if True, use softmax normalization; otherwise, use M / M.sum(dim=1, keepdim=True)
      only useful when use_transition_matrix is False
    min_value: default -100; when using softmax normalization, 
      the 0 entries in the adjacency matrix will be set to this value so that the attention to them will be ~0
    Ms: the adjacency matrix from source to target; 
      if given, then do not call adj_list_to_mat and thus most of the arguments won't be used;
      default None, no use
    Mt: the adjacency matrix from target to source;
      if given, then do not call adj_list_to_mat; default None, not used
    device: default torch.device('cpu');
      I use torch.Tensor to manipulate adj mats; 
      it is convenient to use torch.nn.functional.softmax; but it is better to just use numpy
      
  Returns:
    attention_mats: 
      for bipartite graph: attention_mats = [source_attention_mats, target_attention_mats];
        source_attention_mats stores attention mats from source to the nodes in previous layers;
        target_attention_mats stores attention mats from target to the nodes in previous layers;
        Let Ms be the adjacency matrix from source to target, and Mt from target to source, then:
          source_attention_mats = [Ms.T, (Ms*Mt).T, (Ms*Mt*Ms).T, ...],
          target_attention_mats = [Mt.T, (Mt*Ms).T, (Mt*Ms*Mt).T, ...];
          source_attention_mats stores transition mat from source with 1,2,... steps,
          target_attention_mats stores transition mat from target with 1,2,... steps,
        These transition mats are transposed
      for non-bipartite graph: attention_mats = source_attention_mats as in bipartite graph
        because now source and target are the same set
      
    id_to_name: a dictionary, the reverse of name_to_id

  Examples:
    adj_list = [[3, 4], [5, 6], [5, 4], [6, 4], [3, 6]]
    adj_mat, _ = adj_list_to_mat(adj_list, bipartite=False, add_self_loop=True)
    in_features, out_features = adj_mat.shape
    attention_mats, _ = adj_list_to_attention_mats(adj_list, bipartite=False, add_self_loop=True)
  """
  if Ms is None:
    if bipartite and name_to_id is None and target_to_source is not None:
      # for directed bipartite graph, we have to merge the source and target nodes in two adj lists
      adj_list = np.array(adj_list) # i.e. source_to_target
      target_to_source = np.array(target_to_source)
      source_names = np.unique(np.concatenate([adj_list[:,0], target_to_source[:,1]]))
      source_name_to_id = {n: i for i, n in enumerate(sorted(source_names))}
      target_names = np.unique(np.concatenate([adj_list[:,1], target_to_source[:,0]]))
      target_name_to_id = {n: i for i, n in enumerate(sorted(target_names))}
      name_to_id = [source_name_to_id, target_name_to_id]
    # Ms is the adjacency mat from source to target (source --> target, or left --> right)
    Ms, id_to_name = adj_list_to_mat(adj_list, name_to_id=name_to_id, bipartite=bipartite, 
                                    add_self_loop=add_self_loop, symmetric=symmetric, return_list=False)
  else:
    Ms = np.array(Ms) # make sure it is np.array
    id_to_name = None # provide the same interface as for the other case
  Ms = torch.tensor(Ms, device=device).float()
  num_edges_Ms = int((Ms>0).float().sum().item())
  if bipartite:
    if target_to_source is None and Mt is None:
      # Assume the bipartite graph is undirected; Mt is the adj_mat from target to source
      Mt = Ms.t()
    else:
      if Mt is None:
        # name_to_id is important!
        Mt, _ = adj_list_to_mat(target_to_source, name_to_id=[name_to_id[1], name_to_id[0]], 
                                bipartite=True, add_self_loop=False, symmetric=False, return_list=False)
      else:
        Mt = np.array(Mt)
      Mt = torch.tensor(Mt, device=device).float()
  if use_transition_matrix:
    if torch.any(Ms.sum(dim=1)==0):
      print(f'Warning: there are {torch.sum(Ms.sum(dim=1)==0).item()} isolated node(s)!')
    if softmax_normalization or torch.any(Ms.sum(dim=1)==0):
      Ms[Ms==0] = min_value
      Ms = nn.functional.softmax(Ms, dim=1)
    else:
      Ms = Ms / Ms.sum(dim=1, keepdim=True)
    if bipartite:
      if softmax_normalization:
        Mt[Mt==0] = min_value
        Mt = nn.functional.softmax(Mt, dim=1)
      else:
        Mt = Mt / Mt.sum(dim=1, keepdim=True)
  in_features, out_features = Ms.size()
  if bipartite:
    print(f'Bipartite graph: in_features={in_features}, out_features={out_features}')
    attention_mats = [[], []]
  else:
    assert in_features == out_features
    print(f'Graph with {in_features} nodes and {num_edges_Ms} edges')
    attention_mats = []
  for i in range(1, num_steps+1):
    # there may be in-place modification, thus use torch.clone()
    new_Ms = Ms.clone()
    if bipartite:
      new_Mt = Mt.clone()  # Mt is the transition matrix from target to source
      for j in reversed(range(i-1)): # this is very tricky
        # j starts with i-2 and ends with 0
        if (i-j)%2 == 0:
          # in the first step (j=i-2), Ms*Mt, or Mt*Ms
          new_Ms = torch.mm(new_Ms, Mt)
          new_Mt = torch.mm(new_Mt, Ms)
        else:
          new_Ms = torch.mm(new_Ms, Ms)
          new_Mt = torch.mm(new_Mt, Mt)
      if not use_transition_matrix:
        if softmax_normalization:
          # normalize attention mats
          # 0 entries should have ~0 normalized attention
          new_Ms[new_Ms == 0] = min_value
          new_Mt[new_Mt == 0] = min_value
          new_Ms = torch.nn.functional.softmax(new_Ms, dim=1)
          new_Mt = torch.nn.functional.softmax(new_Mt, dim=1)
        else:
          new_Ms = new_Ms / new_Ms.sum(dim=1, keepdim=True)
          new_Mt = new_Mt / new_Mt.sum(dim=1, keepdim=True)
      attention_mats[0].append(new_Ms.t())
      attention_mats[1].append(new_Mt.t())
    else:
      for j in range(i-1):
        new_Ms = torch.mm(new_Ms, Ms)
      if not use_transition_matrix:
        if softmax_normalization:
          # 0 entries should have ~0 normalized attention
          new_Ms[new_Ms == 0] = min_value
          new_Ms = torch.nn.functional.softmax(new_Ms, dim=1)
        else:
          new_Ms = new_Ms / new_Ms.sum(dim=1, keepdim=True)
      attention_mats.append(new_Ms.t())
  return attention_mats, id_to_name