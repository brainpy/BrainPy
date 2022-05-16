# -*- coding: utf-8 -*-

from typing import Dict, Type, Union, Tuple, List
import logging
import pickle

import numpy as np

from brainpy import errors
from brainpy.base.collector import TensorCollector

logger = logging.getLogger('brainpy.base.io')

__all__ = [
  'SUPPORTED_FORMATS',
  'save_as_h5',
  'save_as_npz',
  'save_as_pkl',
  'save_as_mat',
  'load_by_h5',
  'load_by_npz',
  'load_by_pkl',
  'load_by_mat',
]

SUPPORTED_FORMATS = ['.h5', '.hdf5', '.npz', '.pkl', '.mat']


def check_dict_data(
    a_dict: Dict,
    key_type: Union[Type, Tuple[Type, ...]] = None,
    val_type: Union[Type, Tuple[Type, ...]] = None,
    name: str = None
):
  """Check the dict data."""
  name = '' if (name is None) else f'"{name}"'
  if not isinstance(a_dict, dict):
    raise ValueError(f'{name} must be a dict, while we got {type(a_dict)}')
  if key_type is not None:
    for key, value in a_dict.items():
      if not isinstance(key, key_type):
        raise ValueError(f'{name} must be a dict of ({key_type}, {val_type}), '
                         f'while we got ({type(key)}, {type(value)})')
  if val_type is not None:
    for key, value in a_dict.items():
      if not isinstance(value, val_type):
        raise ValueError(f'{name} must be a dict of ({key_type}, {val_type}), '
                         f'while we got ({type(key)}, {type(value)})')


def _check_module(module, module_name, ext):
  """Check whether the required module is installed."""
  if module is None:
    raise errors.PackageMissingError(
      '"{package}" must be installed when you want to save/load data with {ext} '
      'format. \nPlease install {package} through "pip install {package}" or '
      '"conda install {package}".'.format(package=module_name, ext=ext)
    )


def _check_missing(variables, filename):
  if len(variables):
    logger.warning(f'There are variable states missed in {filename}. '
                   f'The missed variables are: {list(variables.keys())}.')


def _check_target(target):
  from .base import Base
  if not isinstance(target, Base):
    raise TypeError(f'"target" must be instance of "{Base.__name__}", but we got {type(target)}')


not_found_msg = ('"{key}" is stored in {filename}. But we does '
                 'not find it is defined as variable in {target}.')
id_dismatch_msg = ('{key1} and {key2} is the same data in {filename}. '
                   'But we found they are different in {target}.')

DUPLICATE_KEY = 'duplicate_keys'
DUPLICATE_TARGET = 'duplicate_targets'


def _load(
    target,
    verbose: bool,
    filename: str,
    load_vars: dict,
    duplicates: Tuple[List[str], List[str]],
    remove_first_axis: bool = False
):
  from brainpy import math as bm

  # get variables
  _check_target(target)
  variables = target.vars(method='absolute', level=-1)
  all_names = list(variables.keys())

  # read data from file
  for key in load_vars.keys():
    if verbose:
      print(f'Loading {key} ...')
    if key not in variables:
      raise KeyError(not_found_msg.format(key=key, target=target.name, filename=filename))
    if remove_first_axis:
      value = load_vars[key][0]
    else:
      value = load_vars[key]
    variables[key].value = bm.asarray(value)
    all_names.remove(key)

  # check duplicate names
  duplicate_keys = duplicates[0]
  duplicate_targets = duplicates[1]
  for key1, key2 in zip(duplicate_keys, duplicate_targets):
    if key1 not in all_names:
      raise KeyError(not_found_msg.format(key=key1, target=target.name, filename=filename))
    if id(variables[key1]) != id(variables[key2]):
      raise ValueError(id_dismatch_msg.format(key1=key1, key2=target, filename=filename, target=target.name))
    all_names.remove(key1)

  # check missing names
  if len(all_names):
    logger.warning(f'There are variable states missed in {filename}. '
                   f'The missed variables are: {all_names}.')


def _unique_and_duplicate(collector: dict):
  gather = TensorCollector()
  id2name = dict()
  duplicates = ([], [])
  for k, v in collector.items():
    id_ = id(v)
    if id_ not in id2name:
      gather[k] = v
      id2name[id_] = k
    else:
      k2 = id2name[id_]
      duplicates[0].append(k)
      duplicates[1].append(k2)
  duplicates = (duplicates[0], duplicates[1])
  return gather, duplicates


def save_as_h5(filename: str, variables: dict):
  """Save variables into a HDF5 file.

  Parameters
  ----------
  filename: str
    The filename to save.
  variables: dict
    All variables to save.
  """
  if not (filename.endswith('.hdf5') or filename.endswith('.h5')):
    raise ValueError(f'Cannot save variables as a HDF5 file. We only support file with '
                     f'postfix of ".hdf5" and ".h5". But we got {filename}')

  from brainpy import math as bm
  import h5py

  # check variables
  check_dict_data(variables, name='variables')
  variables, duplicates = _unique_and_duplicate(variables)

  # save
  f = h5py.File(filename, "w")
  for key, data in variables.items():
    f[key] = bm.as_numpy(data)
  if len(duplicates[0]):
    f.create_dataset(DUPLICATE_TARGET, data='+'.join(duplicates[1]))
    f.create_dataset(DUPLICATE_KEY, data='+'.join(duplicates[0]))
  f.close()


def load_by_h5(filename: str, target, verbose: bool = False):
  """Load variables in a HDF5 file.

  Parameters
  ----------
  filename: str
    The filename to load variables.
  target: Base
    The instance of :py:class:`~.brainpy.Base`.
  verbose: bool
    Whether report the load progress.
  """
  if not (filename.endswith('.hdf5') or filename.endswith('.h5')):
    raise ValueError(f'Cannot load variables from a HDF5 file. We only support file with '
                     f'postfix of ".hdf5" and ".h5". But we got {filename}')

  # read data
  import h5py
  load_vars = dict()
  with h5py.File(filename, "r") as f:
    for key in f.keys():
      if key in [DUPLICATE_KEY, DUPLICATE_TARGET]: continue
      load_vars[key] = np.asarray(f[key])
    if DUPLICATE_KEY in f:
      duplicate_keys = np.asarray(f[DUPLICATE_KEY]).item().decode("utf-8").split('+')
      duplicate_targets = np.asarray(f[DUPLICATE_TARGET]).item().decode("utf-8").split('+')
      duplicates = (duplicate_keys, duplicate_targets)
    else:
      duplicates = ([], [])

  # assign values
  _load(target, verbose, filename, load_vars, duplicates)


def save_as_npz(filename, variables, compressed=False):
  """Save variables into a numpy file.
  
  Parameters
  ----------
  filename: str
    The filename to store.
  variables: dict
    Variables to save.
  compressed: bool
    Whether we use the compressed mode.
  """
  if not filename.endswith('.npz'):
    raise ValueError(f'Cannot save variables as a .npz file. We only support file with '
                     f'postfix of ".npz". But we got {filename}')

  from brainpy import math as bm
  check_dict_data(variables, name='variables')
  variables, duplicates = _unique_and_duplicate(variables)

  # save
  variables = {k: bm.as_numpy(v) for k, v in variables.items()}
  if len(duplicates[0]):
    variables[DUPLICATE_KEY] = np.asarray(duplicates[0])
    variables[DUPLICATE_TARGET] = np.asarray(duplicates[1])
  if compressed:
    np.savez_compressed(filename, **variables)
  else:
    np.savez(filename, **variables)


def load_by_npz(filename, target, verbose=False):
  """Load variables from a numpy file.

  Parameters
  ----------
  filename: str
    The filename to load variables.
  target: Base
    The instance of :py:class:`~.brainpy.Base`.
  verbose: bool
    Whether report the load progress.
  """
  if not filename.endswith('.npz'):
    raise ValueError(f'Cannot load variables from a .npz file. We only support file with '
                     f'postfix of ".npz". But we got {filename}')

  # load data
  load_vars = dict()
  all_data = np.load(filename)
  for key in all_data.files:
    if key in [DUPLICATE_KEY, DUPLICATE_TARGET]: continue
    load_vars[key] = all_data[key]
  if DUPLICATE_KEY in all_data:
    duplicate_keys = all_data[DUPLICATE_KEY].tolist()
    duplicate_targets = all_data[DUPLICATE_TARGET].tolist()
    duplicates = (duplicate_keys, duplicate_targets)
  else:
    duplicates = ([], [])

  # assign values
  _load(target, verbose, filename, load_vars, duplicates)


def save_as_pkl(filename, variables):
  """Save variables into a pickle file.

  Parameters
  ----------
  filename: str
    The filename to save.
  variables: dict
    All variables to save.
  """
  if not (filename.endswith('.pkl') or filename.endswith('.pickle')):
    raise ValueError(f'Cannot save variables into a pickle file. We only support file with '
                     f'postfix of ".pkl" and ".pickle". But we got {filename}')

  check_dict_data(variables, name='variables')
  variables, duplicates = _unique_and_duplicate(variables)
  import brainpy.math as bm
  targets = {k: bm.as_numpy(v) for k, v in variables.items()}
  if len(duplicates[0]) > 0:
    targets[DUPLICATE_KEY] = np.asarray(duplicates[0])
    targets[DUPLICATE_TARGET] = np.asarray(duplicates[1])
  with open(filename, 'wb') as f:
    pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_by_pkl(filename, target, verbose=False):
  """Load variables from a pickle file.

  Parameters
  ----------
  filename: str
    The filename to load variables.
  target: Base
    The instance of :py:class:`~.brainpy.Base`.
  verbose: bool
    Whether report the load progress.
  """
  if not (filename.endswith('.pkl') or filename.endswith('.pickle')):
    raise ValueError(f'Cannot load variables from a pickle file. We only support file with '
                     f'postfix of ".pkl" and ".pickle". But we got {filename}')

  # load variables
  load_vars = dict()
  with open(filename, 'rb') as f:
    all_data = pickle.load(f)
    for key, data in all_data.items():
      if key in [DUPLICATE_KEY, DUPLICATE_TARGET]: continue
      load_vars[key] = data
    if DUPLICATE_KEY in all_data:
      duplicate_keys = all_data[DUPLICATE_KEY].tolist()
      duplicate_targets = all_data[DUPLICATE_TARGET].tolist()
      duplicates = (duplicate_keys, duplicate_targets)
    else:
      duplicates = ([], [])

  # assign data
  _load(target, verbose, filename, load_vars, duplicates)


def save_as_mat(filename, variables):
  """Save variables into a HDF5 file.

  Parameters
  ----------
  filename: str
    The filename to save.
  variables: dict
    All variables to save.
  """
  if not filename.endswith('.mat'):
    raise ValueError(f'Cannot save variables into a .mat file. We only support file with '
                     f'postfix of ".mat". But we got {filename}')

  from brainpy import math as bm
  import scipy.io as sio

  check_dict_data(variables, name='variables')
  variables, duplicates = _unique_and_duplicate(variables)
  variables = {k: np.expand_dims(bm.as_numpy(v), axis=0) for k, v in variables.items()}
  if len(duplicates[0]):
    variables[DUPLICATE_KEY] = np.expand_dims(np.asarray(duplicates[0]), axis=0)
    variables[DUPLICATE_TARGET] = np.expand_dims(np.asarray(duplicates[1]), axis=0)
  sio.savemat(filename, variables)


def load_by_mat(filename, target, verbose=False):
  """Load variables from a numpy file.

  Parameters
  ----------
  filename: str
    The filename to load variables.
  target: Base
    The instance of :py:class:`~.brainpy.Base`.
  verbose: bool
    Whether report the load progress.
  """
  if not filename.endswith('.mat'):
    raise ValueError(f'Cannot load variables from a .mat file. We only support file with '
                     f'postfix of ".mat". But we got {filename}')

  import scipy.io as sio

  # load data
  load_vars = dict()
  all_data = sio.loadmat(filename)
  for key, data in all_data.items():
    if key.startswith('__'):
      continue
    if key in [DUPLICATE_KEY, DUPLICATE_TARGET]:
      continue
    load_vars[key] = data[0]
  if DUPLICATE_KEY in all_data:
    duplicate_keys = [a.strip() for a in all_data[DUPLICATE_KEY].tolist()[0]]
    duplicate_targets = [a.strip() for a in all_data[DUPLICATE_TARGET].tolist()[0]]
    duplicates = (duplicate_keys, duplicate_targets)
  else:
    duplicates = ([], [])

  # assign values
  _load(target, verbose, filename, load_vars, duplicates)
