# -*- coding: utf-8 -*-

import logging
import os
import pickle

import numpy as np

from brainpy import errors
from brainpy.base.collector import TensorCollector

Base = math = None
logger = logging.getLogger('brainpy.base.io')

try:
  import h5py
except (ModuleNotFoundError, ImportError):
  h5py = None

try:
  import scipy.io as sio
except (ModuleNotFoundError, ImportError):
  sio = None

__all__ = [
  'SUPPORTED_FORMATS',
  'save_h5',
  'save_npz',
  'save_pkl',
  'save_mat',
  'load_h5',
  'load_npz',
  'load_pkl',
  'load_mat',
]

SUPPORTED_FORMATS = ['.h5', '.hdf5', '.npz', '.pkl', '.mat']


def _check(module, module_name, ext):
  if module is None:
    raise errors.PackageMissingError(
      '"{package}" must be installed when you want to save/load data with .{ext} '
      'format. \nPlease install {package} through "pip install {package}" or '
      '"conda install {package}".'.format(package=module_name, ext=ext)
    )


def _check_missing(vars, filename):
  if len(vars):
    logger.warning(f'There are variable states missed in {filename}. '
                   f'The missed variables are: {list(vars.keys())}.')


def save_h5(filename, all_vars):
  _check(h5py, module_name='h5py', ext=os.path.splitext(filename))
  assert isinstance(all_vars, dict)
  all_vars = TensorCollector(all_vars).unique()

  # save
  f = h5py.File(filename, "w")
  for key, data in all_vars.items():
    f[key] = np.asarray(data.value)
  f.close()


def load_h5(filename, target, verbose=False, check=False):
  global math, Base
  if Base is None: from brainpy.base.base import Base
  if math is None: from brainpy import math
  assert isinstance(target, Base)
  _check(h5py, module_name='h5py', ext=os.path.splitext(filename))

  all_vars = target.vars(method='relative')
  f = h5py.File(filename, "r")
  for key in f.keys():
    if verbose: print(f'Loading {key} ...')
    var = all_vars.pop(key)
    var[:] = math.asarray(f[key][:])
  f.close()
  if check: _check_missing(all_vars, filename=filename)


def save_npz(filename, all_vars, compressed=False):
  assert isinstance(all_vars, dict)
  all_vars = TensorCollector(all_vars).unique()
  all_vars = {k.replace('.', '--'): np.asarray(v.value) for k, v in all_vars.items()}
  if compressed:
    np.savez_compressed(filename, **all_vars)
  else:
    np.savez(filename, **all_vars)


def load_npz(filename, target, verbose=False, check=False):
  global math, Base
  if Base is None: from brainpy.base.base import Base
  if math is None: from brainpy import math
  assert isinstance(target, Base)

  all_vars = target.vars(method='relative')
  all_data = np.load(filename)
  for key in all_data.files:
    if verbose: print(f'Loading {key} ...')
    var = all_vars.pop(key)
    var[:] = math.asarray(all_data[key])
  if check: _check_missing(all_vars, filename=filename)


def save_pkl(filename, all_vars):
  assert isinstance(all_vars, dict)
  all_vars = TensorCollector(all_vars).unique()
  targets = {k: np.asarray(v) for k, v in all_vars.items()}
  f = open(filename, 'wb')
  pickle.dump(targets, f, protocol=pickle.HIGHEST_PROTOCOL)
  f.close()


def load_pkl(filename, target, verbose=False, check=False):
  global math, Base
  if Base is None: from brainpy.base.base import Base
  if math is None: from brainpy import math
  assert isinstance(target, Base)
  f = open(filename, 'rb')
  all_data = pickle.load(f)
  f.close()

  all_vars = target.vars(method='relative')
  for key, data in all_data.items():
    if verbose: print(f'Loading {key} ...')
    var = all_vars.pop(key)
    var[:] = math.asarray(data)
  if check: _check_missing(all_vars, filename=filename)


def save_mat(filename, all_vars):
  assert isinstance(all_vars, dict)
  all_vars = TensorCollector(all_vars).unique()
  _check(sio, module_name='scipy', ext=os.path.splitext(filename))
  all_vars = {k.replace('.', '--'): np.asarray(v.value) for k, v in all_vars.items()}
  sio.savemat(filename, all_vars)


def load_mat(filename, target, verbose=False, check=False):
  global math, Base
  if Base is None: from brainpy.base.base import Base
  if math is None: from brainpy import math
  assert isinstance(target, Base)

  all_data = sio.loadmat(filename)
  all_vars = target.vars(method='relative')
  for key, data in all_data.items():
    if verbose: print(f'Loading {key} ...')
    var = all_vars.pop(key)
    var[:] = math.asarray(data)
  if check: _check_missing(all_vars, filename=filename)
