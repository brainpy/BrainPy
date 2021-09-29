# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np

from brainpy import errors, math
from brainpy.base.base import Base
from brainpy.tools.collector import ArrayCollector

try:
  import h5py
except ModuleNotFoundError:
  h5py = None

try:
  import scipy.io as sio
except ModuleNotFoundError:
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


def _check(module, module_name, ext):
  if module is None:
    raise errors.PackageMissingError(
      '"{package}" must be installed when you want to save/load data with .{ext} '
      'format. \nPlease install {package} through "pip install {package}" or '
      '"conda install {package}".'.format(package=module_name, ext=ext)
    )


SUPPORTED_FORMATS = ['.h5', '.hdf5', '.npz', '.pkl', '.mat']


def save_h5(filename, all_vars):
  _check(h5py, module_name='h5py', ext=os.path.splitext(filename))
  assert isinstance(all_vars, dict)
  all_vars = ArrayCollector(all_vars).unique()

  # save
  f = h5py.File(filename, "w")
  for key, data in all_vars.items():
    host_name, val_name = key.split('.')
    if host_name not in f:
      g = f.create_group(host_name)
    else:
      g = f[host_name]
    d = g.create_dataset(val_name, data=np.asarray(data.value))
    d.attrs['type'] = data.type
  f.close()


def load_h5(filename, target):
  _check(h5py, module_name='h5py', ext=os.path.splitext(filename))
  assert isinstance(target, Base)

  # load
  all_vars = target.vars()
  f = h5py.File(filename, "r")
  for g_key in f.keys():
    g = f[g_key]
    for d_key in g.keys():
      d = f[g_key][d_key]
      var = all_vars.pop(g_key + '.' + d_key)
      var[:] = math.asarray(d.value)
      assert var.type == d.attrs['type']
  f.close()


def save_npz(filename, all_vars, compressed=False):
  assert isinstance(all_vars, dict)
  all_vars = ArrayCollector(all_vars).unique()
  all_vars = {k.replace('.', '--'): np.asarray(v.value) for k, v in all_vars.items()}
  if compressed:
    np.savez_compressed(filename, **all_vars)
  else:
    np.savez(filename, **all_vars)


def load_npz(filename, target):
  assert isinstance(target, Base)

  # load
  all_vars = target.vars()
  all_data = np.load(filename)
  for key in all_data.files:
    host_key, data_key = key.split('--')
    all_vars[host_key + '.' + data_key][:] = math.asarray(all_data[key])


def save_pkl(filename, all_vars):
  assert isinstance(all_vars, dict)
  all_vars = ArrayCollector(all_vars).unique()
  targets = {k: np.asarray(v) for k, v in all_vars.items()}
  f = open(filename, 'w')
  pickle.dump(targets, f)
  f.close()


def load_pkl(filename, target):
  assert isinstance(target, Base)
  f = open(filename, 'r')
  all_data = pickle.load(f)
  f.close()
  all_vars = target.vars()
  for key, data in all_data.items():
    all_vars[key] = math.asarray(data)


def save_mat(filename, all_vars):
  assert isinstance(all_vars, dict)
  all_vars = ArrayCollector(all_vars).unique()
  _check(sio, module_name='scipy', ext=os.path.splitext(filename))
  all_vars = {k.replace('.', '--'): np.asarray(v.value) for k, v in all_vars.items()}
  sio.savemat(filename, all_vars)


def load_mat(filename, target):
  assert isinstance(target, Base)
  all_data = sio.loadmat(filename)
  all_vars = target.vars()
  for key, data in all_data.items():
    all_vars[key] = math.asarray(data)
