# -*- coding: utf-8 -*-

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.check import is_dict_data
from brainpy._src.dynsys import DynamicalSystem

__all__ = [
  'format_ys'
]

msg = '''

Given the data with (x_train, y_train) is no longer supported.

Please control your data by yourself. For example, using `torchvision` or `tensorflow-datasets`. 

A simple way to convert your `(x_train, y_train)` data is defining it as a python function:

.. code::

   def data(batch_size):
     x_data = bm.random.shuffle(x_data, key=123)
     y_data = bm.random.shuffle(y_data, key=123)
     for i in range(0, x_data.shape[0], batch_size):
       yield x_data[i: i + batch_size], y_data[i: i + batch_size], 

'''


def format_ys(cls, ys):
  if isinstance(ys, (bm.Array, jnp.ndarray)):
    if len(cls.train_nodes) == 1:
      ys = {cls.train_nodes[0].name: ys}
    else:
      raise ValueError(f'The network\n {cls.target} \nhas {len(cls.train_nodes)} '
                       f'training nodes, while we only got one target data.')
  is_dict_data(ys, key_type=str, val_type=(bm.Array, jnp.ndarray))

  # check data path
  abs_node_names = [node.name for node in cls.train_nodes]
  formatted_ys = {}
  ys_not_included = {}
  for k, v in ys.items():
    if k in abs_node_names:
      formatted_ys[k] = v
    else:
      ys_not_included[k] = v
  if len(ys_not_included):
    rel_nodes = cls.target.nodes('relative', level=-1, include_self=True).subset(DynamicalSystem).unique()
    for k, v in ys_not_included.items():
      if k in rel_nodes:
        formatted_ys[rel_nodes[k].name] = v
      else:
        raise ValueError(f'Unknown target "{k}" for fitting.')

  # check data shape
  for key, val in formatted_ys.items():
    if val.ndim < 3:
      raise ValueError("Targets must be a tensor with shape of "
                       "(batch, time, feature, ...) or (time, batch, feature, ...)"
                       f"but we got {val.shape}")
  return formatted_ys
