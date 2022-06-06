# -*- coding: utf-8 -*-

from typing import Dict, Optional
from brainpy import math as bm

from brainpy.tools.checking import check_dict_data
from jax.tree_util import tree_flatten

__all__ = [
  'serialize_kwargs',
  # 'check_rnn_data_time_step',
  # 'check_data_batch_size',
]


def serialize_kwargs(shared_kwargs: Optional[Dict]):
  """Serialize kwargs."""
  shared_kwargs = dict() if shared_kwargs is None else shared_kwargs
  check_dict_data(shared_kwargs,
                  key_type=str,
                  val_type=(bool, float, int, complex),
                  name='shared_kwargs')
  shared_kwargs = {key: shared_kwargs[key] for key in sorted(shared_kwargs.keys())}
  return str(shared_kwargs)


def check_rnn_data_time_step(data: Dict, num_step=None):
  if len(data) == 1:
    time_step = list(data.values())[0].shape[1]
  else:
    steps = []
    for key, val in data.items():
      steps.append(val.shape[1])
    if len(set(steps)) != 1:
      raise ValueError('Time steps are not consistent among the given data. '
                       f'Got {set(steps)}. We expect only one time step.')
    time_step = steps[0]
  if (num_step is not None) and time_step != num_step:
    raise ValueError(f'Time step is not consistent with the expected {time_step} != {num_step}')
  return time_step


def check_data_batch_size(data, num_batch=None, batch_idx=0):
  leaves, tree = tree_flatten(data, is_leaf=lambda x: isinstance(x, bm.JaxArray))
  batches = [leaf.shape[batch_idx] for leaf in leaves]
  if len(set(batches)) != 1:
    raise ValueError('Batch sizes are not consistent among the given data. '
                     f'Got {set(batches)}. We expect only one batch size.')
  batch_size = batches[0]
  if (num_batch is not None) and batch_size != num_batch:
    raise ValueError(f'Batch size is not consistent with the expected {batch_size} != {num_batch}')
  return batch_size
