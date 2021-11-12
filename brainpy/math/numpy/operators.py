# -*- coding: utf-8 -*-

import numpy as np
from brainpy.tools import numba_jit

__all__ = [
  'pre2syn', 'syn2post',
  'segment_sum', 'segment_prod', 'segment_max', 'segment_min',
]


@numba_jit
def pre2syn(pre_values, pre_ids):
  # for 'jax' backend, please see `pre2syn()` in 'brainpy.math.jax' module
  res = np.zeros(pre_ids.size, dtype=pre_values.dtype)
  for i, pre_id in enumerate(pre_ids):
    res[i] = pre_values[pre_id]
  return res


@numba_jit
def syn2post(syn_values, post_ids, post_num):
  # for 'jax' backend, please see `syn2post()` in 'brainpy.math.jax' module
  res = np.zeros(post_num, dtype=syn_values.dtype)
  for i, post_id in enumerate(post_ids):
    res[post_id] += syn_values[i]
  return res


def segment_sum(*args, **kwargs):
  """Computes the sum within segments of an array."""
  raise NotImplementedError('Please see "segment_sum" in "brainpy.math.jax"')


def segment_prod(*args, **kwargs):
  """Computes the product within segments of an array."""
  raise NotImplementedError('Please see "segment_prod" in "brainpy.math.jax"')


def segment_max(*args, **kwargs):
  raise NotImplementedError('Please see "segment_max" in "brainpy.math.jax"')


def segment_min(*args, **kwargs):
  """Computes the product within segments of an array."""
  raise NotImplementedError('Please see "segment_min" in "brainpy.math.jax"')
