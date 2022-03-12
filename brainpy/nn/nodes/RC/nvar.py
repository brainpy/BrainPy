# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from typing import Union

import numpy as np

import brainpy.math as bm
from brainpy.nn.base import RecurrentNode
from brainpy.tools.checking import (check_shape_consistency,
                                    check_float,
                                    check_integer)

__all__ = [
  'NVAR'
]


def _comb(N, k):
  r"""The number of combinations of N things taken k at a time.

  .. math::

     \frac{N!}{(N-k)! k!}

  """
  if N > k:
    val = 1
    for j in range(min(k, N - k)):
      val = (val * (N - j)) // (j + 1)
    return val
  elif N == k:
    return 1
  else:
    return 0


class NVAR(RecurrentNode):
  """Nonlinear vector auto-regression (NVAR) node.

  This class has the following features:

  - it supports batch size,

  Parameters
  ----------
  delay: int
    The number of delay step.
  order: int
    The nonlinear order.
  stride: int
    The stride to sample linear part vector in the delays.
  constant: optional, float
    The constant value.

  References
  ----------
  .. [1] Gauthier, D.J., Bollt, E., Griffith, A. et al. Next generation
         reservoir computing. Nat Commun 12, 5564 (2021).
         https://doi.org/10.1038/s41467-021-25801-2

  """

  def __init__(self,
               delay: int,
               order: int,
               stride: int = 1,
               constant: Union[float, int] = None,
               **kwargs):
    super(NVAR, self).__init__(**kwargs)

    self.delay = delay
    self.order = order
    self.stride = stride
    self.constant = constant
    check_integer(delay, 'delay', allow_none=False)
    check_integer(order, 'order', allow_none=False)
    check_integer(stride, 'stride', allow_none=False)
    check_float(constant, 'constant', allow_none=True, allow_int=True)

  def init_ff(self):
    # input dimension
    batch_size, free_size = check_shape_consistency(self.input_shapes, -1, True)
    self.input_dim = sum(free_size)
    assert batch_size == (None,), f'batch_size must be None, but got {batch_size}'

    # linear dimension
    linear_dim = self.delay * self.input_dim
    # for each monomial created in the non linear part, indices
    # of the n components involved, n being the order of the
    # monomials. Precompute them to improve efficiency.
    idx = np.array(list(combinations_with_replacement(np.arange(linear_dim), self.order)))
    self.comb_ids = bm.asarray(idx)
    # number of non linear components is (d + n - 1)! / (d - 1)! n!
    # i.e. number of all unique monomials of order n made from the
    # linear components.
    nonlinear_dim = len(self.comb_ids)
    # output dimension
    output_dim = int(linear_dim + nonlinear_dim)
    if self.constant is not None:
      output_dim += 1
    self.set_output_shape((None, output_dim))

    # delay variables
    self.num_delay = self.delay * self.stride
    self.idx = bm.Variable(bm.array([0], dtype=bm.uint32))
    self.store = None

  def init_state(self, num_batch=1):
    # to store the k*s last inputs, k being the delay and s the strides
    state = bm.zeros((self.num_delay, num_batch, self.input_dim), dtype=bm.float_)
    if self.store is None:
      self.store = bm.Variable(state)
    else:
      self.store.value = state

  def forward(self, ff, fb=None, **kwargs):
    # 1. store the current input
    ff = bm.concatenate(ff, axis=-1)
    self.store[self.idx[0]] = ff
    self.idx.value = (self.idx + 1) % self.num_delay
    # 2. Linear part:
    # select all previous inputs, including the current, with strides
    select_ids = (self.idx[0] + bm.arange(self.num_delay)[::self.stride]) % self.num_delay
    linear_parts = bm.moveaxis(self.store[select_ids], 0, 1)  # (num_batch, num_time, num_feature)
    linear_parts = bm.reshape(linear_parts, (linear_parts.shape[0], -1))
    # 3. Nonlinear part:
    # select monomial terms and compute them
    nonlinear_parts = bm.prod(linear_parts[:, self.comb_ids], axis=2)
    if self.constant is None:
      return bm.concatenate([linear_parts, nonlinear_parts], axis=-1)
    else:
      constant = bm.broadcast_to(self.constant, linear_parts.shape[:-1] + (1,))
      return bm.concatenate([constant, linear_parts, nonlinear_parts], axis=-1)

