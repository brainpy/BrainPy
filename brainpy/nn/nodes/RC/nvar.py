# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from typing import Union

import numpy as np

import brainpy.math as bm
from brainpy.dyn.base import ConstantDelay
from brainpy.nn.base import Node
from brainpy.tools.checking import (check_shape_consistency,
                                    check_float)

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


class NVAR(Node):
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
    check_float(constant, 'constant', allow_none=True, allow_int=True)

  def ff_init(self):
    # input dimension
    unique_size, free_size = check_shape_consistency(self.input_shapes, -1, True)
    input_dim = sum(free_size)
    self.batch_size = unique_size
    assert len(unique_size) in [0, 1]

    # linear dimension
    linear_dim = self.delay * input_dim
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
    self.set_output_shape(unique_size + (output_dim,))

    # to store the k*s last inputs, k being the delay and s the strides
    self.store = ConstantDelay(unique_size + (input_dim,), self.delay * self.stride, dt=1)

  def forward(self, ff, fb=None, **kwargs):
    # 1. store the current input
    ff = bm.concatenate(ff, axis=-1)
    self.store.push(ff)
    self.store.update()
    # 2. Linear part:
    # select all previous inputs, including the current, with strides
    select_ids = (self.store.out_idx + bm.arange(self.store.num_step - 1)[::self.stride]) % self.store.num_step
    if len(self.batch_size) == 1:
      linear_parts = bm.moveaxis(self.store.data[select_ids], 0, 1)
      linear_parts = bm.reshape(linear_parts, self.batch_size + (-1,))
    else:
      linear_parts = bm.ravel(self.store.data[select_ids])
    # 3. Nonlinear part:
    # select monomial terms and compute them
    if len(self.batch_size) == 1:
      nonlinear_parts = bm.prod(linear_parts[:, self.comb_ids], axis=2)
    else:
      nonlinear_parts = bm.prod(linear_parts[self.comb_ids], axis=1)
    if self.constant is None:
      return bm.concatenate([linear_parts, nonlinear_parts], axis=-1)
    else:
      constant = bm.broadcast_to(self.constant, linear_parts.shape[:-1] + (1,))
      return bm.concatenate([constant, linear_parts, nonlinear_parts], axis=-1)

  def reset_state(self, to_state=None):
    self.store.data[:] = 0.
