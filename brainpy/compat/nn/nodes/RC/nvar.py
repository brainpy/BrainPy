# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from typing import Union, Sequence

import numpy as np
import jax.numpy as jnp

import brainpy.math as bm
from brainpy.compat.nn.base import RecurrentNode
from brainpy.compat.nn.datatypes import MultipleData
from brainpy.tools.checking import (check_shape_consistency,
                                    check_integer,
                                    check_sequence)

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
  - it supports multiple orders,

  Parameters
  ----------
  delay: int
    The number of delay step.
  order: int, sequence of int
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
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      delay: int,
      order: Union[int, Sequence[int]] = None,
      stride: int = 1,
      constant: bool = False,
      trainable: bool = False,
      **kwargs
  ):
    super(NVAR, self).__init__(trainable=trainable, **kwargs)

    # parameters
    order = tuple() if order is None else order
    if not isinstance(order, (tuple, list)):
      order = (order,)
    self.order = tuple(order)
    check_sequence(order, 'order', allow_none=False)
    for o in order: check_integer(o, 'delay', allow_none=False, min_bound=2)
    check_integer(delay, 'delay', allow_none=False, min_bound=1)
    check_integer(stride, 'stride', allow_none=False, min_bound=1)
    assert isinstance(constant, bool), f'Must be an instance of boolean, but got {constant}.'
    self.delay = delay
    self.stride = stride
    self.constant = constant
    self.num_delay = 1 + (self.delay - 1) * self.stride

    # attributes
    self.comb_ids = []
    self.feature_names = []
    self.input_dim = None
    self.output_dim = None
    self.linear_dim = None
    self.nonlinear_dim = None

    # delay variables
    self.idx = bm.Variable(jnp.asarray([0]))
    self.store = None

  def init_ff_conn(self):
    """Initialize feedforward connections."""
    # input dimension
    batch_size, free_size = check_shape_consistency(self.feedforward_shapes, -1, True)
    self.input_dim = sum(free_size)
    assert batch_size == (None,), f'batch_size must be None, but got {batch_size}'
    # linear dimension
    self.linear_dim = self.delay * self.input_dim
    # For each monomial created in the non-linear part, indices
    # of the n components involved, n being the order of the
    # monomials. Precompute them to improve efficiency.
    for order in self.order:
      assert order >= 2, f'"order" must be a integer >= 2, while we got {order}.'
      idx = np.array(list(combinations_with_replacement(np.arange(self.linear_dim), order)))
      self.comb_ids.append(jnp.asarray(idx))
    # number of non-linear components is (d + n - 1)! / (d - 1)! n!
    # i.e. number of all unique monomials of order n made from the
    # linear components.
    self.nonlinear_dim = sum([len(ids) for ids in self.comb_ids])
    # output dimension
    self.output_dim = int(self.linear_dim + self.nonlinear_dim)
    if self.constant:
      self.output_dim += 1
    self.set_output_shape((None, self.output_dim))

  def init_state(self, num_batch=1):
    """Initialize the node state which depends on batch size."""
    # To store the last inputs.
    # Note, the batch axis is not in the first dimension, so we
    # manually handle the state of NVAR, rather return it.
    state = jnp.zeros((self.num_delay, num_batch, self.input_dim))
    if self.store is None:
      self.store = bm.Variable(state)
    else:
      self.store._value = state

  def forward(self, ff, fb=None, **shared_kwargs):
    all_parts = []
    # 1. Store the current input
    ff = bm.concatenate(ff, axis=-1)
    self.store[self.idx[0]] = ff
    # 2. Linear part:
    # select all previous inputs, including the current, with strides
    select_ids = (self.idx[0] - jnp.arange(0, self.num_delay, self.stride)) % self.num_delay
    linear_parts = jnp.moveaxis(self.store[select_ids], 0, 1)  # (num_batch, num_time, num_feature)
    linear_parts = jnp.reshape(linear_parts, (linear_parts.shape[0], -1))
    # 3. constant
    if self.constant:
      constant = jnp.ones((linear_parts.shape[0], 1), dtype=ff.dtype)
      all_parts.append(constant)
    all_parts.append(linear_parts)
    # 3. Nonlinear part:
    # select monomial terms and compute them
    for ids in self.comb_ids:
      all_parts.append(jnp.prod(linear_parts[:, ids], axis=2))
    # 4. Finally
    self.idx.value = (self.idx + 1) % self.num_delay
    return jnp.concatenate(all_parts, axis=-1)

  def get_feature_names(self, for_plot=False):
    """Get output feature names for transformation.

    Returns
    -------
    feature_names_out : list of str
        Transformed feature names.
    """
    if not self.is_initialized:
      raise ValueError('Please initialize the node first.')
    linear_names = [f'x{i}(t)' for i in range(self.input_dim)]
    for di in range(1, self.delay):
      linear_names.extend([((f'x{i}_' + r'{t-%d}' % (di * self.stride))
                            if for_plot else
                            f'x{i}(t-{di * self.stride})')
                           for i in range(self.input_dim)])
    nonlinear_names = []
    for ids in self.comb_ids:
      for id_ in np.asarray(ids):
        uniques, counts = np.unique(id_, return_counts=True)
        nonlinear_names.append(" ".join(
          "%s^%d" % (linear_names[ind], exp) if (exp != 1) else linear_names[ind]
          for ind, exp in zip(uniques, counts)
        ))
    if for_plot:
      all_names = [f'${n}$' for n in linear_names] + [f'${n}$' for n in nonlinear_names]
    else:
      all_names = linear_names + nonlinear_names
    if self.constant:
      all_names = ['1'] + all_names
    return all_names

  def get_feature_names_for_plot(self):
    """Get output feature names for matplotlib plotting.

    Returns
    -------
    feature_names_out : list of str
        Transformed feature names.
    """
    return self.get_feature_names(for_plot=True)
