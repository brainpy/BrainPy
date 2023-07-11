# -*- coding: utf-8 -*-

from itertools import combinations_with_replacement
from typing import Union, Sequence, List, Optional

import jax.numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy import check
from brainpy._src.dnn.base import Layer

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


class NVAR(Layer):
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

  def __init__(
      self,
      num_in: int,
      delay: int,
      order: Optional[Union[int, Sequence[int]]] = None,
      stride: int = 1,
      constant: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(NVAR, self).__init__(mode=mode, name=name)
    check.is_subclass(self.mode, (bm.BatchingMode, bm.NonBatchingMode), self.__class__.__name__)

    # parameters
    order = tuple() if order is None else order
    if not isinstance(order, (tuple, list)):
      order = (order,)
    self.order = tuple(order)
    check.is_sequence(order, 'order', allow_none=False)
    for o in order:
      check.is_integer(o, 'order', allow_none=False, min_bound=2)
    check.is_integer(delay, 'delay', allow_none=False, min_bound=1)
    check.is_integer(stride, 'stride', allow_none=False, min_bound=1)
    assert isinstance(constant, bool), f'Must be an instance of boolean, but got {constant}.'
    self.delay = delay
    self.stride = stride
    self.constant = constant
    self.num_delay = 1 + (self.delay - 1) * self.stride
    self.num_in = num_in

    # delay variables
    self.idx = bm.Variable(jnp.asarray([0]))
    if isinstance(self.mode, bm.BatchingMode):
      batch_size = 1  # first initialize the state with batch size = 1
      self.store = bm.Variable(jnp.zeros((self.num_delay, batch_size, self.num_in)), batch_axis=1)
    else:
      self.store = bm.Variable(jnp.zeros((self.num_delay, self.num_in)))

    # linear dimension
    self.linear_dim = self.delay * num_in
    # For each monomial created in the non-linear part, indices
    # of the n components involved, n being the order of the
    # monomials. Precompute them to improve efficiency.
    self.comb_ids = []
    for order in self.order:
      assert order >= 2, f'"order" must be a integer >= 2, while we got {order}.'
      idx = np.array(list(combinations_with_replacement(np.arange(self.linear_dim), order)))
      self.comb_ids.append(jnp.asarray(idx))
    # number of non-linear components is (d + n - 1)! / (d - 1)! n!
    # i.e. number of all unique monomials of order n made from the
    # linear components.
    self.nonlinear_dim = sum([len(ids) for ids in self.comb_ids])
    # output dimension
    self.num_out = int(self.linear_dim + self.nonlinear_dim)
    if self.constant:
      self.num_out += 1

  def reset_state(self, batch_size=None):
    """Reset the node state which depends on batch size."""
    self.idx[0] = 0
    # To store the last inputs.
    # Note, the batch axis is not in the first dimension, so we
    # manually handle the state of NVAR, rather return it.
    if batch_size is None:
      self.store.value = jnp.zeros((self.num_delay, self.num_in))
    else:
      self.store.value = jnp.zeros((self.num_delay, batch_size, self.num_in))

  def update(self, x):
    all_parts = []
    select_ids = (self.idx[0] - jnp.arange(0, self.num_delay, self.stride)) % self.num_delay
    # 1. Store the current input
    self.store[self.idx[0]] = x

    if isinstance(self.mode, bm.BatchingMode):
      # 2. Linear part:
      # select all previous inputs, including the current, with strides
      linear_parts = jnp.moveaxis(self.store[select_ids], 0, 1)  # (num_batch, num_time, num_feature)
      linear_parts = jnp.reshape(linear_parts, (linear_parts.shape[0], -1))
      # 3. constant
      if self.constant:
        constant = jnp.ones((linear_parts.shape[0], 1), dtype=x.dtype)
        all_parts.append(constant)
      all_parts.append(linear_parts)
      # 3. Nonlinear part:
      # select monomial terms and compute them
      for ids in self.comb_ids:
        all_parts.append(jnp.prod(linear_parts[:, ids], axis=2))

    else:
      # 2. Linear part:
      # select all previous inputs, including the current, with strides
      linear_parts = self.store[select_ids].flatten()  # (num_time x num_feature,)
      # 3. constant
      if self.constant:
        constant = jnp.ones((1,), dtype=x.dtype)
        all_parts.append(constant)
      all_parts.append(linear_parts)
      # 3. Nonlinear part:
      # select monomial terms and compute them
      for ids in self.comb_ids:
        all_parts.append(jnp.prod(linear_parts[ids], axis=1))

    # 4. Finally
    self.idx.value = (self.idx + 1) % self.num_delay
    return jnp.concatenate(all_parts, axis=-1)

  def get_feature_names(self, for_plot=False) -> List[str]:
    """Get output feature names for transformation.

    Parameters
    ----------
    for_plot: bool
      Use the feature names for plotting or not? (Default False)
    """
    if for_plot:
      linear_names = [f'x{i}_t' for i in range(self.num_in)]
    else:
      linear_names = [f'x{i}(t)' for i in range(self.num_in)]
    for di in range(1, self.delay):
      linear_names.extend([((f'x{i}_' + r'{t-%d}' % (di * self.stride))
                            if for_plot else f'x{i}(t-{di * self.stride})')
                           for i in range(self.num_in)])
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
