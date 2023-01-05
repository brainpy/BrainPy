# -*- coding: utf-8 -*-

from typing import Union, Tuple, List

import jax
import jax.numpy as jnp
import numpy as np

from brainpy.errors import ConnectorError
from .base import *

__all__ = [
  'One2One', 'one2one',
  'All2All', 'all2all',
  'GridFour', 'grid_four',
  'GridEight', 'grid_eight',
  'GridN',
]


class One2One(TwoEndConnector):
  """Connect two neuron groups one by one. This means
  The two neuron groups should have the same size.
  """

  def __init__(self, *args, **kwargs):
    super(One2One, self).__init__(*args, **kwargs)

  def __call__(self, pre_size, post_size):
    super(One2One, self).__call__(pre_size, post_size)
    try:
      assert self.pre_num == self.post_num
    except AssertionError:
      raise ConnectorError(f'One2One connection must be defined in two groups with the '
                           f'same size, but {self.pre_num} != {self.post_num}.')
    return self

  def build_coo(self):
    if self.pre_num != self.post_num:
      raise ConnectorError(f'One2One connection must be defined in two groups with the '
                           f'same size, but {self.pre_num} != {self.post_num}.')
    return np.arange(self.pre_num, dtype=IDX_DTYPE), np.arange(self.post_num, dtype=IDX_DTYPE),

  def build_csr(self):
    if self.pre_num != self.post_num:
      raise ConnectorError(f'One2One connection must be defined in two groups with the '
                           f'same size, but {self.pre_num} != {self.post_num}.')
    ind = np.arange(self.pre_num)
    indptr = np.arange(self.pre_num + 1)
    return (np.asarray(ind, dtype=IDX_DTYPE), np.asarray(indptr, dtype=IDX_DTYPE))

  def build_mat(self):
    if self.pre_num != self.post_num:
      raise ConnectorError(f'One2One connection must be defined in two groups with the '
                           f'same size, but {self.pre_num} != {self.post_num}.')
    mat = np.zeros((self.pre_num, self.post_num), dtype=MAT_DTYPE)
    np.fill_diagonal(mat, True)
    return mat


one2one = One2One()


class All2All(TwoEndConnector):
  """Connect each neuron in first group to all neurons in the
  post-synaptic neuron groups. It means this kind of conn
  will create (num_pre x num_post) synapses.
  """

  def __init__(self, *args, include_self: bool = True, **kwargs):
    self.include_self = include_self
    super(All2All, self).__init__(*args, **kwargs)

  def __repr__(self):
    return f'{self.__class__.__name__}(include_self={self.include_self})'

  def build_mat(self):
    mat = np.ones((self.pre_num, self.post_num), dtype=MAT_DTYPE)
    if not self.include_self:
      np.fill_diagonal(mat, False)
    return mat


all2all = All2All(include_self=True)


def get_size_length(sizes: Union[Tuple, List]):
  if not isinstance(sizes, (tuple, list)):
    raise TypeError
  lengths = []
  a = 1
  for s in reversed(sizes):
    lengths.insert(0, a)
    a *= s
  return np.asarray(lengths)


class GridConn(OneEndConnector):
  def __init__(
      self,
      strides,
      include_self: bool = False,
      periodic_boundary: bool = False,
      **kwargs
  ):
    super(GridConn, self).__init__(**kwargs)
    self.strides = strides
    self.include_self = include_self
    self.periodic_boundary = periodic_boundary

  def __repr__(self):
    return f'{self.__class__.__name__}(include_self={self.include_self}, periodic_boundary={self.periodic_boundary})'

  def _format(self):
    dim = len(self.post_size)
    if self.pre_num != self.post_num:
      raise ConnectorError(f'{self.__class__.__name__} is used to for connection within '
                           f'a same population. But we detect pre_num != post_num '
                           f'({self.pre_num} != {self.post_num}).')
    # point indices
    indices = jnp.meshgrid(*(jnp.arange(size) for size in self.post_size), indexing='ij')
    indices = jnp.asarray(indices)
    indices = indices.reshape(dim, self.post_num).T
    lengths = jnp.asarray(self.post_size)
    return lengths, dim, indices

  def _get_strides(self, dim):
    # increments
    increments = np.asarray(np.meshgrid(*(self.strides for _ in range(dim)))).reshape(dim, -1).T
    select_ids = self._select_stride(increments)
    increments = jnp.asarray(increments[select_ids])
    return increments

  def _select_stride(self, stride: np.ndarray) -> np.ndarray:
    raise NotImplementedError

  def _select_dist(self, dist: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError

  def build_mat(self):
    sizes, _, indices = self._format()

    @jax.vmap
    def f_connect(pre_id):
      # pre_id: R^(num_dim)
      dist = jnp.abs(pre_id - indices)
      if self.periodic_boundary:
        dist = jnp.where(dist > sizes / 2, sizes - dist, dist)
      return self._select_dist(dist)

    return jnp.asarray(f_connect(indices), dtype=MAT_DTYPE)

  def build_coo(self):
    sizes, dim, indices = self._format()
    strides = self._get_strides(dim)

    @jax.vmap
    def f_connect(pre_id):
      # pre_id: R^(num_dim)
      post_ids = pre_id + strides
      if self.periodic_boundary:
        post_ids = post_ids % sizes
      else:
        post_ids = jnp.where(post_ids < sizes, post_ids, -1)
      size = len(post_ids)
      pre_ids = jnp.repeat(pre_id, size).reshape(dim, size).T
      return pre_ids, post_ids

    pres, posts = f_connect(indices)
    pres = pres.reshape(-1, dim)
    posts = posts.reshape(-1, dim)
    idx = jnp.nonzero(jnp.all(posts >= 0, axis=1))[0]
    pres = pres[idx]
    posts = posts[idx]
    if dim == 1:
      pres = pres.flatten()
      posts = posts.flatten()
    else:
      strides = jnp.asarray(get_size_length(self.post_size))
      pres = jnp.sum(pres * strides, axis=1)
      posts = jnp.sum(posts * strides, axis=1)
    return jnp.asarray(pres, dtype=IDX_DTYPE), jnp.asarray(posts, dtype=IDX_DTYPE)


class GridFour(GridConn):
  """The nearest four neighbors connection method.

  Parameters
  ----------
  periodic_boundary : bool
    Whether the neuron encode the value space with the periodic boundary.
    .. versionadded:: 2.2.3.2

  include_self : bool
    Whether create connection at the same position.
  """

  def __init__(
      self,
      include_self: bool = False,
      periodic_boundary: bool = False,
      **kwargs
  ):
    super(GridFour, self).__init__(strides=np.asarray([-1, 0, 1]),
                                   include_self=include_self,
                                   periodic_boundary=periodic_boundary,
                                   **kwargs)
    self.include_self = include_self
    self.periodic_boundary = periodic_boundary

  def _select_stride(self, stride: np.ndarray) -> np.ndarray:
    temp = abs(stride).sum(axis=1)
    return (temp <= 1) if self.include_self else (temp == 1)

  def _select_dist(self, dist: jnp.ndarray) -> jnp.ndarray:
    dist = jnp.linalg.norm(dist, axis=1)
    return dist <= 1 if self.include_self else dist == 1


grid_four = GridFour()


class GridN(GridConn):
  """The nearest (2*N+1) * (2*N+1) neighbors conn method.

  Parameters
  ----------
  N : int
      Extend of the conn scope. For example:
      When N=1,
          [x x x]
          [x I x]
          [x x x]
      When N=2,
          [x x x x x]
          [x x x x x]
          [x x I x x]
          [x x x x x]
          [x x x x x]
  include_self : bool
    Whether create (i, i) conn ?
  periodic_boundary: bool
    Whether the neuron encode the value space with the periodic boundary.
    .. versionadded:: 2.2.3.2
  """

  def __init__(
      self,
      N: int = 1,
      include_self: bool = False,
      periodic_boundary: bool = False,
      **kwargs
  ):
    super(GridN, self).__init__(strides=np.arange(-N, N + 1, 1),
                                include_self=include_self,
                                periodic_boundary=periodic_boundary,
                                **kwargs)
    self.N = N

  def __repr__(self):
    return (f'{self.__class__.__name__}(N={self.N}, '
            f'include_self={self.include_self}, '
            f'periodic_boundary={self.periodic_boundary})')

  def _select_stride(self, stride: np.ndarray) -> np.ndarray:
    return (np.ones(len(stride), dtype=bool)
            if self.include_self else
            (np.sum(np.abs(stride), axis=1) > 0))

  def _select_dist(self, dist: jnp.ndarray) -> jnp.ndarray:
    if self.include_self:
      return jnp.all(dist <= self.N, axis=1)
    else:
      return jnp.logical_and(jnp.all(dist <= self.N, axis=1),
                             jnp.logical_not(jnp.all(dist == 0, axis=1)))


class GridEight(GridN):
  """The nearest eight neighbors conn method.

  Parameters
  ----------
  include_self : bool
    Whether create (i, i) conn ?
  periodic_boundary: bool
    Whether the neurons encode the value space with the periodic boundary.
    .. versionadded:: 2.2.3.2
  """

  def __init__(self,
               include_self: bool = False,
               periodic_boundary: bool = False,
               **kwargs):
    super(GridEight, self).__init__(N=1,
                                    include_self=include_self,
                                    periodic_boundary=periodic_boundary,
                                    **kwargs)


grid_eight = GridEight()
