# -*- coding: utf-8 -*-

from functools import partial
from typing import Optional

from jax import vmap, jit, numpy as jnp
import numpy as np

import brainpy.math as bm
from brainpy.errors import ConnectorError
from brainpy.tools import numba_seed, numba_jit, numba_range, format_seed
from brainpy._src.tools.package import SUPPORT_NUMBA
from .base import *


__all__ = [
  'FixedProb',
  'FixedPreNum',
  'FixedPostNum',
  'FixedTotalNum',
  'GaussianProb',
  'ProbDist',

  'SmallWorld',
  'ScaleFreeBA',
  'ScaleFreeBADual',
  'PowerLaw',
]


class FixedProb(TwoEndConnector):
  """Connect the post-synaptic neurons with fixed probability.

  Parameters
  ----------
  prob: float
    The conn probability.
  pre_ratio: float
    The ratio of pre-synaptic neurons to connect.
  include_self : bool
    Whether create (i, i) conn?
  allow_multi_conn: bool
    Allow one pre-synaptic neuron connects to multiple post-synaptic neurons?

    .. versionadded:: 2.2.3.2

  seed : optional, int
    Seed the random generator.
  """

  def __init__(self,
               prob,
               pre_ratio=1.,
               include_self=True,
               allow_multi_conn=False,
               seed=None,
               **kwargs):
    super(FixedProb, self).__init__(**kwargs)
    assert 0. <= prob <= 1.
    assert 0. <= pre_ratio <= 1.
    self.prob = prob
    self.pre_ratio = pre_ratio
    self.include_self = include_self
    self.seed = format_seed(seed)
    self.allow_multi_conn = allow_multi_conn
    self._jaxrand = bm.random.default_rng(self.seed)
    self._nprand = np.random.RandomState(self.seed)

  def __repr__(self):
    return (f'{self.__class__.__name__}(prob={self.prob}, pre_ratio={self.pre_ratio}, '
            f'include_self={self.include_self}, allow_multi_conn={self.allow_multi_conn}, '
            f'seed={self.seed})')

  def _iii(self):
    if (not self.include_self) and (self.pre_num != self.post_num):
      raise ConnectorError(f'We found pre_num != post_num ({self.pre_num} != {self.post_num}). '
                           f'But `include_self` is set to True.')

    if self.pre_ratio < 1.:
      pre_num_to_select = int(self.pre_num * self.pre_ratio)
      pre_ids = self._jaxrand.choice(self.pre_num, size=(pre_num_to_select,), replace=False)
    else:
      pre_num_to_select = self.pre_num
      pre_ids = jnp.arange(self.pre_num)

    post_num_total = self.post_num
    post_num_to_select = int(self.post_num * self.prob)

    if self.allow_multi_conn:
      selected_post_ids = self._jaxrand.randint(0, post_num_total, (pre_num_to_select, post_num_to_select))

    else:
      if SUPPORT_NUMBA:
        rng = np.random
        numba_seed(self._nprand.randint(0, int(1e8)))
      else:
        rng = self._nprand

      @numba_jit  # (parallel=True, nogil=True)
      def single_conn():
        posts = np.zeros((pre_num_to_select, post_num_to_select), dtype=IDX_DTYPE)
        for i in numba_range(pre_num_to_select):
          posts[i] = rng.choice(post_num_total, post_num_to_select, replace=False)
        return posts

      selected_post_ids = jnp.asarray(single_conn())
    return pre_num_to_select, post_num_to_select, bm.as_jax(selected_post_ids), bm.as_jax(pre_ids)

  def build_coo(self):
    _, post_num_to_select, selected_post_ids, pre_ids = self._iii()
    selected_post_ids = selected_post_ids.flatten()
    selected_pre_ids = jnp.repeat(pre_ids, post_num_to_select)
    if not self.include_self:
      true_ids = selected_pre_ids != selected_post_ids
      selected_pre_ids = selected_pre_ids[true_ids]
      selected_post_ids = selected_post_ids[true_ids]
    return selected_pre_ids.astype(get_idx_type()), selected_post_ids.astype(get_idx_type())

  def build_csr(self):
    pre_num_to_select, post_num_to_select, selected_post_ids, pre_ids = self._iii()
    pre_nums = jnp.ones(pre_num_to_select) * post_num_to_select
    if not self.include_self:
      true_ids = selected_post_ids == jnp.reshape(pre_ids, (-1, 1))
      pre_nums -= jnp.sum(true_ids, axis=1)
      selected_post_ids = selected_post_ids.flatten()[jnp.logical_not(true_ids).flatten()]
    else:
      selected_post_ids = selected_post_ids.flatten()
    selected_pre_inptr = jnp.cumsum(jnp.concatenate([jnp.zeros(1), pre_nums]))
    return selected_post_ids.astype(get_idx_type()), selected_pre_inptr.astype(get_idx_type())

  def build_mat(self):
    if self.pre_ratio < 1.:
      pre_state = self._jaxrand.uniform(size=(self.pre_num, 1)) < self.pre_ratio
      mat = (self._jaxrand.uniform(size=(self.pre_num, self.post_num)) < self.prob) * pre_state
    else:
      mat = (self._jaxrand.uniform(size=(self.pre_num, self.post_num)) < self.prob)
    mat = bm.asarray(mat)
    if not self.include_self:
      bm.fill_diagonal(mat, False)
    return mat.astype(MAT_DTYPE)


class FixedTotalNum(TwoEndConnector):
  """Connect the synaptic neurons with fixed total number.

  Parameters
  ----------
  num : float,int
    The conn total number.
  allow_multi_conn : bool, optional
    Whether allow one pre-synaptic neuron connects to multiple post-synaptic neurons.
  seed: int, optional
    The random number seed.
  """

  def __init__(self,
               num,
               allow_multi_conn=False,
               seed=None, **kwargs):
    super().__init__(**kwargs)
    if isinstance(num, int):
      assert num >= 0, '"num" must be a non-negative integer.'
    elif isinstance(num, float):
      assert 0. <= num <= 1., '"num" must be in [0., 1.).'
    else:
      raise ConnectorError(f'Unknown type: {type(num)}')
    self.num = num
    self.seed = format_seed(seed)
    self.allow_multi_conn = allow_multi_conn
    self.rng = bm.random.RandomState(self.seed)

  def build_coo(self):
    mat_element_num = self.pre_num * self.post_num
    if self.num > mat_element_num:
      raise ConnectorError(f'"num" must be smaller than "all2all num", '
                           f'but got {self.num} > {mat_element_num}')
    if self.allow_multi_conn:
      selected_pre_ids = self.rng.randint(0, self.pre_num, (self.num,))
      selected_post_ids = self.rng.randint(0, self.post_num, (self.num,))
    else:
      index = self.rng.choice(mat_element_num, size=(self.num,), replace=False)
      selected_pre_ids = index // self.post_num
      selected_post_ids = index % self.post_num
    return selected_pre_ids.astype(get_idx_type()), selected_post_ids.astype(get_idx_type())

  def __repr__(self):
    return f'{self.__class__.__name__}(num={self.num}, seed={self.seed})'


class FixedNum(TwoEndConnector):
  def __init__(self,
               num,
               include_self=True,
               allow_multi_conn=False,
               seed=None,
               **kwargs):
    super(FixedNum, self).__init__(**kwargs)
    if isinstance(num, int):
      assert num >= 0, '"num" must be a non-negative integer.'
    elif isinstance(num, float):
      assert 0. <= num <= 1., '"num" must be in [0., 1.).'
    else:
      raise ConnectorError(f'Unknown type: {type(num)}')
    self.num = num
    self.seed = format_seed(seed)
    self.include_self = include_self
    self.allow_multi_conn = allow_multi_conn
    self.rng = bm.random.RandomState(self.seed) if allow_multi_conn else np.random.RandomState(self.seed)

  def __repr__(self):
    return f'{self.__class__.__name__}(num={self.num}, include_self={self.include_self}, seed={self.seed})'


class FixedPreNum(FixedNum):
  """Connect a fixed number pf pre-synaptic neurons for each post-synaptic neuron.

  Parameters
  ----------
  num : float, int
      The conn probability (if "num" is float) or the fixed number of
      connectivity (if "num" is int).
  include_self : bool
      Whether create (i, i) conn ?
  seed : None, int
      Seed the random generator.
  allow_multi_conn: bool
    Allow one pre-synaptic neuron connects to multiple post-synaptic neurons?

    .. versionadded:: 2.2.3.2

  """

  def build_coo(self):
    if isinstance(self.num, int) and self.num > self.pre_num:
      raise ConnectorError(f'"num" must be smaller than "pre_num", '
                           f'but got {self.num} > {self.pre_num}')
    if (not self.include_self) and (self.pre_num != self.post_num):
      raise ConnectorError(f'We found pre_num != post_num ({self.pre_num} != {self.post_num}). '
                           f'But `include_self` is set to True.')
    pre_num_to_select = int(self.pre_num * self.num) if isinstance(self.num, float) else self.num
    pre_num_total = self.pre_num
    post_num_total = self.post_num

    if self.allow_multi_conn:
      selected_pre_ids = self.rng.randint(0, pre_num_total, (post_num_total, pre_num_to_select,))

    else:
      if SUPPORT_NUMBA:
        rng = np.random
        numba_seed(self.rng.randint(0, int(1e8)))
      else:
        rng = self.rng

      @numba_jit  # (parallel=True, nogil=True)
      def single_conn():
        posts = np.zeros((post_num_total, pre_num_to_select), dtype=IDX_DTYPE)
        for i in numba_range(post_num_total):
          posts[i] = rng.choice(pre_num_total, pre_num_to_select, replace=False)
        return posts

      selected_pre_ids = jnp.asarray(single_conn())

    post_nums = jnp.ones((post_num_total,), dtype=get_idx_type()) * pre_num_to_select
    if not self.include_self:
      true_ids = selected_pre_ids == jnp.reshape(jnp.arange(pre_num_total), (-1, 1))
      post_nums -= jnp.sum(true_ids, axis=1)
      selected_pre_ids = selected_pre_ids.flatten()[jnp.logical_not(true_ids).flatten()]
    else:
      selected_pre_ids = selected_pre_ids.flatten()
    selected_post_ids = jnp.repeat(jnp.arange(post_num_total), post_nums)
    return selected_pre_ids.astype(get_idx_type()), selected_post_ids.astype(get_idx_type())


class FixedPostNum(FixedNum):
  """Connect the fixed number of post-synaptic neurons for each pre-synaptic neuron.

  Parameters
  ----------
  num : float, int
      The conn probability (if "num" is float) or the fixed number of
      connectivity (if "num" is int).
  include_self : bool
      Whether create (i, i) conn ?
  seed : None, int
      Seed the random generator.
  allow_multi_conn: bool
    Allow one pre-synaptic neuron connects to multiple post-synaptic neurons?

    .. versionadded:: 2.2.3.2

  """

  def _ii(self):
    if isinstance(self.num, int) and self.num > self.post_num:
      raise ConnectorError(f'"num" must be smaller than "post_num", '
                           f'but got {self.num} > {self.post_num}')
    if (not self.include_self) and (self.pre_num != self.post_num):
      raise ConnectorError(f'We found pre_num != post_num ({self.pre_num} != {self.post_num}). '
                           f'But `include_self` is set to True.')
    post_num_to_select = int(self.post_num * self.num) if isinstance(self.num, float) else self.num
    pre_num_to_select = self.pre_num
    pre_ids = jnp.arange(self.pre_num)
    post_num_total = self.post_num

    if self.allow_multi_conn:
      selected_post_ids = self.rng.randint(0, post_num_total, (pre_num_to_select, post_num_to_select,))

    else:
      if SUPPORT_NUMBA:
        rng = np.random
        numba_seed(self.rng.randint(0, int(1e8)))
      else:
        rng = self.rng

      @numba_jit  # (parallel=True, nogil=True)
      def single_conn():
        posts = np.zeros((pre_num_to_select, post_num_to_select), dtype=IDX_DTYPE)
        for i in numba_range(pre_num_to_select):
          posts[i] = rng.choice(post_num_total, post_num_to_select, replace=False)
        return posts

      selected_post_ids = jnp.asarray(single_conn())
    return pre_num_to_select, post_num_to_select, bm.as_jax(selected_post_ids), bm.as_jax(pre_ids)

  def build_coo(self):
    _, post_num_to_select, selected_post_ids, pre_ids = self._ii()
    selected_post_ids = selected_post_ids.flatten()
    selected_pre_ids = jnp.repeat(pre_ids, post_num_to_select)
    if not self.include_self:
      true_ids = selected_pre_ids != selected_post_ids
      selected_pre_ids = selected_pre_ids[true_ids]
      selected_post_ids = selected_post_ids[true_ids]
    return selected_pre_ids.astype(get_idx_type()), selected_post_ids.astype(get_idx_type())

  def build_csr(self):
    pre_num_to_select, post_num_to_select, selected_post_ids, pre_ids = self._ii()
    pre_nums = jnp.ones(pre_num_to_select) * post_num_to_select
    if not self.include_self:
      true_ids = selected_post_ids == jnp.reshape(pre_ids, (-1, 1))
      pre_nums -= jnp.sum(true_ids, axis=1)
      selected_post_ids = selected_post_ids.flatten()[jnp.logical_not(true_ids).flatten()]
    else:
      selected_post_ids = selected_post_ids.flatten()
    selected_pre_inptr = jnp.cumsum(jnp.concatenate([jnp.zeros(1), pre_nums]))
    return selected_post_ids.astype(get_idx_type()), selected_pre_inptr.astype(get_idx_type())


@jit
@partial(vmap, in_axes=(0, None, None))
def gaussian_prob_dist_cal1(i_value, post_values, sigma):
  dists = jnp.abs(i_value - post_values)
  exp_dists = jnp.exp(-(jnp.sqrt(jnp.sum(dists ** 2, axis=0)) / sigma) ** 2 / 2)
  return bm.asarray(exp_dists)


@jit
@partial(vmap, in_axes=(0, None, None, None))
def gaussian_prob_dist_cal2(i_value, post_values, value_sizes, sigma):
  dists = jnp.abs(i_value - post_values)
  dists = jnp.where(dists > (value_sizes / 2), value_sizes - dists, dists)
  exp_dists = jnp.exp(-(jnp.sqrt(jnp.sum(dists ** 2, axis=0)) / sigma) ** 2 / 2)
  return bm.asarray(exp_dists)


class GaussianProb(OneEndConnector):
  r"""Builds a Gaussian connectivity pattern within a population of neurons,
  where the connection probability decay according to the gaussian function.

  Specifically, for any pair of neurons :math:`(i, j)`,

  .. math::

      p(i, j)=\exp(-\frac{\sum_{k=1}^n |v_k^i - v_k^j|^2 }{2\sigma^2})

  where :math:`v_k^i` is the :math:`i`-th neuron's encoded value at dimension :math:`k`.

  Parameters
  ----------
  sigma : float
      Width of the Gaussian function.
  encoding_values : optional, list, tuple, int, float
    The value ranges to encode for neurons at each axis.

    - If `values` is not provided, the neuron only encodes each positional
      information, i.e., :math:`(i, j, k, ...)`, where :math:`i, j, k` is
      the index in the high-dimensional space.
    - If `values` is a single tuple/list of int/float, neurons at each dimension
      will encode the same range of values. For example, ``values=(0, np.pi)``,
      neurons at each dimension will encode a continuous value space ``[0, np.pi]``.
    - If `values` is a tuple/list of list/tuple, it means the value space will be
      different for each dimension. For example, ``values=((-np.pi, np.pi), (10, 20), (0, 2 * np.pi))``.

  periodic_boundary : bool
    Whether the neuron encode the value space with the periodic boundary.
  normalize : bool
      Whether normalize the connection probability .
  include_self : bool
      Whether create the connection at the same position.
  seed : int
      The random seed.
  """

  def __init__(
      self,
      sigma: float,
      encoding_values: Optional[np.ndarray] = None,
      normalize: bool = True,
      include_self: bool = True,
      periodic_boundary: bool = False,
      seed: int = None,
      **kwargs
  ):
    super(GaussianProb, self).__init__(**kwargs)
    self.sigma = sigma
    self.encoding_values = encoding_values
    self.normalize = normalize
    self.include_self = include_self
    self.periodic_boundary = periodic_boundary
    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(self.seed)

  def __repr__(self):
    return (f'{self.__class__.__name__}(sigma={self.sigma}, '
            f'normalize={self.normalize}, '
            f'periodic_boundary={self.periodic_boundary}, '
            f'include_self={self.include_self}, '
            f'seed={self.seed})')

  def build_mat(self, isOptimized=True):
    self.rng = np.random.RandomState(self.seed)
    # value range to encode
    if self.encoding_values is None:
      value_ranges = tuple([(0, s) for s in self.pre_size])
    elif isinstance(self.encoding_values, (tuple, list)):
      if len(self.encoding_values) == 0:
        raise ConnectorError(f'encoding_values has a length of 0.')
      elif isinstance(self.encoding_values[0], (int, float)):
        assert len(self.encoding_values) == 2
        assert self.encoding_values[0] < self.encoding_values[1]
        value_ranges = tuple([self.encoding_values for _ in self.pre_size])
      elif isinstance(self.encoding_values[0], (tuple, list)):
        if len(self.encoding_values) != len(self.pre_size):
          raise ConnectorError(f'The network size has {len(self.pre_size)} dimensions, while '
                               f'the encoded values provided only has {len(self.encoding_values)}-D. '
                               f'Error in {str(self)}.')
        for v in self.encoding_values:
          assert isinstance(v[0], (int, float))
          assert len(v) == 2
        value_ranges = tuple(self.encoding_values)
      else:
        raise ConnectorError(f'Unsupported encoding values: {self.encoding_values}')
    else:
      raise ConnectorError(f'Unsupported encoding values: {self.encoding_values}')

    # values
    values = [np.linspace(vs[0], vs[1], n + 1)[:n] for vs, n in zip(value_ranges, self.pre_size)]
    # post_values = np.stack([v.flatten() for v in np.meshgrid(*values, indexing='ij')])
    post_values = np.stack([v.flatten() for v in np.meshgrid(*values)])
    value_sizes = np.array([v[1] - v[0] for v in value_ranges])
    if value_sizes.ndim < post_values.ndim:
      value_sizes = np.expand_dims(value_sizes, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))

    # probability of connections
    if isOptimized:
      i_value_list = np.zeros(shape=(self.pre_num, len(self.pre_size), 1))
      for i in range(self.pre_num):
        list_index = i
        # values for node i
        i_coordinate = tuple()
        for s in self.pre_size[:-1]:
          i, pos = divmod(i, s)
          i_coordinate += (pos,)
        i_coordinate += (i,)
        i_value = np.array([values[i][c] for i, c in enumerate(i_coordinate)])
        if i_value.ndim < post_values.ndim:
          i_value = np.expand_dims(i_value, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))
        i_value_list[list_index] = i_value

      if self.periodic_boundary:
        prob_mat = gaussian_prob_dist_cal2(i_value_list, post_values, value_sizes, self.sigma)
      else:
        prob_mat = gaussian_prob_dist_cal1(i_value_list, post_values, self.sigma)
    else:
      prob_mat = []
      for i in range(self.pre_num):
        # values for node i
        i_coordinate = tuple()
        for s in self.pre_size[:-1]:
          i, pos = divmod(i, s)
          i_coordinate += (pos,)
        i_coordinate += (i,)
        i_value = np.array([values[i][c] for i, c in enumerate(i_coordinate)])
        if i_value.ndim < post_values.ndim:
          i_value = np.expand_dims(i_value, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))
        # distances
        dists = np.abs(i_value - post_values)
        if self.periodic_boundary:
          dists = np.where(dists > value_sizes / 2, value_sizes - dists, dists)
        exp_dists = np.exp(-(np.linalg.norm(dists, axis=0) / self.sigma) ** 2 / 2)
        prob_mat.append(exp_dists)
      prob_mat = np.stack(prob_mat)

    if self.normalize:
      prob_mat /= prob_mat.max()

    # connectivity
    conn_mat = np.asarray(prob_mat) >= self.rng.random(prob_mat.shape)
    if not self.include_self:
      np.fill_diagonal(conn_mat, False)
    return conn_mat


class SmallWorld(TwoEndConnector):
  """Build a Watts–Strogatz small-world graph.

  Parameters
  ----------
  num_neighbor : int
      Each node is joined with its `k` nearest neighbors in a ring
      topology.
  prob : float
      The probability of rewiring each edge
  directed : bool
      Whether the graph is a directed graph.
  include_self : bool
      Whether include the node self.

  Notes
  -----
  First create a ring over :math:`num\_node` nodes [1]_.  Then each node in the ring is
  joined to its :math:`num\_neighbor` nearest neighbors (or :math:`num\_neighbor - 1` neighbors
  if :math:`num\_neighbor` is odd). Then shortcuts are created by replacing some edges as
  follows: for each edge :math:`(u, v)` in the underlying ":math:`num\_node`-ring with
  :math:`num\_neighbor` nearest neighbors" with probability :math:`prob` replace it with a new
  edge :math:`(u, w)` with uniformly random choice of existing node :math:`w`.

  References
  ----------
  .. [1] Duncan J. Watts and Steven H. Strogatz,
         Collective dynamics of small-world networks,
         Nature, 393, pp. 440--442, 1998.
  """

  def __init__(
      self,
      num_neighbor,
      prob,
      directed=False,
      include_self=False,
      seed=None,
      **kwargs
  ):
    super(SmallWorld, self).__init__(**kwargs)
    self.prob = prob
    self.directed = directed
    self.num_neighbor = num_neighbor
    self.include_self = include_self

    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(seed=self.seed)
    rng = np.random if SUPPORT_NUMBA else self.rng

    def _smallworld_rewire(i, all_j):
      if rng.random(1) < prob:
        non_connected = np.where(np.logical_not(all_j))[0]
        if len(non_connected) <= 1:
          return -1
        # Enforce no self-loops or multiple edges
        w = rng.choice(non_connected)
        while (not include_self) and w == i:
          # non_connected.remove(w)
          w = rng.choice(non_connected)
        return w
      else:
        return -1

    self._connect = numba_jit(_smallworld_rewire)

  def __repr__(self):
    return (f'{self.__class__.__name__}(prob={self.prob}, '
            f'directed={self.directed}, '
            f'num_neighbor={self.num_neighbor}, '
            f'include_self={self.include_self}, '
            f'seed={self.seed})')

  def build_conn(self):
    assert self.pre_size == self.post_size

    # seed
    self.seed = self.rng.randint(1, int(1e7))
    numba_seed(self.seed)

    if isinstance(self.pre_size, int) or (isinstance(self.pre_size, (tuple, list)) and len(self.pre_size) == 1):
      num_node = self.pre_num

      if self.num_neighbor > num_node:
        raise ConnectorError("num_neighbor > num_node, choose smaller num_neighbor or larger num_node")
      # If k == n, the graph is complete not Watts-Strogatz
      if self.num_neighbor == num_node:
        conn = np.ones((num_node, num_node), dtype=MAT_DTYPE)
      else:
        conn = np.zeros((num_node, num_node), dtype=MAT_DTYPE)
        nodes = np.array(list(range(num_node)))  # nodes are labeled 0 to n-1
        # connect each node to k/2 neighbors
        for j in range(1, self.num_neighbor // 2 + 1):
          targets = np.concatenate([nodes[j:], nodes[0:j]])  # first j nodes are now last in list
          conn[nodes, targets] = True
          conn[targets, nodes] = True

        # rewire edges from each node
        # loop over all nodes in order (label) and neighbors in order (distance)
        # no self loops or multiple edges allowed
        for j in range(1, self.num_neighbor // 2 + 1):  # outer loop is neighbors
          targets = np.concatenate([nodes[j:], nodes[0:j]])  # first j nodes are now last in list
          if self.directed:
            # inner loop in node order
            for u, v in zip(nodes, targets):
              w = self._connect(prob=self.prob, i=u, all_j=conn[u])
              if w != -1:
                conn[u, v] = False
                conn[u, w] = True
              w = self._connect(prob=self.prob, i=u, all_j=conn[:, u])
              if w != -1:
                conn[v, u] = False
                conn[w, u] = True
          else:
            # inner loop in node order
            for u, v in zip(nodes, targets):
              w = self._connect(i=u, all_j=conn[u])
              if w != -1:
                conn[u, v] = False
                conn[v, u] = False
                conn[u, w] = True
                conn[w, u] = True
        # conn = np.asarray(conn, dtype=MAT_DTYPE)
    else:
      raise ConnectorError('Currently only support 1D ring connection.')

    return 'mat', conn


# def _random_subset(seq, m, rng):
#   """Return m unique elements from seq.
#
#   This differs from random.sample which can return repeated
#   elements if seq holds repeated elements.
#
#   Note: rng is a random.Random or numpy.random.RandomState instance.
#   """
#   targets = set()
#   while len(targets) < m:
#     x = rng.choice(seq)
#     targets.add(x)
#   return targets


class ScaleFreeBA(TwoEndConnector):
  """Build a random graph according to the Barabási–Albert preferential
  attachment model.

  A graph of :math:`num\_node` nodes is grown by attaching new nodes each with
  :math:`m` edges that are preferentially attached to existing nodes
  with high degree.

  Parameters
  ----------
  m : int
      Number of edges to attach from a new node to existing nodes
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Raises
  ------
  ConnectorError
      If `m` does not satisfy ``1 <= m < n``.

  References
  ----------
  .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
         random networks", Science 286, pp 509-512, 1999.
  """

  def __init__(self, m, directed=False, seed=None, **kwargs):
    super(ScaleFreeBA, self).__init__(**kwargs)
    self.m = m
    self.directed = directed
    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(self.seed)
    rng = np.random if SUPPORT_NUMBA else self.rng

    def _random_subset(seq, m):
      targets = set()
      while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
      return targets

    self._connect = numba_jit(_random_subset)

  def __repr__(self):
    return (f'{self.__class__.__name__}(m={self.m}, '
            f'directed={self.directed}, '
            f'seed={self.seed})')

  def build_mat(self, isOptimized=True):
    assert self.pre_num == self.post_num

    # seed
    self.rng = np.random.RandomState(self.seed)
    numba_seed(self.seed)

    num_node = self.pre_num
    if self.m < 1 or self.m >= num_node:
      raise ConnectorError(f"Barabási–Albert network must have m >= 1 and "
                           f"m < n, while m = {self.m} and n = {num_node}")

    # Add m initial nodes (m0 in barabasi-speak)
    conn = np.zeros((num_node, num_node), dtype=MAT_DTYPE)
    # Target nodes for new edges
    targets = list(range(self.m))
    # List of existing nodes, with nodes repeated once for each adjacent edge

    if not isOptimized:
      repeated_nodes = []
      # Start adding the other n-m nodes. The first node is m.
      source = self.m
      while source < num_node:
        # Add edges to m nodes from the source.
        origins = [source] * self.m
        conn[origins, targets] = True
        if not self.directed:
          conn[targets, origins] = True
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * self.m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = list(self._connect(np.asarray(repeated_nodes), self.m))
        source += 1
      return conn

    # List of existing nodes, with nodes repeated once for each adjacent edge
    # Preallocate repeated_nodes as a numpy array
    repeated_nodes = np.empty(2 * num_node * self.m, dtype=int)
    size_repeated_nodes = 0
    # Start adding the other n-m nodes. The first node is m.
    source = self.m
    while source < num_node:
      # Add edges to m nodes from the source.
      origins = [source] * self.m
      conn[origins, targets] = True
      if not self.directed:
        conn[targets, origins] = True
      # Add one node to the list for each new edge just created.
      repeated_nodes[size_repeated_nodes:size_repeated_nodes + self.m] = targets
      size_repeated_nodes += self.m
      # And the new node "source" has m edges to add to the list.
      repeated_nodes[size_repeated_nodes:size_repeated_nodes + self.m] = source
      size_repeated_nodes += self.m
      # Now choose m unique nodes from the existing nodes
      # Pick uniformly from repeated_nodes (preferential attachment)
      targets = list(self._connect(repeated_nodes[:size_repeated_nodes], self.m))
      source += 1

    return conn


class ScaleFreeBADual(TwoEndConnector):
  r"""Build a random graph according to the dual Barabási–Albert preferential
  attachment model.

  A graph of :math::`num\_node` nodes is grown by attaching new nodes each with either $m_1$
  edges (with probability :math:`p`) or :math:`m_2` edges (with probability :math:`1-p`) that
  are preferentially attached to existing nodes with high degree.

  Parameters
  ----------
  m1 : int
      Number of edges to attach from a new node to existing nodes with probability :math:`p`
  m2 : int
      Number of edges to attach from a new node to existing nodes with probability :math:`1-p`
  p : float
      The probability of attaching :math:`m\_1` edges (as opposed to :math:`m\_2` edges)
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Raises
  ------
  ConnectorError
      If `m1` and `m2` do not satisfy ``1 <= m1,m2 < n`` or `p` does not satisfy ``0 <= p <= 1``.

  References
  ----------
  .. [1] N. Moshiri "The dual-Barabasi-Albert model", arXiv:1810.10538.
  """

  def __init__(self, m1, m2, p, directed=False, seed=None, **kwargs):
    super(ScaleFreeBADual, self).__init__(**kwargs)
    self.m1 = m1
    self.m2 = m2
    self.p = p
    self.directed = directed
    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(self.seed)
    rng = np.random if SUPPORT_NUMBA else self.rng

    def _random_subset(seq, m):
      targets = set()
      while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
      return targets

    self._connect = numba_jit(_random_subset)

  def __repr__(self):
    return (f'{self.__class__.__name__}(m1={self.m1}, m2={self.m2}, '
            f'p={self.p}, directed={self.directed}, seed={self.seed})')

  def build_mat(self, isOptimized=True):
    assert self.pre_num == self.post_num
    # seed
    self.rng = np.random.RandomState(self.seed)
    numba_seed(self.seed)

    num_node = self.pre_num
    if self.m1 < 1 or self.m1 >= num_node:
      raise ConnectorError(f"Dual Barabási–Albert network must have m1 >= 1 and m1 < num_node, "
                           f"while m1 = {self.m1} and num_node = {num_node}.")
    if self.m2 < 1 or self.m2 >= num_node:
      raise ConnectorError(f"Dual Barabási–Albert network must have m2 >= 1 and m2 < num_node, "
                           f"while m2 = {self.m2} and num_node = {num_node}.")
    if self.p < 0 or self.p > 1:
      raise ConnectorError(f"Dual Barabási–Albert network must have 0 <= p <= 1, while p = {self.p}")

    # Add max(m1,m2) initial nodes (m0 in barabasi-speak)
    conn = np.zeros((num_node, num_node), dtype=MAT_DTYPE)

    if not isOptimized:
      # List of existing nodes, with nodes repeated once for each adjacent edge
      repeated_nodes = []
      # Start adding the remaining nodes.
      source = max(self.m1, self.m2)
      # Pick which m to use first time (m1 or m2)
      m = self.m1 if self.rng.random() < self.p else self.m2
      # Target nodes for new edges
      targets = list(range(m))
      while source < num_node:
        # Add edges to m nodes from the source.
        origins = [source] * m
        conn[origins, targets] = True
        if not self.directed:
          conn[targets, origins] = True
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Pick which m to use next time (m1 or m2)
        m = self.m1 if self.rng.random() < self.p else self.m2
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = list(self._connect(np.asarray(repeated_nodes), m))
        source += 1
      return conn

    # List of existing nodes, with nodes repeated once for each adjacent edge
    # Preallocate repeated_nodes as a numpy array
    repeated_nodes = np.empty(2 * num_node * max(self.m1, self.m2), dtype=int)
    size_repeated_nodes = 0
    # Start adding the remaining nodes.
    source = max(self.m1, self.m2)
    # Pick which m to use first time (m1 or m2)
    m = self.m1 if self.rng.random() < self.p else self.m2
    # Target nodes for new edges
    targets = list(range(m))
    while source < num_node:
      # Add edges to m nodes from the source.
      origins = [source] * m
      conn[origins, targets] = True
      if not self.directed:
        conn[targets, origins] = True
      # Add one node to the list for each new edge just created.
      repeated_nodes[size_repeated_nodes:size_repeated_nodes + m] = targets
      size_repeated_nodes += m
      # And the new node "source" has m edges to add to the list.
      repeated_nodes[size_repeated_nodes:size_repeated_nodes + m] = source
      size_repeated_nodes += m
      # Pick which m to use next time (m1 or m2)
      m = self.m1 if self.rng.random() < self.p else self.m2
      # Now choose m unique nodes from the existing nodes
      # Pick uniformly from repeated_nodes (preferential attachment)
      targets = list(self._connect(repeated_nodes[:size_repeated_nodes], m))
      source += 1

    return conn


class PowerLaw(TwoEndConnector):
  """Holme and Kim algorithm for growing graphs with powerlaw
  degree distribution and approximate average clustering.

  Parameters
  ----------
  m : int
      the number of random edges to add for each new node
  p : float,
      Probability of adding a triangle after adding a random edge
  seed : integer, random_state, or None (default)
      Indicator of random number generation state.

  Notes
  -----
  The average clustering has a hard time getting above a certain
  cutoff that depends on :math:`m`.  This cutoff is often quite low.  The
  transitivity (fraction of triangles to possible triangles) seems to
  decrease with network size.

  It is essentially the Barabási–Albert (BA) growth model with an
  extra step that each random edge is followed by a chance of
  making an edge to one of its neighbors too (and thus a triangle).

  This algorithm improves on BA in the sense that it enables a
  higher average clustering to be attained if desired.

  It seems possible to have a disconnected graph with this algorithm
  since the initial :math:`m` nodes may not be all linked to a new node
  on the first iteration like the BA model.

  Raises
  ------
  ConnectorError
      If :math:`m` does not satisfy :math:`1 <= m <= n` or :math:`p` does not
      satisfy :math:`0 <= p <= 1`.

  References
  ----------
  .. [1] P. Holme and B. J. Kim,
         "Growing scale-free networks with tunable clustering",
         Phys. Rev. E, 65, 026107, 2002.
  """

  def __init__(self, m: int, p: float, directed=False, seed=None, **kwargs):
    super(PowerLaw, self).__init__(**kwargs)
    self.m = m
    self.p = p
    if self.p > 1 or self.p < 0:
      raise ConnectorError(f"p must be in [0,1], while p={self.p}")
    self.directed = directed
    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(self.seed)
    rng = np.random if SUPPORT_NUMBA else self.rng

    def _random_subset(seq, m):
      targets = set()
      while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
      return targets

    self._connect = numba_jit(_random_subset)

  def __repr__(self):
    return (f'{self.__class__.__name__}(m={self.m}, p={self.p}, directed={self.directed}, seed={self.seed})')

  def build_mat(self, isOptimized=True):
    assert self.pre_num == self.post_num
    # seed
    self.rng = np.random.RandomState(self.seed)
    numba_seed(self.seed)
    num_node = self.pre_num
    if self.m < 1 or num_node < self.m:
      raise ConnectorError(f"Must have m>1 and m<n, while m={self.m} and n={num_node}")
    # add m initial nodes (m0 in barabasi-speak)
    conn = np.zeros((num_node, num_node), dtype=MAT_DTYPE)

    if not isOptimized:
      repeated_nodes = list(range(self.m))  # list of existing nodes to sample from
      # with nodes repeated once for each adjacent edge
      source = self.m  # next node is m
      while source < num_node:  # Now add the other n-1 nodes
        possible_targets = self._connect(np.asarray(repeated_nodes), self.m)
        # do one preferential attachment for new node
        target = possible_targets.pop()
        conn[source, target] = True
        if not self.directed:
          conn[target, source] = True
        repeated_nodes.append(target)  # add one node to list for each new link
        count = 1
        while count < self.m:  # add m-1 more new links
          if self.rng.random() < self.p:  # clustering step: add triangle
            neighbors = np.where(conn[target])[0]
            neighborhood = [nbr for nbr in neighbors if not conn[source, nbr] and not nbr == source]
            if neighborhood:  # if there is a neighbor without a link
              nbr = self.rng.choice(neighborhood)
              conn[source, nbr] = True  # add triangle
              if not self.directed:
                conn[nbr, source] = True
              repeated_nodes.append(nbr)
              count = count + 1
              continue  # go to top of while loop
          # else do preferential attachment step if above fails
          target = possible_targets.pop()
          conn[source, target] = True
          if not self.directed:
            conn[target, source] = True
          repeated_nodes.append(target)
          count = count + 1
        repeated_nodes.extend([source] * self.m)  # add source node to list m times
        source += 1
      return conn

    # Preallocate repeated_nodes as a numpy array
    repeated_nodes = np.empty(2 * num_node * self.m, dtype=int)
    repeated_nodes[:self.m] = np.arange(self.m)
    size_repeated_nodes = self.m

    source = self.m  # next node is m
    while source < num_node:  # Now add the other n-1 nodes
      possible_targets = list(self._connect(repeated_nodes[:size_repeated_nodes], self.m))
      possible_targets.reverse()

      # do one preferential attachment for new node
      target = possible_targets.pop()
      conn[source, target] = True
      if not self.directed:
        conn[target, source] = True
      repeated_nodes[size_repeated_nodes] = target
      size_repeated_nodes += 1

      count = 1
      while count < self.m:  # add m-1 more new links
        if self.rng.random() < self.p:  # clustering step: add triangle
          neighbors = np.where(conn[target])[0]
          neighborhood = [nbr for nbr in neighbors if not conn[source, nbr] and nbr != source]
          if neighborhood:  # if there is a neighbor without a link
            nbr = self.rng.choice(neighborhood)
            conn[source, nbr] = True  # add triangle
            if not self.directed:
              conn[nbr, source] = True
            repeated_nodes[size_repeated_nodes] = nbr
            size_repeated_nodes += 1
            count += 1
            continue  # go to top of while loop

        # else do preferential attachment step if above fails
        target = possible_targets.pop()
        conn[source, target] = True
        if not self.directed:
          conn[target, source] = True
        repeated_nodes[size_repeated_nodes] = target
        size_repeated_nodes += 1
        count += 1

      repeated_nodes[size_repeated_nodes:size_repeated_nodes + self.m] = source
      size_repeated_nodes += self.m
      source += 1
    return conn


@numba_jit
def pos2ind(pos, size):
  idx = 0
  for i, p in enumerate(pos):
    idx += p * np.prod(size[i + 1:])
  return idx


class ProbDist(TwoEndConnector):
  """Connection with a maximum distance under a probability `p`.

  .. versionadded:: 2.1.13

  Parameters
  ----------
  dist: float, int
    The maximum distance between two points.
  prob: float
    The connection probability, within 0. and 1.
  pre_ratio: float
    The ratio of pre-synaptic neurons to connect.
  seed: optional, int
    The random seed.
  include_self: bool
    Whether include the point at the same position.

  """

  def __init__(self, dist=1, prob=1., pre_ratio=1., seed=None, include_self=True, **kwargs):
    super(ProbDist, self).__init__(**kwargs)

    self.prob = prob
    self.pre_ratio = pre_ratio
    self.dist = dist
    self.seed = format_seed(seed)
    self.rng = np.random.RandomState(self.seed)
    self.include_self = include_self

    rng = np.random if SUPPORT_NUMBA else self.rng

    @numba_jit
    def _connect_1d_jit(pre_pos, pre_size, post_size, n_dim):
      all_post_ids = np.zeros(post_size[0], dtype=IDX_DTYPE)
      all_pre_ids = np.zeros(post_size[0], dtype=IDX_DTYPE)
      size = 0

      if rng.random() < pre_ratio:
        normalized_pos = np.zeros(n_dim)
        for i in range(n_dim):
          pre_len = pre_size[i]
          post_len = post_size[i]
          normalized_pos[i] = pre_pos[i] * post_len / pre_len
        for i in range(post_size[0]):
          post_pos = np.asarray((i,))
          d = np.abs(pre_pos[0] - post_pos[0])
          if d <= dist:
            if d == 0. and not include_self:
              continue
            if rng.random() <= prob:
              all_post_ids[size] = pos2ind(post_pos, post_size)
              all_pre_ids[size] = pos2ind(pre_pos, pre_size)
              size += 1
      return all_pre_ids[:size], all_post_ids[:size]

    @numba_jit
    def _connect_2d_jit(pre_pos, pre_size, post_size, n_dim):
      max_size = post_size[0] * post_size[1]
      all_post_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      all_pre_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      size = 0

      if rng.random() < pre_ratio:
        normalized_pos = np.zeros(n_dim)
        for i in range(n_dim):
          pre_len = pre_size[i]
          post_len = post_size[i]
          normalized_pos[i] = pre_pos[i] * post_len / pre_len
        for i in range(post_size[0]):
          for j in range(post_size[1]):
            post_pos = np.asarray((i, j))
            d = np.sqrt(np.sum(np.square(pre_pos - post_pos)))
            if d <= dist:
              if d == 0. and not include_self:
                continue
              if rng.random() <= prob:
                all_post_ids[size] = pos2ind(post_pos, post_size)
                all_pre_ids[size] = pos2ind(pre_pos, pre_size)
                size += 1
      return all_pre_ids[:size], all_post_ids[:size]  # Return filled part of the arrays

    @numba_jit
    def _connect_3d_jit(pre_pos, pre_size, post_size, n_dim):
      max_size = post_size[0] * post_size[1] * post_size[2]
      all_post_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      all_pre_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      size = 0

      if rng.random() < pre_ratio:
        normalized_pos = np.zeros(n_dim)
        for i in range(n_dim):
          pre_len = pre_size[i]
          post_len = post_size[i]
          normalized_pos[i] = pre_pos[i] * post_len / pre_len
        for i in range(post_size[0]):
          for j in range(post_size[1]):
            for k in range(post_size[2]):
              post_pos = np.asarray((i, j, k))
              d = np.sqrt(np.sum(np.square(pre_pos - post_pos)))
              if d <= dist:
                if d == 0. and not include_self:
                  continue
                if rng.random() <= prob:
                  all_post_ids[size] = pos2ind(post_pos, post_size)
                  all_pre_ids[size] = pos2ind(pre_pos, pre_size)
                  size += 1
      return all_pre_ids[:size], all_post_ids[:size]

    @numba_jit
    def _connect_4d_jit(pre_pos, pre_size, post_size, n_dim):
      max_size = post_size[0] * post_size[1] * post_size[2] * post_size[3]
      all_post_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      all_pre_ids = np.zeros(max_size, dtype=IDX_DTYPE)
      size = 0

      if rng.random() < pre_ratio:
        normalized_pos = np.zeros(n_dim)
        for i in range(n_dim):
          pre_len = pre_size[i]
          post_len = post_size[i]
          normalized_pos[i] = pre_pos[i] * post_len / pre_len
        for i in range(post_size[0]):
          for j in range(post_size[1]):
            for k in range(post_size[2]):
              for l in range(post_size[3]):
                post_pos = np.asarray((i, j, k, l))
                d = np.sqrt(np.sum(np.square(pre_pos - post_pos)))
                if d <= dist:
                  if d == 0. and not include_self:
                    continue
                  if rng.random() <= prob:
                    all_post_ids[size] = pos2ind(post_pos, post_size)
                    all_pre_ids[size] = pos2ind(pre_pos, pre_size)
                    size += 1
      return all_pre_ids[:size], all_post_ids[:size]

    self._connect_1d_jit = _connect_1d_jit
    self._connect_2d_jit = _connect_2d_jit
    self._connect_3d_jit = _connect_3d_jit
    self._connect_4d_jit = _connect_4d_jit

  def build_coo(self, isOptimized=True):
    if len(self.pre_size) != len(self.post_size):
      raise ValueError('The dimensions of shapes of two objects to establish connections should '
                       f'be the same. But we got dimension {len(self.pre_size)} != {len(self.post_size)}. '
                       f'Specifically, pre size = {self.pre_size}, post size = {self.post_size}')
    self.rng = np.random.RandomState(self.seed)
    numba_seed(self.seed)

    # connections
    n_dim = len(self.pre_size)
    if n_dim == 1:
      f = self._connect_1d_jit
    elif n_dim == 2:
      f = self._connect_2d_jit
    elif n_dim == 3:
      f = self._connect_3d_jit
    elif n_dim == 4:
      f = self._connect_4d_jit
    else:
      raise NotImplementedError('Does not support the network dimension bigger than 4.')

    pre_size = np.asarray(self.pre_size)
    post_size = np.asarray(self.post_size)
    connected_pres = []
    connected_posts = []
    pre_ids = np.meshgrid(*(np.arange(p) for p in self.pre_size), indexing='ij')
    pre_ids = tuple([(np.moveaxis(p, 0, 1).flatten()) if p.ndim > 1 else p.flatten() for p in pre_ids])
    size = np.prod(pre_size)

    for i in range(size):
      pre_pos = np.asarray([p[i] for p in pre_ids])
      pres, posts = f(pre_pos, pre_size=pre_size, post_size=post_size, n_dim=n_dim)
      connected_pres.extend(pres)
      connected_posts.extend(posts)
    return np.asarray(connected_pres), np.asarray(connected_posts)
