# -*- coding: utf-8 -*-

import numpy as np

from brainpy import math as bm
from brainpy.tools.others import to_size, size2num
from .base import IntraLayerInitializer


__all__ = [
  'GaussianDecay',
  'DOGDecay',
]


class GaussianDecay(IntraLayerInitializer):
  r"""Builds a Gaussian connectivity pattern within a population of neurons,
  where the weights decay with gaussian function.

  Specifically, for any pair of neurons :math:`(i, j)`, the weight is computed as

  .. math::

      w(i, j) = w_{max} \cdot \exp(-\frac{\sum_{k=1}^n |v_k^i - v_k^j|^2 }{2\sigma^2})

  where :math:`v_k^i` is the $i$-th neuron's encoded value at dimension $k$.

  Parameters
  ----------
  sigma : float
    Width of the Gaussian function.
  max_w : float
    The weight amplitude of the Gaussian function.
  min_w : float, None
    The minimum weight value below which synapses are not created (default: :math:`0.005 * max\_w`).
  include_self : bool
    Whether create the conn at the same position.
  encoding_values : optional, list, tuple, int, float
    The value ranges to encode for neurons at each axis.

    - If `values` is not provided, the neuron only encodes each positional
      information, i.e., :math:`(i, j, k, ...)`, where :math:`i, j, k` is
      the index in the high-dimensional space.
    - If `values` is a single tuple/list of int/float, neurons at each dimension
      will encode the same range of values. For example, `values=(0, np.pi)`,
      neurons at each dimension will encode a continuous value space `[0, np.pi]`.
    - If `values` is a tuple/list of list/tuple, it means the value space will be
      different for each dimension. For example, `values=((-np.pi, np.pi), (10, 20), (0, 2 * np.pi))`.
  periodic_boundary : bool
    Whether the neuron encode the value space with the periodic boundary.
  normalize : bool
      Whether normalize the connection probability.
  """

  def __init__(self, sigma, max_w, min_w=None, encoding_values=None,
               periodic_boundary=False, include_self=True, normalize=False):
    super(GaussianDecay, self).__init__()
    self.sigma = sigma
    self.max_w = max_w
    self.min_w = max_w * 0.005 if min_w is None else min_w
    self.encoding_values = encoding_values
    self.periodic_boundary = periodic_boundary
    self.include_self = include_self
    self.normalize = normalize

  def __call__(self, shape, dtype=None):
    """Build the weights.

    Parameters
    ----------
    shape : tuple of int, list of int, int
      The network shape. Note, this is not the weight shape.
    """
    shape = to_size(shape)
    net_size = size2num(shape)

    # value ranges to encode
    if self.encoding_values is None:
      value_ranges = tuple([(0, s) for s in shape])
    elif isinstance(self.encoding_values, (tuple, list)):
      if len(self.encoding_values) == 0:
        raise ValueError
      elif isinstance(self.encoding_values[0], (int, float)):
        assert len(self.encoding_values) == 2
        assert self.encoding_values[0] < self.encoding_values[1]
        value_ranges = tuple([self.encoding_values for _ in shape])
      elif isinstance(self.encoding_values[0], (tuple, list)):
        if len(self.encoding_values) != len(shape):
          raise ValueError(f'The network size has {len(shape)} dimensions, while '
                           f'the encoded values provided only has {len(self.encoding_values)}-D. '
                           f'Error in {str(self)}.')
        for v in self.encoding_values:
          assert isinstance(v[0], (int, float))
          assert len(v) == 2
        value_ranges = tuple(self.encoding_values)
      else:
        raise ValueError(f'Unsupported encoding values: {self.encoding_values}')
    else:
      raise ValueError(f'Unsupported encoding values: {self.encoding_values}')

    # values
    values = [np.linspace(vs[0], vs[1], n + 1)[:n] for vs, n in zip(value_ranges, shape)]
    post_values = np.stack([v.flatten() for v in np.meshgrid(*values)])
    value_sizes = np.array([v[1] - v[0] for v in value_ranges])
    if value_sizes.ndim < post_values.ndim:
      value_sizes = np.expand_dims(value_sizes, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))

    # connectivity matrix
    conn_mat = []
    for i in range(net_size):
      # values for node i
      i_coordinate = tuple()
      for s in shape[:-1]:
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
      conn_mat.append(exp_dists)
    conn_mat = np.stack(conn_mat)
    if self.normalize:
      conn_mat /= conn_mat.max()
    if not self.include_self:
      np.fill_diagonal(conn_mat, 0.)

    # connectivity weights
    conn_weights = conn_mat * self.max_w
    conn_weights = np.where(conn_weights < self.min_w, 0., conn_weights)
    return bm.asarray(conn_weights, dtype=dtype)

  def __repr__(self):
    name = self.__class__.__name__
    bank = ' ' * len(name)
    return (f'{name}(sigma={self.sigma}, max_w={self.max_w}, min_w={self.min_w}, \n'
            f'{bank}periodic_boundary={self.periodic_boundary}, '
            f'include_self={self.include_self}, '
            f'normalize={self.normalize})')


class DOGDecay(IntraLayerInitializer):
  r"""Builds a Difference-Of-Gaussian (dog) connectivity pattern within a population of neurons.

  Mathematically, for the given pair of neurons :math:`(i, j)`, the weight between them is computed as

  .. math::

      w(i, j) = w_{max}^+ \cdot \exp(-\frac{\sum_{k=1}^n |v_k^i - v_k^j|^2}{2\sigma_+^2}) -
                w_{max}^- \cdot \exp(-\frac{\sum_{k=1}^n |v_k^i - v_k^j|^2}{2\sigma_-^2})

  where weights smaller than :math:`0.005 * max(w_{max}, w_{min})` are not created and
  self-connections are avoided by default (parameter allow_self_connections).

  Parameters
  ----------
  sigmas : tuple
      Widths of the positive and negative Gaussian functions.
  max_ws : tuple
      The weight amplitudes of the positive and negative Gaussian functions.
  min_w : float, None
      The minimum weight value below which synapses are not created (default: :math:`0.005 * min(max\_ws)`).
  include_self : bool
    Whether create the connections at the same position (self-connections).
  normalize : bool
    Whether normalize the connection probability .
  encoding_values : optional, list, tuple, int, float
    The value ranges to encode for neurons at each axis.

    - If `values` is not provided, the neuron only encodes each positional
      information, i.e., :math:`(i, j, k, ...)`, where :math:`i, j, k` is
      the index in the high-dimensional space.
    - If `values` is a single tuple/list of int/float, neurons at each dimension
      will encode the same range of values. For example, `values=(0, np.pi)`,
      neurons at each dimension will encode a continuous value space `[0, np.pi]`.
    - If `values` is a tuple/list of list/tuple, it means the value space will be
      different for each dimension. For example, `values=((-np.pi, np.pi), (10, 20), (0, 2 * np.pi))`.
  periodic_boundary : bool
    Whether the neuron encode the value space with the periodic boundary.
  """

  def __init__(self, sigmas, max_ws, min_w=None, encoding_values=None,
               periodic_boundary=False, normalize=True, include_self=True):
    super(DOGDecay, self).__init__()
    self.sigma_p, self.sigma_n = sigmas
    self.max_w_p, self.max_w_n = max_ws
    self.min_w = 0.005 * min(self.max_w_p, self.max_w_n) if min_w is None else min_w
    self.normalize = normalize
    self.include_self = include_self
    self.encoding_values = encoding_values
    self.periodic_boundary = periodic_boundary

  def __call__(self, shape, dtype=None):
    """Build the weights.

    Parameters
    ----------
    shape : tuple of int, list of int, int
      The network shape. Note, this is not the weight shape.
    """
    shape = to_size(shape)

    # value ranges to encode
    if self.encoding_values is None:
      value_ranges = tuple([(0, s) for s in shape])
    elif isinstance(self.encoding_values, (tuple, list)):
      if len(self.encoding_values) == 0:
        raise ValueError
      elif isinstance(self.encoding_values[0], (int, float)):
        assert len(self.encoding_values) == 2
        assert self.encoding_values[0] < self.encoding_values[1]
        value_ranges = tuple([self.encoding_values for _ in shape])
      elif isinstance(self.encoding_values[0], (tuple, list)):
        if len(self.encoding_values) != len(shape):
          raise ValueError(f'The network size has {len(shape)} dimensions, while '
                           f'the encoded values provided only has {len(self.encoding_values)}-D. '
                           f'Error in {str(self)}.')
        for v in self.encoding_values:
          assert isinstance(v[0], (int, float))
          assert len(v) == 2
        value_ranges = tuple(self.encoding_values)
      else:
        raise ValueError(f'Unsupported encoding values: {self.encoding_values}')
    else:
      raise ValueError(f'Unsupported encoding values: {self.encoding_values}')

    # values
    values = [np.linspace(vs[0], vs[1], n + 1)[:n] for vs, n in zip(value_ranges, shape)]
    post_values = np.stack([v.flatten() for v in np.meshgrid(*values)])
    value_sizes = np.array([v[1] - v[0] for v in value_ranges])
    if value_sizes.ndim < post_values.ndim:
      value_sizes = np.expand_dims(value_sizes, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))
    voxel_ids = np.meshgrid(*[np.arange(s) for s in shape])
    if np.ndim(voxel_ids[0]) > 1:
      voxel_ids = tuple(np.moveaxis(m, 0, 1).flatten() for m in voxel_ids)

    # connectivity matrix
    ndim = len(voxel_ids)
    conn_weights = []
    for v_id in range(voxel_ids[0].shape[0]):
      i_value = []
      for i in range(ndim):
        p_id = voxel_ids[i][v_id]  # position id
        i_value.append(values[i][p_id])
      i_value = np.array(i_value)
      if i_value.ndim < post_values.ndim:
        i_value = np.expand_dims(i_value, axis=tuple([i + 1 for i in range(post_values.ndim - 1)]))
      # distances
      dists = np.abs(i_value - post_values)
      if self.periodic_boundary:
        dists = np.where(dists > value_sizes / 2, value_sizes - dists, dists)
      dists_exp_p = self.max_w_p * np.exp(-(np.linalg.norm(dists, axis=0) / self.sigma_p) ** 2 / 2)
      dists_exp_n = self.max_w_n * np.exp(-(np.linalg.norm(dists, axis=0) / self.sigma_n) ** 2 / 2)
      conn_weights.append(dists_exp_p - dists_exp_n)
    conn_weights = np.stack(conn_weights)
    if not self.include_self:
      np.fill_diagonal(conn_weights, 0.)

    # connectivity weights
    conn_weights = np.where(np.abs(conn_weights) < self.min_w, 0., conn_weights)
    return bm.asarray(conn_weights, dtype=dtype)

  def __repr__(self):
    name = self.__class__.__name__
    bank = ' ' * len(name)
    return (f'{name}(sigmas={(self.sigma_p, self.sigma_n)}, '
            f'max_ws={(self.max_w_p, self.max_w_n)}, min_w={self.min_w}, \n'
            f'{bank}periodic_boundary={self.periodic_boundary}, '
            f'include_self={self.include_self}, '
            f'normalize={self.normalize})')
