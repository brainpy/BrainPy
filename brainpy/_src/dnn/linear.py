# -*- coding: utf-8 -*-


import numbers
from typing import Dict, Optional, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np

from brainpy import math as bm
from brainpy._src import connect, initialize as init
from brainpy._src.context import share
from brainpy._src.dependency_check import import_taichi
from brainpy._src.dnn.base import Layer
from brainpy._src.mixin import SupportOnline, SupportOffline, SupportSTDP
from brainpy.check import is_initializer
from brainpy.connect import csr2csc
from brainpy.errors import MathError, PackageMissingError
from brainpy.initialize import XavierNormal, ZeroInit, Initializer, parameter
from brainpy.types import ArrayType, Sharding

ti = import_taichi(error_if_not_found=False)

__all__ = [
  'Dense', 'Linear',
  'Identity',
  'AllToAll',
  'OneToOne',
  'MaskedLinear',
  'CSRLinear', 'EventCSRLinear',
  'JitFPHomoLinear', 'JitFPUniformLinear', 'JitFPNormalLinear',
  'EventJitFPHomoLinear', 'EventJitFPNormalLinear', 'EventJitFPUniformLinear',
]


class Dense(Layer, SupportSTDP, SupportOnline, SupportOffline):
  r"""A linear transformation applied over the last dimension of the input.

  Mathematically, this node can be defined as:

  .. math::

     y = x  \cdot weight + b

  Parameters
  ----------
  num_in: int
    The number of the input feature. A positive integer.
  num_out: int
    The number of the output features. A positive integer.
  W_initializer: optional, Initializer
    The weight initialization.
  b_initializer: optional, Initializer
    The bias initialization.
  mode: Mode
    Enable training this node or not. (default True)
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      W_initializer: Union[Initializer, Callable, ArrayType] = XavierNormal(),
      b_initializer: Optional[Union[Initializer, Callable, ArrayType]] = ZeroInit(),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super(Dense, self).__init__(mode=mode, name=name)

    # shape
    self.num_in = num_in
    self.num_out = num_out
    if num_in < 0:
      raise ValueError(f'Received an invalid value for `num_out`, expected '
                       f'a positive integer. Received: num_in={num_in}')
    if num_out < 0:
      raise ValueError(f'Received an invalid value for `num_out`, expected '
                       f'a positive integer. Received: num_out={num_out}')

    # weight initializer
    self.W_initializer = W_initializer
    self.bias_initializer = b_initializer
    is_initializer(W_initializer, 'weight_initializer')
    is_initializer(b_initializer, 'bias_initializer', allow_none=True)

    # parameter initialization
    W = parameter(self.W_initializer, (num_in, self.num_out))
    b = parameter(self.bias_initializer, (self.num_out,))
    if isinstance(self.mode, bm.TrainingMode):
      W = bm.TrainVar(W)
      b = None if (b is None) else bm.TrainVar(b)
    self.W = W
    self.b = b

    # fitting parameters
    self.online_fit_by = None  # support online training
    self.offline_fit_by = None  # support offline training
    self.fit_record = dict()

  def __repr__(self):
    return (f'{self.__class__.__name__}(name={self.name}, '
            f'num_in={self.num_in}, '
            f'num_out={self.num_out}, '
            f'mode={self.mode})')

  def update(self, x):
    x = bm.as_jax(x)
    res = x @ self.W
    if self.b is not None:
      res += self.b

    # online fitting data
    if share.load('fit', False) and self.online_fit_by is not None:
      self.fit_record['input'] = x
      self.fit_record['output'] = res

    # offline fitting data
    if share.load('fit', False) and self.offline_fit_by is not None:
      self.fit_record['input'] = x
      self.fit_record['output'] = res
    return res

  def online_init(self):
    if self.b is None:
      num_input = self.num_in
    else:
      num_input = self.num_in + 1
    self.online_fit_by.register_target(feature_in=num_input, identifier=self.name)

  def online_fit(self,
                 target: ArrayType,
                 fit_record: Dict[str, ArrayType]):
    if not isinstance(target, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"target" must be a tensor, but got {type(target)}')
    x = fit_record['input']
    y = fit_record['output']
    if x.ndim != 2:
      raise ValueError(f'"ff" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {x.shape}')
    if target.ndim != 2:
      raise ValueError(f'"target" must be a 2D tensor with shape of (num_sample, '
                       f'num_feature), but we got {target.shape}')
    if x.shape[0] != target.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {x.shape[0]} != {target.shape[0]}.')
    if target.shape[1] != y.shape[1]:
      raise MathError(f'The output dimension of output and target data should be '
                      f'the same, while we got {target.shape[1]} != {y.shape[1]}')

    # data
    if self.b is not None:
      x = jnp.concatenate([jnp.ones((x.shape[0], 1)), x], axis=-1)

    # fitting
    dW = self.online_fit_by.call(target=target, input=x, output=y, identifier=self.name)

    # assign trained weights
    if self.b is None:
      self.W += dW
    else:
      db, dW = jnp.split(dW, [1])
      self.b += db[0]
      self.W += dW

  def offline_fit(self,
                  target: ArrayType,
                  fit_record: Dict[str, ArrayType]):
    """The offline training interface for the Dense node."""
    # data checking
    if not isinstance(target, (bm.ndarray, jnp.ndarray)):
      raise MathError(f'"targets" must be a tensor, but got {type(target)}')
    xs = fit_record['input']
    ys = fit_record['output']
    if xs.ndim != 3:
      raise ValueError(f'"ffs" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {xs.shape}')
    if target.ndim != 3:
      raise ValueError(f'"targets" must be a 3D tensor with shape of (num_sample, num_time, '
                       f'num_feature), but we got {target.shape}')
    if ys.shape != target.shape:
      raise ValueError(f'The shapes of output and target data should be '
                       f'the same, while we got {ys.shape} != {target.shape}.')
    if xs.shape[0] != target.shape[0]:
      raise ValueError(f'Batch size of the input and target data should be '
                       f'the same, while we got {xs.shape[0]} != {target.shape[0]}.')
    if xs.shape[1] != target.shape[1]:
      raise MathError(f'The time dimension of input and target data should be '
                      f'the same, while we got {xs.shape[1]} != {target.shape[1]}')

    # get input and target training data
    if self.b is not None:
      xs = jnp.concatenate([jnp.ones(xs.shape[:2] + (1,)), xs], axis=-1)  # (..., 1 + num_ff_input)

    # solve weights by offline training methods
    weights = self.offline_fit_by(target, xs, ys)

    # assign trained weights
    if self.b is None:
      self.W.value = weights
    else:
      bias, Wff = jnp.split(weights, [1])
      self.W.value = Wff
      self.b.value = bias[0]

  def stdp_update(
      self,
      on_pre: Dict = None,
      on_post: Dict = None,
      w_min: numbers.Number = None,
      w_max: numbers.Number = None
  ):
    if isinstance(self.W, float):
      raise ValueError(f'Cannot update the weight of a constant node.')
    if not isinstance(self.W, bm.Variable):
      self.tracing_variable('W', self.W, self.W.shape)
    if on_pre is not None:
      spike = on_pre['spike']
      trace = on_pre['trace']
      self.W.value = dense_on_pre(self.W.value, spike, trace, w_min, w_max)
    if on_post is not None:
      spike = on_post['spike']
      trace = on_post['trace']
      self.W.value = dense_on_post(self.W.value, spike, trace, w_min, w_max)


Linear = Dense


class Identity(Layer):
  r"""A placeholder identity operator that is argument-insensitive.
  """

  def __init__(self, *args, **kwargs) -> None:
    super(Identity, self).__init__(*args, **kwargs)

  def update(self, x):
    return x


if ti is not None:

  # @numba.njit(nogil=True, fastmath=True, parallel=False)
  # def _cpu_dense_on_post(weight, spike, trace, w_min, w_max, out_w):
  #   out_w[:] = weight
  #   for i in numba.prange(spike.shape[0]):
  #     if spike[i]:
  #       out_w[:, i] = np.clip(out_w[:, i] + trace, w_min, w_max)

  @ti.kernel
  def _dense_on_post(
      old_w: ti.types.ndarray(ndim=2),
      post_spike: ti.types.ndarray(ndim=1),
      pre_trace: ti.types.ndarray(ndim=1),
      w_min: ti.types.ndarray(ndim=1),
      w_max: ti.types.ndarray(ndim=1),
      out_w: ti.types.ndarray(ndim=2)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_pre, num_post = out_w.shape

    for i, j in ti.ndrange(num_pre, num_post):
      if post_spike[j]:
        new_value = out_w[i, j] + pre_trace[i]
        if new_value < w_min0:
          out_w[i, j] = w_min0
        elif new_value > w_max0:
          out_w[i, j] = w_max0
        else:
          out_w[i, j] = new_value
      else:
        out_w[i, j] = old_w[i, j]


  dense_on_post_prim = bm.XLACustomOp(cpu_kernel=_dense_on_post, gpu_kernel=_dense_on_post)


  # @numba.njit(nogil=True, fastmath=True, parallel=False)
  # def _cpu_dense_on_pre(weight, spike, trace, w_min, w_max, out_w):
  #   out_w[:] = weight
  #   for i in numba.prange(spike.shape[0]):
  #     if spike[i]:
  #       out_w[i] = np.clip(out_w[i] + trace, w_min, w_max)

  @ti.kernel
  def _dense_on_pre(
      old_w: ti.types.ndarray(ndim=2),
      pre_spike: ti.types.ndarray(ndim=1),
      post_trace: ti.types.ndarray(ndim=1),
      w_min: ti.types.ndarray(ndim=1),
      w_max: ti.types.ndarray(ndim=1),
      out_w: ti.types.ndarray(ndim=2)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_pre, num_post = out_w.shape

    for i, j in ti.ndrange(num_pre, num_post):
      if pre_spike[i]:
        new_value = out_w[i, j] + post_trace[j]
        if new_value < w_min0:
          out_w[i, j] = w_min0
        elif new_value > w_max0:
          out_w[i, j] = w_max0
        else:
          out_w[i, j] = new_value
      else:
        out_w[i, j] = old_w[i, j]


  dense_on_pre_prim = bm.XLACustomOp(cpu_kernel=_dense_on_pre, gpu_kernel=_dense_on_pre)

else:
  dense_on_pre_prim = None
  dense_on_post_prim = None


def dense_on_pre(weight, spike, trace, w_min, w_max):
  if dense_on_pre_prim is None:
    raise PackageMissingError.by_purpose('taichi', 'custom operators')

  if w_min is None:
    w_min = -np.inf
  if w_max is None:
    w_max = np.inf
  w_min = jnp.atleast_1d(w_min)
  w_max = jnp.atleast_1d(w_max)
  return dense_on_pre_prim(weight, spike, trace, w_min, w_max,
                           outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)])[0]


def dense_on_post(weight, spike, trace, w_min, w_max):
  if dense_on_post_prim is None:
    raise PackageMissingError.by_purpose('taichi', 'custom operators')

  if w_min is None:
    w_min = -np.inf
  if w_max is None:
    w_max = np.inf
  w_min = jnp.atleast_1d(w_min)
  w_max = jnp.atleast_1d(w_max)
  return dense_on_post_prim(weight, spike, trace, w_min, w_max,
                            outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)])[0]


class AllToAll(Layer, SupportSTDP):
  """Synaptic matrix multiplication with All2All connections.

  Args:
    num_pre: int. The number of neurons in the presynaptic neuron group.
    num_post: int. The number of neurons in the postsynaptic neuron group.
    weight: The synaptic weights.
    sharding: The sharding strategy. 
    include_self: bool. Whether connect the neuron with at the same position.
    mode: Mode. The computing mode.
    name: str. The object name.
  """

  def __init__(
      self,
      num_pre: int,
      num_post: int,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      include_self: bool = True,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(mode=mode, name=name)

    self.num_pre = num_pre
    self.num_post = num_post
    self.include_self = include_self
    self.sharding = sharding

    weight = init.parameter(weight, (self.num_pre, self.num_post), sharding=sharding)
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, pre_val):
    if bm.ndim(self.weight) == 0:  # weight is a scalar
      if isinstance(self.mode, bm.BatchingMode):
        assert pre_val.ndim == 2, 'Under the batching mode, the input should be a 2D array.'
        post_val = bm.sum(pre_val, keepdims=True, axis=1)
      else:
        assert pre_val.ndim == 1, 'Under the NonBatching mode, the input should be a 1D array.'
        post_val = bm.sum(pre_val)
      if not self.include_self:
        if self.num_pre == self.num_post:
          post_val = post_val - pre_val
        elif self.num_pre > self.num_post:
          val = pre_val[:self.num_post]
          post_val = post_val - val
        else:
          val = bm.concatenate([pre_val, bm.zeros(self.num_post - self.num_pre)])
          post_val = post_val - val
      post_val = self.weight * post_val

    else:  # weight is a matrix
      assert self.weight.ndim == 2, '"weight" must be a 2D matrix.'
      if not self.include_self:
        post_val = pre_val @ bm.fill_diagonal(self.weight, 0., inplace=False)
      else:
        post_val = pre_val @ self.weight
    return post_val

  def stdp_update(
      self,
      on_pre: Dict = None,
      on_post: Dict = None,
      w_min: numbers.Number = None,
      w_max: numbers.Number = None
  ):
    if isinstance(self.weight, float):
      raise ValueError(f'Cannot update the weight of a constant node.')
    if not isinstance(self.weight, bm.Variable):
      self.tracing_variable('weight', self.weight, self.weight.shape)
    if on_pre is not None:
      spike = on_pre['spike']
      trace = on_pre['trace']
      self.weight.value = dense_on_pre(self.weight.value, spike, trace, w_min, w_max)
    if on_post is not None:
      spike = on_post['spike']
      trace = on_post['trace']
      self.weight.value = dense_on_post(self.weight.value, spike, trace, w_min, w_max)


class OneToOne(Layer, SupportSTDP):
  """Synaptic matrix multiplication with One2One connection.

  Args:
    num: int. The number of neurons.
    weight: The synaptic weight.
    sharding: The sharding strategy. 
    mode: The computing mode.
    name: The object name.

  """

  def __init__(
      self,
      num: int,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(mode=mode, name=name)

    self.num = num
    self.sharding = sharding

    weight = init.parameter(weight, (self.num,), sharding=sharding)
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, pre_val):
    return pre_val * self.weight

  def stdp_update(
      self,
      on_pre: Dict = None,
      on_post: Dict = None,
      w_min: numbers.Number = None,
      w_max: numbers.Number = None
  ):
    if isinstance(self.weight, float):
      raise ValueError(f'Cannot update the weight of a constant node.')
    if not isinstance(self.weight, bm.Variable):
      self.tracing_variable('weight', self.weight, self.weight.shape)
    if on_pre is not None:
      spike = on_pre['spike']
      trace = on_pre['trace']
      self.weight.value += spike * trace
    if on_post is not None:
      spike = on_post['spike']
      trace = on_post['trace']
      self.weight.value += spike * trace


class MaskedLinear(Layer, SupportSTDP):
  r"""Synaptic matrix multiplication with masked dense computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a dense matrix.

  >>> import brainpy as bp
  >>> l = bp.dnn.MaskedLinear(bp.conn.FixedProb(0.1, pre=100, post=100),
  >>>                         weight=0.1)

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    mask_fun: Masking function.
    sharding: The sharding strategy. 
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      mask_fun: Callable = Identity(),
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    self.conn = conn
    self.sharding = sharding
    self.mask_fun = mask_fun

    # weight
    weight = init.parameter(weight, (conn.pre_num, conn.post_num), sharding=sharding)
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

    # connection
    self.mask = bm.sharding.partition(self.conn.require('conn_mat'), sharding=sharding)

  def update(self, x):
    return x @ self.mask_fun(self.weight * self.mask)

  def stdp_update(
      self,
      on_pre: Dict = None,
      on_post: Dict = None,
      w_min: numbers.Number = None,
      w_max: numbers.Number = None
  ):
    if isinstance(self.weight, float):
      raise ValueError(f'Cannot update the weight of a constant node.')
    if not isinstance(self.weight, bm.Variable):
      self.tracing_variable('weight', self.weight, self.weight.shape)
    if on_pre is not None:
      spike = on_pre['spike']
      trace = on_pre['trace']
      self.weight.value = dense_on_pre(self.weight.value, spike, trace, w_min, w_max)
    if on_post is not None:
      spike = on_post['spike']
      trace = on_post['trace']
      self.weight.value = dense_on_post(self.weight.value, spike, trace, w_min, w_max)


class _CSRLayer(Layer, SupportSTDP):
  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = True,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    assert sharding is None, 'Currently this model does not support sharding.'
    self.conn = conn
    self.sharding = sharding
    self.transpose = transpose

    # connection
    self.indices, self.indptr = self.conn.require('csr')

    # weight
    weight = init.parameter(weight, (self.indices.size,))
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def stdp_update(
      self,
      on_pre: Dict = None,
      on_post: Dict = None,
      w_min: numbers.Number = None,
      w_max: numbers.Number = None
  ):
    if bm.isscalar(self.weight):
      raise ValueError(f'When using STDP to update synaptic weights, the weight cannot be a scalar.')
    if self.weight.shape != self.indices.shape:
      raise ValueError(f'The shape of weight should be the same as the shape of sparse weight {self.weight.shape}.')
    if not isinstance(self.weight, bm.Variable):
      self.tracing_variable('weight', self.weight, self.weight.shape)
    if on_pre is not None:  # update on presynaptic spike
      spike = on_pre['spike']
      trace = on_pre['trace']
      self.weight.value = csr_on_pre_update(self.weight.value, self.indices, self.indptr, spike, trace, w_min, w_max)
    if on_post is not None:  # update on postsynaptic spike
      if not hasattr(self, '_pre_ids'):
        with jax.ensure_compile_time_eval():
          self._pre_ids, self._post_indptr, self.w_indices = csr2csc(
            [self.indices, self.indptr], self.conn.post_num, data=np.arange(self.weight.size)
          )
      spike = on_post['spike']
      trace = on_post['trace']
      self.weight.value = csc_on_post_update(self.weight.value, self._pre_ids, self._post_indptr,
                                             self.w_indices, spike, trace, w_min, w_max)


class CSRLinear(_CSRLayer):
  r"""Synaptic matrix multiplication with CSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a CSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy. 
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      method: str = None,
      transpose: bool = True,
  ):
    super().__init__(name=name, mode=mode, conn=conn, weight=weight, sharding=sharding, transpose=transpose)
    self.method = method

  def update(self, x):
    if x.ndim == 1:
      return bm.sparse.csrmv(self.weight, self.indices, self.indptr, x,
                             shape=(self.conn.pre_num, self.conn.post_num), transpose=self.transpose)
    elif x.ndim > 1:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_csrmv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_csrmv(self, x):
    return bm.sparse.csrmv(self.weight, self.indices, self.indptr, x,
                           shape=(self.conn.pre_num, self.conn.post_num), transpose=self.transpose)


class EventCSRLinear(_CSRLayer):
  r"""Synaptic matrix multiplication with event CSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
  :math:`M` the synaptic weight using a CSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = True,
  ):
    super().__init__(name=name, mode=mode, conn=conn, weight=weight, sharding=sharding, transpose=transpose)

  def update(self, x):
    if x.ndim == 1:
      return bm.event.csrmv(self.weight, self.indices, self.indptr, x,
                            shape=(self.conn.pre_num, self.conn.post_num),
                            transpose=self.transpose)
    elif x.ndim > 1:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_csrmv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_csrmv(self, x):
    return bm.event.csrmv(self.weight, self.indices, self.indptr, x,
                          shape=(self.conn.pre_num, self.conn.post_num),
                          transpose=self.transpose)


if ti is not None:
  @ti.kernel
  def _csr_on_pre_update(
      old_w: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      indices: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      indptr: ti.types.ndarray(ndim=1),  # vector with shape of (num_pre + 1)
      spike: ti.types.ndarray(ndim=1),  # vector with shape of (num_pre,)
      trace: ti.types.ndarray(ndim=1),  # vector with shape of (num_post,)
      w_min: ti.types.ndarray(ndim=1),  # scalar
      w_max: ti.types.ndarray(ndim=1),  # scalar
      out_w: ti.types.ndarray(ndim=1)  # vector with shape of (num_syn)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_pre = spike.shape[0]
    for i_pre in range(num_pre):
      if spike[i_pre]:
        for i_syn in range(indptr[i_pre], indptr[i_pre + 1]):
          out_w[i_syn] = min(max(old_w[i_syn] + trace[indices[i_syn]], w_min0), w_max0)
      else:
        for i_syn in range(indptr[i_pre], indptr[i_pre + 1]):
          out_w[i_syn] = old_w[i_syn]


  csr_on_pre_update_prim = bm.XLACustomOp(cpu_kernel=_csr_on_pre_update, gpu_kernel=_csr_on_pre_update)


  @ti.kernel
  def _coo_on_pre_update(
      old_w: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      pre_ids: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      post_ids: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      pre_spike: ti.types.ndarray(ndim=1),  # vector with shape of (num_pre,)
      post_trace: ti.types.ndarray(ndim=1),  # vector with shape of (num_post,)
      w_min: ti.types.ndarray(ndim=1),  # scalar
      w_max: ti.types.ndarray(ndim=1),  # scalar
      out_w: ti.types.ndarray(ndim=1)  # vector with shape of (num_syn)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_syn = old_w.shape[0]
    for i_syn in range(num_syn):
      if pre_spike[pre_ids[i_syn]]:  # pre spike
        out_w[i_syn] = min(max(old_w[i_syn] + post_trace[post_ids[i_syn]], w_min0), w_max0)
      else:
        out_w[i_syn] = old_w[i_syn]


  coo_on_pre_update_prim = bm.XLACustomOp(cpu_kernel=_coo_on_pre_update, gpu_kernel=_coo_on_pre_update)


  @ti.kernel
  def _coo_on_post_update(
      old_w: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      pre_ids: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      post_ids: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      post_spike: ti.types.ndarray(ndim=1),  # vector with shape of (num_pre,)
      pre_trace: ti.types.ndarray(ndim=1),  # vector with shape of (num_post,)
      w_min: ti.types.ndarray(ndim=1),  # scalar
      w_max: ti.types.ndarray(ndim=1),  # scalar
      out_w: ti.types.ndarray(ndim=1)  # vector with shape of (num_syn)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_syn = old_w.shape[0]
    for i_syn in range(num_syn):
      if post_spike[post_ids[i_syn]]:  # pre spike
        out_w[i_syn] = min(max(old_w[i_syn] + pre_trace[pre_ids[i_syn]], w_min0), w_max0)
      else:
        out_w[i_syn] = old_w[i_syn]


  coo_on_post_update_prim = bm.XLACustomOp(cpu_kernel=_coo_on_post_update, gpu_kernel=_coo_on_post_update)


  # @numba.njit(nogil=True, fastmath=True, parallel=False)
  # def _cpu_csc_on_pre_update(w, post_ids, indptr, w_ids, spike, trace, w_min, w_max, out_w):
  #   out_w[:] = w
  #   w_min = w_min[()]
  #   w_max = w_max[()]
  #   for i in numba.prange(spike.shape[0]):  # post id
  #     if spike[i]:
  #       for k in range(indptr[i], indptr[i + 1]):
  #         j = post_ids[k]  # pre id
  #         l = w_ids[k]  # syn id
  #         out_w[l] = np.minimum(np.maximum(out_w[l] + trace[j], w_min), w_max)

  @ti.kernel
  def _csc_on_post_update(
      old_w: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      indices: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      indptr: ti.types.ndarray(ndim=1),  # vector with shape of (num_post + 1)
      w_ids: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
      post_spike: ti.types.ndarray(ndim=1),  # vector with shape of (num_post,)
      pre_trace: ti.types.ndarray(ndim=1),  # vector with shape of (num_pre,)
      w_min: ti.types.ndarray(ndim=1),  # scalar
      w_max: ti.types.ndarray(ndim=1),  # scalar
      out_w: ti.types.ndarray(ndim=1),  # vector with shape of (num_syn)
  ):
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    num_post = post_spike.shape[0]
    for i_post in range(num_post):
      if post_spike[i_post]:
        for k in range(indptr[i_post], indptr[i_post + 1]):
          i_syn = w_ids[k]  # syn id
          out_w[i_syn] = min(max(old_w[i_syn] + pre_trace[indices[k]], w_min0), w_max0)
      else:
        for k in range(indptr[i_post], indptr[i_post + 1]):
          i_syn = w_ids[k]  # syn id
          out_w[i_syn] = old_w[i_syn]


  csc_on_post_update_prim = bm.XLACustomOp(cpu_kernel=_csc_on_post_update, gpu_kernel=_csc_on_post_update)


else:
  csr_on_pre_update_prim = None
  coo_on_pre_update_prim = None
  csc_on_post_update_prim = None


def csr_on_pre_update(w, indices, indptr, spike, trace, w_min=None, w_max=None):
  if csr_on_pre_update_prim is None:
    raise PackageMissingError.by_purpose('taichi', 'customized operators')

  if w_min is None:
    w_min = -np.inf
  if w_max is None:
    w_max = np.inf
  w_min = jnp.atleast_1d(w_min)
  w_max = jnp.atleast_1d(w_max)
  return csr_on_pre_update_prim(w, indices, indptr, spike, trace, w_min, w_max,
                                outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)])[0]


def coo_on_pre_update(w, pre_ids, post_ids, spike, trace, w_min=None, w_max=None):
  if coo_on_pre_update_prim is None:
    raise PackageMissingError.by_purpose('taichi', 'customized operators')

  if w_min is None:
    w_min = -np.inf
  if w_max is None:
    w_max = np.inf
  w_min = jnp.atleast_1d(w_min)
  w_max = jnp.atleast_1d(w_max)
  return coo_on_pre_update_prim(w, pre_ids, post_ids, spike, trace, w_min, w_max,
                                outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)])[0]


def csc_on_post_update(w, post_ids, indptr, w_ids, post_spike, pre_trace, w_min=None, w_max=None):
  if csc_on_post_update_prim is None:
    raise PackageMissingError.by_purpose('taichi', 'customized operators')

  if w_min is None:
    w_min = -np.inf
  if w_max is None:
    w_max = np.inf
  w_min = jnp.atleast_1d(w_min)
  w_max = jnp.atleast_1d(w_max)
  return csc_on_post_update_prim(w, post_ids, indptr, w_ids, post_spike, pre_trace, w_min, w_max,
                                 outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)])[0]


class CSCLinear(Layer):
  r"""Synaptic matrix multiplication with CSC sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a CSC sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    self.conn = conn
    self.sharding = sharding


class BcsrMM(Layer):
  r"""Synaptic matrix multiplication with BCSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a BCSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy. 
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    self.conn = conn
    self.sharding = sharding


class BcscMM(Layer):
  r"""Synaptic matrix multiplication with BCSC sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a BCSC sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    sharding: The sharding strategy. 
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    self.conn = conn
    self.sharding = sharding


class JitLinear(Layer):
  def get_conn_matrix(self):
    pass


class JitFPHomoLayer(JitLinear):
  def get_conn_matrix(self):
    return bm.jitconn.get_homo_weight_matrix(self.weight, self.prob, self.seed,
                                             shape=(self.num_out, self.num_in),
                                             transpose=self.transpose,
                                             outdim_parallel=not self.atomic)


class JitFPUniformLayer(JitLinear):
  def get_conn_matrix(self):
    return bm.jitconn.get_uniform_weight_matrix(self.w_low, self.w_high, self.prob, self.seed,
                                                shape=(self.num_out, self.num_in),
                                                transpose=self.transpose,
                                                outdim_parallel=not self.atomic)


class JitFPNormalLayer(JitLinear):
  def get_conn_matrix(self):
    return bm.jitconn.get_normal_weight_matrix(self.w_mu, self.w_sigma, self.prob, self.seed,
                                               shape=(self.num_out, self.num_in),
                                               transpose=self.transpose,
                                               outdim_parallel=not self.atomic)


class JitFPHomoLinear(JitFPHomoLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is the same :math:`weight`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    weight: float. The synaptic value at each position.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      weight: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = False,
      atomic: bool = False,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 100000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.mv_prob_homo(x, self.weight, self.prob, self.seed,
                                     shape=(self.num_out, self.num_in),
                                     transpose=self.transpose,
                                     outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.mv_prob_homo(x, self.weight, self.prob, self.seed,
                                   shape=(self.num_out, self.num_in),
                                   transpose=self.transpose,
                                   outdim_parallel=not self.atomic)


class JitFPUniformLinear(JitFPUniformLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is sample from a uniform distribution :math:`U(w_{low}, w_{high})`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    w_low: float. The lowest value of the uniform distribution.
    w_high: float. The highest value of the uniform distribution.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      w_low: float,
      w_high: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = False,
      atomic: bool = False,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 100000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    self.w_low = w_low
    self.w_high = w_high

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                        shape=(self.num_out, self.num_in),
                                        transpose=self.transpose,
                                        outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                      shape=(self.num_out, self.num_in),
                                      transpose=self.transpose,
                                      outdim_parallel=not self.atomic)


class JitFPNormalLinear(JitFPNormalLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic variable,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is sample from a normal distribution :math:`N(\mu, \sigma)`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    w_mu: float. The center of the normal distribution.
    w_sigma: float. The standard variance of the normal distribution.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      w_mu: float,
      w_sigma: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      transpose: bool = False,
      atomic: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 100000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    self.w_mu = w_mu
    self.w_sigma = w_sigma

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                       shape=(self.num_out, self.num_in),
                                       transpose=self.transpose,
                                       outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                     shape=(self.num_out, self.num_in),
                                     transpose=self.transpose,
                                     outdim_parallel=not self.atomic)


class EventJitFPHomoLinear(JitFPHomoLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is the same :math:`weight`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    weight: float. The synaptic value at each position.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      weight: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = False,
      atomic: bool = True,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 1000000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    self.weight = weight

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.event_mv_prob_homo(x, self.weight, self.prob, self.seed,
                                           shape=(self.num_out, self.num_in),
                                           transpose=self.transpose,
                                           outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.event_mv_prob_homo(x, self.weight, self.prob, self.seed,
                                         shape=(self.num_out, self.num_in),
                                         transpose=self.transpose,
                                         outdim_parallel=not self.atomic)


class EventJitFPUniformLinear(JitFPUniformLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is sample from a uniform distribution :math:`U(w_{low}, w_{high})`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    w_low: float. The lowest value of the uniform distribution.
    w_high: float. The highest value of the uniform distribution.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      w_low: float,
      w_high: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
      transpose: bool = False,
      atomic: bool = True,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 100000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    self.w_low = w_low
    self.w_high = w_high

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.event_mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                              shape=(self.num_out, self.num_in),
                                              transpose=self.transpose,
                                              outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.event_mv_prob_uniform(x, self.w_low, self.w_high, self.prob, self.seed,
                                            shape=(self.num_out, self.num_in),
                                            transpose=self.transpose,
                                            outdim_parallel=not self.atomic)


class EventJitFPNormalLinear(JitFPNormalLayer):
  r"""Synaptic matrix multiplication with the just-in-time connectivity.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic spikes,
  :math:`M` the synaptic weights which has the fixed sparse connectivity and weights.
  Particularly, the connectivity in :math:`M` is sampled from a fixed probability :math:`prob`,
  and at each connection, the synaptic value is sample from a normal distribution :math:`N(\mu, \sigma)`.

  Args:
    num_in: int. The number of the input feature. A positive integer.
    num_out: int. The number of the input feature. A positive integer.
    prob: float. The connectivity probability.
    w_mu: float. The center of the normal distribution.
    w_sigma: float. The standard variance of the normal distribution.
    seed: int. The random seed used to keep the reproducibility of the connectivity.
    transpose: bool. Transpose the JIT matrix or not. Default False.
    atomic: bool. Compute the post-synaptic value with the atomic summation. Default False.
       May be changed in the future.
    sharding: The sharding strategy.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      num_in: int,
      num_out: int,
      prob: float,
      w_mu: float,
      w_sigma: float,
      seed: Optional[int] = None,
      sharding: Optional[Sharding] = None,
      transpose: bool = False,
      atomic: bool = True,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    self.prob = prob
    self.sharding = sharding
    self.transpose = transpose
    self.seed = np.random.randint(0, 100000) if seed is None else seed
    self.atomic = atomic
    self.num_in = num_in
    self.num_out = num_out

    # weight
    self.w_mu = w_mu
    self.w_sigma = w_sigma

  def update(self, x):
    if x.ndim == 1:
      return bm.jitconn.event_mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                             shape=(self.num_out, self.num_in),
                                             transpose=self.transpose,
                                             outdim_parallel=not self.atomic)
    elif x.ndim == 2:
      return jax.vmap(self._batch_mv)(x)
    elif x.ndim > 2:
      shapes = x.shape[:-1]
      x = bm.flatten(x, end_dim=-2)
      y = jax.vmap(self._batch_mv)(x)
      return bm.reshape(y, shapes + (y.shape[-1],))
    else:
      raise ValueError

  def _batch_mv(self, x):
    return bm.jitconn.event_mv_prob_normal(x, self.w_mu, self.w_sigma, self.prob, self.seed,
                                           shape=(self.num_out, self.num_in),
                                           transpose=self.transpose,
                                           outdim_parallel=not self.atomic)
