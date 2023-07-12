# -*- coding: utf-8 -*-

from typing import Union, Tuple, Sequence, Optional, Callable, List, Any

import jax
import jax.numpy as jnp
import numpy as np

from brainpy import math as bm, check
from brainpy._src.dnn.base import Layer

__all__ = [
  'MaxPool',
  'MinPool',
  'AvgPool',
  'AvgPool1d',
  'AvgPool2d',
  'AvgPool3d',
  'MaxPool1d',
  'MaxPool2d',
  'MaxPool3d',
  'AdaptiveAvgPool1d',
  'AdaptiveAvgPool2d',
  'AdaptiveAvgPool3d',
  'AdaptiveMaxPool1d',
  'AdaptiveMaxPool2d',
  'AdaptiveMaxPool3d',
]


class Pool(Layer):
  """Pooling functions are implemented using the ReduceWindow XLA op.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``kernel_size`` or ``stride`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      init_value,
      computation,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]],
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: bm.Mode = None,
      name: Optional[str] = None,
  ):
    super(Pool, self).__init__(mode=mode, name=name)

    check.is_subclass(self.mode, [bm.NonBatchingMode, bm.TrainingMode])

    self.init_value = init_value
    self.computation = computation
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.channel_axis = channel_axis
    if isinstance(padding, str):
      if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")
    else:
      assert all([isinstance(x, (tuple, list)) for x in padding]), \
        f'padding should be sequence of Tuple[int, int]. {padding}'
      assert all([len(x) == 2 for x in padding]), f"each entry in padding {padding} must be length 2"

  def update(self, x):
    x = bm.as_jax(x)
    window_shape = self._infer_shape(x.ndim, self.kernel_size)
    stride = self._infer_shape(x.ndim, self.stride)
    padding = (self.padding
               if isinstance(self.padding, str) else
               self._infer_shape(x.ndim, self.padding, element=(0, 0), element_type=(tuple, list)))
    r = jax.lax.reduce_window(bm.as_jax(x),
                              init_value=self.init_value,
                              computation=self.computation,
                              window_dimensions=window_shape,
                              window_strides=stride,
                              padding=padding)
    return r

  def _infer_shape(self,
                   x_dim: int,
                   size: Union[Any, Sequence[Any]],
                   element: Any = 1,
                   element_type: Union[type, Sequence[type]] = int):
    """Infer shape for pooling window or stride."""

    # channel axis
    channel_axis = self.channel_axis
    if channel_axis and not 0 <= abs(channel_axis) < x_dim:
      raise ValueError(f"Invalid channel axis {channel_axis} for input with {x_dim} dimensions")
    if channel_axis and channel_axis < 0:
      channel_axis = x_dim + channel_axis

    if isinstance(size, (tuple, list)) and isinstance(size[0], element_type):
      size = tuple(size)
      if len(size) > x_dim:
        raise ValueError(f'Invalid size {size}. Its dimension is bigger than its input.')
      elif len(size) == x_dim:
        return size
      else:
        if isinstance(self.mode, bm.BatchingMode):
          size = (element,) + size
        if len(size) + 1 == x_dim:
          if channel_axis is None:
            raise ValueError('"channel_axis" should be provided.')
          size = size[:channel_axis] + (element,) + size[channel_axis:]
        else:
          raise ValueError(f'size {size} is invalid. Please provide more elements.')
        return size
    else:
      if isinstance(self.mode, bm.BatchingMode):
        return (element,) + tuple((size if d != channel_axis else element) for d in range(1, x_dim))
      else:
        return tuple((size if d != channel_axis else element) for d in range(0, x_dim))


class MaxPool(Pool):
  """Pools the input by taking the maximum over a window.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``kernel_size`` or ``stride`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: bm.Mode = None,
      name: Optional[str] = None,
  ):
    super(MaxPool, self).__init__(init_value=-jax.numpy.inf,
                                  computation=jax.lax.max,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)


class MinPool(Pool):
  """Pools the input by taking the minimum over a window.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``kernel_size`` or ``stride`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: bm.Mode = None,
      name: Optional[str] = None,
  ):
    super(MinPool, self).__init__(init_value=jax.numpy.inf,
                                  computation=jax.lax.min,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)


class AvgPool(Pool):
  """Pools the input by taking the average over a window.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``kernel_size`` or ``stride`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.
  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: bm.Mode = None,
      name: Optional[str] = None,
  ):
    super(AvgPool, self).__init__(init_value=0.,
                                  computation=jax.lax.add,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)

  def update(self, x):
    x = bm.as_jax(x)
    window_shape = self._infer_shape(x.ndim, self.kernel_size)
    strides = self._infer_shape(x.ndim, self.stride)
    padding = (self.padding if isinstance(self.padding, str) else
               self._infer_shape(x.ndim, self.padding, element=(0, 0), element_type=(tuple, list)))
    pooled = jax.lax.reduce_window(bm.as_jax(x),
                                   init_value=self.init_value,
                                   computation=self.computation,
                                   window_dimensions=window_shape,
                                   window_strides=strides,
                                   padding=padding)
    if padding == "VALID":
      # Avoid the extra reduce_window.
      return pooled / np.prod(window_shape)
    else:
      # Count the number of valid entries at each input point, then use that for
      # computing average. Assumes that any two arrays of same shape will be
      # padded the same.
      window_counts = jax.lax.reduce_window(jnp.ones_like(bm.as_jax(x)),
                                            init_value=self.init_value,
                                            computation=self.computation,
                                            window_dimensions=window_shape,
                                            window_strides=strides,
                                            padding=padding)
      assert pooled.shape == window_counts.shape
      return pooled / window_counts


class _MaxPoolNd(Layer):
  def __init__(
      self,
      init_value,
      computation,
      pool_dim: int,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = None,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(name=name, mode=mode)

    self.init_value = init_value
    self.computation = computation
    self.pool_dim = pool_dim

    # kernel_size
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,) * pool_dim
    elif isinstance(kernel_size, Sequence):
      check.is_sequence(kernel_size, elem_type=int)
      if len(kernel_size) != pool_dim:
        raise ValueError(f'kernel_size should a tuple with {pool_dim} ints, but got {len(kernel_size)}')
    else:
      raise TypeError(f'kernel_size should be a int or a tuple with {pool_dim} ints.')
    self.kernel_size = kernel_size

    # stride
    if stride is None:
      stride = kernel_size
    if isinstance(stride, int):
      stride = (stride,) * pool_dim
    elif isinstance(stride, Sequence):
      check.is_sequence(stride, elem_type=int)
      if len(stride) != pool_dim:
        raise ValueError(f'stride should a tuple with {pool_dim} ints, but got {len(kernel_size)}')
    else:
      raise TypeError(f'stride should be a int or a tuple with {pool_dim} ints.')
    self.stride = stride

    # padding
    if isinstance(padding, str):
      if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")
    elif isinstance(padding, int):
      padding = [(padding, padding) for _ in range(pool_dim)]
    elif isinstance(padding, (list, tuple)):
      if isinstance(padding[0], int):
        if len(padding) == pool_dim:
          padding = [(x, x) for x in padding]
        else:
          raise ValueError(f'If padding is a sequence of ints, it '
                           f'should has the length of {pool_dim}.')
      else:
        if not all([isinstance(x, (tuple, list)) for x in padding]):
          raise ValueError(f'padding should be sequence of Tuple[int, int]. {padding}')
        if not all([len(x) == 2 for x in padding]):
          raise ValueError(f"Each entry in padding must be tuple of 2 ints. {padding} ")
        if len(padding) == 1:
          padding = tuple(padding) * pool_dim
        assert len(padding) == pool_dim, f'padding should has the length of {pool_dim}. {padding}'
    else:
      raise ValueError
    self.padding = padding

    # channel_axis
    self.channel_axis = check.is_integer(channel_axis, allow_none=True)

  def update(self, x):
    x = bm.as_jax(x)
    x_dim = self.pool_dim + (0 if self.channel_axis is None else 1)
    if x.ndim < x_dim:
      raise ValueError(f'Excepted input with >= {x_dim} dimensions, but got {x.ndim}.')
    window_shape = self._infer_shape(x.ndim, self.kernel_size, 1)
    stride = self._infer_shape(x.ndim, self.stride, 1)
    padding = (self.padding
               if isinstance(self.padding, str) else
               self._infer_shape(x.ndim, self.padding, element=(0, 0)))
    r = jax.lax.reduce_window(bm.as_jax(x),
                              init_value=self.init_value,
                              computation=self.computation,
                              window_dimensions=window_shape,
                              window_strides=stride,
                              padding=padding)
    return r

  def _infer_shape(self, x_dim, inputs, element):
    channel_axis = self.channel_axis
    if channel_axis and not 0 <= abs(channel_axis) < x_dim:
      raise ValueError(f"Invalid channel axis {channel_axis} for input with {x_dim} dimensions")
    if channel_axis and channel_axis < 0:
      channel_axis = x_dim + channel_axis
    all_dims = list(range(x_dim))
    if channel_axis is not None:
      all_dims.pop(channel_axis)
    pool_dims = all_dims[-self.pool_dim:]
    results = [element] * x_dim
    for i, dim in enumerate(pool_dims):
      results[dim] = inputs[i]
    return results


class MaxPool1d(_MaxPoolNd):
  """Applies a 1D max pooling over an input signal composed of several input
    planes.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = None,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=-jax.numpy.inf,
                     computation=jax.lax.max,
                     pool_dim=1,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name,
                     mode=mode)


class MaxPool2d(_MaxPoolNd):
  """Applies a 1D max pooling over an input signal composed of several input
      planes.

    Parameters
    ----------
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    mode: Mode
      The computation mode.
    name: optional, str
      The object name.

    """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = None,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=-jax.numpy.inf,
                     computation=jax.lax.max,
                     pool_dim=2,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name, mode=mode)


class MaxPool3d(_MaxPoolNd):
  """Applies a 1D max pooling over an input signal composed of several input
      planes.

    Parameters
    ----------
    kernel_size: int, sequence of int
      An integer, or a sequence of integers defining the window to reduce over.
    stride: int, sequence of int
      An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
    padding: str, int, sequence of tuple
      Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of n `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    channel_axis: int, optional
      Axis of the spatial channels for which pooling is skipped.
      If ``None``, there is no channel axis.
    mode: Mode
      The computation mode.
    name: optional, str
      The object name.

    """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = None,
      padding: Union[str, int, Tuple[int], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=-jax.numpy.inf,
                     computation=jax.lax.max,
                     pool_dim=3,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name, mode=mode)


class _AvgPoolNd(_MaxPoolNd):
  def update(self, x):
    x = bm.as_jax(x)
    x_dim = self.pool_dim + (0 if self.channel_axis is None else 1)
    if x.ndim < x_dim:
      raise ValueError(f'Excepted input with >= {x_dim} dimensions, but got {x.ndim}.')
    dims = self._infer_shape(x.ndim, self.kernel_size, 1)
    stride = self._infer_shape(x.ndim, self.stride, 1)
    padding = (self.padding
               if isinstance(self.padding, str) else
               self._infer_shape(x.ndim, self.padding, element=(0, 0)))
    pooled = jax.lax.reduce_window(bm.as_jax(x),
                                   init_value=self.init_value,
                                   computation=self.computation,
                                   window_dimensions=dims,
                                   window_strides=stride,
                                   padding=padding)
    if padding == "VALID":
      # Avoid the extra reduce_window.
      return pooled / np.prod(dims)
    else:
      # Count the number of valid entries at each input point, then use that for
      # computing average. Assumes that any two arrays of same shape will be
      # padded the same.
      window_counts = jax.lax.reduce_window(jnp.ones_like(bm.as_jax(x)),
                                            init_value=self.init_value,
                                            computation=self.computation,
                                            window_dimensions=dims,
                                            window_strides=stride,
                                            padding=padding)
      assert pooled.shape == window_counts.shape
      return pooled / window_counts


class AvgPool1d(_AvgPoolNd):
  """Applies a 1D average pooling over an input signal composed of several input
    planes.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=0.,
                     computation=jax.lax.add,
                     pool_dim=1,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name,
                     mode=mode)


class AvgPool2d(_AvgPoolNd):
  """Applies a 2D average pooling over an input signal composed of several input
    planes.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.
  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=0.,
                     computation=jax.lax.add,
                     pool_dim=2,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name,
                     mode=mode)


class AvgPool3d(_AvgPoolNd):
  """Applies a 3D average pooling over an input signal composed of several input
    planes.

  Parameters
  ----------
  kernel_size: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  stride: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window stride (default: `(1, ..., 1)`).
  padding: str, int, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      kernel_size: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, int, Tuple[int, ...], Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = -1,
      mode: bm.Mode = None,
      name: Optional[str] = None
  ):
    super().__init__(init_value=0.,
                     computation=jax.lax.add,
                     pool_dim=3,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     channel_axis=channel_axis,
                     name=name,
                     mode=mode)


def _adaptive_pool1d(x, target_size: int, operation: Callable):
  """Adaptive pool 1D.

  Args:
    x: The input. Should be a JAX array of shape `(dim,)`.
    target_size: The shape of the output after the pooling operation `(target_size,)`.
    operation: The pooling operation to be performed on the input array.

  Returns:
    A JAX array of shape `(target_size, )`.
  """
  x = bm.as_jax(x)
  size = jnp.size(x)
  num_head_arrays = size % target_size
  num_block = size // target_size
  if num_head_arrays != 0:
    head_end_index = num_head_arrays * (num_block + 1)
    heads = jax.vmap(operation)(x[:head_end_index].reshape(num_head_arrays, -1))
    tails = jax.vmap(operation)(x[head_end_index:].reshape(-1, num_block))
    outs = jnp.concatenate([heads, tails])
  else:
    outs = jax.vmap(operation)(x.reshape(-1, num_block))
  return outs


def _generate_vmap(fun: Callable, map_axes: List[int]):
  map_axes = sorted(map_axes)
  for axis in map_axes:
    fun = jax.vmap(fun, in_axes=(axis, None, None), out_axes=axis)
  return fun


class AdaptivePool(Layer):
  """General N dimensional adaptive down-sampling to a target shape.

  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  num_spatial_dims: int
    The number of spatial dimensions.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  operation: Callable
    The down-sampling operation.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(
      self,
      target_shape: Union[int, Sequence[int]],
      num_spatial_dims: int,
      operation: Callable,
      channel_axis: Optional[int] = -1,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    self.channel_axis = channel_axis
    self.operation = operation
    if isinstance(target_shape, int):
      self.target_shape = (target_shape,) * num_spatial_dims
    elif isinstance(target_shape, Sequence) and (len(target_shape) == num_spatial_dims):
      self.target_shape = target_shape
    else:
      raise ValueError("`target_size` must either be an int or tuple of length "
                       f"{num_spatial_dims} containing ints.")

  def update(self, x):
    """Input-output mapping.

    Parameters
    ----------
    x: Array
      Inputs. Should be a JAX array of shape `(..., dim_1, dim_2, channels)`
      or `(..., dim_1, dim_2)`.
    """
    x = bm.as_jax(x)

    # channel axis
    channel_axis = self.channel_axis


    if channel_axis:
      if not 0 <= abs(channel_axis) < x.ndim:
        raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
      if channel_axis < 0:
        channel_axis = x.ndim + channel_axis
    # input dimension
    if (x.ndim - (0 if channel_axis is None else 1)) < len(self.target_shape):
      raise ValueError(f"Invalid input dimension. Except >={len(self.target_shape)} "
                       f"dimensions (channel_axis={self.channel_axis}). "
                       f"But got {x.ndim} dimensions.")
    # pooling dimensions
    pool_dims = list(range(x.ndim))
    if channel_axis:
      pool_dims.pop(channel_axis)

    # pooling
    for i, di in enumerate(pool_dims[-len(self.target_shape):]):
      poo_axes = [j for j in range(x.ndim) if j != di]
      op = _generate_vmap(_adaptive_pool1d, poo_axes)
      x = op(x, self.target_shape[i], self.operation)
    return x


class AdaptiveAvgPool1d(AdaptivePool):
  """Adaptive one-dimensional average down-sampling.

  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=1,
                     operation=jnp.mean,
                     name=name,
                     mode=mode)


class AdaptiveAvgPool2d(AdaptivePool):
  """Adaptive two-dimensional average down-sampling.


  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=2,
                     operation=jnp.mean,
                     name=name,
                     mode=mode)


class AdaptiveAvgPool3d(AdaptivePool):
  """Adaptive three-dimensional average down-sampling.


  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=3,
                     operation=jnp.mean,
                     name=name,
                     mode=mode)


class AdaptiveMaxPool1d(AdaptivePool):
  """Adaptive one-dimensional maximum down-sampling.

  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=1,
                     operation=jnp.max,
                     name=name,
                     mode=mode)


class AdaptiveMaxPool2d(AdaptivePool):
  """Adaptive two-dimensional maximum down-sampling.

  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=2,
                     operation=jnp.max,
                     name=name,
                     mode=mode)


class AdaptiveMaxPool3d(AdaptivePool):
  """Adaptive three-dimensional maximum down-sampling.

  Parameters
  ----------
  target_shape: int, sequence of int
    The target output shape.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped.
    If ``None``, there is no channel axis.
  name: str
    The class name.
  mode: Mode
    The computing mode.
  """

  def __init__(self,
               target_shape: Union[int, Sequence[int]],
               channel_axis: Optional[int] = -1,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(target_shape,
                     channel_axis=channel_axis,
                     num_spatial_dims=3,
                     operation=jnp.max,
                     name=name,
                     mode=mode)
