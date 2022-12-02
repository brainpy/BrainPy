# -*- coding: utf-8 -*-

from typing import Union, Tuple, Sequence, Optional, Any, TypeVar

import numpy as np
from jax import lax

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.modes import Mode, training, BatchingMode
from brainpy.types import Array

__all__ = [
  'MaxPool',
  'AvgPool',
  'MinPool'
]

T = TypeVar('T')


def _infer_shape(x: Array,
                 mode: Mode,
                 size: Union[T, Sequence[T]],
                 channel_axis: Optional[int] = None,
                 element: T = 1):
  """Infer shape for pooling window or strides."""

  # channel axis
  if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
    raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
  if channel_axis and channel_axis < 0:
    channel_axis = x.ndim + channel_axis

  if isinstance(size, (tuple, list)):
    assert isinstance(size, (tuple, list)), "Should be a tuple/list of integer."
    size = tuple(size)
    if len(size) > x.ndim:
      raise ValueError(f'Invalid size {size}. Its dimension is bigger than its input.')
    elif len(size) == x.ndim:
      return size
    else:
      if isinstance(mode, BatchingMode):
        size = (element,) + size
      if len(size) + 1 == x.ndim:
        if channel_axis is None:
          raise ValueError('"channel_axis" should be provided.')
        size = size[:channel_axis] + (element,) + size[channel_axis:]
      else:
        raise ValueError(f'size {size} is invalid. Please provide more elements.')
      return size

  else:
    if isinstance(mode, BatchingMode):
      return (element,) + tuple((size if d != channel_axis else element) for d in range(1, x.ndim))
    else:
      return tuple((size if d != channel_axis else element) for d in range(0, x.ndim))


class Pool(DynamicalSystem):
  """Pooling functions are implemented using the ReduceWindow XLA op.

  Parameters
  ----------
  window_shape: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  strides: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window strides (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``window_shape`` or ``strides`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      init_value,
      computation,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: Mode = training,
      name: Optional[str] = None,
  ):
    super(Pool, self).__init__(mode=mode, name=name)

    self.init_value = init_value
    self.computation = computation
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis
    if isinstance(padding, str):
      if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")
    else:
      assert all([isinstance(x, (tuple, list)) for x in padding]), \
        f'padding should be sequence of Tuple[int, int]. {padding}'
      assert all([len(x) == 2 for x in padding]), f"each entry in padding {padding} must be length 2"

  def update(self, sha, x):
    window_shape = _infer_shape(x, self.mode, self.window_shape, self.channel_axis)
    strides = _infer_shape(x, self.mode, self.strides, self.channel_axis)
    padding = (self.padding if isinstance(self.padding, str) else
               _infer_shape(x, self.mode, self.padding, self.channel_axis, element=(0, 0)))
    return lax.reduce_window(bm.as_jax(x),
                             init_value=self.init_value,
                             computation=self.computation,
                             window_dimensions=window_shape,
                             window_strides=strides,
                             padding=padding)

  def reset_state(self, batch_size=None):
    pass


class MaxPool(Pool):
  """Pools the input by taking the maximum over a window.

  Parameters
  ----------
  window_shape: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  strides: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window strides (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``window_shape`` or ``strides`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: Mode = training,
      name: Optional[str] = None,
  ):
    super(MaxPool, self).__init__(init_value=-bm.inf,
                                  computation=lax.max,
                                  window_shape=window_shape,
                                  strides=strides,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)


class MinPool(Pool):
  """Pools the input by taking the minimum over a window.

  Parameters
  ----------
  window_shape: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  strides: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window strides (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``window_shape`` or ``strides`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: Mode = training,
      name: Optional[str] = None,
  ):
    super(MinPool, self).__init__(init_value=bm.inf,
                                  computation=lax.min,
                                  window_shape=window_shape,
                                  strides=strides,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)


class AvgPool(Pool):
  """Pools the input by taking the average over a window.


  Parameters
  ----------
  window_shape: int, sequence of int
    An integer, or a sequence of integers defining the window to reduce over.
  strides: int, sequence of int
    An integer, or a sequence of integers, representing the inter-window strides (default: `(1, ..., 1)`).
  padding: str, sequence of tuple
    Either the string `'SAME'`, the string `'VALID'`, or a sequence
    of n `(low, high)` integer pairs that give the padding to apply before
    and after each spatial dimension.
  channel_axis: int, optional
    Axis of the spatial channels for which pooling is skipped,
    used to infer ``window_shape`` or ``strides`` if they are an integer.
  mode: Mode
    The computation mode.
  name: optional, str
    The object name.

  """

  def __init__(
      self,
      window_shape: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]],
      padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
      channel_axis: Optional[int] = None,
      mode: Mode = training,
      name: Optional[str] = None,
  ):
    super(AvgPool, self).__init__(init_value=0.,
                                  computation=lax.add,
                                  window_shape=window_shape,
                                  strides=strides,
                                  padding=padding,
                                  channel_axis=channel_axis,
                                  mode=mode,
                                  name=name)

  def update(self, sha, x):
    window_shape = _infer_shape(x, self.mode, self.window_shape, self.channel_axis)
    strides = _infer_shape(x, self.mode, self.strides, self.channel_axis)
    padding = (self.padding if isinstance(self.padding, str) else
               _infer_shape(x, self.mode, self.padding, self.channel_axis, element=(0, 0)))
    pooled = lax.reduce_window(bm.as_jax(x),
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
      window_counts = lax.reduce_window(bm.ones_like(x).value,
                                        init_value=self.init_value,
                                        computation=self.computation,
                                        window_dimensions=window_shape,
                                        window_strides=strides,
                                        padding=padding)
      assert pooled.shape == window_counts.shape
      return pooled / window_counts
