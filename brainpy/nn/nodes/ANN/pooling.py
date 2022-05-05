# -*- coding: utf-8 -*-


import jax.lax
import brainpy.math as bm
from brainpy.nn.base import Node

__all__ = [
  'Pool',
  'MaxPool',
  'AvgPool',
  'MinPool'
]


class Pool(Node):
  def __init__(self, init_v, reduce_fn, window_shape, strides, padding, **kwargs):
    """Pooling functions are implemented using the ReduceWindow XLA op.

     Args:
       init_v: scalar
          the initial value for the reduction
       reduce_fn: callable
          a reduce function of the form `(T, T) -> T`.
       window_shape: tuple
          a shape tuple defining the window to reduce over.
       strides: sequence[int]
          a sequence of `n` integers, representing the inter-window strides.
       padding: str, sequence[int]
          either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension.

      Returns:
          The output of the reduction for each window slice.
     """
    super(Pool, self).__init__(**kwargs)
    self.init_v = init_v
    self.reduce_fn = reduce_fn
    self.window_shape = window_shape
    self.strides = strides or (1,) * len(window_shape)
    assert len(self.window_shape) == len(self.strides), (
      f"len({self.window_shape}) must equal len({self.strides})")
    self.strides = (1,) + self.strides + (1,)
    self.dims = (1,) + window_shape + (1,)
    self.is_single_input = False

    if not isinstance(padding, str):
      padding = tuple(map(tuple, padding))
      assert len(padding) == len(window_shape), (
        f"padding {padding} must specify pads for same number of dims as "
        f"window_shape {window_shape}")
      assert all([len(x) == 2 for x in padding]), (
        f"each entry in padding {padding} must be length 2")
      padding = ((0, 0),) + padding + ((0, 0),)
    self.padding = padding

  def init_ff_conn(self):
    input_shapes = tuple((0,)) + tuple(d for d in self.feedforward_shapes if d is not None)
    assert len(input_shapes) == len(self.dims), f"len({len(input_shapes)}) != len({self.dims})"

    padding_vals = jax.lax.padtype_to_pads(input_shapes, self.dims, self.strides, self.padding)
    ones = (1,) * len(self.dims)
    out_shapes = jax.lax.reduce_window_shape_tuple(
      input_shapes, self.dims, self.strides, padding_vals, ones, ones)

    out_shapes = tuple((None,)) + tuple(d for i, d in enumerate(out_shapes) if i != 0)
    self.set_output_shape(out_shapes)

  def forward(self, ff, fb=None, **shared_kwargs):
    y = jax.lax.reduce_window(ff, self.init_v, self.reduce_fn, self.dims, self.strides, self.padding)

    return y


class AvgPool(Pool):
  """Pools the input by taking the average over a window.

  Args:
    window_shape: tuple
      a shape tuple defining the window to reduce over.
    strides: sequence[int]
      a sequence of `n` integers, representing the inter-window strides (default: `(1, ..., 1)`).
    padding: str, sequence[int]
      either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).

  Returns:
    The average for each window slice.
  """

  def __init__(self, window_shape, strides=None, padding="VALID"):
    super(AvgPool, self).__init__(
      init_v=0.,
      reduce_fn=jax.lax.add,
      window_shape=window_shape,
      strides=strides,
      padding=padding
    )

  def forward(self, ff, fb=None, **shared_kwargs):
    y = jax.lax.reduce_window(ff, self.init_v, self.reduce_fn, self.dims, self.strides, self.padding)
    y = y / bm.prod(bm.asarray(self.window_shape))
    return y


class MaxPool(Pool):
  """Pools the input by taking the maximum over a window.

    Args:
      window_shape: tuple
        a shape tuple defining the window to reduce over.
      strides: sequence[int]
        a sequence of `n` integers, representing the inter-window strides (default: `(1, ..., 1)`).
      padding: str, sequence[int]
        either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension (default: `'VALID'`).

    Returns:
      The maximum for each window slice.
  """
  def __init__(self, window_shape, strides=None, padding="VALID"):
    super(MaxPool, self).__init__(
      init_v=-bm.inf,
      reduce_fn=jax.lax.max,
      window_shape=window_shape,
      strides=strides,
      padding=padding
    )


class MinPool(Pool):
  """Pools the input by taking the minimum over a window.

      Args:
        window_shape: tuple
          a shape tuple defining the window to reduce over.
        strides: sequence[int]
          a sequence of `n` integers, representing the inter-window strides (default: `(1, ..., 1)`).
        padding: str, sequence[int]
          either the string `'SAME'`, the string `'VALID'`, or a sequence
          of `n` `(low, high)` integer pairs that give the padding to apply before
          and after each spatial dimension (default: `'VALID'`).

      Returns:
        The minimum for each window slice.
    """
  def __init__(self, window_shape, strides=None, padding="VALID"):
    super(MinPool, self).__init__(
      init_v=bm.inf,
      reduce_fn=jax.lax.min,
      window_shape=window_shape,
      strides=strides,
      padding=padding
    )