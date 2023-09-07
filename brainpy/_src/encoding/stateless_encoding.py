# -*- coding: utf-8 -*-

from typing import Optional

import brainpy.math as bm
from brainpy import check
from .base import Encoder

__all__ = [
  'PoissonEncoder',
  'DiffEncoder',
]


class PoissonEncoder(Encoder):
  r"""Encode the rate input as the Poisson spike train.

  Expected inputs should be between 0 and 1. If not, the input :math:`x` will be
  normalized to :math:`x_{\text{normalize}}` within ``[0, 1]`` according
  to :math:`x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}`.

  Given the input :math:`x`, the poisson encoder will output
  spikes whose firing probability is :math:`x_{\text{normalize}}`.


  Examples::

     import brainpy as bp
     import brainpy.math as bm

     img = bm.random.random((10, 2))  # image to encode (normalized to [0., 1.])
     encoder = bp.encoding.PoissonEncoder()  # the encoder

     # encode the image at each time
     for run_index in range(100):
       spike = encoder.single_step(img)
       # do something

     # or, encode the image at multiple times once
     spikes = encoder.multi_steps(img, n_time=10.)


  Args:
    min_val: float. The minimal value in the given data `x`, used to the data normalization.
    max_val: float. The maximum value in the given data `x`, used to the data normalization.
    gain: float. Scale input features by the gain, defaults to ``1``.
    offset: float. Shift input features by the offset, defaults to ``0``.
    first_spk_time: float. The time to first spike, defaults to ``0``.
  """

  def __init__(
      self,
      min_val: Optional[float] = None,
      max_val: Optional[float] = None,
      gain: float = 1.0,
      offset: float = 0.0,
      first_spk_time: float = 0.,
  ):
    super().__init__()

    self.min_val = check.is_float(min_val, 'min_val', allow_none=True)
    self.max_val = check.is_float(max_val, 'max_val', allow_none=True)
    self.gain = check.is_float(gain, allow_none=False)
    self.offset = check.is_float(offset, allow_none=False)
    self.first_spk_time = check.is_float(first_spk_time)
    self.first_spk_step = int(self.first_spk_time / bm.get_dt())

  def single_step(self, x, i_step: int = None):
    """Generate spikes at the single step according to the inputs.

    Args:
      x: Array. The rate input.
      i_step: int. The time step to generate spikes.

    Returns:
      out: Array. The encoded spike train.
    """
    if i_step is None:
      return self.multi_steps(x, n_time=None)
    else:
      return bm.cond(bm.as_jax(i_step < self.first_spk_step), self._zero_out, self.multi_steps, x)

  def multi_steps(self, x, n_time: Optional[float]):
    """Generate spikes at multiple steps according to the inputs.

    Args:
      x: Array. The rate input.
      n_time: float. Encode rate values as spike trains in the given time length.
        ``n_time`` is converted into the ``n_step`` according to `n_step = int(n_time / brainpy.math.dt)`.
        - If ``n_time=None``, encode the rate values at the current time step.
          Users should repeatedly call it to encode `x` as a spike train.
        - Else, given the ``x`` with shape ``(S, ...)``, the encoded
          spike train is the array with shape ``(n_step, S, ...)``.

    Returns:
      out: Array. The encoded spike train.
    """
    n_time = int(n_time / bm.get_dt())

    if (self.min_val is not None) and (self.max_val is not None):
      x = (x - self.min_val) / (self.max_val - self.min_val)
    x = x * self.gain + self.offset
    if n_time is not None and self.first_spk_step > 0:
      pre = bm.zeros((self.first_spk_step,) + x.shape, dtype=x.dtype)
      shape = ((n_time - self.first_spk_step,) + x.shape)
      post = bm.asarray(bm.random.rand(*shape) < x, dtype=x.dtype)
      return bm.cat([pre, post], axis=0)
    else:
      shape = x.shape if (n_time is None) else ((n_time - self.first_spk_step,) + x.shape)
      return bm.asarray(bm.random.rand(*shape) < x, dtype=x.dtype)

  def _zero_out(self, x):
    return bm.zeros_like(x)


class DiffEncoder(Encoder):
  """Generate spike only when the difference between two subsequent
  time steps meets a threshold.

  Optionally include `off_spikes` for negative changes.

  Example::

    >>> a = bm.array([1, 2, 2.9, 3, 3.9])
    >>> encoder = DiffEncoder(threshold=1)
    >>> encoder.multi_steps(a)
    Array([1., 0., 0., 0.])

    >>> encoder = DiffEncoder(threshold=1, padding=True)
    >>> encoder.multi_steps(a)
    Array([0., 1., 0., 0., 0.])

    >>> b = bm.array([1, 2, 0, 2, 2.9])
    >>> encoder = DiffEncoder(threshold=1, off_spike=True)
    >>> encoder.multi_steps(b)
    Array([ 1.,  1., -1.,  1.,  0.])

    >>> encoder = DiffEncoder(threshold=1, padding=True, off_spike=True)
    >>> encoder.multi_steps(b)
    Array([ 0.,  1., -1.,  1.,  0.])

  Args:
    threshold: float. Input features with a change greater than the thresold
      across one timestep will generate a spike, defaults to ``0.1``.
    padding: bool. Used to change how the first time step of spikes are
      measured. If ``True``, the first time step will be repeated with itself
      resulting in ``0``'s for the output spikes.
      If ``False``, the first time step will be padded with ``0``'s, defaults
      to ``False``.
    off_spike: bool. If ``True``, negative spikes for changes less than
      ``-threshold``, defaults to ``False``.
  """

  def __init__(
      self,
      threshold: float = 0.1,
      padding: bool = False,
      off_spike: bool = False,
  ):
    super().__init__()

    self.threshold = threshold
    self.padding = padding
    self.off_spike = off_spike

  def single_step(self, *args, **kwargs):
    raise NotImplementedError(f'{DiffEncoder.__class__.__name__} does not support single-step encoding.')

  def multi_steps(self, x):
    """Encoding multistep inputs with the spiking trains.

    Args:
      x: Array. The array with the shape of `(num_step, ....)`.

    Returns:
      out: Array. The spike train.
    """
    if self.padding:
      diff = bm.diff(x, axis=0, prepend=x[:1])
    else:
      diff = bm.diff(x, axis=0, prepend=bm.zeros((1,) + x.shape[1:], dtype=x.dtype))

    if self.off_spike:
      on_spk = bm.asarray(diff >= self.threshold, dtype=x.dtype)
      off_spk = -bm.asarray(diff <= -self.threshold, dtype=x.dtype)
      return on_spk + off_spk

    else:
      return bm.asarray(diff >= self.threshold, dtype=x.dtype)
