# -*- coding: utf-8 -*-

from typing import Union, Optional

import jax
import brainpy.math as bm
from brainpy import check
from brainpy.types import ArrayType
from .base import Encoder

__all__ = [
  'PoissonEncoder',
]


class PoissonEncoder(Encoder):
  r"""Encode the rate input as the Poisson spike train.

  Given the input :math:`x`, the poisson encoder will output
  spikes whose firing probability is :math:`x_{\text{normalize}}`, where
  :math:`x_{\text{normalize}}` is normalized into ``[0, 1]`` according
  to :math:`x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}`.

  Parameters
  ----------
  min_val: float
    The minimal value in the given data `x`, used to the data normalization.
  max_val: float
    The maximum value in the given data `x`, used to the data normalization.
  seed: int, ArrayType
    The seed or key for random generation.
  """

  def __init__(self,
               min_val: Optional[float] = None,
               max_val: Optional[float] = None):
    super().__init__()

    self.min_val = check.is_float(min_val, 'min_val', allow_none=True)
    self.max_val = check.is_float(max_val, 'max_val', allow_none=True)

  def __call__(self, x: ArrayType, num_step: int = None):
    """

    Parameters
    ----------
    x: ArrayType
      The rate input.
    num_step: int
      Encode rate values as spike trains in the given time length.

      - If ``time_len=None``, encode the rate values at the current time step.
        Users should repeatedly call it to encode `x` as a spike train.
      - Else, given the ``x`` with shape ``(S, ...)``, the encoded
        spike train is the array with shape ``(time_len, S, ...)``.

    Returns
    -------
    out: ArrayType
      The encoded spike train.
    """
    with jax.ensure_compile_time_eval():
      check.is_integer(num_step, 'time_len', min_bound=1, allow_none=True)
    if not (self.min_val is None or self.max_val is None):
      x = (x - self.min_val) / (self.max_val - self.min_val)
    shape = x.shape if (num_step is None) else ((num_step,) + x.shape)
    d = bm.as_jax(bm.random.rand(*shape)) < x
    return d.astype(x.dtype)
