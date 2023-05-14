# -*- coding: utf-8 -*-

import math
from typing import Union, Callable

import jax

import brainpy.math as bm
from brainpy import check
from brainpy.types import ArrayType
from .base import Encoder

__all__ = [
  'LatencyEncoder',
  'WeightedPhaseEncoder',
]


class WeightedPhaseEncoder(Encoder):
  r"""Encode the rate input into the spike train according to [1]_.

  The main idea of the weighted spikes is assigning different weights
  to different phases (or to spikes in those phases) in order to pack
  more information into the spikes. This is the major difference from
  a conventional rate coding scheme that assigns the same weight to every spike [1]_.

  Parameters
  ----------
  min_val: float
    The minimal value in the given data `x`, used to the data normalization.
  max_val: float
    The maximum value in the given data `x`, used to the data normalization.
  num_phase: int
    The number of the encoding period.
  weight_fun: Callable
    The function to generate weight at the phase :math:`i`.

  References
  ----------
  .. [1] Kim, Jaehyun et al. “Deep neural networks with weighted spikes.” Neurocomputing 311 (2018): 373-386.
  """

  def __init__(self,
               min_val: float,
               max_val: float,
               num_phase: int,
               weight_fun: Callable = None):
    super().__init__()

    check.is_integer(num_phase, 'num_phase', min_bound=1)
    check.is_float(min_val, 'min_val')
    check.is_float(max_val, 'max_val')
    check.is_callable(weight_fun, 'weight_fun', allow_none=True)
    self.num_phase = num_phase
    self.min_val = min_val
    self.max_val = max_val
    self.weight_fun = (lambda i: 2 ** (-(i % num_phase + 1))) if weight_fun is None else weight_fun
    self.scale = (1 - self.weight_fun(self.num_phase - 1)) / (self.max_val - self.min_val)

  def __call__(self, x: ArrayType, num_step: int):
    """Encoding function.

    Parameters
    ----------
    x: ArrayType
      The input rate value.
    num_step: int
      The number of time steps.

    Returns
    -------
    out: ArrayType
      The encoded spike train.
    """
    # normalize all input signals to fit into the range [1, 1-2^K]
    x = (x - self.min_val) * self.scale

    # run
    inputs = bm.Variable(x)

    def f(i):
      w = self.weight_fun(i)
      spike = inputs >= w
      inputs.value -= w * spike
      return spike

    return bm.for_loop(f, bm.arange(num_step).value)


class LatencyEncoder(Encoder):
  r"""Encode the rate input as the spike train.

  The latency encoder will encode ``x`` (normalized into ``[0, 1]`` according to
  :math:`x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}`)
  to spikes whose firing time is :math:`0 \le t_f \le \text{num_period}-1`.
  A larger ``x`` will cause the earlier firing time.

  Parameters
  ----------
  min_val: float
    The minimal value in the given data `x`, used to the data normalization.
  max_val: float
    The maximum value in the given data `x`, used to the data normalization.
  num_period: int
    The periodic firing time step.
  method: str
    How to convert intensity to firing time. Currently, we support `linear` or `log`.

    - If ``method='linear'``, the firing rate is calculated as
      :math:`t_f(x) = (\text{num_period} - 1)(1 - x)`.
    - If ``method='log'``, the firing rate is calculated as
      :math:`t_f(x) = (\text{num_period} - 1) - ln(\alpha * x + 1)`,
      where :math:`\alpha` satisfies :math:`t_f(1) = \text{num_period} - 1`.
  """

  def __init__(self,
               min_val: float,
               max_val: float,
               num_period: int,
               method: str = 'linear'):
    super().__init__()

    check.is_integer(num_period, 'num_period', min_bound=1)
    check.is_float(min_val, 'min_val')
    check.is_float(max_val, 'max_val')
    assert method in ['linear', 'log']
    self.num_period = num_period
    self.min_val = min_val
    self.max_val = max_val
    self.method = method

  def __call__(self, x: ArrayType, i_step: Union[int, ArrayType]):
    """Encoding function.

    Parameters
    ----------
    x: ArrayType
      The input rate value.
    i_step: int, ArrayType
      The indices of the time step.

    Returns
    -------
    out: ArrayType
      The encoded spike train.
    """
    _temp = self.num_period - 1.
    if self.method == 'log':
      alpha = math.exp(_temp) - 1.
      t_f = bm.round(_temp - bm.log(alpha * x + 1.)).astype(bm.int_)
    else:
      t_f = bm.round(_temp * (1. - x)).astype(bm.int_)

    def f(i):
      return bm.as_jax(t_f == (i % self.num_period), dtype=x.dtype)

    if isinstance(i_step, int):
      return f(i_step)
    else:
      assert isinstance(i_step, (jax.Array, bm.Array))
      return jax.vmap(f, i_step)
