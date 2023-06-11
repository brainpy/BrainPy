# -*- coding: utf-8 -*-


import warnings

import jax.numpy as jnp
import numpy as np

from brainpy import math as bm
from brainpy.check import is_float, is_integer

__all__ = [
  'section_input',
  'constant_input', 'constant_current',
  'spike_input', 'spike_current',
  'ramp_input', 'ramp_current',
  'wiener_process',
  'ou_process',
  'sinusoidal_input',
  'square_input',
]


def section_input(values, durations, dt=None, return_length=False):
  """Format an input current with different sections.

  For example:

  If you want to get an input where the size is 0 bwteen 0-100 ms,
  and the size is 1. between 100-200 ms.

  >>> section_input(values=[0, 1],
  >>>               durations=[100, 100])

  Parameters
  ----------
  values : list, np.ndarray
      The current values for each period duration.
  durations : list, np.ndarray
      The duration for each period.
  dt : float
      Default is None.
  return_length : bool
      Return the final duration length.

  Returns
  -------
  current_and_duration
  """
  if len(durations) != len(values):
    raise ValueError(f'"values" and "durations" must be the same length, while '
                     f'we got {len(values)} != {len(durations)}.')

  dt = bm.get_dt() if dt is None else dt

  # get input current shape, and duration
  I_duration = sum(durations)
  I_shape = ()
  for val in values:
    shape = jnp.shape(val)
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = bm.zeros((int(np.ceil(I_duration / dt)),) + I_shape)
  for c_size, duration in zip(values, durations):
    length = int(duration / dt)
    I_current[start: start + length] = c_size
    start += length

  if return_length:
    return I_current, I_duration
  else:
    return I_current


def constant_input(I_and_duration, dt=None):
  """Format constant input in durations.

  For example:

  If you want to get an input where the size is 0 bwteen 0-100 ms,
  and the size is 1. between 100-200 ms.

  >>> import brainpy.math as bm
  >>> constant_input([(0, 100), (1, 100)])
  >>> constant_input([(bm.zeros(100), 100), (bm.random.rand(100), 100)])

  Parameters
  ----------
  I_and_duration : list
      This parameter receives the current size and the current
      duration pairs, like `[(Isize1, duration1), (Isize2, duration2)]`.
  dt : float
      Default is None.

  Returns
  -------
  current_and_duration : tuple
      (The formatted current, total duration)
  """
  dt = bm.get_dt() if dt is None else dt

  # get input current dimension, shape, and duration
  I_duration = 0.
  I_shape = ()
  for I in I_and_duration:
    I_duration += I[1]
    shape = jnp.shape(I[0])
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = bm.zeros((int(np.ceil(I_duration / dt)),) + I_shape)
  for c_size, duration in I_and_duration:
    length = int(duration / dt)
    I_current[start: start + length] = c_size
    start += length
  return I_current, I_duration


def constant_current(*args, **kwargs):
  """Format constant input in durations.

  .. deprecated:: 2.1.13
     Use ``constant_current()`` instead.
  """
  warnings.warn('Please use "brainpy.inputs.constant_input()" instead. '
                '"brainpy.inputs.constant_current()" is deprecated since version 2.1.13.',
                DeprecationWarning)
  return constant_input(*args, **kwargs)


def spike_input(sp_times, sp_lens, sp_sizes, duration, dt=None):
  """Format current input like a series of short-time spikes.

  For example:

  If you want to generate a spike train at 10 ms, 20 ms, 30 ms, 200 ms, 300 ms,
  and each spike lasts 1 ms and the spike current is 0.5, then you can use the
  following funtions:

  >>> spike_input(sp_times=[10, 20, 30, 200, 300],
  >>>             sp_lens=1.,  # can be a list to specify the spike length at each point
  >>>             sp_sizes=0.5,  # can be a list to specify the current size at each point
  >>>             duration=400.)

  Parameters
  ----------
  sp_times : list, tuple
      The spike time-points. Must be an iterable object.
  sp_lens : int, float, list, tuple
      The length of each point-current, mimicking the spike durations.
  sp_sizes : int, float, list, tuple
      The current sizes.
  duration : int, float
      The total current duration.
  dt : float
      The default is None.

  Returns
  -------
  current : bm.ndarray
      The formatted input current.
  """
  dt = bm.get_dt() if dt is None else dt
  assert isinstance(sp_times, (list, tuple))
  if isinstance(sp_lens, (float, int)):
    sp_lens = [sp_lens] * len(sp_times)
  if isinstance(sp_sizes, (float, int)):
    sp_sizes = [sp_sizes] * len(sp_times)

  current = bm.zeros(int(np.ceil(duration / dt)))
  for time, dur, size in zip(sp_times, sp_lens, sp_sizes):
    pp = int(time / dt)
    p_len = int(dur / dt)
    current[pp: pp + p_len] = size
  return current


def spike_current(*args, **kwargs):
  """Format current input like a series of short-time spikes.

  .. deprecated:: 2.1.13
     Use ``spike_current()`` instead.
  """
  warnings.warn('Please use "brainpy.inputs.spike_input()" instead. '
                '"brainpy.inputs.spike_current()" is deprecated since version 2.1.13.',
                DeprecationWarning)
  return constant_input(*args, **kwargs)


def ramp_input(c_start, c_end, duration, t_start=0, t_end=None, dt=None):
  """Get the gradually changed input current.

  Parameters
  ----------
  c_start : float
      The minimum (or maximum) current size.
  c_end : float
      The maximum (or minimum) current size.
  duration : int, float
      The total duration.
  t_start : float
      The ramped current start time-point.
  t_end : float
      The ramped current end time-point. Default is the None.
  dt : float, int, optional
      The numerical precision.

  Returns
  -------
  current : bm.ndarray
    The formatted current
  """
  dt = bm.get_dt() if dt is None else dt
  t_end = duration if t_end is None else t_end

  current = bm.zeros(int(np.ceil(duration / dt)))
  p1 = int(np.ceil(t_start / dt))
  p2 = int(np.ceil(t_end / dt))
  cc = jnp.array(jnp.linspace(c_start, c_end, p2 - p1))
  current[p1: p2] = cc
  return current


def ramp_current(*args, **kwargs):
  """Get the gradually changed input current.

  .. deprecated:: 2.1.13
     Use ``ramp_input()`` instead.
  """
  warnings.warn('Please use "brainpy.inputs.ramp_input()" instead. '
                '"brainpy.inputs.ramp_current()" is deprecated since version 2.1.13.',
                DeprecationWarning)
  return constant_input(*args, **kwargs)


def wiener_process(duration, dt=None, n=1, t_start=0., t_end=None, seed=None):
  """Stimulus sampled from a Wiener process, i.e.
  drawn from standard normal distribution N(0, sqrt(dt)).

  Parameters
  ----------
  duration: float
    The input duration.
  dt: float
    The numerical precision.
  n: int
    The variable number.
  t_start: float
    The start time.
  t_end: float
    The end time.
  seed: int
    The noise seed.
  """
  dt = bm.get_dt() if dt is None else dt
  is_float(dt, 'dt', allow_none=False, min_bound=0.)
  is_integer(n, 'n', allow_none=False, min_bound=0)
  rng = bm.random.default_rng(seed, clone=False)
  t_end = duration if t_end is None else t_end
  i_start = int(t_start / dt)
  i_end = int(t_end / dt)
  noises = rng.standard_normal((i_end - i_start, n)) * jnp.sqrt(dt)
  currents = bm.zeros((int(duration / dt), n))
  currents[i_start: i_end] = noises
  return currents


def ou_process(mean, sigma, tau, duration, dt=None, n=1, t_start=0., t_end=None, seed=None):
  r"""Ornstein–Uhlenbeck input.

  .. math::

     dX = (mu - X)/\tau * dt + \sigma*dW

  Parameters
  ----------
  mean: float
    Drift of the OU process.
  sigma: float
    Standard deviation of the Wiener process, i.e. strength of the noise.
  tau: float
    Timescale of the OU process, in ms.
  duration: float
    The input duration.
  dt: float
    The numerical precision.
  n: int
    The variable number.
  t_start: float
    The start time.
  t_end: float
    The end time.
  seed: optional, int
    The random seed.
  """
  dt = bm.get_dt() if dt is None else dt
  dt_sqrt = jnp.sqrt(dt)
  is_float(dt, 'dt', allow_none=False, min_bound=0.)
  is_integer(n, 'n', allow_none=False, min_bound=0)
  rng = bm.random.default_rng(seed, clone=False)
  x = bm.Variable(jnp.ones(n) * mean)

  def _f(t):
    x.value = x + dt * ((mean - x) / tau) + sigma * dt_sqrt * rng.rand(n)
    return x.value

  noises = bm.for_loop(_f, jnp.arange(t_start, t_end, dt))

  t_end = duration if t_end is None else t_end
  i_start = int(t_start / dt)
  i_end = int(t_end / dt)
  currents = bm.zeros((int(duration / dt), n))
  currents[i_start: i_end] = noises
  return currents


def sinusoidal_input(amplitude, frequency, duration, dt=None, t_start=0., t_end=None, bias=False):
  """Sinusoidal input.

  Parameters
  ----------
  amplitude: float
    Amplitude of the sinusoid.
  frequency: float
    Frequency of the sinus oscillation, in Hz
  duration: float
    The input duration.
  t_start: float
    The start time.
  t_end: float
    The end time.
  dt: float
    The numerical precision.
  bias: bool
    Whether the sinusoid oscillates around 0 (False), or
    has a positive DC bias, thus non-negative (True).
  """
  dt = bm.get_dt() if dt is None else dt
  is_float(dt, 'dt', allow_none=False, min_bound=0.)
  if t_end is None:
    t_end = duration
  times = jnp.arange(0, t_end - t_start, dt)
  start_i = int(t_start / dt)
  end_i = int(t_end / dt)
  sin_inputs = amplitude * jnp.sin(2 * jnp.pi * times * (frequency / 1000.0))
  if bias: sin_inputs += amplitude
  currents = bm.zeros(int(duration / dt))
  currents[start_i:end_i] = sin_inputs
  return currents


def _square(t, duty=0.5):
  t, w = np.asarray(t), np.asarray(duty)
  w = np.asarray(w + (t - t))
  t = np.asarray(t + (w - w))
  if t.dtype.char in 'fFdD':
    ytype = t.dtype.char
  else:
    ytype = 'd'

  y = np.zeros(t.shape, ytype)

  # width must be between 0 and 1 inclusive
  mask1 = (w > 1) | (w < 0)
  np.place(y, mask1, np.nan)

  # on the interval 0 to duty*2*pi function is 1
  tmod = np.mod(t, 2 * np.pi)
  mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
  np.place(y, mask2, 1)

  # on the interval duty*2*pi to 2*pi function is
  #  (pi*(w+1)-tmod) / (pi*(1-w))
  mask3 = (1 - mask1) & (1 - mask2)
  np.place(y, mask3, -1)
  return y


def square_input(amplitude, frequency, duration, dt=None, bias=False, t_start=0., t_end=None):
  """Oscillatory square input.

  Parameters
  ----------
  amplitude: float
    Amplitude of the square oscillation.
  frequency: float
    Frequency of the square oscillation, in Hz.
  duration: float
    The input duration.
  t_start: float
    The start time.
  t_end: float
    The end time.
  dt: float
    The numerical precision.
  bias: bool
    Whether the sinusoid oscillates around 0 (False), or
    has a positive DC bias, thus non-negative (True).
  """
  dt = bm.get_dt() if dt is None else dt
  is_float(dt, 'dt', allow_none=False, min_bound=0.)
  if t_end is None: t_end = duration
  times = np.arange(0, t_end - t_start, dt)
  sin_inputs = amplitude * _square(2 * np.pi * times * (frequency / 1000.0))
  if bias: sin_inputs += amplitude
  currents = bm.zeros(int(duration / dt))
  start_i = int(t_start / dt)
  end_i = int(t_end / dt)
  currents[start_i:end_i] = bm.asarray(sin_inputs)
  return currents
