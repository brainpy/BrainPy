# -*- coding: utf-8 -*-


"""
This module provides various methods to form current inputs.
You can access them through ``brainpy.inputs.XXX``.
"""

import numpy as np

from brainpy import math as bm

__all__ = [
  'section_input',
  'constant_input', 'constant_current',
  'spike_input', 'spike_current',
  'ramp_input', 'ramp_current',
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
  current_and_duration : tuple
      (The formatted current, total duration)
  """
  assert len(durations) == len(values), f'"values" and "durations" must be the same length, while ' \
                                        f'we got {len(values)} != {len(durations)}.'

  dt = bm.get_dt() if dt is None else dt

  # get input current shape, and duration
  I_duration = sum(durations)
  I_shape = ()
  for val in values:
    shape = bm.shape(val)
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = bm.zeros((int(np.ceil(I_duration / dt)),) + I_shape, dtype=bm.float_)
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
    shape = bm.shape(I[0])
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = bm.zeros((int(np.ceil(I_duration / dt)),) + I_shape, dtype=bm.float_)
  for c_size, duration in I_and_duration:
    length = int(duration / dt)
    I_current[start: start + length] = c_size
    start += length
  return I_current, I_duration


constant_current = constant_input


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

  current = bm.zeros(int(np.ceil(duration / dt)), dtype=bm.float_)
  for time, dur, size in zip(sp_times, sp_lens, sp_sizes):
    pp = int(time / dt)
    p_len = int(dur / dt)
    current[pp: pp + p_len] = size
  return current


spike_current = spike_input


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
  current_and_duration : tuple
      (The formatted current, total duration)
  """
  dt = bm.get_dt() if dt is None else dt
  t_end = duration if t_end is None else t_end

  current = bm.zeros(int(np.ceil(duration / dt)), dtype=bm.float_)
  p1 = int(np.ceil(t_start / dt))
  p2 = int(np.ceil(t_end / dt))
  current[p1: p2] = bm.array(bm.linspace(c_start, c_end, p2 - p1), dtype=bm.float_)
  return current


ramp_current = ramp_input

