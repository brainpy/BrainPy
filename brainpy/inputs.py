# -*- coding: utf-8 -*-

import numpy as np

from brainpy import profile

__all__ = [
    'constant_current',
    'spike_current',
    'ramp_current',
]


def constant_current(Iext, dt=None):
    """Format constant input in durations.

    For example:

    If you want to get an input where the size is 0 bwteen 0-100 ms,
    and the size is 1. between 100-200 ms.
    >>> constant_current([(0, 100), (1, 100)])

    Parameters
    ----------
    Iext : list
        This parameter receives the current size and the current
        duration pairs, like `[(size1, duration1), (size2, duration2)]`.
    dt : float
        Default is None.

    Returns
    -------
    current_and_duration : tuple
        (The formatted current, total duration)
    """
    dt = profile.get_dt() if dt is None else dt

    total_duration = sum([a[1] for a in Iext])
    current = np.zeros(int(np.ceil(total_duration / dt)))
    start = 0
    for c_size, duration in Iext:
        length = int(duration / dt)
        current[start: start + length] = c_size
        start += length
    return current, total_duration


def spike_current(points, lengths, sizes, duration, dt=None):
    """Format current input like a series of short-time spikes.

    Parameters
    ----------
    points : a_list, tuple
        The spike time-points. Must be an iterable object.
    lengths : int, float, a_list, tuple
        The length of each point-current, mimicking the spike durations.
    sizes : int, float, a_list, tuple
        The current sizes.
    duration : int, float
        The total current duration.
    dt : float
        The default is None.

    Returns
    -------
    current_and_duration : tuple
        (The formatted current, total duration)
    """
    dt = profile.get_dt() if dt is None else dt
    assert isinstance(points, (list, tuple))
    if isinstance(lengths, (float, int)):
        lengths = [lengths] * len(points)
    if isinstance(sizes, (float, int)):
        sizes = [sizes] * len(points)

    current = np.zeros(int(np.ceil(duration / dt)))
    for time, dur, size in zip(points, lengths, sizes):
        pp = int(time / dt)
        p_len = int(dur / dt)
        current[pp: pp + p_len] = size
    return current


def ramp_current(c_start, c_end, duration, t_start=0, t_end=None, dt=None):
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
    dt

    Returns
    -------
    current_and_duration : tuple
        (The formatted current, total duration)
    """
    dt = profile.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end

    current = np.zeros(int(np.ceil(duration / dt)))
    p1 = int(np.ceil(t_start / dt))
    p2 = int(np.ceil(t_end / dt))
    current[p1: p2] = np.linspace(c_start, c_end, p2 - p1)
    return current

