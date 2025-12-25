# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import warnings

import brainstate
import braintools

import brainpy.math

__all__ = [
    'section_input',
    'constant_input',
    'constant_current',
    'spike_input',
    'spike_current',
    'ramp_input',
    'ramp_current',
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

    >>> section_input(values=[0, 1], durations=[100, 100])

    Parameters::

    values : list, np.ndarray
        The current values for each period duration.
    durations : list, np.ndarray
        The duration for each period.
    dt : float
        Default is None.
    return_length : bool
        Return the final duration length.

    Returns::

    current_and_duration
    """
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.section(values, durations, return_length=return_length)


def constant_input(I_and_duration, dt=None):
    """Format constant input in durations.

    For example:

    If you want to get an input where the size is 0 bwteen 0-100 ms,
    and the size is 1. between 100-200 ms.

    >>> import brainpy.math as bm
    >>> constant_input([(0, 100), (1, 100)])
    >>> constant_input([(bm.zeros(100), 100), (bm.random.rand(100), 100)])

    Parameters::

    I_and_duration : list
        This parameter receives the current size and the current
        duration pairs, like `[(Isize1, duration1), (Isize2, duration2)]`.
    dt : float
        Default is None.

    Returns::

    current_and_duration : tuple
        (The formatted current, total duration)
    """
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.constant(I_and_duration)


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

    Parameters::

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

    Returns::

    current : bm.ndarray
        The formatted input current.
    """
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.spike(sp_times, sp_lens, sp_sizes, duration)


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

    Parameters::

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

    Returns::

    current : bm.ndarray
      The formatted current
    """
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.ramp(c_start, c_end, duration, t_start, t_end)


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

    Parameters::

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
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.wiener_process(duration, sigma=1.0, n=n, t_start=t_start, t_end=t_end, seed=seed)


def ou_process(mean, sigma, tau, duration, dt=None, n=1, t_start=0., t_end=None, seed=None):
    r"""Ornstein–Uhlenbeck input.

    .. math::

       dX = (mu - X)/\tau * dt + \sigma*dW

    Parameters::

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
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.ou_process(mean, sigma, tau, duration, n=n, t_start=t_start, t_end=t_end, seed=seed)


def sinusoidal_input(amplitude, frequency, duration, dt=None, t_start=0., t_end=None, bias=False):
    """Sinusoidal input.

    Parameters::

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
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.sinusoidal(amplitude, frequency, duration, t_start=t_start, t_end=t_end, bias=bias)


def square_input(amplitude, frequency, duration, dt=None, bias=False, t_start=0., t_end=None):
    """Oscillatory square input.

    Parameters::

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
    with brainstate.environ.context(dt=brainpy.math.get_dt() if dt is None else dt):
        return braintools.input.square(amplitude, frequency, duration, t_start=t_start, t_end=t_end, duty_cycle=0.5,
                                       bias=bias)
