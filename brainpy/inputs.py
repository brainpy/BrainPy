# -*- coding: utf-8 -*-

from . import numpy as np
from . import profile
from .core_system import NeuGroup
from .core_system import NeuType
from .core_system.types import Array
from .core_system.types import NeuState

__all__ = [
    'constant_current',
    'spike_current',
    'ramp_current',
    'PoissonInput',
    'SpikeTimeInput',
    'FreqInput',
]


def constant_current(Iext, dt=None):
    """Format constant input in durations.

    For example:

    If you want to get an input where the size is 0 bwteen 0-100 ms,
    and the size is 1. between 100-200 ms.
    >>> constant_current([(0, 100), (1, 100)])
    >>> constant_current([(np.zeros(100), 100), (np.random.rand(100), 100)])

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

    # get input current dimension, shape, and duration
    I_duration = 0.
    I_dim = 0
    I_shape = ()
    for I in Iext:
        I_duration += I[1]
        dim = np.ndim(I[0])
        if dim > I_dim:
            I_dim = dim
            I_shape = np.shape(I[0])

    # get the current
    I_current = np.zeros((int(np.ceil(I_duration / dt)),) + I_shape)
    start = 0
    for c_size, duration in Iext:
        length = int(duration / dt)
        I_current[start: start + length] = c_size
        start += length
    return I_current, I_duration


def spike_current(points, lengths, sizes, duration, dt=None):
    """Format current input like a series of short-time spikes.

    For example:

    If you want to generate a spike train at 10 ms, 20 ms, 30 ms, 200 ms, 300 ms,
    and each spike lasts 1 ms and the spike current is 0.5, then you can use the
    following funtions:

    >>> spike_current(points=[10, 20, 30, 200, 300],
    >>>               lengths=1.,  # can be a list to specify the spike length at each point
    >>>               sizes=0.5,  # can be a list to specify the current size at each point
    >>>               duration=400.)

    Parameters
    ----------
    points : list, tuple
        The spike time-points. Must be an iterable object.
    lengths : int, float, list, tuple
        The length of each point-current, mimicking the spike durations.
    sizes : int, float, list, tuple
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


class PoissonInput(NeuGroup):
    """The Poisson input neuron group.

    Note: The ``PoissonGroup`` does not work for high-frequency rates. This is because
    more than one spike might fall into a single time step (``dt``).
    However, you can split high frequency rates into several neurons with lower frequency rates.
    For example, use ``PoissonGroup(10, 100)`` instead of ``PoissonGroup(1, 1000)``.

    Parameters
    ----------
    geometry : int, tuple, list
        The neuron group geometry.
    freqs : float, int, np.ndarray
        The spike rates.
    monitors : list, tuple
        The targets for monitoring.
    name : str
        The neuron group name.
    """

    def __init__(self, geometry, freqs, monitors=None, name=None):
        # firing rate
        if isinstance(freqs, np.ndarray):
            freqs = freqs.flatten()
        if not np.all(freqs <= 1000. / profile.get_dt()):
            print(f'WARNING: The maximum supported frequency at dt={profile.get_dt()} ms '
                  f'is {1000. / profile.get_dt()} Hz. While we get your "freq" setting which '
                  f'is bigger than that.')

        # neuron model
        dt = profile.get_dt() / 1000.

        def update(ST):
            ST['spike'] = np.random.random(ST['spike'].shape) < freqs * dt

        model = NeuType(name='poisson_input',
                        requires=dict(ST=NeuState(['spike'])),
                        steps=update,
                        mode='vector')

        # neuron group
        super(PoissonInput, self).__init__(model=model, geometry=geometry, monitors=monitors, name=name)


class SpikeTimeInput(NeuGroup):
    """The input neuron group characterized by spikes emitting at given times.

    >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
    >>> SpikeTimeInput(2, times=[10, 20])
    >>> # or
    >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
    >>> SpikeTimeInput(2, times=[10, 20], indices=0)
    >>> # or
    >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
    >>> SpikeTimeInput(2, times=[10, 20, 30], indices=[0, 1, 0])
    >>> # or
    >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
    >>> # at 30 ms, neuron 1 fires.
    >>> SpikeTimeInput(2, times=[10, 20, 30], indices=[0, [0, 1], 1])

    Parameters
    ----------
    geometry : int, tuple, list
        The neuron group geometry.
    indices : int, list, tuple
        The neuron indices at each time point to emit spikes.
    times : list, np.ndarray
        The time points which generate the spikes.
    monitors : list, tuple
        The targets for monitoring.
    name : str
        The group name.
    """

    def __init__(self, geometry, times, indices=None, monitors=None, name=None):
        # times
        assert (isinstance(times, (list, tuple)) and isinstance(times[0], (int, float))) or \
               (isinstance(times, np.ndarray) and np.ndim(times) == 1)
        times = np.array(times)
        num_times = len(times)

        def update(ST, _t_, input_idx):
            in_idx = input_idx[0]
            if (in_idx < num_times) and (_t_ >= times[in_idx]):
                ST['spike'] = indices[in_idx]
                input_idx += 1
            else:
                ST['spike'] = 0.

        model = NeuType(name='time_input',
                        requires=dict(ST=NeuState(['spike']),
                                      input_idx=Array(dim=1, help='The current index.')),
                        steps=update,
                        mode='vector')

        # neuron group
        super(SpikeTimeInput, self).__init__(model=model, geometry=geometry, monitors=monitors, name=name)
        self.input_idx = np.array([0], dtype=np.int_)

        # indices
        if indices is None:
            indices = np.ones((len(times), self.size), dtype=np.bool_)
        elif isinstance(indices, int):
            idx = indices
            indices = np.zeros((len(times), self.size), dtype=np.bool_)
            indices[:, idx] = True
        elif isinstance(indices, (tuple, list)):
            old_idx = indices
            indices = np.zeros((len(times), self.size), dtype=np.bool_)
            for i, it in enumerate(old_idx):
                if isinstance(it, int):
                    indices[i, it] = True
                elif isinstance(it, (tuple, list)):
                    indices[i][it] = True
                else:
                    raise ValueError('Unknown type.')
        else:
            raise ValueError('Unknown type.')


class FreqInput(NeuGroup):
    """The input neuron group characterized by frequency.

    For examples:

    >>> # Get 2 neurons, with 10 Hz firing rate.
    >>> FreqInput(2, freq=10.)
    >>> # Get 4 neurons, with 20 Hz firing rate. The neurons
    >>> # start firing at [10, 30] ms randomly.
    >>> FreqInput(4, freq=20., start_time=np.random.randint(10, 30, (4,)))

    Parameters
    ----------
    geometry : int, list, tuple
        The geometry of neuron group.
    freqs : int, float, np.ndarray
        The output spike frequency.
    start_time : float
        The time of the first spike.
    monitors : list, tuple
        The targets for monitoring.
    name : str
        The name of the neuron group.
    """
    def __init__(self, geometry, freqs, start_time=0., monitors=None, name=None):
        if not np.allclose(freqs <= 1000. / profile.get_dt()):
            print(f'WARNING: The maximum supported frequency at dt={profile.get_dt()} ms '
                  f'is {1000. / profile.get_dt()} Hz. While we get your "freq" setting which '
                  f'is bigger than that.')

        ST = NeuState({'spike': 0., 't_next_spike': 0., 't_last_spike': -1e7})

        if profile.is_jit_backend():
            def update_state(ST, _t_):
                if _t_ >= ST['t_next_spike']:
                    ST['spike'] = 1.
                    ST['t_last_spike'] = _t_
                    ST['t_next_spike'] += 1000. / freqs
                else:
                    ST['spike'] = 0.

            model = NeuType(name='poisson_input',
                            requires=dict(ST=ST),
                            steps=update_state,
                            mode='scalar')

        else:
            if np.size(freqs) == 1:
                def update_state(ST, _t_):
                    should_spike = _t_ >= ST['t_next_spike']
                    ST['spike'] = should_spike
                    spike_ids = np.where(should_spike)[0]
                    ST['t_last_spike'][spike_ids] = _t_
                    ST['t_next_spike'][spike_ids] += 1000. / freqs

            else:
                def update_state(ST, _t_):
                    should_spike = _t_ >= ST['t_next_spike']
                    ST['spike'] = should_spike
                    spike_ids = np.where(should_spike)[0]
                    ST['t_last_spike'][spike_ids] = _t_
                    ST['t_next_spike'][spike_ids] += 1000. / freqs[spike_ids]

            model = NeuType(name='poisson_input',
                            requires=dict(ST=ST),
                            steps=update_state,
                            mode='vector')

        # neuron group
        super(FreqInput, self).__init__(model=model, geometry=geometry, monitors=monitors, name=name)

        self.ST['t_next_spike'] = start_time
        self.pars['freqs'] = freqs
