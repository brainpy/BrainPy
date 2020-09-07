# -*- coding: utf-8 -*-

from .. import _numpy as np

from ..core.neuron import Neurons
from ..core.neuron import format_geometry
from ..core.neuron import init_neu_state
from ..utils import autojit

__all__ = [
    'FreqInput',
    'TimeInput',
]


def FreqInput(geometry, freq, start_time=0., name='FreqInput'):
    """The input neuron group characterized by frequency.

    For examples:


    >>> # Get 2 neurons, with 10 Hz firing rate.
    >>> FreqInput(2, 10.)
    >>> # Get 4 neurons, with 20 Hz firing rate. The neurons
    >>> # start firing at 50 ms.
    >>> FreqInput(4, 20., 50.)

    Parameters
    ----------
    geometry : int, list, tuple
        The geometry of neuron group.
    freq : int, float
        The output spike frequency.
    start_time : float
        The time of the first spike.
    name : str
        The name of the neuron group.

    Returns
    -------
    neurons : Neurons
        The created neuron group.
    """
    var2index = dict(syn_sp_time=0)
    num, geometry = format_geometry(geometry)
    state = init_neu_state(num, [('syn_sp_time', start_time)])

    @autojit('void(f[:, :], f)')
    def update_state(neu_state, t):
        if t >= neu_state[0, 0]:
            neu_state[-3] = 1.
            neu_state[-2] = t
            neu_state[0, 0] += 1000 / freq
        else:
            neu_state[-3] = 0.

    return Neurons(**locals())


def TimeInput(geometry, times, indices=None, name='TimeInput'):
    """The input neuron group characterized by specific times.

    For examples:

    >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
    >>> TimeInput(2, [10, 20])
    >>> # or
    >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
    >>> TimeInput(2, [10, 20], 0)
    >>> # or
    >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
    >>> TimeInput(2, [10, 20, 30], [0, 1, 0])
    >>> # or
    >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
    >>> # at 30 ms, neuron 1 fires.
    >>> TimeInput(2, [10, 20, 30], [0, [0, 1], 1])

    Parameters
    ----------
    geometry : int, list, tuple
        The geometry of neuron group.
    times : list
        The time points which generate the spikes.
    indices : None, int, list, tuple
        The neuron indices at each time point.
    name : str
        The name of the neuron group.

    Returns
    -------
    neurons : Neurons
        The created neuron group.
    """
    var2index = dict()
    num, geometry = format_geometry(geometry)

    # times
    assert (isinstance(times, (list, tuple)) and isinstance(times[0], (int, float))) or \
           (isinstance(times, np.ndarray) and np.ndim(times) == 1)
    times = np.array(times)
    num_times = len(times)

    # indices
    if indices is None:
        indices = np.ones((len(times), num), dtype=np.bool_)
    elif isinstance(indices, int):
        idx = indices
        indices = np.zeros((len(times), num), dtype=np.bool_)
        indices[:, idx] = True
    elif isinstance(indices, (tuple, list)):
        old_idx = indices
        indices = np.zeros((len(times), num), dtype=np.bool_)
        for i, it in enumerate(old_idx):
            if isinstance(it, int):
                indices[i, it] = True
            elif isinstance(it, (tuple, list)):
                indices[i][it] = True
            else:
                raise ValueError('Unknown type.')
    else:
        raise ValueError('Unknown type.')

    state = init_neu_state(1, num)
    state[0, 0] = 0.  # current index

    @autojit('void(f[:, :], f)')
    def update_state(neu_state, t):
        current_idx = int(neu_state[0, 0])
        if (current_idx < num_times) and (t >= times[current_idx]):
            idx = np.where(indices[current_idx])[0]
            neu_state[-3][idx] = 1.
            neu_state[-2][idx] = t
            neu_state[0, 0] += 1
        else:
            neu_state[-3] = 0.

    return Neurons(**locals())
