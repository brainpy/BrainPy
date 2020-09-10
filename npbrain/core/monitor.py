# -*- coding: utf-8 -*-

from numba import typed, types, prange

from .neuron_group import NeuronGroup
from .synapse_group import SynapseGroup
from .. import _numpy as bnp
from .. import profile
from ..utils import helper

__all__ = [
    'Monitor',
    'SpikeMonitor',
    'StateMonitor',

    'raster_plot',
    'firing_rate',
]


class Monitor(object):
    """Base monitor class.
    """

    def __init__(self, target, variables=None):
        self.target = target

        # check `variables`
        if variables is None:
            if isinstance(target, NeuronGroup):
                variables = ['V']
            elif isinstance(target, SynapseGroup):
                variables = ['g_out']
            else:
                raise ValueError('When `vars=None`, NumpyBrain only supports the recording '
                                 'of "V" for NeuronGroup and "g_out" for SynapseGroup.')
        if isinstance(variables, str):
            variables = [variables]
        assert isinstance(variables, (list, tuple))
        self.variables = tuple(variables)

        # fake initialization
        for k in self.variables:
            setattr(self, k, bnp.zeros((1, 1), dtype=bnp.float_))


        # generate update_function
        if profile.is_numba_bk():
            self.state = []

            # monitor of synapse object
            if 'g_out' in self.variables or 'g_in' in self.variables:
                if len([v for v in self.variables if v != 'g_out' and v != 'g_in']):
                    func_str = '''def func(S, D, mon_state, out_idx, in_idx, i):'''
                else:
                    func_str = '''def func(D, mon_state, out_idx, in_idx, i):'''
            # monitor of neuron object
            else:
                func_str = '''def func(S, mon_state, i):'''

            # define the monitor function
            for j, k in enumerate(self.variables):
                if k == 'g_out':
                    func_str += '\n\tmon_state[{}][i] = D[out_idx]'.format(j)
                elif k == 'g_in':
                    func_str += '\n\tmon_state[{}][i] = D[in_idx]'.format(j)
                else:
                    func_str += '\n\tmon_state[{}][i] = S[{}]'.format(j, target.var2index[k])

            # compile the function
            exec(compile(func_str, '', 'exec'))

            if profile.debug:
                print('\nMonitor function:')
                print('-' * 30)
                print(func_str)

            self.update = helper.autojit(locals()['func'])

        else:
            if isinstance(target, SynapseGroup):
                if len([v for v in self.variables if v != 'g_out' and v != 'g_in']):
                    def func(S, D, mon_state, out_idx, in_idx, i):
                        pass
                else:
                    def func(D, mon_state, out_idx, in_idx, i):
                        for v in self.variables:
                            getattr(self, )

            if isinstance(target, NeuronGroup):
                def func(S, mon_state, i):
                    pass


class SpikeMonitor(Monitor):
    """Monitor class to record spikes.

    Parameters
    ----------
    target : Neurons
        The neuron group to monitor.
    """

    def __init__(self, target):
        # check `variables`
        self.vars = ('index', 'time')
        num = target.state.shape[1]

        # check `target`
        assert isinstance(target, Neurons), 'Cannot monitor spikes in synapses.'

        # fake initialization
        if profile.is_numba_bk():
            self.index = typed.List.empty_list(types.int64)
            self.time = typed.List.empty_list(types.float64)
        else:
            self.index = []
            self.time = []

        @helper.autojit
        def update_state(neu_state, mon_time, mon_index, t):
            for idx in prange(num):
                if neu_state[-3, idx] > 0.:
                    mon_index.append(idx)
                    mon_time.append(t)

        self.update_state = update_state

        # super class initialization
        super(SpikeMonitor, self).__init__(target)

    def __str__(self):
        return 'SpikeMonitor of {}'.format(str(self.target))

    def __repr__(self):
        return 'SpikeMonitor of {}'.format(repr(self.target))


class StateMonitor():
    """Monitor class to record states.

    Parameters
    ----------
    target : Neurons, Synapses
        The object to monitor.
    vars : str, list, tuple
        The variable need to be recorded for the ``target``.
    """

    def __init__(self, target, vars=None):
        # check `variables`
        if vars is None:
            if isinstance(target, Neurons):
                vars = ['V']
            elif isinstance(target, Synapses):
                vars = ['g_out']
            else:
                raise ValueError('When `vars=None`, NumpyBrain only supports the recording '
                                 'of "V" for Neurons and "g" for Synapses.')
        if isinstance(vars, str):
            vars = [vars]
        assert isinstance(vars, (list, tuple))
        vars = tuple(vars)
        for var in vars:
            if var not in target.var2index:
                raise ValueError('Variable "{}" is not in target "{}".'.format(var, target))
        self.vars = vars

        # fake initialization
        for k in self.vars:
            setattr(self, k, bnp.zeros((1, 1)))
        self.state = []

        if 'g_out' in vars or 'g_in' in vars:  # monitor of synapse object
            if len([v for v in vars if v != 'g_out' and v != 'g_in']):
                func_str = '''def func(obj_state, delay_state, mon_state, out_idx, in_idx, i):'''
            else:
                func_str = '''def func(delay_state, mon_state, out_idx, in_idx, i):'''
        else:  # monitor of neuron object
            func_str = '''def func(obj_state, mon_state, i):'''
        for j, k in enumerate(vars):
            if k == 'g_out':
                func_str += '\n\tmon_state[{}][i] = delay_state[out_idx]'.format(j)
            elif k == 'g_in':
                func_str += '\n\tmon_state[{}][i] = delay_state[in_idx]'.format(j)
            else:
                func_str += '\n\tmon_state[{}][i] = obj_state[{}]'.format(j, target.var2index[k])
        exec(compile(func_str, '', 'exec'))

        if profile.debug:
            print('\nMonitor function:')
            print('-' * 30)
            print(func_str)

        self.update_state = helper.autojit(locals()['func'])

        # super class initialization
        super(StateMonitor, self).__init__(target)

    def init_state(self, length):
        assert isinstance(length, int)

        mon_states = []
        for i, k in enumerate(self.vars):
            if k in ['g_out', 'g_in']:
                v = self.target.delay_state[0]
            else:
                v = self.target.state[self.target.var2index[k]]
            shape = (length,) + v.shape
            state = bnp.zeros(shape, dtype=profile.ftype)
            setattr(self, k, state)
            mon_states.append(state)
        self.state = tuple(mon_states)

    def __str__(self):
        return 'StateMonitor of {}'.format(str(self.target))

    def __repr__(self):
        return 'StateMonitor of {}'.format(repr(self.target))


def raster_plot(mon, times=None):
    """Get spike raster plot which displays the spiking activity
    of a group of neurons over time.

    Parameters
    ----------
    mon : Monitor
        The monitor which record spiking activities.
    times : None, numpy.ndarray
        The time steps.

    Returns
    -------
    raster_plot : tuple
        Include (neuron index, spike time).
    """
    if isinstance(mon, StateMonitor):
        elements = bnp.where(mon.spike > 0.)
        index = elements[1]
        if hasattr(mon, 'spike_time'):
            time = mon.spike_time[elements]
        else:
            assert times is not None, 'Must provide "times" when StateMonitor has no "spike_time" attribute.'
            time = times[elements[0]]
    elif isinstance(mon, SpikeMonitor):
        index = bnp.array(mon.index)
        time = bnp.array(mon.time)
    else:
        raise ValueError
    return index, time


def firing_rate(mon, width, window='gaussian'):
    """Calculate the mean firing rate over in a neuron group.

    This method is adopted from Brian2.

    The firing rate in trial :math:`k` is the spike count :math:`n_{k}^{sp}`
    in an interval of duration :math:`T` divided by :math:`T`:

    .. math::

        v_k = {n_k^{sp} \\over T}

    Parameters
    ----------
    mon : StateMonitor
        The monitor which record spiking activities.
    width : int, float
        The width of the ``window`` in millisecond.
    window : str
        The window to use for smoothing. It can be a string to chose a
        predefined window:

        - `flat`: a rectangular,
        - `gaussian`: a Gaussian-shaped window.

        For the `Gaussian` window, the `width` parameter specifies the
        standard deviation of the Gaussian, the width of the actual window
        is `4 * width + dt`.
        For the `flat` window, the width of the actual window
        is `2 * width/2 + dt`.

    Returns
    -------
    rate : numpy.ndarray
        The population rate in Hz, smoothed with the given window.
    """
    # rate
    assert hasattr(mon, 'spike'), 'Must record the "spike" of the neuron group to get firing rate.'
    rate = bnp.sum(mon.spike, axis=1)

    # window
    dt = profile.get_dt()
    if window == 'gaussian':
        width1 = 2 * width / dt
        width2 = int(bnp.round(width1))
        window = bnp.exp(-bnp.arange(-width2, width2 + 1) ** 2 / (width1 ** 2 / 2))
    elif window == 'flat':
        width1 = int(width / 2 / dt) * 2 + 1
        window = bnp.ones(width1)
    else:
        raise ValueError('Unknown window type "{}".'.format(window))
    window = bnp.float_(window)

    return bnp.convolve(rate, window / sum(window), mode='same')
