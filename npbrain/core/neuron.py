# -*- coding: utf-8 -*-

import numpy as np

from npbrain.utils import helper
from ..utils import profile


__all__ = [
    'judge_spike',
    'get_spike_judger',
    'init_neu_state',
    'format_geometry',
    'format_refractory',
    'Neurons',
    'generate_fake_neuron'
]


def judge_spike(neu_state, vth, t):
    """Judge and record the spikes of the given neuron group.

    Parameters
    ----------
    neu_state : np.ndarray
        The state of the neuron group.
    vth : float, int, np.ndarray
        The spike threshold.
    t : float
        The current time point.

    Returns
    -------
    spike_indexes : list
        The neuron indexes that are spiking.
    """
    above_threshold = (neu_state[0] >= vth).astype(np.float64)
    prev_above_th = neu_state[-4]
    spike_st = above_threshold * (1. - prev_above_th)
    spike_idx = np.where(spike_st > 0.)[0]
    neu_state[-4] = above_threshold
    neu_state[-3] = spike_st
    neu_state[-2][spike_idx] = t
    return spike_idx


def get_spike_judger():
    @helper.autojit('i[:](f[:, :], f, f)')
    def f(neu_state, vth, t):
        above_threshold = (neu_state[0] >= vth).astype(np.float64)
        prev_above_th = neu_state[-4]
        spike_st = above_threshold * (1. - prev_above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        neu_state[-4] = above_threshold
        neu_state[-3] = spike_st
        neu_state[-2][spike_idx] = t
        return spike_idx

    return f


default_neu_state = [
    ('not_ref', 1., -5),
    ('above_th', 0., -4),
    ('spike', 0., -3),
    ('spike_time', -1e5, -2),
    ('Isyn', 0., -1)
]


def _format_vars(variables, num):
    def digit2array(digit, var_name):
        if np.size(digit) == 1:
            return np.ones(num, dtype=profile.ftype) * digit
        elif np.size(digit) == num:
            return np.asarray(digit, dtype=profile.ftype)
        else:
            raise ValueError('The dimension value of var "{}" doesn\'t match "{}".'.format(var_name, num))

    if variables is None:
        var_names, var_values = [], []
    else:
        if isinstance(variables, int):
            var_names = [None] * variables
            var_values = [0.] * variables
        elif isinstance(variables, (list, tuple)):
            if isinstance(variables[0], str):
                var_names = list(variables)
                var_values = [0.] * len(variables)
            elif isinstance(variables[0], (list, tuple)):
                var_names = [k for k, v in variables]
                var_values = [digit2array(v, k) for k, v in variables]
            else:
                var_names = [None] * len(variables)
                var_values = [digit2array(v, '?') for v in variables]
        else:
            raise ValueError('Unknown variables type "{}".'.format(type(variables)))

    return var_names, var_values


def init_neu_state(num_neu, variables=None, parameters=None):
    """Initialize the state of the given neuron group.

    Parameters
    ----------
    num_neu : int
        Number of the neurons in the group.
    variables : tuple, list, dict, int
        The variables of the neuron model.
        Each variable has the shape of (num_neu, ).
        If `variables` is an instance of `list` or `tuple`, each of them is
        initialized as `zeros`.
        If `variables` is an instance of `dict`, each of `key` in the dict is
        initialized as the corresponding `value`.
    parameters : tuple, list
        The parameter of the neuron models. Each of them can be modified in
        the model running.

    Returns
    -------
    state : np.ndarray
        The state of the neuron group.
    """

    if variables is None and parameters is None:
        raise ValueError('variables and parameters cannot be both None.')

    # get names and values of variables, parameters
    var_names, var_values = _format_vars(variables, num_neu)
    par_names, par_values = _format_vars(parameters, num_neu)
    default_names = [k for k, v, i in default_neu_state]
    default_values = [v for k, v, i in default_neu_state]

    # get states
    names = var_names + par_names + default_names
    values = var_values + par_values + default_values

    state = np.zeros((len(names), num_neu), dtype=profile.ftype)
    for i, v in enumerate(values):
        state[i] = v

    return state


def format_geometry(geometry):
    """Format the geometry of the neuron group.

    Parameters
    ----------
    geometry : int, list, tuple
        The size (geometry) of the neuron group.

    Returns
    -------
    num_and_geo : tuple
        (Number of neurons, geometry of the neuron group),
        where the shape of geometry is (height, width).
    """
    # define network geometry
    if isinstance(geometry, (int, float)):
        geometry = (1, int(geometry))
    elif isinstance(geometry, (tuple, list)):
        # a tuple is given, can be 1 .. N dimensional
        if len(geometry) == 1:
            height, width = 1, geometry[0]
        elif len(geometry) == 2:
            height, width = geometry[0], geometry[1]
        else:
            raise ValueError('Do not support 3+ dimensional networks.')
        geometry = (height, width)
    else:
        raise ValueError()
    num = int(np.prod(geometry))
    return num, geometry


def format_refractory(ref=None):
    """Format the refractory period in the given neuron group.

    Parameters
    ----------
    ref : None, int, float
        The refractory period.

    Returns
    -------
    tau_ref : float
        The formatted refractory period.
    """
    if ref is None:
        tau_ref = 0
    elif isinstance(ref, (int, float)):
        if ref > 0:
            tau_ref = float(ref)
        elif ref == 0:
            tau_ref = 0
        else:
            raise ValueError
    elif isinstance(ref, np.ndarray):
        assert np.alltrue(ref >= 0)
        tau_ref = ref
    else:
        raise ValueError()
    return tau_ref


class NeuronsView(object):
    def __init__(self, target, ranks):
        self.target = target
        self.ranks = ranks
        self.num = len(ranks)


_neuron_no = 0


class Neurons(object):
    """The base neurons class.

    Parameters
    ----------
    kwargs : dict
        Parameters of the given neuron group.
    """
    def __init__(self, **kwargs):
        if 'args' in kwargs:
            kwargs.pop('args')
        if 'kwargs' in kwargs:
            kwargs.pop('kwargs')
        for k, v in kwargs.items():
            setattr(self, k, v)

        # define external connections
        self.pre_synapses = []
        self.post_synapses = []

        # check functions
        assert 'update_state' in kwargs
        wrapper = helper.autojit('(f8[:, :], f8)')
        self.update_state = wrapper(self.update_state)

        # check `geometry`
        assert 'geometry' in kwargs, 'Must define "geometry".'
        assert 'num' in kwargs, 'Must define "num".'

        # check `name`
        if 'name' not in kwargs:
            global _neuron_no
            self.name = "Neurons-{}".format(_neuron_no)
            _neuron_no += 1

        # check `state`
        assert 'state' in kwargs, 'Must define "state".'

        # check `var2index`
        if 'var2index' not in kwargs:
            raise ValueError('Must define "var2index".')
        assert isinstance(self.var2index, dict), '"var2index" must be a dict.'
        for k, _, _ in default_neu_state:
            if k in self.var2index:
                if k == 'V':
                    if self.var2index['V'] != 0:
                        print('The position of "V" is not 0.')
                else:
                    raise ValueError('"{}" is a pre-defined variable, cannot '
                                     'be defined in "var2index".'.format(k))
        self.var2index.update({k: i for k, _, i in default_neu_state})

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __getitem__(self, item):
        if isinstance(item, int):  # a single neuron
            return NeuronsView(self, [int(item)])
        elif isinstance(item, (list, np.ndarray)):
            if isinstance(item, np.ndarray):
                if item.ndim != 1:
                    raise ValueError('Only one-dimensional lists/arrays are allowed to address a neuron group.')
                item = list(item.astype(int))
            return NeuronsView(self, list(item))
        elif isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            if item.start is None:
                start = 0
            if item.stop is None:
                stop = self.num
            if item.step is None:
                step = 1
            rk_range = list(range(start, stop, step))
            return NeuronsView(self, rk_range)
        else:
            raise ValueError('Can not address the population with', item)

    @property
    def available_monitors(self):
        return sorted(list(self.var2index.keys()))


def generate_fake_neuron(num, V=0.):
    """Generate the fake neuron group for testing synapse function.

    Parameters
    ----------
    num : int
        Number of neurons in the group.
    V : int, float, numpy.ndarray
        Initial membrane potential.

    Returns
    -------
    neurons : dict
        An instance of ``Dict`` for simulating neurons.
    """

    var2index = dict(V=0)
    num, geometry = num, (num,)
    state = np.zeros((5, num))
    state[0] = V
    update_state = helper.autojit(lambda neu_state, t: 1)
    return Neurons(**locals())
