# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

from npbrain.utils import helper, profile

__all__ = [
    'syn_delay',
    'get_conductance_recorder',
    'format_delay',
    'initial_syn_state',
    'Synapses',
]

synapse_no = 0


def record_conductance(syn_state, var_index, g):
    """Record the conductance of the synapses.

    Parameters
    ----------
    syn_state : tuple
        The state of the synapses.
    var_index : np.ndarray
        The indexes of variables.
    g : np.ndarray
        The conductance to record at current time.
    """
    # get `delay_len`
    delay_len = var_index[-1, 0]
    # update `output_idx`
    output_idx = (var_index[-2, 1] + 1) % delay_len
    var_index[-2, 1] = output_idx
    # update `delay_idx`
    delay_idx = (var_index[-3, 1] + 1) % delay_len
    var_index[-3, 1] = delay_idx
    # update `conductance`
    syn_state[1][delay_idx] = g


def get_conductance_recorder():
    @helper.autojit
    def record_conductance(syn_state, var_index, g):
        # get `delay_len`
        delay_len = var_index[-1, 0]
        # update `output_idx`
        output_idx = (var_index[-2, 1] + 1) % delay_len
        var_index[-2, 1] = output_idx
        # update `delay_idx`
        delay_idx = (var_index[-3, 1] + 1) % delay_len
        var_index[-3, 1] = delay_idx
        # update `conductance`
        syn_state[1][delay_idx] = g
    return record_conductance


def syn_delay(func):

    func = helper.autojit(func)

    @helper.autojit
    def f(syn_state, t, var2index):
        # get `g`
        g = func(syn_state, t)
        # get `delay_len`
        delay_len = var2index[-1, 0]
        # update `output_idx`
        output_idx = (var2index[-2, 1] + 1) % delay_len
        var2index[-2, 1] = output_idx
        # update `delay_idx`
        delay_idx = (var2index[-3, 1] + 1) % delay_len
        var2index[-3, 1] = delay_idx
        # update `conductance`
        syn_state[1][delay_idx] = g

    return f


def format_delay(delay, dt=None):
    """Format the given delay and get the delay length.

    Parameters
    ----------
    delay : None, int, float, np.ndarray
        The delay.
    dt : float, None
        The precision of the numerical integration.

    Returns
    -------
    delay_len : int
        Delay length.
    """
    if delay is None:
        delay_len = 1
    elif isinstance(delay, (int, float)):
        dt = profile.get_dt() if dt is None else dt
        delay_len = int(np.ceil(delay / dt)) + 1
    else:
        raise ValueError()
    return delay_len


def initial_syn_state(delay,
                      num_pre: int,
                      num_post: int,
                      num_syn: int,
                      num_pre_shape_var: int = 0,
                      num_post_shape_var: int = 0,
                      num_syn_shape_var: int = 0):
    """For each state, it is composed by
    (pre_shape_state, post_shape_state, syn_shape_state).

    Parameters
    ----------
    delay : float, int, None
        The length of synapse delay.
    num_pre : int
        Number of neurons in pre-synaptic group.
    num_post : int
        Number of neurons in post-synaptic group.
    num_syn : int
        Number of synapses.
    num_pre_shape_var : int
        Number of variables with (num_pre, ) shape.
    num_post_shape_var : int
        Number of variables with (num_post, ) shape.
    num_syn_shape_var : int
        Number of variables with (num_syn, ) shape.

    Returns
    -------
    state : tuple
        Synapse state.
    """

    # state with (pre_num, ) shape #
    ################################
    # The state is:
    #   pre_spike                 [[..........],
    # --------------------------   [..........],
    #   vars with num_pre shape    [..........],
    # --------------------------   [..........]]
    pre_shape_state = np.zeros((1 + num_pre_shape_var, num_pre))

    # state with (post_num, ) shape #
    #################################
    # The state is:
    # ----------- [[..........],
    # delays       [..........],
    # -----------  [..........],
    # other vars   [..........],
    # -----------  [..........]]

    delay_len = format_delay(delay)
    post_shape_state = np.zeros((delay_len + num_post_shape_var, num_post))

    # state with (num_syn, ) shape #
    ################################
    # The state is:
    # -------------------------  [[..........],
    #  vars with num_syn shape    [..........]
    # -------------------------   [..........]]
    syn_shape_state = np.zeros((num_syn_shape_var, num_syn))

    state = (pre_shape_state, post_shape_state, syn_shape_state)
    return state


def output_synapse(syn_state, var_index, neu_state):
    output_idx = var_index[-2]
    neu_state[-1] += syn_state[output_idx[0]][output_idx[1]]


def collect_spike(syn_state, pre_neu_state, post_neu_state):
    syn_state[0][-1] = pre_neu_state[-3]


class Synapses(object):
    """The base synapses class.

    Parameters
    ----------
    kwargs : dict
        Parameters of the synapses.
    """

    def __init__(self, **kwargs):
        if 'kwargs' in kwargs:
            kwargs.pop('kwargs')
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.post.pre_synapses.append(self)
        self.pre.post_synapses.append(self)

        # check functions
        assert 'update_state' in kwargs, 'Must provide "update_state" function.'

        if 'output_synapse' not in kwargs:
            self.output_synapse = output_synapse

        if 'collect_spike' not in kwargs:
            self.collect_spike = collect_spike

        self.update_state = helper.autojit(self.update_state)
        self.output_synapse = helper.autojit(self.output_synapse)
        self.collect_spike = helper.autojit(self.collect_spike)

        # check `name`
        if 'name' not in kwargs:
            global synapse_no
            self.name = "Synapses-{}".format(synapse_no)
            synapse_no += 1

        # check `num`, `num_pre` and `num_post`
        assert 'num' in kwargs, 'Must provide "num" attribute.'
        if 'num_pre' not in kwargs:
            self.num_pre = self.pre.num
        if 'num_post' not in kwargs:
            self.num_post = self.post.num

        # check `delay_len`
        if 'delay_len' not in kwargs:
            if 'delay' not in kwargs:
                raise ValueError('Must define "delay".')
            else:
                dt = kwargs.get('dt', profile.get_dt())
                self.delay_len = format_delay(self.delay, dt)

        # check `var2index`
        if 'var2index' not in kwargs:
            raise ValueError('Must define "var2index".')
        assert isinstance(self.var2index, dict), '"var2index" must be a dict.'
        # "g" is the "delay_idx"
        # 'g_post' is the "output_idx"
        default_variables = [('pre_spike', (0, -1)),
                             ('g', (1, self.delay_len - 1)),
                             ('g_post', (1, 0)), ]
        self.default_variables = default_variables
        for k, _ in default_variables:
            if k in self.var2index:
                raise ValueError('"{}" is a pre-defined variable, '
                                 'cannot be defined in "var2index".'.format(k))
        user_defined_variables = sorted(list(self.var2index.items()), key=lambda a: a[1])
        syn_variables = user_defined_variables + default_variables
        var2index_array = np.zeros((len(syn_variables) + 1, 2), dtype=np.int32)
        var2index_array[-1, 0] = self.delay_len
        vars = dict(delay_len=-1)
        for i, (var, index) in enumerate(syn_variables):
            var2index_array[i] = list(index)
            vars[var] = i
        self.var2index = vars
        self.var2index_array = var2index_array

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def available_monitors(self):
        return sorted(list(self.var2index.keys()))

    def set_state(self, key, value):
        if key not in self.var2index:
            raise ValueError('Variable "{}" is not in the synapses.'.format(key))
        i0, i1 = self.var2index[key]
        self.state[i0][i1] = value
