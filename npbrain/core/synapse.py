# -*- coding: utf-8 -*-
import numpy as np
import numba as nb

from npbrain.utils import helper, profile

__all__ = [
    'format_delay',
    'initial_syn_state',
    'Synapses',
]

synapse_no = 0


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

        wrapper = helper.autojit('(UniTuple(f8[:, :], 3), f8, i8)')
        self.update_state = wrapper(self.update_state)

        wrapper = helper.autojit('(UniTuple(f8[:, :], 3), i8, f8[:, :])')
        self.output_synapse = wrapper(self.output_synapse)

        wrapper = helper.autojit('(UniTuple(f8[:, :], 3), f8[:, :], f8[:, :])')
        self.collect_spike = wrapper(self.collect_spike)

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
        # "g_in" is the "delay_idx"
        # 'g_out' is the "output_idx"
        default_variables = {'pre_spike': (0, -1), 'g_in': [1, self.delay_len - 1], 'g_out': [1, 0]}
        self.default_variables = default_variables
        for k in default_variables.keys():
            if k in self.var2index:
                raise ValueError('"{}" is a pre-defined variable, '
                                 'cannot be defined in "var2index".'.format(k))
        self.var2index.update(default_variables)

    def update_conductance_index(self):
        self.var2index['g_in'][1] = (self.var2index['g_in'][1] + 1) % self.delay_len
        self.var2index['g_out'][1] = (self.var2index['g_out'][1] + 1) % self.delay_len

    def delay_idx(self):
        return self.var2index['g_in'][1]

    def output_idx(self):
        return self.var2index['g_out'][1]

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
