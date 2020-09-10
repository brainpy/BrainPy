# -*- coding: utf-8 -*-

from .. import _numpy as bnp
from .. import profile
from ..utils import helper
from collections import OrderedDict
from .common_func import Group
from .common_func import numbify_func

_group_no = 0


def init_syn_state(num_syn, variables=None, parameters=None):
    """Initialize the synapse state with (num_syn, ) shape.

    Parameters
    ----------
    num_syn : int
        Number of the synapses.
    variables : tuple, list, int
        The variables of the neuron model.
        Each variable has the shape of (num_syn, ).
        If `variables` is an instance of `list` or `tuple`, each of them is
        initialized as `zeros`.
    parameters : dict
        The parameter of the neuron models. Each of them can be modified in
        the model running.

    Returns
    -------
    state : np.ndarray
        The state of the synapse.
    """
    if variables is None and parameters is None:
        raise ValueError('variables and parameters cannot be both None.')

    # get names and values of variables, parameters
    var_names, var_values = _format_vars(variables, num_syn)
    par_names, par_values = _format_vars(parameters, num_syn)

    # get state
    names = var_names + par_names
    values = var_values + par_values

    state = bnp.zeros((len(names), num_syn), dtype=profile.ftype)
    for i, v in enumerate(values):
        state[i] = v

    return state


def init_delay_state(num_post, delay, variables=None, parameters=None):
    delay_len = format_delay(delay)

    # get names and values of variables, parameters
    var_names, var_values = _format_vars(variables, num_post)
    par_names, par_values = _format_vars(parameters, num_post)

    # get state
    names = var_names + par_names
    values = var_values + par_values

    state = bnp.zeros((delay_len + len(names), num_post), dtype=profile.ftype)
    for i, v in enumerate(values):
        state[delay_len + i] = v

    return state


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

        assert 'pre' in kwargs, 'Must define "pre" in synapses.'
        assert 'post' in kwargs, 'Must define "post" in synapses.'
        self.post.pre_synapses.append(self)
        self.pre.post_synapses.append(self)

        # check `num`, `num_pre` and `num_post`
        assert 'num' in kwargs, 'Must provide "num" attribute.'
        if 'num_pre' not in kwargs:
            self.num_pre = self.pre.num
        if 'num_post' not in kwargs:
            self.num_post = self.post.num

        # check functions
        assert 'update_state' in kwargs, 'Must provide "update_state" function.'
        assert 'output_synapse' in kwargs, 'Must provide "output_synapse" function.'

        self.update_state = helper.autojit(self.update_state)
        self.output_synapse = helper.autojit(self.output_synapse)

        # check `name`
        if 'name' not in kwargs:
            global synapse_no
            self.name = "Synapses-{}".format(synapse_no)
            synapse_no += 1

        # check `delay_len`
        if 'delay_len' not in kwargs:
            if 'delay' not in kwargs:
                raise ValueError('Must define "delay".')
            else:
                dt = kwargs.get('dt', profile.get_dt())
                self.delay_len = format_delay(self.delay, dt)

        # check `state`
        assert 'delay_state' in kwargs, 'Must define "delay_state" in synapses.'
        if 'state' not in kwargs:
            print('Synapses "{}" do not define "state" item.'.format(self.name))
            self.state = None

        # check `var2index`
        if 'var2index' not in kwargs:
            raise ValueError('Must define "var2index".')
        assert isinstance(self.var2index, dict), '"var2index" must be a dict.'
        # "g_in" is the "delay_idx"
        # 'g_out' is the "output_idx"
        default_var2index = {'g_in': self.delay_len - 1, 'g_out': 0}
        self.default_var2index = default_var2index
        for k in default_var2index.keys():
            if k in self.var2index:
                raise ValueError('"{}" is a pre-defined variable, '
                                 'cannot be defined in "var2index".'.format(k))
        self.var2index.update(default_var2index)

    def update_conductance_index(self):
        self.var2index['g_in'] = (self.var2index['g_in'] + 1) % self.delay_len
        self.var2index['g_out'] = (self.var2index['g_out'] + 1) % self.delay_len

    @property
    def delay_idx(self):
        return self.var2index['g_in']

    @property
    def output_idx(self):
        return self.var2index['g_out']

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def available_monitors(self):
        return sorted(list(self.var2index.keys()))


class SynapseGroup(Group):

    def __init__(self, pre, post, delay=0., monitors=None,
                 vars_init=None, pars_updates=None):
        # essential
        # -----------
        self.num_pre = pre.num
        self.num_post = post.num

        # connection
        # -----------
        self.num = ...

        # delay and delay state
        # ----------------------
        if delay is None:
            delay_len = 1
        elif isinstance(delay, (int, float)):
            dt = profile.get_dt()
            delay_len = int(bnp.ceil(delay / dt)) + 1
        else:
            raise ValueError('NumpyBrain currently does not support other kinds of delay.')
        self.delay_len = delay_len
        self.delay_state = bnp.zeros((delay_len, self.num_post), dtype=bnp.float_)
        self.D = self.delay_state
        self._delay_indices = bnp.array([0, delay_len-1])

        # super class initialization
        super(SynapseGroup, self).__init__(vars_init=vars_init, pars_updates=pars_updates)

        # monitors
        # ----------
        self.mon = dict()
        self._mon_vars = monitors
        self._mon_update = None

        if monitors is not None:
            for k in monitors:
                self.mon[k] = bnp.zeros((1, 1), dtype=bnp.float_)

            # generate function
            if profile.is_numba_bk():
                def update(i):
                    for k in self._mon_vars:
                        if k == 'g_out':
                            self.mon[k][i] = self.D[self._delay_indices[0]]
                        elif k == 'g_in':
                            self.mon[k][i] = self.D[self._delay_indices[1]]
                        else:
                            self.mon[k][i] = self.state[self.var2index[k]]
            else:
                def update(i):
                    if k == 'g_out':
                        self.mon[k][i] = self.D[self._delay_indices[0]]
                    elif k == 'g_in':
                        self.mon[k][i] = self.D[self._delay_indices[1]]
                    else:
                        self.mon[k][i] = self.state[k]
            self._mon_update = update

    def update_delay_indices(self):
        self._delay_indices[0] = (self._delay_indices[0] + 1) % self.delay_len
        self._delay_indices[1] = (self._delay_indices[1] + 1) % self.delay_len


def create_synapse_model(parameters=None, variables=None, update_func=None, name=None):
    # handle "update"
    # -----------------
    assert update_func is not None, '"update_func" cannot be None.'

    # handle "name"
    # --------------
    if name is None:
        global _group_no
        name_ = 'syn_group_{}'.format(_group_no)
        _group_no += 1
    else:
        name_ = name

    # handle "parameters"
    # --------------------
    if parameters is None:
        parameters = OrderedDict()
    elif isinstance(parameters, (list, tuple)):
        parameters = OrderedDict((par, 0.) for par in parameters)
    elif isinstance(parameters, dict):
        parameters = OrderedDict(parameters)
    else:
        raise ValueError('Unknown parameters type: {}'.format(type(parameters)))

    # handle "variables"
    # --------------------
    if variables is None:
        variables = OrderedDict()
    elif isinstance(variables, (list, tuple)):
        variables = OrderedDict((var_, 0.) for var_ in variables)
    elif isinstance(variables, dict):
        variables = OrderedDict(variables)
    else:
        raise ValueError('Unknown variables type: {}'.format(type(variables)))

    # generate class
    # --------------------

    class CreatedSynapseGroup(SynapseGroup):
        pars = parameters
        vars = variables
        update = update_func
        name = name_

        def __str__(self):
            return self.name

        def __repr__(self):
            return self.name

    return CreatedSynapseGroup

