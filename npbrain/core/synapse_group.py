# -*- coding: utf-8 -*-

from .. import _numpy as bnp
from .. import profile
from ..utils import helper
from collections import OrderedDict
from .common_func import NodeGroup
from .common_func import numbify_func
from .connectivity import Connector

_group_no = 0


class SynapseGroup(NodeGroup):

    def __init__(self, pre, post, connection, delay=0., monitors=None,
                 vars_init=None, pars_updates=None):
        # connection
        # -----------
        if isinstance(connection, Connector):
            self.connector = connection
            self.pre_idx = ...
            self.post_idx = ...
        else:
            assert isinstance(connection, dict), '"connection" only support "dict".'
            assert 'i' in connection, '"connection" must provide "i" item.'
            assert 'j' in connection, '"connection" must provide "j" item.'
            self.pre_idx = connection['i']
            self.post_idx = connection['j']

        # essential
        # -----------
        self.num_pre = pre.num
        self.num_post = post.num
        self.num = len(self.pre_idx)

        # super class initialization
        super(SynapseGroup, self).__init__(vars_init=vars_init, pars_updates=pars_updates)

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
        self.state['delay'] = self.delay_state
        self.state['delay_in'] = 0
        self.state['delay_out'] = self.delay_len - 1

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
                            self.mon[k][i] = self.D[self.delay_indices[0]]
                        elif k == 'g_in':
                            self.mon[k][i] = self.D[self.delay_indices[1]]
                        else:
                            self.mon[k][i] = self.state[self.var2index[k]]
            else:
                def update(i):
                    if k == 'g_out':
                        self.mon[k][i] = self.D[self.delay_indices[0]]
                    elif k == 'g_in':
                        self.mon[k][i] = self.D[self.delay_indices[1]]
                    else:
                        self.mon[k][i] = self.state[k]
            self._mon_update = update

    def update_delay_indices(self):
        # in_index
        self.state['delay_in'] = (self.state['delay_in'] + 1) % self.delay_len
        # out_index
        self.state['delay_out'] = (self.state['delay_out'] + 1) % self.delay_len


def create_synapse_model(parameters=None, variables=None, update_func=None, name=None):
    # handle "update"
    # -----------------
    assert update_func is not None, '"update_func" cannot be None.'

    # handle "name"
    # --------------
    if name is None:
        global _group_no
        name_ = 'SynGroup{}'.format(_group_no)
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

    cls_str = '''
class {name}(SynapseGroup):
    pars = {name}_pars
    vars = {name}_vars
    update = {name}_update
    name = "{name}"

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
'''.format(name=name_)

    globals()[name_ + '_pars'] = parameters
    globals()[name_ + '_vars'] = variables
    globals()[name_ + '_update'] = update_func
    exec(cls_str, globals())

    return globals()[name_]

