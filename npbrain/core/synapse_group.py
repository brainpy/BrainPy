# -*- coding: utf-8 -*-

from .. import _numpy as bnp
from .. import profile
from ..utils import helper
from collections import OrderedDict
from .common_func import BaseGroup
from .common_func import numbify_func
from .connectivity import Connector

_group_no = 0


class RunnableSynapse(object):
    pass


class SynapseGroup(BaseGroup):
    def __call__(self, pre, post, connection, delay=0., monitors=None, vars_init=None, pars_updates=None):
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

        # variables and "state" ("S")
        # ----------------------------
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        for k, v in vars_init:
            if k not in self.vars:
                raise KeyError('variable "{}" is not defined in "{}".'.format(k, self.name))
            self.vars[k] = v

        if profile.is_numba_bk():
            import numba as nb

            self.var2index = dict()
            self.state = bnp.zeros((len(self.vars), self.num), dtype=bnp.float_)
            for i, (k, v) in enumerate(self.vars.items()):
                self.state[i] = v
                self.var2index[k] = i
        else:
            self.var2index = None
            self.state = dict()
            for k, v in self.vars.items():
                self.state[k] = bnp.ones(self.num, dtype=bnp.float_) * v
        self.S = self.state

        # parameters and "P"
        # -------------------
        assert isinstance(pars_updates, dict), '"pars_updates" must be a dict.'
        for k, v in pars_updates:
            val_size = bnp.size(v)
            if val_size != 1:
                if val_size != self.num:
                    raise ValueError('The size of parameter "{k}" is wrong, "{s}" != 1 and '
                                     '"{s}" != group.num.'.format(k=k, s=val_size))
            self.pars[k] = v

        if profile.is_numba_bk():
            max_size = bnp.max([bnp.size(v) for v in self.pars.values()])
            if max_size > 1:
                self.P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float64[:])
                for k, v in self.pars.items():
                    self.P[k] = bnp.ones(self.num, dtype=bnp.float_) * v
            else:
                self.P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float64)
                for k, v in self.pars.items():
                    self.P[k] = v
        else:
            self.P = self.pars

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

