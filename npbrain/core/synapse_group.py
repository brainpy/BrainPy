# -*- coding: utf-8 -*-

from .. import _numpy as bnp
from .. import profile
from ..utils.helper import Dict
from collections import OrderedDict
from .common_func import BaseType
from .common_func import numbify_func
from .connectivity import Connector
from .connectivity import FixedProb
from .neuron_group import NeuGroup

from copy import deepcopy

__all__ = [
    'SynConn'
]


class SynConn(BaseType):
    def __init__(self, create_func, name=None):
        super(SynConn, self).__init__(create_func=create_func, name=name, type_='syn')

    def __call__(self, pre=None, post=None, conn=FixedProb(prob=0.1), num=None,
                 delay=0., monitors=None, vars_init=None, pars_update=None):
        if pre is not None and post is not None:
            # check
            # ------
            assert isinstance(pre, NeuGroup), '"pre" must be an instance of NeuGroup.'
            assert isinstance(post, NeuGroup), '"post" must be an instance of NeuGroup.'

            # connections
            # ------------
            if isinstance(conn, Connector):
                self.pre_idx, self.post_idx = conn(pre.geometry, post.geometry)
            elif isinstance(conn, bnp.ndarray):
                assert bnp.ndim(conn) == 2, f'"conn" must be a 2D array, not {bnp.ndim(conn)}D.'
                shape = bnp.shape(conn)
                assert shape[0] == pre.num and shape[1] == post.num, f'The shape of "conn" must be ({pre.num}, {post.num})'
                self.pre_idx, self.post_idx = [], []
                for i in enumerate(pre.num):
                    idx = bnp.where(conn[i] > 0)[0]
                    self.pre_idx.extend([i * len(idx)])
                    self.post_idx.extend(idx)
                self.pre_idx = bnp.asarray(self.pre_idx, dtype=bnp.int_)
                self.post_idx = bnp.asarray(self.post_idx, dtype=bnp.int_)
            else:
                assert isinstance(conn, dict), '"conn" only support "dict" or a 2D "array".'
                assert 'i' in conn, '"conn" must provide "i" item.'
                assert 'j' in conn, '"conn" must provide "j" item.'
                self.pre_idx = bnp.asarray(conn['i'], dtype=bnp.int_)
                self.post_idx = bnp.asarray(conn['j'], dtype=bnp.int_)

            # essential
            # ---------
            self.num_pre = pre.num
            self.num_post = post.num
            self.num = len(self.pre_idx)

        else:
            assert num is not None, '"num" must be provided when "pre" and "post" are none.'
            assert 0 < num, '"num" must be a positive number.'
            self.num = num

        # variables and "state" ("S")
        # ------------------------
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.variables)
        for k, v in vars_init:
            if k not in self.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.name}".')
            variables[k] = v

        if profile.is_numba_bk():
            self.var2index = Dict()
            self.state = Dict()
            self._state_mat = bnp.zeros((len(variables), self.num), dtype=bnp.float_)
            for i, (k, v) in enumerate(variables.items()):
                self._state_mat[i] = v
                self.state[k] = self._state_mat[i]
                self.var2index[k] = i
        else:
            self.var2index = None
            self.state = Dict()
            for k, v in variables.items():
                self.state[k] = bnp.ones(self.num, dtype=bnp.float_) * v
        self.S = self.state

        # parameters and "P"
        # -------------------
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.parameters)
        for k, v in pars_update:
            val_size = bnp.size(v)
            if val_size != 1:
                if val_size != self.num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != group.num.')
            parameters[k] = v

        if profile.is_numba_bk():
            import numba as nb
            max_size = max([bnp.size(v) for v in parameters.values()])
            if max_size > 1:
                self.P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float_[:])
                for k, v in parameters.items():
                    self.P[k] = bnp.ones(self.num, dtype=bnp.float_) * v
                self.parameters = self.P
            else:
                self.P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float_)
                for k, v in parameters.items():
                    self.P[k] = v
                self.parameters = self.P
        else:
            self.P = self.parameters = parameters

        # define update functions
        # -------------------------
        self.func_returns = self.create_func(**parameters)
        step_funcs = self.func_returns['step_funcs']
        if callable(step_funcs):
            self.update_funcs = [step_funcs, ]
        elif isinstance(step_funcs, (tuple, list)):
            self.update_funcs = list(step_funcs)
        else:
            raise ValueError('"step_funcs" must be a callable, or a list/tuple of callable functions.')

        # delay and delay state
        # ----------------------
        if delay is None:
            delay_len = 1
        elif isinstance(delay, (int, float)):
            dt = profile.get_dt()
            delay_len = int(bnp.ceil(delay / dt)) + 1
        else:
            raise ValueError("NumpyBrain currently doesn't support other kinds of delay.")
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
            def update(i):
                for k in self._mon_vars:
                    if k == 'g_out':
                        self.mon[k][i] = self.delay_state[self.state['delay_out']]
                    elif k == 'g_in':
                        self.mon[k][i] = self.delay_state[self.state['delay_in']]
                    else:
                        self.mon[k][i] = self.state[k]

            self._mon_update = update
            self.update_funcs.append(update)

        # update function
        self.update_funcs.append(self.update_delay_indices)
        self.update_funcs = tuple(self.update_funcs)

    def update_delay_indices(self):
        # in_index
        self.state['delay_in'] = (self.state['delay_in'] + 1) % self.delay_len
        # out_index
        self.state['delay_out'] = (self.state['delay_out'] + 1) % self.delay_len
