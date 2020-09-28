# -*- coding: utf-8 -*-

from collections import OrderedDict

from .. import _numpy as np
from .. import profile
from ..utils.helper import Dict
from .base import BaseType
from .base import BaseGroup
from .connectivity import Connector
from .connectivity import FixedProb
from .neuron_group import NeuGroup

from copy import deepcopy

__all__ = [
    'SynType',
    'SynConn',
]


class SynType(BaseType):
    """Abstract Synapse Type.

    It can be defined based on a collection of synapses or a single synapse model.
    """
    def __init__(self, create_func, group_based=True, name=None):
        super(SynType, self).__init__(create_func=create_func, name=name, group_based=group_based, type_='syn')


class SynConn(BaseGroup):
    """Synaptic connections.

    """
    def __init__(self, model, pre_group=None, post_group=None, conn=None, num=None,
                 delay=0., monitors=None, vars_init=None, pars_update=None):
        assert isinstance(model, SynType), 'Must provide a SynType class.'
        self.model = model

        if pre_group is not None and post_group is not None:
            # check
            # ------
            assert isinstance(pre_group, NeuGroup), '"pre" must be an instance of NeuGroup.'
            assert isinstance(post_group, NeuGroup), '"post" must be an instance of NeuGroup.'

            # connections
            # ------------
            if conn is None:
                pass
            elif isinstance(conn, Connector):
                self.pre_idx, self.post_idx = conn(pre_group.geometry, post_group.geometry)
            elif isinstance(conn, np.ndarray):
                assert np.ndim(conn) == 2, f'"conn" must be a 2D array, not {np.ndim(conn)}D.'
                shape = np.shape(conn)
                assert shape[0] == pre_group.num and shape[1] == post_group.num, \
                    f'The shape of "conn" must be ({pre_group.num}, {post_group.num})'
                self.pre_idx, self.post_idx = [], []
                for i in enumerate(pre_group.num):
                    idx = np.where(conn[i] > 0)[0]
                    self.pre_idx.extend([i * len(idx)])
                    self.post_idx.extend(idx)
                self.pre_idx = np.asarray(self.pre_idx, dtype=np.int_)
                self.post_idx = np.asarray(self.post_idx, dtype=np.int_)
            else:
                assert isinstance(conn, dict), '"conn" only support "dict" or a 2D "array".'
                assert 'i' in conn, '"conn" must provide "i" item.'
                assert 'j' in conn, '"conn" must provide "j" item.'
                self.pre_idx = np.asarray(conn['i'], dtype=np.int_)
                self.post_idx = np.asarray(conn['j'], dtype=np.int_)

            # essential
            # ---------
            self.pre_group = pre_group
            self.post_group = post_group
            self.num_pre = pre_group.num
            self.num_post = post_group.num
            self.num = len(self.pre_idx)

        else:
            assert num is not None, '"num" must be provided when "pre" and "post" are none.'
            assert 0 < num, '"num" must be a positive number.'
            self.num = num

        # delay
        # -------
        if delay is None:
            delay_len = 1
        elif isinstance(delay, (int, float)):
            dt = profile.get_dt()
            delay_len = int(np.ceil(delay / dt)) + 1
        else:
            raise ValueError("NumpyBrain currently doesn't support other kinds of delay.")
        self.delay_len = delay_len
        self.delay_in = 0
        self.delay_out = self.delay_len - 1

        # variables and "state" ("S")
        # ------------------------
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.model.variables)
        for k, v in vars_init:
            if k not in self.model.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.model.name}".')
            variables[k] = v

        if profile.is_numba_bk():
            self.var2index = Dict()
            self.state = Dict()
            self._state_mat = np.zeros((len(variables), self.delay_len, self.num), dtype=np.float_)
            for i, (k, v) in enumerate(variables.items()):
                self._state_mat[i] = v
                self.state[k] = self._state_mat[i]
                self.var2index[k] = i
        else:
            self.var2index = None
            self.state = Dict()
            for k, v in variables.items():
                self.state[k] = np.ones((self.delay_len, self.num), dtype=np.float_) * v
        self.S = self.state

        # parameters and "P"
        # -------------------
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.parameters)
        for k, v in pars_update:
            val_size = np.size(v)
            if val_size != 1:
                if val_size != self.num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != group.num.')
            parameters[k] = v

        if profile.is_numba_bk():
            import numba as nb
            max_size = max([np.size(v) for v in parameters.values()])
            if max_size > 1:
                self.P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float_[:])
                for k, v in parameters.items():
                    self.P[k] = np.ones(self.num, dtype=np.float_) * v
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
        if profile.is_numpy_bk():
            func_returns = self.model.create_func(**parameters)
            step_funcs = func_returns['step_funcs']
            if callable(step_funcs):
                step_funcs = [step_funcs, ]
            elif isinstance(step_funcs, (tuple, list)):
                step_funcs = list(step_funcs)
            else:
                raise ValueError('"step_funcs" must be a callable, or a list/tuple of callable functions.')
        else:
            raise NotImplementedError('Please wait, other backends currently are not supported.')

        # monitors
        # ----------
        self.mon = dict()
        self._mon_vars = monitors
        self._mon_update = None

        if monitors is not None:
            for k in monitors:
                self.mon[k] = np.zeros((1, 1), dtype=np.float_)

            # generate function
            def mon_step_function(i):
                for k in self._mon_vars:
                    self.mon[k][i] = self.state[k][self.delay_in]

            self._mon_update = mon_step_function
            step_funcs.append(mon_step_function)

        # step functions
        # ---------------
        step_funcs.append(self.update_delay_indices)
        self.step_funcs = OrderedDict()
        for func in step_funcs:
            func_name = func.__name__
            self.step_funcs[func_name] = func

    def update_delay_indices(self):
        # in_index
        self.delay_in = (self.delay_in + 1) % self.delay_len
        # out_index
        self.delay_out = (self.delay_out + 1) % self.delay_len
