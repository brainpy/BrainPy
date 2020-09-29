# -*- coding: utf-8 -*-

from collections import OrderedDict

from .. import _numpy as np
from .. import profile
from ..utils.helper import Dict
from .base import BaseType
from .types import ObjState
from .base import BaseEnsemble
from .connectivity import Connector
from .connectivity import pre2syn
from .connectivity import post2syn
from .connectivity import FixedProb
from .neuron_group import NeuGroup

from copy import deepcopy

__all__ = [
    'SynType',
    'SynConn',
]


_syn_no = 0


class SynType(BaseType):
    """Abstract Synapse Type.

    It can be defined based on a collection of synapses or a single synapse model.
    """
    def __init__(self, create_func, group_based=True, name=None):
        super(SynType, self).__init__(create_func=create_func, name=name, group_based=group_based, type_='syn')


class SynConn(BaseEnsemble):
    """Synaptic connections.

    """
    def __init__(self, model, delay=0.,
                 pre_group=None, post_group=None, conn=None,
                 num=None,
                 monitors=None, vars_init=None, pars_update=None, name=None):
        assert isinstance(model, SynType), 'Must provide an instance of SynType class.'
        self.model = model

        # name
        # ----
        if name is None:
            global _syn_no
            self.name = f'SynConn{_syn_no}'
            _syn_no += 1
        else:
            self.name = name

        if pre_group is not None and post_group is not None:
            # check
            # ------
            assert isinstance(pre_group, NeuGroup), '"pre" must be an instance of NeuGroup.'
            assert isinstance(post_group, NeuGroup), '"post" must be an instance of NeuGroup.'

            # connections
            # ------------
            if conn is None:
                if isinstance(conn, Connector):
                    pre_idx, post_idx = conn(pre_group.geometry, post_group.geometry)
                elif isinstance(conn, np.ndarray):
                    assert np.ndim(conn) == 2, f'"conn" must be a 2D array, not {np.ndim(conn)}D.'
                    conn_shape = np.shape(conn)
                    assert conn_shape[0] == pre_group.num and conn_shape[1] == post_group.num, \
                        f'The shape of "conn" must be ({pre_group.num}, {post_group.num})'
                    pre_idx, post_idx = [], []
                    for i in enumerate(pre_group.num):
                        idx = np.where(conn[i] > 0)[0]
                        pre_idx.extend([i * len(idx)])
                        post_idx.extend(idx)
                    pre_idx = np.asarray(pre_idx, dtype=np.int_)
                    post_idx = np.asarray(post_idx, dtype=np.int_)
                else:
                    assert isinstance(conn, dict), '"conn" only support "dict" or a 2D "array".'
                    assert 'i' in conn, '"conn" must provide "i" item.'
                    assert 'j' in conn, '"conn" must provide "j" item.'
                    pre_idx = np.asarray(conn['i'], dtype=np.int_)
                    post_idx = np.asarray(conn['j'], dtype=np.int_)

                self.num = len(pre_idx)
                self.pre2syn = pre2syn(pre_idx, post_idx, pre_group.num)
                self.post2syn = post2syn(pre_idx, post_idx, post_group.num)

            else:
                assert num is not None, '"num" must be provided when "conn" are none.'
                assert 0 < num, '"num" must be a positive number.'
                self.num = num

            # essential
            # ---------
            self.pre_group = pre_group
            self.post_group = post_group

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
        self.dlen = delay_len  # delay length
        self.din = 0  # delay in position
        self.dout = self.dlen - 1  # delay out position

        # variables and "state" ("S")
        # ----------------------------
        vars_init = dict() if vars_init is None else vars_init
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.model.variables)
        for k, v in vars_init:
            if k not in self.model.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.model.name}".')
            variables[k] = v
        self.vars_init = variables
        self.ST = ObjState(variables)((delay_len, self.num))

        # parameters and "P"
        # -------------------
        pars_update = dict() if pars_update is None else pars_update
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.model.parameters)
        for k, v in pars_update:
            val_size = np.size(v)
            if val_size != 1:
                if val_size != self.num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != group.num.')
            parameters[k] = v
        self.pars_update = parameters

        if profile.is_numba_bk():
            import numba as nb
            max_size = max([np.size(v) for v in parameters.values()])
            if max_size > 1:
                self.PA = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float_[:])
                for k, v in parameters.items():
                    self.PA[k] = np.ones(self.num, dtype=np.float_) * v
            else:
                self.PA = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float_)
                for k, v in parameters.items():
                    self.PA[k] = v
        else:
            self.PA = parameters

        # step functions
        # ---------------
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

        for func in step_funcs:
            func_name = func.__name__
            setattr(self, func_name, func)

        step_funcs.append(self.update_delay_indices)
        self.step_funcs = step_funcs

        # monitors
        # ----------
        self.mon = dict()
        self._mon_vars = monitors

        if monitors is not None:
            for k in monitors:
                self.mon[k] = np.zeros((1, 1), dtype=np.float_)

    def update_delay_indices(self):
        # in_index
        self.din = (self.din + 1) % self.dlen
        # out_index
        self.dout = (self.dout + 1) % self.dlen

    @property
    def _keywords(self):
        kws = ['model', 'num', 'dlen', 'din', 'dout',
               'ST', 'PA', 'mon', '_mon_vars', 'step_funcs']
        if hasattr(self, 'model'):
            return kws + self.model.step_names
        else:
            return kws

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in SynConn, please change another name.')
        super(SynConn, self).__setattr__(key, value)

