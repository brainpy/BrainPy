# -*- coding: utf-8 -*-

from collections import OrderedDict
import inspect
from .common_func import numbify_func
from .common_func import BaseType
from .. import _numpy as bnp
from .. import profile
from copy import deepcopy
from ..utils import helper

_group_no = 0

__all__ = [
    'NeuronGroup',
    'NeuronType',
]


class NeuronGroup(object):
    '''

    Handle

    '''
    def __init__(self):
        pass


class NeuronType(BaseType):
    def __init__(self, create_func, name=None):
        super(NeuronType, self).__init__(create_func=create_func, name=name, type_='neu')

    def __call__(self, geometry, monitors=None, vars_init=None, pars_updates=None):
        # num and geometry
        # -----------------
        if isinstance(geometry, (int, float)):
            geometry = (1, int(geometry))
        elif isinstance(geometry, (tuple, list)):
            if len(geometry) == 1:
                height, width = 1, geometry[0]
            elif len(geometry) == 2:
                height, width = geometry[0], geometry[1]
            else:
                raise ValueError('Do not support 3+ dimensional networks.')
            geometry = (height, width)
        else:
            raise ValueError()
        num = int(bnp.prod(geometry))

        # variables and "state" ("S")
        # ----------------------------
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.variables)
        for k, v in vars_init:
            if k not in self.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.name}".')
            variables[k] = v

        if profile.is_numba_bk():
            import numba as nb

            var2index = dict()
            state = bnp.zeros((len(variables), num), dtype=bnp.float_)
            for i, (k, v) in enumerate(variables.items()):
                state[i] = v
                var2index[k] = i
            var2index['not_ref'] = -5
            var2index['above_th'] = -4
            var2index['spike'] = -3
            var2index['sp_time'] = -2
            var2index['input'] = -1
        else:
            var2index = None
            state = dict()
            for k, v in variables.items():
                state[k] = bnp.ones(num, dtype=bnp.float_) * v

        # parameters and "P"
        # -------------------
        assert isinstance(pars_updates, dict), '"pars_updates" must be a dict.'
        parameters = deepcopy(self.parameters)
        for k, v in pars_updates:
            val_size = bnp.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     'and "{val_size}" != {num}.')
            parameters[k] = v

        if profile.is_numba_bk():
            max_size = bnp.max([bnp.size(v) for v in parameters.values()])
            if max_size > 1:
                P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float64[:])
                for k, v in parameters.items():
                    P[k] = bnp.ones(num, dtype=bnp.float_) * v
            else:
                P = nb.typed.Dict(key_type=nb.types.unicode_type, value_type=nb.types.float64)
                for k, v in parameters.items():
                    P[k] = v
        else:
            P = parameters

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
                        self.mon[k][i] = self.state[self.var2index[k]]
            else:
                def update(i):
                    for k in self._mon_vars:
                        self.mon[k][i] = self.state[k]
            self._mon_update = update

