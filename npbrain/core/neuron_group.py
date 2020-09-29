# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import OrderedDict

from .types import ObjState
from .base import BaseType
from .base import BaseEnsemble
from .. import _numpy as np
from .. import profile
from ..utils import helper

__all__ = [
    'NeuType',
    'NeuGroup',
]

_neu_no = 0


class NeuType(BaseType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """
    def __init__(self, create_func, group_based=True, name=None):
        super(NeuType, self).__init__(create_func=create_func, name=name, group_based=group_based, type_='neu')


class NeuGroup(BaseEnsemble):
    """Neuron Group.
    """

    def __init__(self, model, geometry, monitors=None, vars_init=None, pars_update=None, name=None):
        assert isinstance(model, NeuType), 'Must provide an instance of NeuType class.'
        self.model = model

        # name
        # ----
        if name is None:
            global _neu_no
            self.name = f'NeuGroup{_neu_no}'
            _neu_no += 1
        else:
            self.name = name

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
        num = int(np.prod(geometry))
        self.num = num
        self.geometry = geometry

        # parameters and "P"
        # -------------------
        pars_update = dict() if pars_update is None else pars_update
        assert isinstance(pars_update, dict), '"pars_update" must be a dict.'
        parameters = deepcopy(self.model.parameters)
        for k, v in pars_update:
            val_size = np.size(v)
            if val_size != 1:
                if val_size != num:
                    raise ValueError(f'The size of parameter "{k}" is wrong, "{val_size}" != 1 '
                                     f'and "{val_size}" != {num}.')
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

        # variables and "state" ("S")
        # ----------------------------
        vars_init = dict() if vars_init is None else vars_init
        assert isinstance(vars_init, dict), '"vars_init" must be a dict.'
        variables = deepcopy(self.model.variables)
        for k, v in vars_init:
            if k not in self.model.variables:
                raise KeyError(f'variable "{k}" is not defined in "{self.model.name}".')
            variables[k] = v
        self.vars_init = vars_init
        self.ST = ObjState(variables)(num)

        # step functions
        # ----------------
        if profile.is_numpy_bk():
            func_return = self.model.create_func(**pars_update)
            step_funcs = func_return['step_funcs']
            if callable(step_funcs):
                step_funcs = [step_funcs, ]
            elif isinstance(step_funcs, (tuple, list)):
                step_funcs = list(step_funcs)
            else:
                raise ValueError('"step_funcs" must be a callable, or a list/tuple of callable functions.')
            self.step_funcs = step_funcs

        elif profile.is_numba_bk():
            raise NotImplementedError

        else:
            raise NotImplementedError

        for func in step_funcs:
            func_name = func.__name__
            setattr(self, func_name, func)

        # monitors
        # ----------
        self.mon = helper.Dict()
        self._mon_vars = monitors
        if monitors is not None:
            for k in monitors:
                self.mon[k] = np.empty((1, 1), dtype=np.float_)

    @property
    def _keywords(self):
        kws = ['model', 'num', 'geometry', 'ST', 'vars_init', 'PA',
               'mon', '_mon_vars', 'step_funcs']
        if hasattr(self, 'model'):
            return kws + self.model.step_names
        else:
            return kws

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in NeuGroup, please change another name.')
        super(NeuGroup, self).__setattr__(key, value)

