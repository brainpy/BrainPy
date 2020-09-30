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
        # model
        # ------
        assert isinstance(model, NeuType), 'Must provide an instance of NeuType class.'

        # name
        # -----
        if name is None:
            global _neu_no
            name = f'NeuGroup{_neu_no}'
            _neu_no += 1
        else:
            name = name

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
        self.geometry = geometry

        # initialize
        # ----------
        super(NeuGroup, self).__init__(model=model, name=name, num=num, pars_update=pars_update,
                                       vars_init=vars_init, monitors=monitors)

        # ST
        # --
        self.ST = ObjState(self.vars_init)(num)

        # model update schedule
        # ---------------------
        self._schedule = ['input', 'step_funcs', 'monitor']

    @property
    def _keywords(self):
        kws = ['model', 'num', 'geometry', 'ST', 'PA',
               'vars_init', 'pars_update',
               'mon', '_mon_vars', 'step_funcs', '_schedule']
        if hasattr(self, 'model'):
            return kws + self.model.step_names
        else:
            return kws

    def __setattr__(self, key, value):
        if key in self._keywords:
            if hasattr(self, key):
                raise KeyError(f'"{key}" is a keyword in NeuGroup, please change another name.')
        super(NeuGroup, self).__setattr__(key, value)

    def _add_input(self, keys, values, types):
        has_iter = False
        for t in types:
            assert t in ['iter', 'fix'], 'Only support inputs with "iter" and "fix" types.'
            if t == 'iter':
                has_iter = True

        if profile.is_numpy_bk():
            if has_iter:
                code = '\ndef add_input(i):'
            else:
                code = '\ndef add_input():'



