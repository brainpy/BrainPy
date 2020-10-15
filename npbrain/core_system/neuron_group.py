# -*- coding: utf-8 -*-

import inspect
from copy import deepcopy

from .base_objects import BaseEnsemble
from .base_objects import BaseType
from .base_objects import _NEU_GROUP
from .base_objects import _NEU_TYPE
from .base_objects import _ARG_KEYWORDS
from .types import NeuState
from ..tools import DictPlus
from .. import profile
from .. import numpy as np

__all__ = [
    'NeuType',
    'NeuGroup',
]

_NEU_NO = 0


class NeuType(BaseType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(self, name, requires, steps, vector_based=True):
        super(NeuType, self).__init__(requires=requires, steps=steps, name=name, vector_based=vector_based, type_=_NEU_TYPE)


class NeuGroup(BaseEnsemble):
    """Neuron Group.
    """

    def __init__(self, create_func, geometry, monitors=None, vars_init=None, pars_update=None, name=None):
        # name
        # -----
        if name is None:
            global _NEU_NO
            name = f'NeuGroup{_NEU_NO}'
            _NEU_NO += 1
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
        super(NeuGroup, self).__init__(create_func=create_func, name=name, num=num, pars_update=pars_update,
                                       vars_init=vars_init, monitors=monitors, cls_type=_NEU_GROUP)

        # ST
        # --
        self.ST = NeuState(self.vars_init)(num)

    @property
    def _keywords(self):
        return super(NeuGroup, self)._keywords + ['geometry', ]
