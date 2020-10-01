# -*- coding: utf-8 -*-

from .base import BaseEnsemble
from .base import BaseType
from .types import ObjState
from .. import _numpy as np

__all__ = [
    'NeuType',
    'NeuGroup',
]

_neu_no = 0


class NeuType(BaseType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(self, name, create_func, group_based=True):
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
                                       vars_init=vars_init, monitors=monitors, cls_type='neu_group')

        # ST
        # --
        self.ST = ObjState(self.vars_init)(num)

    @property
    def _keywords(self):
        return super(NeuGroup, self)._keywords + ['geometry', ]
