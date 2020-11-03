# -*- coding: utf-8 -*-

from collections.abc import Sequence

from .base import BaseEnsemble
from .base import BaseType
from .constants import _NEU_GROUP
from .. import numpy as np
from ..errors import ModelUseError

__all__ = [
    'NeuType',
    'NeuGroup',
    'NeuSubGroup',
]

_NEU_GROUP_NO = 0


class NeuType(BaseType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(self, name, requires, steps, vector_based=True, heter_params_replace=None):
        super(NeuType, self).__init__(requires=requires, steps=steps, name=name, vector_based=vector_based,
                                      heter_params_replace=heter_params_replace)


class NeuGroup(BaseEnsemble):
    """Neuron Group.

    Parameters
    ----------
    model : NeuType
        The instantiated neuron type model.
    geometry : int, tuple
        The neuron group geometry.
    pars_update : dict, None
        Parameters to update.
    monitors : list, tuple, None
        Variables to monitor.
    name : str, None
        The name of the neuron group.
    """

    def __init__(self, model, geometry, pars_update=None, monitors=None, name=None):
        # name
        # -----
        if name is None:
            global _NEU_GROUP_NO
            name = f'NG{_NEU_GROUP_NO}'
            _NEU_GROUP_NO += 1
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

        # model
        # ------
        try:
            assert isinstance(model, NeuType)
        except AssertionError:
            raise ModelUseError(f'{NeuGroup.__name__} receives an instance of {NeuType.__name__}, '
                                f'not {type(model).__name__}.')

        # initialize
        # ----------
        super(NeuGroup, self).__init__(model=model,
                                       pars_update=pars_update,
                                       name=name,
                                       num=num,
                                       monitors=monitors,
                                       cls_type=_NEU_GROUP)

        # ST
        # --
        self.ST = self.requires['ST'].make_copy(num)

    @property
    def _keywords(self):
        return super(NeuGroup, self)._keywords + ['geometry', ]

    def __getitem__(self, item):
        """Return a subset of neuron group.

        Parameters
        ----------
        item : slice, int, sequence

        Returns
        -------
        sub_group : NeuSubGroup
            The subset of the neuron group.
        """

        # get the start and stop value
        # ----------------------------

        if isinstance(item, slice):
            start, end, step = item.indices(self.num)
        elif isinstance(item, int):
            try:
                assert item < self.num
            except AssertionError:
                raise ModelUseError(f'Index error, because the maximum number of neurons'
                                    f'is {self.num}, but got "item={item}".')
            start = item
            end = item + 1
            step = 1
        elif isinstance(item, (Sequence, np.ndarray)):
            if not (len(item) > 0 and np.all(np.diff(item) == 1)):
                raise ModelUseError('Subgroups can only be constructed using contiguous indices.')
            start = int(item[0])
            end = int(item[-1]) + 1
            step = 1
        else:
            raise ModelUseError('Subgroups can only be constructed using slicing syntax, '
                                'a single index, or an array of contiguous indices.')
        if step != 1:
            raise ModelUseError('Subgroups have to be contiguous.')
        if start >= end:
            raise ModelUseError(f'Illegal start/end values for subgroup, {start}>={end}')
        if start >= self.num:
            raise ModelUseError(f'Illegal start value for subgroup, {start}>={self.num}')
        if end > self.num:
            raise ModelUseError(f'Illegal stop value for subgroup, {end}>{self.num}')
        if start < 0:
            raise ModelUseError('Indices have to be positive.')

        return NeuSubGroup(self, start, end)


class NeuSubGroup(object):
    """Subset of a `NeuGroup`.

    """
    def __init__(self, source, start, end):
        try:
            assert isinstance(source, NeuGroup)
        except AssertionError:
            raise ModelUseError('NeuSubGroup only support an instance of NeuGroup.')

        self.source = source
        self.start = start
        self.end = end

    def __getattr__(self, item):
        if item in ['source', 'start', 'end']:
            return getattr(self, item)
        else:
            return getattr(self.source, item)
