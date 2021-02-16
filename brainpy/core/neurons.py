# -*- coding: utf-8 -*-

import numpy as np

from . import base
from . import constants
from . import utils
from .. import errors

__all__ = [
    'NeuType',
    'NeuGroup',
    'NeuSubGroup',
]

_NEU_GROUP_NO = 0


class NeuType(base.ObjType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(self, name, ST, steps, mode='vector', requires=None, hand_overs=None, ):
        if mode not in [constants.SCALAR_MODE, constants.VECTOR_MODE]:
            raise errors.ModelDefError('NeuType only support "scalar" or "vector".')

        super(NeuType, self).__init__(ST=ST,
                                      requires=requires,
                                      steps=steps,
                                      name=name,
                                      mode=mode,
                                      hand_overs=hand_overs)


class NeuGroup(base.Ensemble):
    """Neuron Group.

    Parameters
    ----------
    model : NeuType
        The instantiated neuron type model.
    geometry : int, tuple
        The neuron group geometry.
    pars_update : dict
        Parameters to update.
    monitors : list, tuple
        Variables to monitor.
    name : str
        The name of the neuron group.
    """

    def __init__(self, model, geometry, monitors=None, name=None, satisfies=None, pars_update=None, ):
        # name
        # -----
        if name is None:
            global _NEU_GROUP_NO
            name = f'NeuGroup{_NEU_GROUP_NO}'
            _NEU_GROUP_NO += 1
        else:
            name = name

        # num and geometry
        # -----------------
        if isinstance(geometry, (int, float)):
            geometry = num = int(geometry)
            self.indices = np.asarray(np.arange(int(geometry)), dtype=np.int_)
        elif isinstance(geometry, (tuple, list)):
            if len(geometry) == 1:
                geometry = num = geometry[0]
                indices = np.arange(num)
            elif len(geometry) == 2:
                height, width = geometry[0], geometry[1]
                num = height * width
                indices = np.arange(num).reshape((height, width))
            else:
                raise errors.ModelUseError('Do not support 3+ dimensional networks.')
            self.indices = np.asarray(indices, dtype=np.int_)
        else:
            raise ValueError()
        self.geometry = geometry
        self.size = np.size(self.indices)

        # model
        # ------
        try:
            assert isinstance(model, NeuType)
        except AssertionError:
            raise errors.ModelUseError(f'{NeuGroup.__name__} receives an '
                                       f'instance of {NeuType.__name__}, '
                                       f'not {type(model).__name__}.')

        # initialize
        # ----------
        super(NeuGroup, self).__init__(model=model,
                                       pars_update=pars_update,
                                       name=name,
                                       num=num,
                                       monitors=monitors,
                                       cls_type=constants.NEU_GROUP_TYPE,
                                       satisfies=satisfies)

        # ST
        # --
        self.ST = self.model.ST.make_copy(num)

    def __getitem__(self, item):
        """Return a subset of neuron group.

        Parameters
        ----------
        item : slice, int, tuple of slice

        Returns
        -------
        sub_group : NeuSubGroup
            The subset of the neuron group.
        """

        if isinstance(item, int):
            try:
                assert item < self.num
            except AssertionError:
                raise errors.ModelUseError(f'Index error, because the maximum number of neurons'
                                           f'is {self.num}, but got "item={item}".')
            d1_start, d1_end, d1_step = item, item + 1, 1
            utils.check_slice(d1_start, d1_end, self.num)
            indices = self.indices[d1_start:d1_end:d1_step]
        elif isinstance(item, slice):
            d1_start, d1_end, d1_step = item.indices(self.num)
            utils.check_slice(d1_start, d1_end, self.num)
            indices = self.indices[d1_start:d1_end:d1_step]
        elif isinstance(item, tuple):
            if not isinstance(self.geometry, (tuple, list)):
                raise errors.ModelUseError(f'{self.name} has a 1D geometry, cannot use a tuple of slice.')
            if len(item) != 2:
                raise errors.ModelUseError(f'Only support 2D network, cannot make {len(item)}D slice.')

            if isinstance(item[0], slice):
                d1_start, d1_end, d1_step = item[0].indices(self.geometry[0])
            elif isinstance(item[0], int):
                d1_start, d1_end, d1_step = item[0], item[0] + 1, 1
            else:
                raise errors.ModelUseError("Only support slicing syntax or a single index.")
            utils.check_slice(d1_start, d1_end, self.geometry[0])

            if isinstance(item[1], slice):
                d2_start, d2_end, d2_step = item[1].indices(self.geometry[1])
            elif isinstance(item[1], int):
                d2_start, d2_end, d2_step = item[1], item[1] + 1, 1
            else:
                raise errors.ModelUseError("Only support slicing syntax or a single index.")
            utils.check_slice(d1_start, d1_end, self.geometry[1])

            indices = self.indices[d1_start:d1_end:d1_step, d2_start:d2_end:d2_step]
        else:
            raise errors.ModelUseError('Subgroups can only be constructed using slicing syntax, '
                                       'a single index, or an array of contiguous indices.')

        return NeuSubGroup(source=self, indices=indices)


class NeuSubGroup(object):
    """Subset of a `NeuGroup`.
    """

    def __init__(self, source, indices):
        if not isinstance(source, NeuGroup):
            raise errors.ModelUseError('NeuSubGroup only support an instance of NeuGroup.')

        self.source = source
        self.indices = indices
        self.num = np.size(indices)

    def __getattr__(self, item):
        if item in ['source', 'indices', 'num']:
            return getattr(self, item)
        else:
            return getattr(self.source, item)
