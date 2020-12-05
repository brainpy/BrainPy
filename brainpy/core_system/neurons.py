# -*- coding: utf-8 -*-

import typing

from .base import BrainEnsemble
from .base import BrainType
from .constants import _NEU_GROUP
from .. import numpy as np
from ..errors import ModelDefError
from ..errors import ModelUseError

__all__ = [
    'NeuType',
    'NeuGroup',
    'NeuSubGroup',
]

_NEU_GROUP_NO = 0


class NeuType(BrainType):
    """Abstract Neuron Type.

    It can be defined based on a group of neurons or a single neuron.
    """

    def __init__(
            self,
            name: str,
            requires: dict,
            steps: typing.Union[typing.Callable, typing.List, typing.Tuple],
            mode: str = 'vector',
            heter_params_replace: typing.Dict = None,
            extra_functions: typing.Union[typing.List, typing.Tuple] = (),
            extra_attributes: typing.Dict[str, typing.Any] = None,
    ):

        if mode not in ['scalar', 'vector']:
            raise ModelDefError('NeuType only support "scalar" or "vector".')

        super(NeuType, self).__init__(requires=requires,
                                      steps=steps,
                                      name=name,
                                      mode=mode,
                                      heter_params_replace=heter_params_replace,
                                      extra_functions=extra_functions,
                                      extra_attributes=extra_attributes)


class NeuGroup(BrainEnsemble):
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

    def __init__(
            self,
            model: NeuType,
            geometry: typing.Union[typing.Tuple, typing.List, int],
            pars_update: typing.Dict = None,
            monitors: typing.Union[typing.List, typing.Tuple] = None,
            name: str = None
    ):
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
                raise ModelUseError('Do not support 3+ dimensional networks.')
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
            raise ModelUseError(f'{NeuGroup.__name__} receives an '
                                f'instance of {NeuType.__name__}, '
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
                raise ModelUseError(f'Index error, because the maximum number of neurons'
                                    f'is {self.num}, but got "item={item}".')
            d1_start, d1_end, d1_step = item, item + 1, 1
            check_slice(d1_start, d1_end, self.num)
            indices = self.indices[d1_start:d1_end:d1_step]
        elif isinstance(item, slice):
            d1_start, d1_end, d1_step = item.indices(self.num)
            check_slice(d1_start, d1_end, self.num)
            indices = self.indices[d1_start:d1_end:d1_step]
        elif isinstance(item, tuple):
            if not isinstance(self.geometry, (tuple, list)):
                raise ModelUseError(f'{self.name} has a 1D geometry, cannot use a tuple of slice.')
            if len(item) != 2:
                raise ModelUseError(f'Only support 2D network, cannot make {len(item)}D slice.')

            if isinstance(item[0], slice):
                d1_start, d1_end, d1_step = item[0].indices(self.geometry[0])
            elif isinstance(item[0], int):
                d1_start, d1_end, d1_step = item[0], item[0] + 1, 1
            else:
                raise ModelUseError("Only support slicing syntax or a single index.")
            check_slice(d1_start, d1_end, self.geometry[0])

            if isinstance(item[1], slice):
                d2_start, d2_end, d2_step = item[1].indices(self.geometry[1])
            elif isinstance(item[1], int):
                d2_start, d2_end, d2_step = item[1], item[1] + 1, 1
            else:
                raise ModelUseError("Only support slicing syntax or a single index.")
            check_slice(d1_start, d1_end, self.geometry[1])

            indices = self.indices[d1_start:d1_end:d1_step, d2_start:d2_end:d2_step]
        else:
            raise ModelUseError('Subgroups can only be constructed using slicing syntax, '
                                'a single index, or an array of contiguous indices.')

        return NeuSubGroup(source=self, indices=indices)


def check_slice(start, end, length):
    if start >= end:
        raise ModelUseError(f'Illegal start/end values for subgroup, {start}>={end}')
    if start >= length:
        raise ModelUseError(f'Illegal start value for subgroup, {start}>={length}')
    if end > length:
        raise ModelUseError(f'Illegal stop value for subgroup, {end}>{length}')
    if start < 0:
        raise ModelUseError('Indices have to be positive.')


class NeuSubGroup(object):
    """Subset of a `NeuGroup`.
    """

    def __init__(self,
                 source: NeuGroup,
                 indices: np.ndarray):
        try:
            assert isinstance(source, NeuGroup)
        except AssertionError:
            raise ModelUseError('NeuSubGroup only support an instance of NeuGroup.')

        self.source = source
        self.indices = indices
        self.num = np.size(indices)

    def __getattr__(self, item):
        if item in ['source', 'indices', 'num']:
            return getattr(self, item)
        else:
            return getattr(self.source, item)
