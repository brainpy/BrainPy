# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.simulation import utils
from brainpy.simulation.dynamic_system import DynamicSystem

__all__ = [
    'NeuGroup',
]

_NeuGroup_NO = 0


class NeuGroup(DynamicSystem):
    """Neuron Group.

    Parameters
    ----------
    size : int, tuple
        The neuron group geometry.
    monitors : list, tuple
        Variables to monitor.
    name : str
        The name of the neuron group.
    """

    def __init__(self, size, monitors=None, name=None, show_code=False):
        # name
        # -----
        if name is None:
            name = ''
        else:
            name = '_' + name
        global _NeuGroup_NO
        _NeuGroup_NO += 1
        name = f'NG{_NeuGroup_NO}{name}'

        # size
        # ----
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise errors.ModelDefError('size must be int, or a tuple/list of int.')
            if not isinstance(size[0], int):
                raise errors.ModelDefError('size must be int, or a tuple/list of int.')
            size = tuple(size)
        elif isinstance(size, int):
            size = (size,)
        else:
            raise errors.ModelDefError('size must be int, or a tuple/list of int.')
        self.size = size
        self.num = utils.size2len(size)

        # initialize
        # ----------
        super(NeuGroup, self).__init__(steps={'update': self.update},
                                       monitors=monitors,
                                       name=name,
                                       show_code=show_code)

    def update(self, *args):
        raise NotImplementedError
