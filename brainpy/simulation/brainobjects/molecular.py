# -*- coding: utf-8 -*-

from brainpy.simulation.base import DynamicSystem

__all__ = [
    'Molecular'
]

_Molecular_NO = 0


class Molecular(DynamicSystem):
    """Molecular object for neuron modeling.

    """

    def __init__(self, name, **kwargs):
        if name is None:
            global _Molecular_NO
            name = f'Molecular{_Molecular_NO}'
            _Molecular_NO += 1
        super(Molecular, self).__init__(name=name, **kwargs)
