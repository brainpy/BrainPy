# -*- coding: utf-8 -*-

from brainpy.simulation.base import DynamicSystem

__all__ = [
    'Soma'
]

_Soma_NO = 0


class Soma(DynamicSystem):
    """Soma object for neuron modeling.

    """

    def __init__(self, name, **kwargs):
        if name is None:
            global _Soma_NO
            name = f'Soma{_Soma_NO}'
            _Soma_NO += 1
        super(Soma, self).__init__(name=name, **kwargs)
