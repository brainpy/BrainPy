# -*- coding: utf-8 -*-

from brainpy.version2.dynsys import Dynamic
from brainpy.version2.mixin import SupportAutoDelay, ParamDesc

__all__ = [
    'NeuDyn', 'SynDyn', 'IonChaDyn',
]


class NeuDyn(Dynamic, SupportAutoDelay):
    """Neuronal Dynamics."""
    pass


class SynDyn(Dynamic, SupportAutoDelay, ParamDesc):
    """Synaptic Dynamics."""
    pass


class IonChaDyn(Dynamic):
    """Ion Channel Dynamics."""
    pass
