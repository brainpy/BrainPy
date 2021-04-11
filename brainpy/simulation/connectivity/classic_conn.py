# -*- coding: utf-8 -*-


import numpy as np

from brainpy import backend
from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import Connector

try:
    import numba as nb
except ModuleNotFoundError:
    nb = None

__all__ = [
]

class CompleteGraph(Connector):
    pass

