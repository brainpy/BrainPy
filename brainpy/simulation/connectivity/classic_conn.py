# -*- coding: utf-8 -*-


import numpy as np

from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import TwoEndConnector

try:
  import numba as nb
except ModuleNotFoundError:
  nb = None

__all__ = [
]


class CompleteGraph(TwoEndConnector):
  pass
