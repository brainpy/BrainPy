# -*- coding: utf-8 -*-


from brainpy.simulation.connectivity.base import TwoEndConnector

try:
  import numba as nb
except ModuleNotFoundError:
  nb = None

__all__ = [
]


class CompleteGraph(TwoEndConnector):
  pass
