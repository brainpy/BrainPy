# -*- coding: utf-8 -*-

from brainpy.rnn.base import Node

__all__ = [
  'RNNRunner',
]


class RNNRunner(object):
  def __init__(self, target, monitors, jit=True, dyn_vars=None,
               numpy_mon_after_run=True, progress_bar=True):
    assert isinstance(target, Node)
    self.target = target

  def run(self, X, forced_states=None, reset=False, init_states=None):
    pass
