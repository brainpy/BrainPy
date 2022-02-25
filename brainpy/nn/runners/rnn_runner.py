# -*- coding: utf-8 -*-

from brainpy.nn.base import Node
from brainpy.running.runner import Runner

__all__ = [
  'RNNRunner',
]


class RNNRunner(Runner):
  def __init__(self, target, monitors, jit=True, dyn_vars=None,
               progress_bar=True, numpy_mon_after_run=True, ):
    super(RNNRunner, self).__init__(target=target, monitors=monitors,
                                    jit=jit, progress_bar=progress_bar,
                                    dyn_vars=dyn_vars)
    assert isinstance(target, Node)

    self.numpy_mon_after_run = numpy_mon_after_run

  def run(self, X, forced_states=None, reset=False, init_states=None):
    pass
