# -*- coding: utf-8 -*-


from brainpy.rnn.runner import RNNRunner
from brainpy.rnn.base import Node

__all__ = [
  'RNNTrainer',
]


class RNNTrainer(RNNRunner):
  def __init__(self, target, monitors, jit=True, dyn_vars=None,
               numpy_mon_after_run=True, progress_bar=True ):
    super(RNNTrainer, self).__init__(target=target,
                                     monitors=monitors,
                                     jit=jit,
                                     dyn_vars=dyn_vars,
                                     numpy_mon_after_run=numpy_mon_after_run,
                                     progress_bar=progress_bar)

  def train(self, X, Y,
            forced_states=None, reset=False, init_states=None,
            learn_every=1, ):
    pass
