# -*- coding: utf-8 -*-

from typing import Union, Dict

from brainpy.types import Tensor
from .rnn_runner import RNNRunner

__all__ = [
  'RNNTrainer',
]


class RNNTrainer(RNNRunner):
  def __init__(self, target, monitors, jit=True, dyn_vars=None,
               numpy_mon_after_run=True, progress_bar=True):
    super(RNNTrainer, self).__init__(target=target,
                                     monitors=monitors,
                                     jit=jit,
                                     dyn_vars=dyn_vars,
                                     numpy_mon_after_run=numpy_mon_after_run,
                                     progress_bar=progress_bar)

  def train(self,
            xs: Union[Tensor, Dict[str, Tensor]],
            ys: Union[Tensor, Dict[str, Tensor]],
            forced_states: Dict[str, Tensor] = None,
            initial_states: Dict[str, Tensor] = None,
            initial_feedbacks: Dict[str, Tensor] = None,
            reset=False,
            learn_every=1, ):
    pass
