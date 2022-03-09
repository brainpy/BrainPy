# -*- coding: utf-8 -*-

from typing import Union, Dict

from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'FORCELearning'
]


class FORCELearning(RNNTrainer):
  """Force learning."""

  def __init__(self, target, alpha=1., **kwargs):
    super(FORCELearning, self).__init__(target=target, **kwargs)

    self.train_nodes = self._get_trainable_nodes()
    self._check_interface('__force_init__')
    self._check_interface('__force_train__')
    self.train_pars = dict(alpha=alpha)

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],
          ys: Union[Tensor, Dict[str, Tensor]],
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None,
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          reset=False, ):
    pass
