# -*- coding: utf-8 -*-

from typing import Dict, Union

from brainpy.errors import NoImplementationError
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'OnlineTrainer',
  'FORCELearning',
]


class OnlineTrainer(RNNTrainer):
  pass


class FORCELearning(OnlineTrainer):
  """Force learning."""

  def __init__(self, target, alpha=1., **kwargs):
    super(FORCELearning, self).__init__(target=target, **kwargs)

    self.train_nodes = self._get_trainable_nodes()
    self._check_interface()
    self._check_interface()
    self.train_pars = dict(alpha=alpha)

  def fit(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      ys: Union[Tensor, Dict[str, Tensor]],
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Dict[str, Tensor] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
      reset=False,
  ):
    pass

  def _check_interface(self):
    for node in self.train_nodes:
      if hasattr(node.online_fit, 'not_implemented'):
        if node.online_fit.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_fit()" function. '
          )

