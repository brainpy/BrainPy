# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any, Union

from brainpy.errors import UnsupportedError
from brainpy.nn.base import Node, Network
from brainpy.types import Tensor
from .rnn_runner import RNNRunner

__all__ = [
  'RNNTrainer',
]


class RNNTrainer(RNNRunner):
  """Structural Trainer for Models with Recurrent Dynamics."""

  train_nodes: Sequence[Node]  # need to be initialized by subclass
  train_pars: Dict[str, Any]  # need to be initialized by subclass

  def __init__(self, target, **kwargs):
    super(RNNTrainer, self).__init__(target=target, **kwargs)

  def fit(
      self,
      train_data: Any,
      test_data: Any,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Union[Tensor, Dict[str, Tensor]] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
      reset: bool = False,
      shared_kwargs: Dict = None
  ):  # need to be implemented by subclass
    raise NotImplementedError('Must implement the fit function. ')

  def _get_trainable_nodes(self):
    # check trainable nodes
    if isinstance(self.target, Network):
      train_nodes = [node for node in self.target.lnodes if node.trainable]
    elif isinstance(self.target, Node):
      train_nodes = [self.target]
    else:
      raise UnsupportedError('Must be a brainpy.nn.Node instance, '
                             f'while we got {type(self.target)}: {self.target}')
    return train_nodes



