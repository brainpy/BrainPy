# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any

from brainpy.errors import UnsupportedError, NoImplementationError
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

  def _check_interface(self, name):
    for node in self.train_nodes:
      if not hasattr(node, name):
        raise NoImplementationError(f'The node \n\n{node}\n\n'
                                    f'is set to be trainable with {self.__class__.__name__} method. '
                                    f'However, it does not implement the required training '
                                    f'interface, "{name}()" function. ')
      if not callable(getattr(node, name)):
        raise NoImplementationError(f'The node \n\n{node}\n\n'
                                    f'is set to be trainable with {self.__class__.__name__} method. '
                                    f'However, the required training interface "{name}" is not a '
                                    f'callable function.')


