# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any, Union

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.errors import UnsupportedError
from brainpy.compat.nn.base import Node, Network
from brainpy.types import Tensor
from brainpy.tools.checking import check_dict_data
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

    # get all trainable nodes
    self.train_nodes = self._get_trainable_nodes()

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
      train_nodes = [self.target] if self.target.trainable else []
    else:
      raise UnsupportedError('Must be a brainpy.nn.Node instance, '
                             f'while we got {type(self.target)}: {self.target}')
    return train_nodes

  def _check_ys(self, ys, num_batch, num_step, move_axis=False):
    # output_shapes = {}
    # for node in self.train_nodes:
    #   name = self.target.entry_nodes[0].name
    #   output_shapes[name] = node.output_shape

    if isinstance(ys, (bm.ndarray, jnp.ndarray)):
      if len(self.train_nodes) == 1:
        ys = {self.train_nodes[0].name: ys}
      else:
        raise ValueError(f'The network {self.target} has {len(self.train_nodes)} '
                         f'training nodes, while we only got one target data.')
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    for key, val in ys.items():
      if val.ndim < 3:
        raise ValueError("Targets must be a tensor with shape of "
                         "(num_sample, num_time, feature_dim, ...), "
                         f"but we got {val.shape}")
      if val.shape[0] != num_batch:
        raise ValueError(f'Batch size of the target {key} does not match '
                         f'with the input data {val.shape[0]} != {num_batch}')
      if val.shape[1] != num_step:
        raise ValueError(f'The time step of the target {key} does not match '
                         f'with the input data {val.shape[1]} != {num_step})')
    if move_axis:
      # change shape to (num_time, num_sample, num_feature)
      ys = {k: bm.moveaxis(v, 0, 1) for k, v in ys.items()}
    return ys


