# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dyn.runners import DSRunner
from brainpy.tools.checking import check_dict_data
from brainpy.train.base import TrainingSystem

__all__ = [
  'DSTrainer', 'DSRunner',
]


class DSTrainer(DSRunner):
  """Structural Trainer for Models with Recurrent Dynamics."""

  target: TrainingSystem
  train_nodes: Sequence[TrainingSystem]  # need to be initialized by subclass

  def __init__(
      self,
      target: TrainingSystem,
      **kwargs
  ):
    if not isinstance(target, TrainingSystem):
      raise TypeError(f'"target" must be an instance of {TrainingSystem.__name__}, '
                      f'but we got {type(target)}: {target}')
    super(DSTrainer, self).__init__(target=target, **kwargs)

    # jit
    self.jit['predict'] = self.jit.get('predict', True)
    self.jit['fit'] = self.jit.get('fit', True)

  def fit(
      self,
      train_data: Any,
      reset: bool = False,
      shared_kwargs: Dict = None
  ):  # need to be implemented by subclass
    raise NotImplementedError('Must implement the fit function. ')

  def _get_trainable_nodes(self):
    # check trainable nodes
    nodes = self.target.nodes(level=-1, include_self=True).subset(TrainingSystem).unique()
    return tuple([node for node in nodes.values() if node.trainable])

  def _check_ys(self, ys, num_batch, num_step, move_axis=False):
    if isinstance(ys, (bm.ndarray, jnp.ndarray)):
      if len(self.train_nodes) == 1:
        ys = {self.train_nodes[0].name: ys}
      else:
        raise ValueError(f'The network\n {self.target} \nhas {len(self.train_nodes)} '
                         f'training nodes, while we only got one target data.')
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))

    # check data path
    abs_node_names = [node.name for node in self.train_nodes]
    formatted_ys = {}
    ys_not_included = {}
    for k, v in ys.items():
      if k in abs_node_names:
        formatted_ys[k] = v
      else:
        ys_not_included[k] = v
    if len(ys_not_included):
      rel_nodes = self.target.nodes('relative', level=-1, include_self=True).subset(TrainingSystem).unique()
      for k, v in ys_not_included.items():
        if k in rel_nodes:
          formatted_ys[rel_nodes[k].name] = v
        else:
          raise ValueError(f'Unknown target "{k}" for fitting.')

    # check data shape
    for key, val in formatted_ys.items():
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
      formatted_ys = {k: bm.moveaxis(v, 0, 1) for k, v in formatted_ys.items()}
    return formatted_ys
