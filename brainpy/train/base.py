# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any, Union

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.dyn.runners import DSRunner
from brainpy.tools.checking import check_dict_data
from brainpy.types import Array, Output
from ..running import constants as c

__all__ = [
  'DSTrainer',
]


class DSTrainer(DSRunner):
  """Structural Trainer for Dynamical Systems."""

  target: DynamicalSystem
  train_nodes: Sequence[DynamicalSystem]  # need to be initialized by subclass

  def __init__(
      self,
      target: DynamicalSystem,
      **kwargs
  ):
    if not isinstance(target, (DynamicalSystem, DynamicalSystem)):
      raise TypeError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                      f'but we got {type(target)}: {target}')
    super(DSTrainer, self).__init__(target=target, **kwargs)

    # jit
    self.jit[c.PREDICT_PHASE] = self.jit.get(c.PREDICT_PHASE, True)
    self.jit[c.FIT_PHASE] = self.jit.get(c.FIT_PHASE, True)

  def predict(
      self,
      inputs: Union[Array, Sequence[Array], Dict[str, Array]],
      reset_state: bool = False,
      shared_args: Dict = None,
      eval_time: bool = False
  ) -> Output:
    """Prediction function.

    What's different from `predict()` function in :py:class:`~.DynamicalSystem` is that
    the `inputs_are_batching` is default `True`.

    Parameters
    ----------
    inputs: Array, sequence of Array, dict of Array
      The input values.
    reset_state: bool
      Reset the target state before running.
    shared_args: dict
      The shared arguments across nodes.
    eval_time: bool
      Whether we evaluate the running time or not?

    Returns
    -------
    output: Array, sequence of Array, dict of Array
      The running output.
    """
    return super(DSTrainer, self).predict(duration=None,
                                          inputs=inputs,
                                          inputs_are_batching=True,
                                          reset_state=reset_state,
                                          shared_args=shared_args,
                                          eval_time=eval_time)

  def fit(
      self,
      train_data: Any,
      reset_state: bool = False,
      shared_args: Dict = None
  ) -> Output:  # need to be implemented by subclass
    raise NotImplementedError('Must implement the fit function. ')

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
      rel_nodes = self.target.nodes('relative', level=-1, include_self=True).subset(DynamicalSystem).unique()
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
