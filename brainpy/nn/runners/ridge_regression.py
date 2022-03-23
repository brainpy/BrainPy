# -*- coding: utf-8 -*-

from typing import Dict, Sequence

import jax.numpy as jnp
import tqdm.auto
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy.nn.base import Node, Network
from brainpy.nn.utils import serialize_kwargs
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'RidgeTrainer'
]


class RidgeTrainer(RNNTrainer):
  """
  Trainer of ridge regression, also known as regression with Tikhonov regularization.

  ``RidgeTrainer`` requires that the trainable node implements its training interface
  ``__ridge_train__()`` function. Otherwise an error will raise.

  ``__ridge_train__()`` function has two variants:

  - `__ridge_train__(ffs, targets, train_pars)` if the model only has feedforward connections,
  - `__ridge_train__(ffs, targets, fbs, train_pars)` if the model has the feedback signaling.

  where ``ffs`` means the feedforward inputs with the shape of `(num_sample, num_time, num_feature)`,
  ``targets`` the ground truth with the shape of `(num_sample, num_time, num_feature)`,
  ``fbs`` the feedback signals with the shape of `(num_sample, num_time, num_feature)`,
  and ``train_pars`` the training related parameters.

  Parameters
  ----------
  target: Node
    The target model.
  beta: float
    The regularization coefficient.
  **kwarg
    Other common parameters for :py:class:`brainpy.nn.RNNTrainer``.
  """

  def __init__(self, target, beta=1e-7, **kwargs):
    self.true_numpy_mon_after_run = kwargs.get('numpy_mon_after_run', True)
    kwargs['numpy_mon_after_run'] = False
    super(RidgeTrainer, self).__init__(target=target, **kwargs)

    # get all trainable nodes
    self.train_nodes = self._get_trainable_nodes()
    # check the required interface in the trainable nodes
    self._check_interface('__ridge_train__')
    # add the monitor items which are needed for the training process
    self._added_items = self._add_monitor_items()
    # train parameters
    self.train_pars = dict(beta=beta)
    # training function
    self._f_train = dict()

  def fit(
      self,
      train_data: Sequence,
      test_data=None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      reset=False,
      shared_kwargs: Dict = None,
  ):
    # checking training and testing data
    if not isinstance(train_data, (list, tuple)):
      raise ValueError(f"{self.__class__.__name__} only support "
                       f"training data with the format of (X, Y) pair, "
                       f"but we got a {type(train_data)}.")
    if len(train_data) != 2:
      raise ValueError(f"{self.__class__.__name__} only support "
                       f"training data with the format of (X, Y) pair, "
                       f"but we got a sequence with length {len(train_data)}")
    if test_data is not None:
      raise ValueError(f'{self.__class__.__name__} does not support testing data.')
    xs, ys = train_data

    # prediction, get all needed data
    _ = self.predict(xs=xs,
                     reset=reset,
                     forced_states=forced_states,
                     forced_feedbacks=forced_feedbacks)

    # get all input data
    xs, num_step, num_batch = self._check_xs(xs, move_axis=False)
    if isinstance(self.target, Network):
      for node in self.target.entry_nodes:
        if node in self.train_nodes:
          inputs = node.data_pass_func({node.name: xs[node.name]})
          self.mon.item_contents[f'{node.name}.inputs'] = inputs
          self._added_items.add(f'{node.name}.inputs')
    elif isinstance(self.target, Node):
      if self.target in self.train_nodes:
        inputs = self.target.data_pass_func({self.target.name: xs[self.target.name]})
        self.mon.item_contents[f'{self.target.name}.inputs'] = inputs
        self._added_items.add(f'{self.target.name}.inputs')

    # format all target data
    if isinstance(ys, (bm.ndarray, jnp.ndarray)):
      if len(self.train_nodes) == 1:
        ys = {self.train_nodes[0].name: ys}
      else:
        raise ValueError(f'The network {self.target} has {len(self.train_nodes)} '
                         f'training nodes, while we only got one target data.')
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    for key, val in ys.items():
      if val.ndim != 3:
        raise ValueError("Targets must be a tensor with shape of "
                         "(num_sample, num_time, num_feature), "
                         f"but we got {val.shape}")
      if val.shape[0] != num_batch:
        raise ValueError(f'Batch size of the target {key} does not match '
                         f'with the input data {val.shape[0]} != {num_batch}')
      if val.shape[1] != num_step:
        raise ValueError(f'The time step of the target {key} does not match '
                         f'with the input data {val.shape[1]} != {num_step})')

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=len(self.train_nodes))
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # training
    monitor_data = dict()
    for node in self.train_nodes:
      monitor_data[f'{node.name}.inputs'] = self.mon.item_contents.get(f'{node.name}.inputs', None)
      monitor_data[f'{node.name}.feedbacks'] = self.mon.item_contents.get(f'{node.name}.feedbacks', None)
    self.f_train(shared_kwargs)(monitor_data, ys)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for key in self._added_items:
      self.mon.item_contents.pop(key)
    if self.true_numpy_mon_after_run:
      self.mon.numpy()

  def f_train(self, shared_kwargs: Dict = None):
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_train:
      self._f_train[shared_kwargs_str] = self._make_fit_func(shared_kwargs)
    return self._f_train[shared_kwargs_str]

  def _make_fit_func(self, shared_kwargs):
    shared_kwargs = dict() if shared_kwargs is None else shared_kwargs

    def train_func(monitor_data: Dict[str, Tensor], target_data: Dict[str, Tensor]):
      for node in self.train_nodes:
        ff = monitor_data[f'{node.name}.inputs']
        fb = monitor_data.get(f'{node.name}.feedbacks', None)
        targets = target_data[node.name]
        if fb is None:
          node.__ridge_train__(ff, targets, train_pars=self.train_pars, **shared_kwargs)
        else:
          node.__ridge_train__(ff, targets, fb, train_pars=self.train_pars, **shared_kwargs)
        if self.progress_bar:
          id_tap(lambda *args: self._pbar.update(), ())

    if self.jit:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      train_func = bm.jit(train_func, dyn_vars=dyn_vars.unique())
    return train_func

  def _add_monitor_items(self):
    added_items = set()
    if isinstance(self.target, Network):
      for node in self.train_nodes:
        if node not in self.target.entry_nodes:
          if f'{node.name}.inputs' not in self.mon.item_names:
            self.mon.item_names.append(f'{node.name}.inputs')
            self.mon.item_contents[f'{node.name}.inputs'] = []
            added_items.add(f'{node.name}.inputs')
        if node in self.target.fb_senders:
          if f'{node.name}.feedbacks' not in self.mon.item_names:
            self.mon.item_names.append(f'{node.name}.feedbacks')
            self.mon.item_contents[f'{node.name}.feedbacks'] = []
            added_items.add(f'{node.name}.feedbacks')
    else:
      # brainpy.nn.Node instance does not need to monitor its inputs
      pass
    return added_items
