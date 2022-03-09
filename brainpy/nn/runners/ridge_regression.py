# -*- coding: utf-8 -*-

from typing import Union, Dict

import jax.numpy as jnp
import tqdm.auto
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy.nn.base import Node, Network
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'RidgeTrainer'
]


class RidgeTrainer(RNNTrainer):
  """Trainer of ridge regression, also known as regression with Tikhonov regularization.

  ``RidgeTrainer`` requires that the trainable node implements its training interface
  ``__ridge_train__()`` function. Otherwise an error will raise.

  ``__ridge_train__()`` function has two variants:

  - `__ridge_train__(ffs, targets, train_pars)` if the model only has feedforward connections,
  - `__ridge_train__(ffs, targets, fbs, train_pars)` if the model has the feedback signaling.

  where ``ffs`` means the feedforward inputs, ``targets`` the ground truth,
  ``fbs`` the feedback signals, and ``train_pars`` the training related parameters.

  Parameters
  ----------
  target: Node
    The target model.
  beta: float
    The regularization coefficient.
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
    self._fit_func = None

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],
          ys: Union[Tensor, Dict[str, Tensor]],
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None,
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          reset=False):

    # prediction, get all needed data
    _ = self.predict(xs=xs, reset=reset,
                     forced_states=forced_states, forced_feedbacks=forced_feedbacks,
                     initial_states=initial_states, initial_feedbacks=initial_feedbacks)

    # get all input data
    xs, num_step = self._format_xs(xs)
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

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=len(self.train_nodes))
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # train the network
    if self._fit_func is None:
      self._fit_func = self._make_fit_func()
    monitor_data = dict()
    for node in self.train_nodes:
      monitor_data[f'{node.name}.inputs'] = self.mon.item_contents.get(f'{node.name}.inputs', None)
      monitor_data[f'{node.name}.feedbacks'] = self.mon.item_contents.get(f'{node.name}.feedbacks', None)
    self._fit_func(monitor_data, ys)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for key in self._added_items:
      self.mon.item_contents.pop(key)
    if self.true_numpy_mon_after_run: self.mon.numpy()

  def _make_fit_func(self):
    def train_func(monitor_data: Dict[str, Tensor], target_data: Dict[str, Tensor]):
      for node in self.train_nodes:
        ff = monitor_data[f'{node.name}.inputs']
        fb = monitor_data.get(f'{node.name}.feedbacks', None)
        targets = target_data[node.name]
        if fb is None:
          node.__ridge_train__(ff, targets, train_pars=self.train_pars)
        else:
          node.__ridge_train__(ff, targets, fb, train_pars=self.train_pars)
        if self.progress_bar:
          id_tap(lambda *args: self._pbar.update(), ())

    if self.jit:
      # self.dyn_vars.update(self.target.vars().unique())
      train_func = bm.jit(train_func, dyn_vars=self.dyn_vars.unique())
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
        if node in self.target._fb_senders:
          if f'{node.name}.feedbacks' not in self.mon.item_names:
            self.mon.item_names.append(f'{node.name}.feedbacks')
            self.mon.item_contents[f'{node.name}.feedbacks'] = []
            added_items.add(f'{node.name}.feedbacks')
    else:
      # brainpy.nn.Node instance does not need to monitor its inputs
      pass
    return added_items
