# -*- coding: utf-8 -*-

from typing import Union, Dict

import  tqdm.auto
import jax.numpy as jnp

import brainpy.math as bm
from brainpy.nn.base import Node, Network
from brainpy.nn.utils import check_dict_types
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'RidgeTrainer'
]


class RidgeTrainer(RNNTrainer):
  """Ridge regression trainer."""

  def __init__(self, target, beta=1e-7, **kwargs):
    self.true_numpy_mon_after_run = kwargs.get('numpy_mon_after_run', True)
    kwargs['numpy_mon_after_run'] = False
    super(RidgeTrainer, self).__init__(target=target, **kwargs)

    # check the required interface in the trainable nodes
    self._check_interface('__ridge_train__')
    # add the monitor items which are needed for the training process
    self._added_items = self._add_monitor_items()
    # train parameters
    self.train_pars = dict(beta=beta)

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],
          ys: Union[Tensor, Dict[str, Tensor]],
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None,
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          reset=False):
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
    check_dict_types(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # train the network
    for node in self.train_nodes:
      ff = self.mon.item_contents[f'{node.name}.inputs']
      targets = ys[node.name]
      fb = self.mon.item_contents.get(f'{node.name}.feedbacks', None)
      if fb is None:
        node.__ridge_train__(ff, targets, train_pars=self.train_pars)
      else:
        node.__ridge_train__(ff, targets, fb, train_pars=self.train_pars)
      self._pbar.update()

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for key in self._added_items:
      self.mon.item_contents.pop(key)
    if self.true_numpy_mon_after_run: self.mon.numpy()

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
