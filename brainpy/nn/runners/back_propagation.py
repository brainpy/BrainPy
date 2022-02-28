# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable

import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.tree_util import tree_flatten, tree_map

import brainpy.losses as losses
import brainpy.math as bm
import brainpy.optimizers as optim
from brainpy.errors import UnsupportedError
from brainpy.nn.base import Node, Network
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'BPTT',
  'BPFF',
]


class BPTT(RNNTrainer):
  """The trainer implementing back propagation through time (BPTT)."""

  def __init__(self,
               target: Node,
               loss: Union[str, Callable],
               optimizer: optim.Optimizer = None,
               max_grad_norm=None,
               shuffle_data=True,
               metrics=('loss',),
               **kwargs):
    super(BPTTTrainer, self).__init__(target=target, **kwargs)

    # optimizer
    if optimizer is None:
      lr = optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
      optimizer = optim.Adam(lr=lr, train_vars=self.target.train_vars())
    self.optimizer = optimizer

    # loss
    if loss is None:
      loss = losses.cross_entropy_loss
    elif isinstance(loss, str):
      loss = getattr(losses, loss)
    elif callable(loss):
      loss = loss
    else:
      raise UnsupportedError(f'Do not support {type(loss)} to specify the loss function. '
                             f'We only support str and callable function.')
    self.loss_fun = loss

    # training parameters
    self.max_grad_norm = max_grad_norm  # gradient clipping
    self.shuffle_data = shuffle_data
    self.metrics = metrics

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],  # shape=(num_data, num_time, ...)
          ys: Union[Tensor, Dict[str, Tensor]],  # shape=(num_data, num_time, ...)
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          num_batch=32,
          num_train=100,
          reset=False,
          # unsupported features
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None):
    assert forced_states is None, 'Currently BPTrainer does not support "forced_states"'
    assert forced_feedbacks is None, 'Currently BPTrainer does not support "forced_feedbacks"'

    # format input data
    xs, num_data = self._format_xs(xs)
    # init the target model
    self._init_target(xs)
    # reset the model states
    if reset:
      self.target.reset_state()
    # set initial states/feedbacks
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)
    # format output data
    ys = self._format_ys(ys)

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_train)
      self._pbar.set_description(f"Train {num_train} epochs: ", refresh=True)

    # training
    for train_idx in range(num_train):
      train_losses = []
      for data_idx in range(0, num_data, num_batch):
        f = lambda x: bm.moveaxis(x[data_idx: data_idx + num_batch], 0, 1)
        inputs = tree_map(f, xs)
        targets = tree_map(f, ys)
        loss = self._f_train(inputs, targets)
        train_losses.append(loss)
      self._pbar.update()
      self._pbar.set_postfix(train_loss={round(float(bm.mean(train_losses)), 4)},
                             )

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

  @property
  def _f_loss(self):
    def loss_fun(inputs, targets):
      outputs = self.predict(inputs, progress_bar=False)
      outputs = self._format_ys(outputs)
      _, outputs = tree_flatten(outputs, is_leaf=lambda a: isinstance(a, bm.JaxArray))
      _, targets = tree_flatten(targets, is_leaf=lambda a: isinstance(a, bm.JaxArray))
      all_losses = [self.loss_fun(output, target) for output, target in zip(outputs, targets)]
      return bm.sum(all_losses)
    return loss_fun

  @property
  def _f_grad(self):
    dyn_vars = self.target.vars()
    dyn_vars.update(self.dyn_vars)
    tran_vars = dyn_vars.subset(bm.TrainVar)
    return bm.grad(self._f_loss,
                   dyn_vars=dyn_vars.unique(),
                   grad_vars=tran_vars.unique(),
                   return_value=True)

  @property
  def _f_train(self):
    def train_func(inputs, targets):
      grads, loss = self._f_grad(inputs, targets)
      if self.max_grad_norm > 0.:
        grads = bm.clip_by_norm(grads, self.max_grad_norm)
      self.optimizer.update(grads)
      return loss

    if self.jit:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      dyn_vars.update(self.optimizer.vars())
      train_func = bm.jit(train_func, dyn_vars=dyn_vars)
    return train_func

  def _format_ys(self, ys):
    if isinstance(ys, (bm.ndarray, jnp.ndarray)):
      if isinstance(self.target, Network):
        assert len(self.target.exit_nodes) == 1, (f'The network {self.target} has {len(self.target.exit_nodes)} '
                                                  f'output nodes, while we only got one output data.')
        ys = {self.target.exit_nodes[0].name: ys}
      else:
        ys = {self.target.name: ys}
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    assert len(ys) > 0, 'We got no input data.'
    return ys

  def _init_target(self, xs):
    # we need to initialize the node or the network
    x = dict()
    for key, tensor in xs.items():
      assert isinstance(key, str), ('"xs" must a dict of (str, tensor), while we got '
                                    f'({type(key)}, {type(tensor)})')
      assert isinstance(tensor, (bm.ndarray, jnp.ndarray)), ('"xs" must a dict of (str, tensor), while we got '
                                                             f'({type(key)}, {type(tensor)})')
      x[key] = tensor[0]
    self.target.initialize(x)

  def _shuffle_xy(self, xs, ys):
    seed = np.random.randint(0, 100000)
    xs = tree_map(lambda data: bm.random.RandomState(seed).shuffle(data, axis=0), xs)
    ys = tree_map(lambda data: bm.random.RandomState(seed).shuffle(data, axis=0), ys)
    return xs, ys


class BPFF(BPTTTrainer):
  pass
