# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Sequence

import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.tree_util import tree_map

import brainpy.losses as losses
import brainpy.math as bm
import brainpy.optimizers as optim
from brainpy.errors import UnsupportedError
from brainpy.nn.base import Node, Network
from brainpy.nn.utils import check_rnn_data_batch_size
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'BPTT',
  'BPFF',
]

MANY2ONE = 'many2one'
MANY2MANY = 'many2many'


class BPFF(RNNTrainer):
  pass


class BPTT(RNNTrainer):
  """
  The trainer implementing back propagation through time (BPTT)
  for recurrent neural networks.

  """

  def __init__(
      self,
      target: Node,

      # arguments for BPTT trainer
      loss: Union[str, Callable],
      optimizer: optim.Optimizer = None,
      max_grad_norm=None,
      shuffle_data=True,
      metrics=('loss',),

      # common arguments for RNNTrainer
      **kwargs
  ):
    super(BPTT, self).__init__(target=target, **kwargs)

    # optimizer
    if optimizer is None:
      lr = optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
      optimizer = optim.Adam(lr=lr)
    self.optimizer = optimizer

    # loss
    if isinstance(loss, str):
      loss = getattr(losses, loss)
    elif callable(loss):
      loss = loss
    else:
      raise UnsupportedError(f'Do not support {type(loss)} to specify the loss function. '
                             f'We only support str and callable function.')
    self.loss_fun = loss
    self._train_losses = None
    self._test_losses = None
    self._mapping_type = None

    # training parameters
    self.max_grad_norm = max_grad_norm  # gradient clipping
    self.shuffle_data = shuffle_data
    self.metrics = metrics

    # initialize the optimizer
    if not (self.target.is_ff_initialized and
            self.target.is_fb_initialized and
            self.target.is_state_initialize):
      raise ValueError('Please initialize the target model first by calling "initialize()" function.')
    self.optimizer.register_vars(self.target.vars().subset(bm.TrainVar))

  def predict(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Dict[str, Tensor] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
      reset=True,
      progress_bar=True
  ):
    # check forced states/feedbacks
    assert forced_states is None, (f'Currently {self.__class__.__name__} does '
                                   f'not support "forced_states"')
    assert forced_feedbacks is None, (f'Currently {self.__class__.__name__} does '
                                      f'not support "forced_feedbacks"')
    # format input data
    xs, num_step, num_batch = self._check_xs(xs, move_axis=True)
    # state
    if reset:
      self.target.init_state(num_batch)
    # prediction
    return self._predict(xs=xs,
                         num_step=num_step,
                         initial_states=initial_states,
                         initial_feedbacks=initial_feedbacks,
                         iter_forced_states=dict(),
                         fixed_forced_states=dict(),
                         iter_forced_feedbacks=dict(),
                         fixed_forced_feedbacks=dict(),
                         progress_bar=progress_bar)

  def fit(
      self,
      train_data: Union[Callable, Sequence],
      test_data: Union[Callable, Sequence] = None,
      num_batch: int = 32,
      num_train: int = 100,
      reset: bool = True,

      # other arguments
      initial_states: Dict[str, Tensor] = None,
      initial_feedbacks: Dict[str, Tensor] = None,

      # unsupported features
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None
  ):
    """
    Fit the target model according to the given data ``xs`` and ``ys``.

    Parameters
    ----------
    train_data: callable, sequence of data
      It can be a callable function, or a tuple/list representing XY data.
      - Callable. This function should return a pair of (X, Y) data
      - Sequence. It should be a pair of (X, Y) train set.
        - X: should be a tensor or a dict of tensors with the shape of
          ``(num_sample, num_time, num_feature)``, where `num_sample` is
          the number of samples, `num_time` is the number of the time step,
          and `num_feature` is the number of features.
        - Y: Target values. A tensor or a dict of tensors.
          - If the shape of each tensor is ``(num_sample, num_feature)``,
            then we will only fit the model with the only last output.
          - If the shape of each tensor is ``(num_sample, num_time, num_feature)``,
            then the fitting happens on the whole data series.
    test_data: callable, sequence of data
      Same as the ``train_data``.
    initial_states: optional, dict
      Initial states for the nodes in the target model.
    initial_feedbacks: optional, dict
      Initial feedbacks for the nodes in the target model.
    num_batch: int
    num_train: int
    reset: bool
    forced_states: optional,dict
    forced_feedbacks: optional,dict
    """
    # check forced states/feedbacks
    assert forced_states is None, (f'Currently {self.__class__.__name__} does '
                                   f'not support "forced_states"')
    assert forced_feedbacks is None, (f'Currently {self.__class__.__name__} does '
                                      f'not support "forced_feedbacks"')

    # set initial states/feedbacks
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)

    # progress bar
    if self.progress_bar:
      progress_bar = tqdm.tqdm(range(num_train))
      progress_bar.set_description(f"Train {num_train} epochs: ", refresh=True)
    else:
      progress_bar = range(num_train)

    # training the model
    all_train_losses = []
    all_test_losses = []
    for _ in progress_bar:
      train_data = self._get_train_data(train_data, num_batch)
      postfix = dict()
      # training set
      train_losses = []
      for x, y in train_data():
        self._check_mapping_type(y)
        batch_size = check_rnn_data_batch_size(x)
        if batch_size != num_batch:
          raise ValueError(f'"num_batch" is set to {num_batch}, '
                           f'but we got {batch_size}.')
        if reset:
          self.target.init_state(batch_size)
        loss = self.f_train(x, y)
        train_losses.append(loss)
      loss = round(float(bm.mean(bm.asarray(train_losses))), 5)
      all_train_losses.append(loss)
      postfix['train_loss'] = loss
      # testing set
      test_data = self._get_test_data(test_data, num_batch)
      if test_data is not None:
        test_losses = []
        for x, y in test_data():
          self._check_mapping_type(y)
          batch_size = check_rnn_data_batch_size(x)
          if batch_size != num_batch:
            raise ValueError(f'"num_batch" is set to {num_batch}, '
                             f'but we got {batch_size}.')
          if reset:
            self.target.init_state(batch_size)
          loss = self.f_loss(x, y)
          test_losses.append(loss)
        loss = round(float(bm.mean(bm.asarray(test_losses))), 5)
        all_test_losses.append(loss)
        postfix['test_loss'] = loss
      if self.progress_bar: progress_bar.set_postfix(**postfix, refresh=True)
    self._train_losses = bm.asarray(all_train_losses)
    self._test_losses = bm.asarray(all_test_losses)

  @property
  def train_losses(self):
    """Training loss."""
    return self._train_losses

  @property
  def test_losses(self):
    """Training loss."""
    return self._test_losses

  @property
  def mapping_type(self):
    """Mapping type for the output and the target."""
    return self._mapping_type

  @property
  def _f_loss(self):
    def loss_fun(inputs, targets):
      inputs = {k: bm.moveaxis(v, 0, 1) for k, v in inputs.items()}
      outputs = self._predict(xs=inputs, progress_bar=False)
      outputs = self._format_ys(outputs)
      loss = 0.
      for key, output in outputs.items():
        if self.mapping_type[key] == MANY2ONE:
          output = output[:, -1]
        loss += self.loss_fun(output, targets[key])
      return loss

    return loss_fun

  @property
  def f_loss(self):
    if self.jit:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      return bm.jit(self._f_loss, dyn_vars=dyn_vars)
    else:
      return self._f_loss

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
  def f_train(self):
    def train_func(inputs, targets):
      grads, loss = self._f_grad(inputs, targets)
      if (self.max_grad_norm is not None) and (self.max_grad_norm > 0.):
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
        assert len(self.target.exit_nodes) == 1, (f'The network {self.target} has '
                                                  f'{len(self.target.exit_nodes)} '
                                                  f'output nodes, while we only got '
                                                  f'one output data.')
        ys = {self.target.exit_nodes[0].name: ys}
      else:
        ys = {self.target.name: ys}
    else:
      for node in self.target.exit_nodes:
        if node.name not in ys:
          raise ValueError(f'The network has output node {node.name}, '
                           f'however, we did not get the corresponding '
                           f'output targets.')
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    return ys

  def _check_mapping_type(self, ys):
    if self.mapping_type is None:
      self._mapping_type = dict()
    for (key, y) in ys.items():
      assert y.ndim in [2, 3], ('Each tensor in "ys" must have the shape of '
                                '(num_sample, num_time, num_feature) or '
                                '(num_sample, num_feature), but we '
                                f'got {y.shape}')
      if key not in self._mapping_type:
        self._mapping_type[key] = MANY2MANY if y.ndim == 3 else MANY2ONE
      else:
        if self._mapping_type[key] != (MANY2MANY if y.ndim == 3 else MANY2ONE):
          raise ValueError(f'Mapping type of {key} is {self.mapping_type[key]}, '
                           f'it cannot be changed.')

  def _get_train_data(self, train_data, num_batch):
    # training dataset
    if callable(train_data):
      train_data = train_data
    elif isinstance(train_data, (tuple, list)):
      assert len(train_data) == 2, f"Must be (X, Y) pair, but got a sequence with length {len(train_data)}"
      train_data = self._get_data_iter(train_data,
                                       num_batch=num_batch,
                                       shuffle=self.shuffle_data)
    else:
      raise ValueError(f'Train data does not support {type(train_data)}. ')
    return train_data

  def _get_test_data(self, test_data, num_batch):
    # testing dataset
    if test_data is None:
      test_data = None
    elif callable(test_data):
      test_data = test_data
    elif isinstance(test_data, (tuple, list)):
      assert len(test_data) == 2, f"Must be (X, Y) pair, but got a sequence with length {len(test_data)}"
      test_data = self._get_data_iter(test_data,
                                      num_batch=num_batch,
                                      shuffle=False)
    else:
      raise ValueError(f'Test data does not support {type(test_data)}. ')
    return test_data

  def _get_data_iter(self, dataset, num_batch, shuffle=False, ):
    assert isinstance(dataset, (tuple, list)) and len(dataset) == 2
    xs, ys = dataset
    xs, _, num_sample = self._check_xs(xs, move_axis=False)
    ys = self._format_ys(ys)
    if shuffle:
      seed = np.random.randint(0, 100000)
      xs = tree_map(lambda data: bm.random.RandomState(seed).shuffle(data, axis=0), xs)
      ys = tree_map(lambda data: bm.random.RandomState(seed).shuffle(data, axis=0), ys)

    def data_iter():
      for data_idx in range(0, num_sample, num_batch):
        if (data_idx + num_batch) > num_sample:
          ids = bm.arange(data_idx, data_idx + num_batch) % num_sample
          inputs = {k: v[ids] for k, v in xs.items()}
          targets = {k: v[ids] for k, v in ys.items()}
        else:
          inputs = {k: v[data_idx: data_idx + num_batch] for k, v in xs.items()}
          targets = {k: v[data_idx: data_idx + num_batch] for k, v in ys.items()}
        yield inputs, targets

    return data_iter


