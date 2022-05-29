# -*- coding: utf-8 -*-

import time
from typing import Union, Dict, Callable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import jit, random as jr
from jax.tree_util import tree_map

import brainpy.losses as losses
import brainpy.math as bm
import brainpy.optimizers as optim
from brainpy.errors import UnsupportedError
from brainpy.tools.checking import check_float
from brainpy.train.base import TrainingSystem
from brainpy.train.utils import check_data_batch_size, serialize_kwargs
from brainpy.types import Tensor
from .base_runner import DSTrainer

__all__ = [
  'BPTT',
  'BPFF',
]


class BPTT(DSTrainer):
  """
  The trainer implementing back propagation through time (BPTT)
  algorithm for recurrent neural networks.

  """

  def __init__(
      self,
      target: TrainingSystem,

      # arguments for BPTT trainer
      loss: Union[str, Callable],  # loss function
      optimizer: optim.Optimizer = None,  # optimizer
      max_grad_norm: float = None,
      shuffle_data: bool = True,

      # common arguments for RNNTrainer
      **kwargs
  ):
    super(BPTT, self).__init__(target=target, **kwargs)

    # jit settings
    self.jit['predict'] = self.jit.get('predict', True)
    self.jit['loss'] = self.jit.get('loss', True)
    self.jit['fit'] = self.jit.get('fit', True)

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
    self._f_shuffle = None

    # functions
    self._f_loss = dict()
    self._f_train = dict()
    self._f_grad = dict()

    # training parameters
    self.max_grad_norm = max_grad_norm  # gradient clipping
    self.shuffle_data = shuffle_data

    # initialize the optimizer
    self.optimizer.register_vars(self.target.vars().subset(bm.TrainVar).unique())


  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}jit={self.jit}, \n\t'
            f'{prefix}loss={self.loss_fun}, \n\t'
            f'{prefix}optimizer={self.optimizer})')

  def predict(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      reset_state: bool = True,
      shared_args: Dict = None,
      **kwargs
  ):
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output, if users want.

    Parameters
    ----------
    xs: Tensor, dict
      The feedforward input data. It must be a 3-dimensional data
      which has the shape of `(num_sample, num_time, num_feature)`.
    shared_args: dict
      Shared keyword arguments for the given target model.
    reset_state: bool
      Whether reset the model states. Default True.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # check forced states/feedbacks
    return super(BPTT, self).predict(xs=xs, reset_state=reset_state, shared_args=shared_args, **kwargs)

  def fit(
      self,
      train_data: Union[Callable, Sequence],
      test_data: Union[Callable, Sequence] = None,
      num_batch: int = None,
      num_epoch: int = 100,
      num_report: int = 100,
      reset_state: bool = True,
      shared_args: Dict = None,
  ):
    """
    Fit the target model according to the given training and testing data.

    Parameters
    ----------
    train_data: callable, sequence of data
      It can be a callable function, or a tuple/list representing `(X, Y)` data.
      - Callable. This function should return a pair of `(X, Y)` data
      - Sequence. It should be a pair of `(X, Y)` train set.
        - ``X``: should be a tensor or a dict of tensors with the shape of
          `(num_sample, num_time, num_feature)`, where `num_sample` is
          the number of samples, `num_time` is the number of the time step,
          and `num_feature` is the number of features.
        - ``Y``: Target values. A tensor or a dict of tensors.
          - If the shape of each tensor is `(num_sample, num_feature)`,
            then we will only fit the model with the only last output.
          - If the shape of each tensor is `(num_sample, num_time, num_feature)`,
            then the fitting happens on the whole data series.
    test_data: callable, sequence of data
      Same as the ``train_data``. It can be a callable function,
      or a tuple/list representing `(X, Y)` data.
    num_batch: int
      The batch size. Default 32. This setting is used when users provide
      the ``train_data`` and ``test_data`` as a pair of `(X, Y)` data, rather
      than a function.
    num_epoch: int
      The number of training epoch. Default 100.
    num_report: int
      The number of step to report the progress. Default 100 training steps.
    reset_state: bool
      Whether reset the initial states of the target model.
    shared_args: dict
      The shared keyword arguments for the target models.

    """
    true_progress_bar =  self.progress_bar
    self.progress_bar = False
    # training the model
    all_train_losses = []
    all_test_losses = []
    train_i = 0
    t0 = time.time()
    for _ in range(num_epoch):
      train_data_ = self._get_train_data(train_data, num_batch)

      # training set
      for x, y in train_data_:
        if reset_state:
          self.target.reset_batch_state(check_data_batch_size(x))
        loss = self.f_train(shared_args)(x, y)
        all_train_losses.append(loss)
        train_i += 1
        if train_i % num_report == 0:
          t1 = time.time()
          print(f'Train {train_i} steps, use {t1 - t0:.4f} s, train loss {round(float(loss), 5)}')
          t0 = t1

      # testing set
      test_data_ = self._get_test_data(test_data, num_batch)
      if test_data_ is not None:
        for x, y in test_data_:
          if reset_state:
            self.target.reset_batch_state(check_data_batch_size(x))
          loss = self.f_loss(shared_args)(x, y)
          all_test_losses.append(loss)

    self._train_losses = bm.asarray(all_train_losses)
    self._test_losses = bm.asarray(all_test_losses)
    self.progress_bar = true_progress_bar

  def f_grad(self, shared_kwargs=None) -> Callable:
    """Get gradient function."""
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_grad:
      self._f_grad[shared_kwargs_str] = self._make_f_grad(shared_kwargs)
    return self._f_grad[shared_kwargs_str]

  def f_loss(self, shared_kwargs=None) -> Callable:
    """Get loss function."""
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_loss:
      self._f_loss[shared_kwargs_str] = self._make_f_loss(shared_kwargs)
      if self.jit['loss']:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_loss[shared_kwargs_str] = bm.jit(self._f_loss[shared_kwargs_str],
                                                 dyn_vars=dyn_vars)
    return self._f_loss[shared_kwargs_str]

  def f_train(self, shared_kwargs=None) -> Callable:
    """Get training function."""
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_train:
      self._f_train[shared_kwargs_str] = self._make_f_train(shared_kwargs)
    return self._f_train[shared_kwargs_str]

  @property
  def train_losses(self):
    """Training loss."""
    return self._train_losses

  def _make_f_loss(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'Only supports dict for "shared_kwargs". '
                       f'But got {type(shared_kwargs)}: {shared_kwargs}')

    def loss_fun(inputs, targets):
      inputs, num_step, num_batch = self._check_xs(inputs, move_axis=True)
      times = jnp.linspace(0., self.dt * (num_step - 1), num_step)
      inputs = (times, inputs)
      outputs, _ = self._predict(xs=inputs, shared_args=shared_kwargs)
      loss = self.loss_fun(bm.moveaxis(outputs, 0, 1), targets)
      return loss

    return loss_fun

  def _make_f_grad(self, shared_kwargs: Dict = None):
    _f_loss_internal = self._make_f_loss(shared_kwargs)
    dyn_vars = self.target.vars()
    dyn_vars.update(self.dyn_vars)
    tran_vars = dyn_vars.subset(bm.TrainVar)
    return bm.grad(_f_loss_internal,
                   dyn_vars=dyn_vars.unique(),
                   grad_vars=tran_vars.unique(),
                   return_value=True)

  def _make_f_train(self, shared_kwargs: Dict = None):
    if shared_kwargs is None:
      shared_kwargs = dict()
    elif not isinstance(shared_kwargs, dict):
      raise ValueError(f'Only supports dict for "shared_kwargs". '
                       f'But got {type(shared_kwargs)}: {shared_kwargs}')

    def train_func(inputs, targets):
      grads, loss = self.f_grad(shared_kwargs)(inputs, targets)
      if self.max_grad_norm is not None:
        check_float(self.max_grad_norm, 'max_grad_norm', min_bound=0.)
        grads = bm.clip_by_norm(grads, self.max_grad_norm)
      self.optimizer.update(grads)
      return loss

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      dyn_vars.update(self.optimizer.vars())
      train_func = bm.jit(train_func, dyn_vars=dyn_vars.unique())
    return train_func

  def _get_train_data(self, train_data, num_batch):
    # training dataset
    if callable(train_data):
      train_data = self._get_data_by_method1(train_data, num_batch)
    elif isinstance(train_data, (tuple, list)):
      if len(train_data) != 2:
        raise ValueError(f"Must be (X, Y) pair, but got a sequence with "
                         f"length {len(train_data)}")
      train_data = self._get_data_by_method2(train_data,
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
      test_data = self._get_data_by_method1(test_data, num_batch)
    elif isinstance(test_data, (tuple, list)):
      assert len(test_data) == 2, f"Must be (X, Y) pair, but got a sequence with length {len(test_data)}"
      test_data = self._get_data_by_method2(test_data,
                                            num_batch=num_batch,
                                            shuffle=False)
    else:
      raise ValueError(f'Test data does not support {type(test_data)}. ')
    return test_data

  def _get_data_by_method1(self, dataset, num_batch):
    for xs, ys in dataset():
      yield xs, ys

  def _get_data_by_method2(self, dataset, num_batch=None, shuffle=False):
    if num_batch is None:
      raise ValueError('Must provide "num_batch" when dataset is not a callable function.')
    assert isinstance(dataset, (tuple, list)) and len(dataset) == 2
    xs, ys = dataset
    num_sample = self._get_xs_batch_size(xs)
    if shuffle:
      xs, ys = self._shuffle(xs, ys)
    for data_idx in range(0, num_sample, num_batch):
      if (data_idx + num_batch) > num_sample:
        inputs = {k: v[data_idx:] for k, v in xs.items()}
        targets = {k: v[data_idx:] for k, v in ys.items()}
      else:
        inputs = {k: v[data_idx: data_idx + num_batch] for k, v in xs.items()}
        targets = {k: v[data_idx: data_idx + num_batch] for k, v in ys.items()}
      yield inputs, targets

  def _shuffle(self, xs, ys):
    key = jr.PRNGKey(seed=np.random.randint(0, 100000))
    if self._f_shuffle is None:
      def shuffle(xs, ys, key):
        xs = tree_map(lambda x: jr.permutation(key, x, axis=0), xs)
        ys = tree_map(lambda y: jr.permutation(key, y, axis=0), ys)
        return xs, ys

      self._f_shuffle = jit(shuffle)
    return self._f_shuffle(xs, ys, key)

  def _get_xs_batch_size(self, xs):
    num_batch_sizes = []
    for key, val in xs.items():
      num_batch_sizes.append(val.shape[0])
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across tensors in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    return num_batch_sizes[0]


class BPFF(BPTT):
  """
  The trainer implementing back propagation algorithm
  for feedforward neural networks.

  """

  def __init__(
      self,
      target: TrainingSystem,
      **kwargs
  ):
    super(BPFF, self).__init__(target=target, **kwargs)

  def predict(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      reset_state: bool = True,
      shared_args: Dict = None,
      **kwargs
  ):
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    xs: Tensor, dict
      The feedforward input data. It must be a 3-dimensional data
      which has the shape of `(num_sample, num_time, num_feature)`.
    reset_state: bool
      Whether reset the model states.
    shared_args: optional, dict
      The shared arguments across different layers.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # format input data
    num_batch = self._get_xs_batch_size(xs)
    # reset the model states
    if reset_state:
      self.target.reset_batch_state(num_batch)
    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # prediction
    outputs, hists = self._predict(xs=xs, shared_args=shared_args)
    # post-running for monitors
    for key in self.mon.item_names:
      self.mon.item_contents[key] = hists[key]
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return outputs

  def _predict(
      self,
      xs: Dict[str, Tensor],
      shared_args: Dict = None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    forced_states: dict
      The forced state values.
    forced_feedbacks: dict
      The forced feedback output values.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    return self._get_predict_func(shared_args)(xs)

  def _make_f_loss(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'Only supports dict for "shared_kwargs". '
                       f'But got {type(shared_kwargs)}: {shared_kwargs}')

    def loss_fun(inputs, targets):
      outputs, _ = self._predict(xs=inputs, shared_args=shared_kwargs)
      loss = self.loss_fun(outputs, targets)
      return loss

    return loss_fun

  def _get_predict_func(self, shared_args: Dict = None):
    if shared_args is None: shared_args = dict()
    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._predict_func:
      self._predict_func[shared_kwargs_str] = self._make_predict_func(shared_args)
    return self._predict_func[shared_kwargs_str]

  def _make_predict_func(self, shared_args: Dict):
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, '
                       f'but got {type(shared_args)}')

    def run_func(xs):
      return self.target(xs, shared_args)

    if self.jit['predict']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      run_func = bm.jit(run_func, dyn_vars=dyn_vars.unique())
    return run_func

  def _get_xs_batch_size(self, xs):
    num_batch_sizes = []
    for key, val in xs.items():
      num_batch_sizes.append(val.shape[0])
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across tensors in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    return num_batch_sizes[0]
