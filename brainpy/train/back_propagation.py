# -*- coding: utf-8 -*-

import time
from typing import Union, Dict, Callable, Sequence

import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map, tree_flatten

import brainpy.losses as losses
import brainpy.math as bm
import brainpy.optimizers as optim
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import UnsupportedError
from brainpy.tools.checking import serialize_kwargs
from brainpy.tools.others import DotDict
from brainpy.types import Array, Output
from ..running import constants as c
from .base import DSTrainer

__all__ = [
  'BPTT',
  'BPFF',
  'OnlineBPTT',
]


def _is_jax_array(s):
  return isinstance(s, bm.JaxArray)


class BPTrainer(DSTrainer):
  """Trainer implementing back-propagation algorithm.

  Parameters
  ----------
  target: DynamicalSystem, TrainingSystem
    The target model to train.
  loss_fun: str, callable
    The loss function. If it is a string, it should be the
    function chosen from ``brainpy.losses`` module. Otherwise,
    a callable function which receives argument of `(predicts, targets)`
    should be provided.
  optimizer: optim.Optimizer
    The optimizer used for training.
  shuffle_data: bool
  seed: int
  numpy_mon_after_run: bool
  """

  def __init__(
      self,
      target: DynamicalSystem,
      loss_fun: Union[str, Callable],  # loss function
      optimizer: optim.Optimizer = None,  # optimizer
      loss_has_aux: bool = False,
      shuffle_data: bool = True,  # shuffle data
      seed: int = None,  # random seed for data shuffling
      numpy_mon_after_run: bool = False,
      **kwargs,
  ):
    super(BPTrainer, self).__init__(target=target,
                                    numpy_mon_after_run=numpy_mon_after_run,
                                    **kwargs)

    self.shuffle_data = shuffle_data
    self.rng = bm.random.RandomState(seed)

    # jit settings
    self.jit[c.PREDICT_PHASE] = self.jit.get(c.PREDICT_PHASE, True)
    self.jit[c.LOSS_PHASE] = self.jit.get(c.LOSS_PHASE, True)
    self.jit[c.FIT_PHASE] = self.jit.get(c.FIT_PHASE, True)

    # optimizer
    if optimizer is None:
      lr = optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
      optimizer = optim.Adam(lr=lr)
    self.optimizer: optim.Optimizer = optimizer
    self.optimizer.register_vars(self.target.vars(level=-1, include_self=True).subset(bm.TrainVar).unique())

    # loss
    self.loss_has_aux = loss_has_aux
    if isinstance(loss_fun, str):
      loss_fun = getattr(losses, loss_fun)
    elif callable(loss_fun):
      loss_fun = loss_fun
    else:
      raise UnsupportedError(f'Do not support {type(loss_fun)} to specify the loss function. '
                             f'We only support str and callable function.')
    self._loss_func = loss_fun
    self._train_losses = None
    self._train_loss_aux = None
    self._test_losses = None
    self._f_shuffle = None

    # functions
    self._f_loss_compiled = dict()
    self._f_train_compiled = dict()
    self._f_grad_compiled = dict()

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}jit={self.jit}, \n\t'
            f'{prefix}loss={self._loss_func}, \n\t'
            f'{prefix}optimizer={self.optimizer})')

  @property
  def train_losses(self):
    """Training loss."""
    return self._train_losses

  @property
  def train_loss_aux(self):
    return self._train_loss_aux

  def predict(
      self,
      inputs: Union[Array, Sequence[Array], Dict[str, Array]],
      reset_state: bool = True,
      shared_args: Dict = None,
      eval_time: bool = False
  ) -> Output:
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output, if users want.

    Parameters
    ----------
    inputs: Array, sequence, dict
      The feedforward input data. It must be a 3-dimensional data
      which has the shape of `(num_sample, num_time, num_feature)`.
    shared_args: dict
      Shared keyword arguments for the given target model.
    reset_state: bool
      Whether reset the model states. Default True.
    eval_time: bool
      Whether evaluate the running time or not. Default False.
    """
    return super(BPTrainer, self).predict(inputs=inputs,
                                          reset_state=reset_state,
                                          shared_args=shared_args,
                                          eval_time=eval_time)

  def fit(
      self,
      train_data: Union[Callable, Sequence],
      batch_size: int = None,
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
    batch_size: int
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
    true_progress_bar = self.progress_bar
    self.progress_bar = False

    # training the model
    all_train_losses = []
    all_train_loss_aux = None
    # all_test_losses = []

    train_i = 0
    t0 = time.time()
    for _ in range(num_epoch):
      # training set
      train_data_ = self._get_batchable_data(train_data, batch_size, self.shuffle_data)
      for x, y in train_data_:
        if reset_state:
          self.target.reset_state(self._get_batch_size(x))
          self.reset_state()

        # training
        res = self.f_train(shared_args)(x, y)

        # loss
        loss = res[0]
        all_train_losses.append(loss)
        if self.loss_has_aux:
          if all_train_loss_aux is None:
            all_train_loss_aux = {k: [] for k in res[1].keys()}
          if not isinstance(res[1], dict):
            raise ValueError(f'Auxiliary data in loss function should be a dict. '
                             f'But we got {type(res)}')
          for k, v in res[1].items():
            all_train_loss_aux[k].append(v)

        # report
        train_i += 1
        if train_i % num_report == 0:
          t1 = time.time()
          msg = f'Train {train_i} steps, use {t1 - t0:.4f} s, train loss {round(float(loss), 5)}'
          if self.loss_has_aux:
            msg += ', {}'.format(", ".join([f"{k} {v}" for k, v in res[1].items()]))
          print(msg)
          t0 = t1

    # finally
    self._train_losses = bm.asarray(all_train_losses)
    if all_train_loss_aux is None:
      self._train_loss_aux = dict()
    else:
      self._train_loss_aux = {k: bm.asarray(v) for k, v in all_train_loss_aux.items()}
    self.progress_bar = true_progress_bar

  def _get_batchable_data(self, data, num_batch, shuffle=False):
    if callable(data):
      data = self._get_data_by_callable(data, num_batch)
    elif isinstance(data, (tuple, list)):
      if len(data) != 2:
        raise ValueError(f"Must be (X, Y) pair, but got a sequence with "
                         f"length {len(data)}")
      data = self._get_data_by_tensor(data, num_batch=num_batch, shuffle=shuffle)
    else:
      raise ValueError(f'Train data does not support {type(data)}. ')
    return data

  def _get_batch_size(self, xs, batch_axis=0):
    if isinstance(xs, (bm.JaxArray, jnp.ndarray)):
      return xs.shape[batch_axis]
    else:
      num_batch_sizes = [leaf.shape[batch_axis] for leaf in tree_flatten(xs, is_leaf=_is_jax_array)[0]]
      if len(set(num_batch_sizes)) != 1:
        raise ValueError(f'Number of batch size is different across tensors in '
                         f'the provided "xs". We got {set(num_batch_sizes)}.')
      return num_batch_sizes[0]

  def _get_data_by_callable(self, dataset, num_batch):
    raise NotImplementedError

  def _get_data_by_tensor(self, dataset, num_batch=None, shuffle=False):
    raise NotImplementedError

  def f_train(self, shared_args=None) -> Callable:
    raise NotImplementedError

  def f_loss(self, shared_args=None) -> Callable:
    raise NotImplementedError


class BPTT(BPTrainer):
  """
  The trainer implementing back propagation through time (BPTT)
  algorithm for recurrent neural networks.
  """

  def f_loss(self, shared_args=None, jit=True) -> Callable:
    """Get loss function."""
    if shared_args is None: shared_args = dict()

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args2)
    if shared_args_str not in self._f_loss_compiled:

      def loss_fun(inputs, targets):
        times, indices, inputs, _, _, _, _ = self._format_xs(
          None, inputs, inputs_are_batching=True, move_axis=True)
        inputs = (times, indices, inputs)
        outputs, mon = self._predict(xs=inputs, shared_args=shared_args)
        outputs = bm.moveaxis(outputs, 0, 1)
        predicts = (outputs, mon) if len(mon) > 0 else outputs
        return self._loss_func(predicts, targets)

      self._f_loss_compiled[shared_args_str] = loss_fun
      if self.jit[c.LOSS_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_loss_compiled[shared_args_str] = bm.jit(self._f_loss_compiled[shared_args_str],
                                                        dyn_vars=dyn_vars)
    return self._f_loss_compiled[shared_args_str]

  def f_grad(self, shared_args=None) -> Callable:
    """Get gradient function."""
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_grad_compiled:
      _f_loss_internal = self.f_loss(shared_args, jit=False)
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      tran_vars = dyn_vars.subset(bm.TrainVar)
      grad_f = bm.grad(_f_loss_internal,
                       dyn_vars=dyn_vars.unique(),
                       grad_vars=tran_vars.unique(),
                       return_value=True,
                       has_aux=self.loss_has_aux)
      self._f_grad_compiled[shared_args_str] = grad_f
    return self._f_grad_compiled[shared_args_str]

  def f_train(self, shared_args=None) -> Callable:
    """Get training function."""
    if shared_args is None: shared_args = dict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'Only supports dict for "shared_args". '
                       f'But got {type(shared_args)}: {shared_args}')

    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_train_compiled:

      def train_func(inputs, targets):
        res = self.f_grad(shared_args)(inputs, targets)
        self.optimizer.update(res[0])
        return res[1:]

      if self.jit[c.FIT_PHASE]:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        dyn_vars.update(self.optimizer.vars())
        self._f_train_compiled[shared_args_str] = bm.jit(train_func, dyn_vars=dyn_vars.unique())
      else:
        self._f_train_compiled[shared_args_str] = train_func
    return self._f_train_compiled[shared_args_str]

  def _get_data_by_callable(self, dataset: Callable, num_batch=None):
    for xs, ys in dataset():
      yield xs, ys

  def _get_data_by_tensor(self, dataset, num_batch=None, shuffle=False):
    if num_batch is None:
      raise ValueError('Must provide "batch_size" when dataset is not a callable function.')
    assert isinstance(dataset, (tuple, list)) and len(dataset) == 2
    xs, ys = dataset
    num_sample = self._get_batch_size(xs)
    if shuffle:
      xs, ys = self._shuffle(xs, ys)
    for data_idx in range(0, num_sample, num_batch):
      if (data_idx + num_batch) > num_sample:
        inputs = tree_map(lambda v: v[data_idx:], xs, is_leaf=_is_jax_array)
        targets = tree_map(lambda v: v[data_idx:], ys, is_leaf=_is_jax_array)
      else:
        inputs = tree_map(lambda v: v[data_idx: data_idx + num_batch], xs, is_leaf=_is_jax_array)
        targets = tree_map(lambda v: v[data_idx: data_idx + num_batch], ys, is_leaf=_is_jax_array)
      yield inputs, targets

  def _shuffle(self, xs, ys):
    key = self.rng.split_key()

    if self._f_shuffle is None:
      def shuffle(xs, ys, key):
        xs = tree_map(lambda x: self.rng.permutation(x, key=key), xs)
        ys = tree_map(lambda y: self.rng.permutation(y, key=key), ys)
        return xs, ys

      self._f_shuffle = bm.jit(shuffle)
    return self._f_shuffle(xs, ys, key)


class BPFF(BPTT):
  """
  The trainer implementing back propagation algorithm
  for feedforward neural networks.

  """

  def predict(
      self,
      inputs: Union[Array, Sequence[Array], Dict[str, Array]],
      reset_state: bool = True,
      shared_args: Dict = None,
      eval_time: bool = False
  ) -> Output:
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    inputs: Array, dict
      The feedforward input data. It must be a 3-dimensional data
      which has the shape of `(num_sample, num_time, num_feature)`.
    reset_state: bool
      Whether reset the model states.
    shared_args: optional, dict
      The shared arguments across different layers.
    eval_time: bool
      Evaluate the time used for running.

    Returns
    -------
    output: Array, dict
      The model output.
    """
    # format input data
    num_batch = self._get_batch_size(inputs)
    # reset the model states
    if reset_state:
      self.target.reset_state(num_batch)
      self.reset_state()
    # init monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items
    # prediction
    outputs, hists = self._predict(xs=inputs, shared_args=shared_args)
    # post-running for monitors
    for key in hists.keys():
      self.mon[key] = bm.asarray(hists[key])
    if self.numpy_mon_after_run:
      self.mon.ts = np.asarray(self.mon.ts)
      for key in hists.keys():
        self.mon[key] = np.asarray(self.mon[key])
    return outputs

  def f_loss(self, shared_args=None, jit=True) -> Callable:
    """Get loss function."""
    if shared_args is None: shared_args = dict()

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args2)
    if shared_args_str not in self._f_loss_compiled:

      def loss_fun(inputs, targets):
        outputs, mon = self.f_predict(shared_args)(inputs)
        outs = (outputs, mon) if len(mon) > 0 else outputs
        loss = self._loss_func(outs, targets)
        return loss

      if self.jit[c.LOSS_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_loss_compiled[shared_args_str] = bm.jit(self._f_loss_compiled[shared_args_str],
                                                        dyn_vars=dyn_vars)
      else:
        self._f_loss_compiled[shared_args_str] = loss_fun
    return self._f_loss_compiled[shared_args_str]

  def f_predict(self, shared_args: Dict = None, jit: bool = True):
    if shared_args is None: shared_args = DotDict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_args" must be a dict, '
                       f'but got {type(shared_args)}')

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_predict_compiled:

      monitor_func = self.build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

      def run_func(xs):
        outs = self.target(shared_args, xs)
        hist = monitor_func(shared_args)
        return outs, hist

      if self.jit[c.PREDICT_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_predict_compiled[shared_args_str] = bm.jit(run_func, dyn_vars=dyn_vars.unique())
      else:
        self._f_predict_compiled[shared_args_str] = run_func
    return self._f_predict_compiled[shared_args_str]


class OnlineBPTT(BPTT):

  def f_loss(self, shared_args=None, jit=True) -> Callable:
    """Get loss function."""
    if shared_args is None: shared_args = dict()

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args2)
    if shared_args_str not in self._f_loss_compiled:

      def loss_fun(t, i, input_, target_):
        outputs, mon = self.f_predict_one_step(shared_args)(t, i, input_)
        predicts = (outputs, mon) if len(mon) > 0 else outputs
        return self._loss_func(predicts, target_)

      if self.jit[c.LOSS_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_loss_compiled[shared_args_str] = bm.jit(self._f_loss_compiled[shared_args_str],
                                                        dyn_vars=dyn_vars)
      else:
        self._f_loss_compiled[shared_args_str] = loss_fun
    return self._f_loss_compiled[shared_args_str]

  def f_train(self, shared_args=None) -> Callable:
    """Get training function."""
    if shared_args is None: shared_args = dict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'Only supports dict for "shared_args". '
                       f'But got {type(shared_args)}: {shared_args}')
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_train_compiled:

      def train_step(*x):
        # t, i, input_, target_ = x
        res = self.f_grad(shared_args)(*x)
        self.optimizer.update(res[0])
        return res[1:]

      if self.jit[c.FIT_PHASE]:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        run_func = lambda all_inputs: bm.for_loop(train_step, dyn_vars.unique(), all_inputs)

      else:
        def run_func(xs):
          times, indices, inputs, targets = xs
          losses = []
          for i in range(times.shape[0]):
            # data at time i
            x = tree_map(lambda x: x[i], inputs, is_leaf=_is_jax_array)
            y = tree_map(lambda x: x[i], targets, is_leaf=_is_jax_array)
            # step at the i
            loss = train_step(times[i], indices[i], x, y)
            # append output and monitor
            losses.append(loss)
          return bm.asarray(losses)

      def train_fun(inputs, targets):
        times, indices, inputs, num_step, _, duration, _ = self._format_xs(
          None, inputs, inputs_are_batching=True, move_axis=True)
        targets = tree_map(lambda x: bm.moveaxis(x, 0, 1), targets, is_leaf=_is_jax_array)
        ls = run_func([times, indices, inputs, targets])
        self.i0 += num_step
        self.t0 += duration
        return ls

      self._f_train_compiled[shared_args_str] = train_fun
    return self._f_train_compiled[shared_args_str]

  def f_predict_one_step(self, shared_args: Dict = None, jit: bool = False):
    if shared_args is None: shared_args = DotDict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_args" must be a dict, '
                       f'but got {type(shared_args)}')

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args2['_one_step_'] = True
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_predict_compiled:

      monitor_func = self.build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

      def run_func(t, i, x):
        shared = DotDict(t=t, i=i, dt=self.dt)
        shared.update(shared_args)
        self.target.clear_input()
        outs = self.target(shared, x)
        hist = monitor_func(shared)
        return outs, hist

      if self.jit[c.FIT_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        self._f_predict_compiled[shared_args_str] = bm.jit(run_func, dyn_vars=dyn_vars.unique())
      else:
        self._f_predict_compiled[shared_args_str] = run_func
    return self._f_predict_compiled[shared_args_str]
