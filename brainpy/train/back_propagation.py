# -*- coding: utf-8 -*-

import sys
import time
from collections.abc import Iterable
from functools import partial
from typing import Union, Dict, Callable, Sequence, Any, Optional

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

import brainpy.losses as losses
import brainpy.math as bm
import brainpy.optimizers as optim
from brainpy import tools
from brainpy.check import serialize_kwargs
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import UnsupportedError, NoLongerSupportError
from brainpy.running import constants as c
from brainpy.train.base import DSTrainer
from brainpy.types import ArrayType, Output
from ._utils import msg

__all__ = [
  'BPTT',
  'BPFF',
]


def _is_brainpy_array(s):
  return isinstance(s, bm.Array)


class BPTrainer(DSTrainer):
  """Trainer implementing back-propagation algorithm for supervised trasks.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: DynamicalSystem
    The target model to train.
  loss_fun: str, callable
    The loss function. If it is a string, it should be the
    function chosen from ``brainpy.losses`` module. Otherwise,
    a callable function which receives argument of `(predicts, targets)`
    should be provided.
  loss_has_aux: bool
    To indicate whether the `loss_fun` returns auxiliary data.
  optimizer: optim.Optimizer
    The optimizer used for training.
  numpy_mon_after_run: bool
    Make the monitored results as NumPy arrays.
  logger: Any
    A file-like object (stream); defaults to the current `sys.stdout`.
  shuffle_data: bool
    .. deprecated:: 2.2.4.1
       Control the data shuffling by user self.
  seed: int
    .. deprecated:: 2.2.4.1
       Control the data shuffling by user self.

  kwargs: Any
    Other general parameters please see :py:class:`~.DSRunner`.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      loss_fun: Union[str, Callable],  # loss function
      optimizer: optim.Optimizer = None,  # optimizer
      loss_has_aux: bool = False,  # loss auxiliary
      logger: Any = sys.stdout,

      # -------------
      # API deprecated
      seed: int = None,  # deprecated
      shuffle_data: bool = None,  # deprecated

      **kwargs,
  ):
    super(BPTrainer, self).__init__(target=target,
                                    **kwargs)

    if shuffle_data is not None:
      raise NoLongerSupportError(
        f'''
        "shuffle_data" is no longer supported. '
        To be general, users should shuffle their data by themself.
        
        See https://github.com/brainpy/BrainPy/releases/tag/V2.3.1
        for the solution of how to fix this.
        '''
      )
    if seed is not None:
      NoLongerSupportError('"seed" is no longer supported. '
                           'Please shuffle your data by yourself.')

    # jit settings
    if isinstance(self._origin_jit, bool):
      self.jit[c.PREDICT_PHASE] = self.jit.get(c.PREDICT_PHASE, self._origin_jit)
      self.jit[c.LOSS_PHASE] = self.jit.get(c.LOSS_PHASE, self._origin_jit)
      self.jit[c.FIT_PHASE] = self.jit.get(c.FIT_PHASE, self._origin_jit)
    else:
      self.jit[c.PREDICT_PHASE] = self._origin_jit.get(c.PREDICT_PHASE, True)
      self.jit[c.LOSS_PHASE] = self._origin_jit.get(c.LOSS_PHASE, True)
      self.jit[c.FIT_PHASE] = self._origin_jit.get(c.FIT_PHASE, True)

    # optimizer
    if optimizer is None:
      lr = optim.ExponentialDecay(lr=0.025, decay_steps=1, decay_rate=0.99975)
      optimizer = optim.Adam(lr=lr)
    self.optimizer: optim.Optimizer = optimizer
    if len(self.optimizer.vars_to_train) == 0:
      self.optimizer.register_vars(self.target.vars(level=-1, include_self=True).subset(bm.TrainVar).unique())

    # loss function
    self.loss_has_aux = loss_has_aux
    if isinstance(loss_fun, str):
      loss_fun = getattr(losses, loss_fun)
    elif callable(loss_fun):
      loss_fun = loss_fun
    else:
      raise UnsupportedError(f'Do not support {type(loss_fun)} to specify the loss function. '
                             f'We only support str and callable function.')
    self._loss_func = loss_fun

    # loss data
    self._report_train_metrics = dict()
    self._report_test_metrics = dict()
    self._detailed_train_metrics = dict()
    self._detailed_test_metrics = dict()

    # functions
    self._f_loss_compiled = dict()
    self._f_grad_compiled = dict()

    # others
    self.logger = logger

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}jit={self.jit}, \n\t'
            f'{prefix}loss={self._loss_func}, \n\t'
            f'{prefix}optimizer={self.optimizer})')

  def get_hist_metric(self, phase='fit', metric='loss', which='report'):
    """Get history losses."""
    assert phase in [c.FIT_PHASE, c.TEST_PHASE, c.TRAIN_PHASE, c.PREDICT_PHASE]
    assert which in ['report', 'detailed']
    if phase in [c.FIT_PHASE, c.TRAIN_PHASE]:
      if which == 'report':
        return self._report_train_metrics.get(metric, None)
      elif which == 'detailed':
        return self._detailed_train_metrics.get(metric, None)
    elif phase in [c.TEST_PHASE, c.PREDICT_PHASE]:
      if which == 'report':
        return self._report_test_metrics.get(metric, None)
      elif which == 'detailed':
        return self._detailed_test_metrics.get(metric, None)

  @property
  def train_losses(self):
    return self.get_hist_metric(phase='fit')

  @property
  def test_losses(self):
    return self.get_hist_metric(phase='test')

  def fit(
      self,
      train_data: Union[Callable, Iterable],
      test_data: Optional[Union[Callable, Iterable]] = None,
      num_epoch: int = 100,
      num_report: int = -1,
      reset_state: bool = True,
      shared_args: Optional[Dict] = None,
      fun_after_report: Optional[Callable] = None,

      # ------
      # API deprecated
      batch_size: int = None,
  ):
    """Fit the target model according to the given training data.

    Parameters
    ----------
    train_data: callable, iterable
      It can be a callable function, or a tuple/list representing `(X, Y)` data.
      - Callable. This function should return a pair of `(X, Y)` data.
      - Iterable. It should be a pair of `(X, Y)` train set.
        - ``X``: should be a tensor or a dict of tensors with the shape of
          `(num_sample, num_time, ...)`, where `num_sample` is
          the number of samples, `num_time` is the number of the time step,
          and `num_feature` is the number of features.

        - ``Y``: Target values. A tensor or a dict of tensors.
          - If the shape of each tensor is `(num_sample, num_feature)`,
            then we will only fit the model with the only last output.
          - If the shape of each tensor is `(num_sample, num_time, num_feature)`,
            then the fitting happens on the whole data series.
    test_data: callable, iterable, optional
      Same as ``train_data``.
    num_epoch: int
      The number of training epoch. Default 100.
    num_report: int
      The number of step to report the progress.
      If `num_report=-1`, it will report the training progress each epoch.
    reset_state: bool
      Whether reset the initial states of the target model.
    shared_args: dict
      The shared keyword arguments for the target models.
    fun_after_report: optional, Callable
      The function to call after each report of `fit` phase or `test` phase.
      The function should receive three arguments:
      - ``idx`` for the indicator the current the running index. (If ``report=-1``,
        The running index is the epoch. Otherwise, is the 'fit_idx' for 'fit' phase
        and 'test_idx' for 'test' phase).
      - ``metrics``: the metrics defined in the loss function
      - ``phase``: to indicate the phase of 'fit' or 'test'.

      .. versionadded:: 2.3.1
    batch_size: int

      .. deprecated:: 2.2.4.1
         Please set batch size in your dataset.

    """
    if batch_size is not None:
      raise NoLongerSupportError('Please set batch size in your data. '
                                 'Specifically, make an iterable dataset '
                                 'which return a batch of (X, Y) data.')
    if isinstance(train_data, (tuple, list)):
      if len(train_data) == 2:
        raise UnsupportedError(msg)

    if fun_after_report is not None:
      assert callable(fun_after_report), ('\n'
                                          'Unknown "fun_after_report", '
                                          'it should be a callable function receiving '
                                          'three arguments: idx, metrics, phase')

    if shared_args is None:
      shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', False)

    true_progress_bar = self.progress_bar
    self.progress_bar = False

    # training the model
    detailed_train_metric = dict()
    report_train_metric = dict()
    detailed_test_metric = dict()
    report_test_metric = dict()

    fit_i, fit_t = 0, 0
    test_i, test_t = 0, 0
    for epoch_idx in range(num_epoch):

      # training set
      fit_t0 = time.time()
      fit_epoch_metric = dict(loss=[])
      for x, y in (train_data() if callable(train_data) else train_data):

        # reset state
        if reset_state:
          self.target.reset_state(self._get_input_batch_size(x))
          self.reset_state()

        # training
        res = self._get_f_train(shared_args)(x, y)

        # loss
        fit_epoch_metric['loss'].append(res[0])
        if self.loss_has_aux:
          if not isinstance(res[1], dict):
            raise TypeError(f'Auxiliary data in loss function should be a dict. But we got {type(res)}')
          for k, v in res[1].items():
            if k not in fit_epoch_metric:
              fit_epoch_metric[k] = []
            fit_epoch_metric[k].append(v)

        # report
        fit_i += 1
        if num_report > 0 and fit_i % num_report == 0:
          fit_t1 = time.time()
          aux = {}
          for k, v in fit_epoch_metric.items():
            aux[k] = bm.mean(bm.asarray(v))
            if k not in report_train_metric:
              report_train_metric[k] = []
              detailed_train_metric[k] = []
            report_train_metric[k].append(aux[k])
            detailed_train_metric[k].extend(v)
            v.clear()
          print((f'Train {fit_i} steps, use {fit_t + fit_t1 - fit_t0:.4f} s' +
                 ', {}'.format(", ".join([f"{k} {v}" for k, v in aux.items()]))),
                file=self.logger)
          if fun_after_report is not None:
            fun_after_report(fit_i, aux, 'fit')
          fit_t0 = time.time()
          fit_t = 0

      if num_report <= 0:
        fit_t1 = time.time()
        aux = {}
        for k, v in fit_epoch_metric.items():
          aux[k] = np.mean(np.asarray(v))
          if k not in report_train_metric:
            report_train_metric[k] = []
            detailed_train_metric[k] = []
          report_train_metric[k].append(aux[k])
          detailed_train_metric[k].extend(v)
          v.clear()
        print((f'Train {epoch_idx} epoch, use {fit_t1 - fit_t0:.4f} s' +
               ', {}'.format(", ".join([f"{k} {v}" for k, v in aux.items()]))),
              file=self.logger)
        if fun_after_report is not None:
          fun_after_report(epoch_idx, aux, 'fit')
      else:
        fit_t = time.time() - fit_t0

      # testing set
      if test_data is not None:
        test_t0 = time.time()
        test_epoch_metric = dict(loss=[])
        for x, y in (test_data() if callable(test_data) else test_data):
          # reset state
          if reset_state:
            self.target.reset_state(self._get_input_batch_size(x))
            self.reset_state()

          # testing
          res = self._get_f_loss(shared_args)(x, y)

          # loss
          if self.loss_has_aux:
            test_epoch_metric['loss'].append(res[0])
            if not isinstance(res[1], dict):
              raise TypeError(f'Auxiliary data in loss function should be a dict. But we got {type(res)}')
            for k, v in res[1].items():
              if k not in test_epoch_metric:
                test_epoch_metric[k] = []
              test_epoch_metric[k].append(v)
          else:
            test_epoch_metric['loss'].append(res)

          # report
          test_i += 1
          if num_report > 0 and test_i % num_report == 0:
            test_t1 = time.time()
            aux = {}
            for k, v in test_epoch_metric.items():
              aux[k] = np.mean(np.asarray(v))
              if k not in report_test_metric:
                report_test_metric[k] = []
                detailed_test_metric[k] = []
              report_test_metric[k].append(aux[k])
              detailed_test_metric[k].extend(v)
              v.clear()
            print((f'Test {test_i} steps, use {test_t + test_t1 - test_t0:.4f} s' +
                   ', {}'.format(", ".join([f"{k} {v}" for k, v in aux.items()]))),
                  file=self.logger)
            if fun_after_report is not None:
              fun_after_report(test_i, aux, 'test')
            test_t0 = time.time()
            test_t = 0

        if num_report <= 0:
          test_t1 = time.time()
          aux = {}
          for k, v in test_epoch_metric.items():
            aux[k] = bm.mean(bm.asarray(v))
            if k not in report_test_metric:
              report_test_metric[k] = []
              detailed_test_metric[k] = []
            report_test_metric[k].append(aux[k])
            detailed_test_metric[k].extend(v)
            v.clear()
          print((f'Test {epoch_idx} epoch, use {test_t1 - test_t0:.4f} s' +
                 ', {}'.format(", ".join([f"{k} {v}" for k, v in aux.items()]))),
                file=self.logger)
          if fun_after_report is not None:
            fun_after_report(epoch_idx, aux, 'test')
        else:
          test_t = time.time() - test_t0

    # finally
    self._report_train_metrics = {k: np.asarray(v) for k, v in report_train_metric.items()}
    self._detailed_train_metrics = {k: np.asarray(v) for k, v in detailed_train_metric.items()}
    self._report_test_metrics = {k: np.asarray(v) for k, v in report_test_metric.items()}
    self._detailed_test_metrics = {k: np.asarray(v) for k, v in detailed_test_metric.items()}
    self.progress_bar = true_progress_bar

  def _get_f_loss(self, shared_args=None, jit=True) -> Callable:
    """Get loss function."""
    if shared_args is None:
      shared_args = dict()
    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args2)
    if shared_args_str not in self._f_loss_compiled:
      self._f_loss_compiled[shared_args_str] = partial(self._step_func_loss, shared_args)
      if self.jit[c.LOSS_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self._dyn_vars)
        dyn_vars.update(self.vars(level=0))
        self._f_loss_compiled[shared_args_str] = bm.jit(self._f_loss_compiled[shared_args_str],
                                                        dyn_vars=dyn_vars.unique())
    return self._f_loss_compiled[shared_args_str]

  def _get_f_grad(self, shared_args=None) -> Callable:
    """Get gradient function."""
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_grad_compiled:
      _f_loss_internal = self._get_f_loss(shared_args, jit=False)
      dyn_vars = self.target.vars()
      dyn_vars.update(self._dyn_vars)
      dyn_vars = dyn_vars.unique()
      tran_vars = dyn_vars.subset(bm.TrainVar)
      grad_f = bm.grad(_f_loss_internal,
                       dyn_vars=dyn_vars,
                       grad_vars=tran_vars,
                       return_value=True,
                       has_aux=self.loss_has_aux)
      self._f_grad_compiled[shared_args_str] = grad_f
    return self._f_grad_compiled[shared_args_str]

  def _get_f_train(self, shared_args=None) -> Callable:
    """Get training function."""
    if shared_args is None: shared_args = dict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'Only supports dict for "shared_args". '
                       f'But got {type(shared_args)}: {shared_args}')

    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_fit_compiled:
      self._f_fit_compiled[shared_args_str] = partial(self._step_func_fit, shared_args)
      if self.jit[c.FIT_PHASE]:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.optimizer.vars())
        if isinstance(self._loss_func, bm.BrainPyObject):
          dyn_vars.update(self._loss_func)
        dyn_vars.update(self._dyn_vars)
        dyn_vars.update(self.vars(level=0))
        dyn_vars = dyn_vars.unique()
        self._f_fit_compiled[shared_args_str] = bm.jit(self._f_fit_compiled[shared_args_str],
                                                       dyn_vars=dyn_vars)
    return self._f_fit_compiled[shared_args_str]

  def _step_func_loss(self, shared_args, inputs, targets):
    raise NotImplementedError

  def _step_func_fit(self, shared_args, inputs, targets):
    raise NotImplementedError


class BPTT(BPTrainer):
  """The trainer implementing the back-propagation through time (BPTT)
  algorithm for training dyamical systems.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: DynamicalSystem
    The target model to train.

  loss_fun: str, callable
    The loss function.

    - If it is a string, it should be the function chosen from ``brainpy.losses`` module.
    - Otherwise, a callable function which receives argument of ``(predicts, targets)``
      should be provided.

    .. note::
       If ``monitors`` has been set in the trainer, the ``predicts`` contains two
       parts: the network history prediction outputs, and the monitored values.

       see BrainPy examples for more information.
  loss_has_aux: bool
    To indicate whether the loss function returns auxiliary data expect the loss.
    Moreover, all auxiliary data should be a dict, whose key is used for logging
    item name and its data is used for the corresponding value.
    For example,

    .. code-block:: python

       def loss_fun(predicts, targets):
          return loss, {'acc': acc, 'spike_num': spike_num}
  optimizer: Optimizer
    The optimizer used for training. Should be an instance of :py:class:`~.Optimizer`.
  numpy_mon_after_run: bool
    Make the monitored results as NumPy arrays.
  logger: Any
    A file-like object (stream). Used to output the running results. Default is the current `sys.stdout`.
  data_first_axis: str
    To indicate whether the first axis is the batch size (``data_first_axis='B'``) or the
    time length (``data_first_axis='T'``).
  """

  def _step_func_loss(self, shared_args, inputs, targets):
    num_step = self._get_input_time_step(xs=inputs)
    indices = jnp.arange(num_step, dtype=bm.int_)
    times = indices * self.dt + self.t0
    indices = indices + self.i0
    if isinstance(self.target.mode, bm.BatchingMode) and self.data_first_axis == 'B':
      inputs = tree_map(lambda x: bm.moveaxis(x, 0, 1), inputs, is_leaf=lambda x: isinstance(x, bm.Array))
    inputs = (times, indices, inputs)
    outs, mons = self._predict(xs=inputs, shared_args=shared_args)
    predicts = (outs, mons) if len(mons) > 0 else outs
    return self._loss_func(predicts, targets)

  def _step_func_fit(self, shared_args, inputs, targets):
    res = self._get_f_grad(shared_args)(inputs, targets)
    self.optimizer.update(res[0])
    return res[1:]


class BPFF(BPTrainer):
  """
  The trainer implementing back propagation algorithm
  for feedforward neural networks.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  """

  def _step_func_loss(self, shared_args, inputs, targets):
    outputs, mon = self._get_f_predict(shared_args)(inputs)
    outs = (outputs, mon) if len(mon) > 0 else outputs
    loss = self._loss_func(outs, targets)
    return loss

  def _step_func_fit(self, shared_args, inputs, targets):
    res = self._get_f_grad(shared_args)(inputs, targets)
    self.optimizer.update(res[0])
    return res[1:]

  def _step_func_predict(self, shared, x=None):
    assert self.data_first_axis == 'B', f'There is no time dimension when using the trainer {self.__class__.__name__}.'

    # input step
    self.target.clear_input()
    self._step_func_input(shared)

    # dynamics update step
    args = (shared,) if x is None else (shared, x)
    out = self.target(*args)

    # monitor step
    mon = self._step_func_monitor(shared)
    return out, mon

  def _get_f_predict(self, shared_args: Dict = None, jit: bool = True):
    if shared_args is None:
      shared_args = tools.DotDict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_args" must be a dict, but got {type(shared_args)}')

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_predict_compiled:

      self._f_predict_compiled[shared_args_str] = partial(self._step_func_predict, shared_args)
      if self.jit[c.PREDICT_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self._dyn_vars)
        dyn_vars = dyn_vars - dyn_vars.subset(bm.VariableView)
        self._f_predict_compiled[shared_args_str] = bm.jit(self._f_predict_compiled[shared_args_str],
                                                           dyn_vars=dyn_vars.unique())
    return self._f_predict_compiled[shared_args_str]

  def predict(
      self,
      inputs: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]],
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
    inputs: ArrayType, dict
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
    output: ArrayType, dict
      The model output.
    """
    if shared_args is None: shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', False)

    # reset the model states
    if reset_state:
      self.target.reset_state(self._get_input_batch_size(xs=inputs))
      self.reset_state()
    # init monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items
    # prediction
    if eval_time: t0 = time.time()
    outs, hists = self._predict(xs=inputs, shared_args=shared_args)
    if eval_time: t1 = time.time()
    # post-running for monitors
    for key in hists.keys():
      self.mon[key] = bm.asarray(hists[key])
    if self.numpy_mon_after_run:
      for key in hists.keys():
        self.mon[key] = np.asarray(self.mon[key])
    return (t1 - t0, outs) if eval_time else outs


class OnlineBPTT(BPTT):
  def _step_func_loss(self, shared_args, t, i, input_, target_):
    outputs, mon = self._get_f_predict_one_step(shared_args)(t, i, input_)
    predicts = (outputs, mon) if len(mon) > 0 else outputs
    return self._loss_func(predicts, target_)

  def _get_f_loss(self, shared_args=None, jit=True) -> Callable:
    """Get loss function."""
    if shared_args is None: shared_args = dict()

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args_str = serialize_kwargs(shared_args2)
    if shared_args_str not in self._f_loss_compiled:

      self._f_loss_compiled[shared_args_str] = partial(self._step_func_loss, shared_args)
      if self.jit[c.LOSS_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        dyn_vars = dyn_vars - dyn_vars.subset(bm.VariableView)
        self._f_loss_compiled[shared_args_str] = bm.jit(self._f_loss_compiled[shared_args_str],
                                                        dyn_vars=dyn_vars)
    return self._f_loss_compiled[shared_args_str]

  def _get_f_train(self, shared_args=None) -> Callable:
    """Get training function."""
    if shared_args is None: shared_args = dict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'Only supports dict for "shared_args". '
                       f'But got {type(shared_args)}: {shared_args}')
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_fit_compiled:

      def train_step(*x):
        # t, i, input_, target_ = x
        res = self._get_f_grad(shared_args)(*x)
        self.optimizer.update(res[0])
        return res[1:]

      if self.jit[c.FIT_PHASE]:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        dyn_vars = dyn_vars - dyn_vars.subset(bm.VariableView)
        run_func = lambda all_inputs: bm.for_loop(train_step, all_inputs, dyn_vars=dyn_vars.unique())

      else:
        def run_func(xs):
          times, indices, inputs, targets = xs
          losses = []
          for i in range(times.shape[0]):
            # data at time i
            x = tree_map(lambda x: x[i], inputs, is_leaf=_is_brainpy_array)
            y = tree_map(lambda x: x[i], targets, is_leaf=_is_brainpy_array)
            # step at the i
            loss = train_step(times[i], indices[i], x, y)
            # append output and monitor
            losses.append(loss)
          return bm.asarray(losses)

      def train_fun(inputs, targets):
        times, indices, inputs, num_step, _, duration, _ = self._format_xs(
          None, inputs, inputs_are_batching=True, move_axis=True)
        targets = tree_map(lambda x: bm.moveaxis(x, 0, 1), targets, is_leaf=_is_brainpy_array)
        ls = run_func([times, indices, inputs, targets])
        self.i0 += num_step
        self.t0 += duration
        return ls

      self._f_fit_compiled[shared_args_str] = train_fun
    return self._f_fit_compiled[shared_args_str]

  def _get_f_predict_one_step(self, shared_args: Dict = None, jit: bool = False):
    if shared_args is None: shared_args = tools.DotDict()
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_args" must be a dict, '
                       f'but got {type(shared_args)}')

    shared_args2 = {k: v for k, v in shared_args.items()}
    shared_args2['_local_jit_'] = jit
    shared_args2['_one_step_'] = True
    shared_args_str = serialize_kwargs(shared_args)
    if shared_args_str not in self._f_predict_compiled:

      monitor_func = self._build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

      def run_func(t, i, x):
        shared = tools.DotDict(t=t, i=i, dt=self.dt)
        shared.update(shared_args)
        self.target.clear_input()
        outs = self.target(shared, x)
        hist = monitor_func(shared)
        return outs, hist

      if self.jit[c.FIT_PHASE] and jit:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        dyn_vars = dyn_vars - dyn_vars.subset(bm.VariableView)
        self._f_predict_compiled[shared_args_str] = bm.jit(run_func, dyn_vars=dyn_vars.unique())
      else:
        self._f_predict_compiled[shared_args_str] = run_func
    return self._f_predict_compiled[shared_args_str]


class OTTT(BPTrainer):
  pass
