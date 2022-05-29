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
from brainpy.compat.nn.base import Node, Network
from brainpy.compat.nn.utils import check_data_batch_size, serialize_kwargs
from brainpy.tools.checking import check_dict_data, check_float
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'BPTT',
  'BPFF',
]


class BPTT(RNNTrainer):
  """
  The trainer implementing back propagation through time (BPTT)
  algorithm for recurrent neural networks.

  """

  def __init__(
      self,
      target: Node,

      # arguments for BPTT trainer
      loss: Union[str, Callable],  # loss function
      optimizer: optim.Optimizer = None,  # optimizer
      max_grad_norm=None,
      shuffle_data: bool = True,
      jit: bool = True,

      # common arguments for RNNTrainer
      **kwargs
  ):
    super(BPTT, self).__init__(target=target, **kwargs)

    # jit settings
    if isinstance(jit, bool):
      self.jit = {'fit': jit, 'predict': jit, 'loss': jit}
    elif isinstance(jit, dict):
      jit = {key: val for key, val in jit.items()}
      self.jit = {'fit': jit.pop('fit', True),
                  'predict': jit.pop('predict', True),
                  'loss': jit.pop('loss', True)}
      if len(jit):
        raise ValueError(f'Unknown jit setting for {jit.keys()}')
    else:
      raise ValueError(f'Unknown "jit" setting: {jit}')

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

    # target/output mapping types
    self._mapping_type = None

    # functions
    self._f_loss = dict()
    self._f_train = dict()
    self._f_grad = dict()

    # training parameters
    self.max_grad_norm = max_grad_norm  # gradient clipping
    self.shuffle_data = shuffle_data

    # initialize the optimizer
    if not self.target.is_initialized:
      raise ValueError('Please initialize the target model first by calling "initialize()" function.')
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
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Union[Tensor, Dict[str, Tensor]] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
      reset: bool = True,
      shared_kwargs: Dict = None,
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
    shared_kwargs: dict
      Shared keyword arguments for the given target model.
    reset: bool
      Whether reset the model states. Default True.

    forced_states: dict
      The fixed node states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`. Default None.

      .. versionadded:: 2.1.4

    forced_feedbacks: dict
      The fixed feedback states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`. Default None.

      .. versionadded:: 2.1.4

    initial_states: JaxArray, ndarray, dict
      The initial states. Each tensor in ``initial_states`` must be a
      tensor with the shape of `(num_sample, num_feature)`.

      .. versionadded:: 2.1.4

    initial_feedbacks: dict
      The initial feedbacks for the node in the network model.
      Each tensor in ``initial_feedbacks`` must be a
      tensor with the shape of `(num_sample, num_feature)`.

      .. versionadded:: 2.1.4

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # check forced states/feedbacks
    return super(BPTT, self).predict(xs=xs,
                                     forced_states=forced_states,
                                     forced_feedbacks=forced_feedbacks,
                                     initial_states=initial_states,
                                     initial_feedbacks=initial_feedbacks,
                                     reset=reset,
                                     shared_kwargs=shared_kwargs)

  def fit(
      self,
      train_data: Union[Callable, Sequence],
      test_data: Union[Callable, Sequence] = None,
      num_batch: int = 32,
      num_train: int = 100,
      num_report: int = 100,
      reset: bool = True,
      shared_kwargs: Dict = None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
      initial_states: Union[Tensor, Dict[str, Tensor]] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
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
    num_train: int
      The number of training epoch. Default 100.
    num_report: int
      The number of step to report the progress. Default 100 training steps.
    reset: bool
      Whether reset the initial states of the target model.
    shared_kwargs: dict
      The shared keyword arguments for the target models.
    forced_states: dict
      The fixed node states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`.

      .. versionadded:: 2.1.4

    forced_feedbacks: dict
      The fixed feedback states. Similar with ``xs``, each tensor in
      ``forced_states`` must be a tensor with the shape of
      `(num_sample, num_time, num_feature)`.

      .. versionadded:: 2.1.4

    initial_states: JaxArray, ndarray, dict
      The initial states. Each tensor in ``initial_states`` must be a
      tensor with the shape of `(num_sample, num_feature)`.

      .. versionadded:: 2.1.4

    initial_feedbacks: dict
      The initial feedbacks for the node in the network model.
      Each tensor in ``initial_feedbacks`` must be a
      tensor with the shape of `(num_sample, num_feature)`.

      .. versionadded:: 2.1.4

    """
    # training the model
    all_train_losses = []
    all_test_losses = []
    train_i = 0
    t0 = time.time()
    for _ in range(num_train):
      train_data_ = self._get_train_data(train_data, num_batch)

      # training set
      for x, y in train_data_:
        self._set_initial_states(initial_states)
        self._set_initial_feedbacks(initial_feedbacks)
        batch_size = check_data_batch_size(x)
        if reset:
          self.target.initialize(batch_size)
        loss = self.f_train(shared_kwargs)(x, y,
                                           forced_states=forced_states,
                                           forced_feedbacks=forced_feedbacks)
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
          batch_size = check_data_batch_size(x)
          if reset:
            self.target.initialize(batch_size)
          loss = self.f_loss(shared_kwargs)(x, y,
                                            forced_states=forced_states,
                                            forced_feedbacks=forced_feedbacks)
          all_test_losses.append(loss)

    self._train_losses = bm.asarray(all_train_losses)
    self._test_losses = bm.asarray(all_test_losses)

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

  @property
  def mapping_type(self):
    """Mapping type for the output and the target."""
    return self._mapping_type

  def _make_f_loss(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'Only supports dict for "shared_kwargs". '
                       f'But got {type(shared_kwargs)}: {shared_kwargs}')

    def loss_fun(inputs, targets, forced_states=None, forced_feedbacks=None):
      inputs = self._format_xs(inputs)
      targets = self._format_ys(targets)
      num_batch, num_step = list(inputs.values())[0].shape[:2]
      forced_states = self._check_forced_states(forced_states, num_batch, num_step)
      forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks, num_batch, num_step)
      inputs = {k: bm.moveaxis(v, 0, 1) for k, v in inputs.items()}
      outputs, _ = self._predict(xs=inputs,
                                 shared_kwargs=shared_kwargs,
                                 forced_states=forced_states,
                                 forced_feedbacks=forced_feedbacks)
      outputs = self._format_ys(outputs)
      loss = 0.
      for key, output in outputs.items():
        loss += self.loss_fun(output, targets[key])
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

    def train_func(inputs, targets, forced_states=None, forced_feedbacks=None):
      inputs = self._format_xs(inputs)
      targets = self._format_ys(targets)
      grads, loss = self.f_grad(shared_kwargs)(inputs,
                                               targets,
                                               forced_states=forced_states,
                                               forced_feedbacks=forced_feedbacks)
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

  def _format_ys(self, ys):
    if isinstance(ys, (bm.ndarray, jnp.ndarray)):
      if isinstance(self.target, Network):
        if len(self.target.exit_nodes) != 1:
          raise ValueError(f'The network {self.target} has '
                           f'{len(self.target.exit_nodes)} '
                           f'output nodes, while we only got '
                           f'one output data.')
        ys = {self.target.exit_nodes[0].name: ys}
      else:
        ys = {self.target.name: ys}
    else:
      exit_nodes = self.target.exit_nodes if isinstance(self.target, Network) else [self.target]
      for node in exit_nodes:
        if node.name not in ys:
          raise ValueError(f'The network has output node {node.name}, '
                           f'however, we did not get the corresponding '
                           f'output targets.')
    check_dict_data(ys, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    return ys

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
      xs = self._format_xs(xs)
      ys = self._format_ys(ys)
      yield xs, ys

  def _shuffle(self, xs, ys):
    key = jr.PRNGKey(seed=np.random.randint(0, 100000))
    if self._f_shuffle is None:
      def shuffle(xs, ys, key):
        xs = tree_map(lambda x: jr.permutation(key, x, axis=0), xs)
        ys = tree_map(lambda y: jr.permutation(key, y, axis=0), ys)
        return xs, ys

      self._f_shuffle = jit(shuffle)
    return self._f_shuffle(xs, ys, key)

  def _get_data_by_method2(self, dataset, num_batch, shuffle=False, ):
    assert isinstance(dataset, (tuple, list)) and len(dataset) == 2
    xs, ys = dataset
    xs = self._format_xs(xs)
    num_sample = self._get_xs_info(xs)
    ys = self._format_ys(ys)
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

  def _get_xs_info(self, xs):
    input_shapes = {}
    if isinstance(self.target, Network):
      for node in self.target.entry_nodes:
        name = self.target.entry_nodes[0].name
        input_shapes[name] = node._feedforward_shapes[name]
    else:
      name = self.target.name
      input_shapes[name] = self.target._feedforward_shapes[name]
    num_batch_sizes = []
    for key, val in xs.items():
      if key not in input_shapes:
        raise ValueError(f'Cannot find {key} in the required inputs. Please check!')
      shape = input_shapes[key]
      if bm.ndim(val) != len(shape) + 1:
        raise ValueError(f'Each tensor in "xs" must be a tensor of shape '
                         f'(num_sample, num_time, {str(shape[1:])[1:-1]}). '
                         f'But we got {val.shape}.')
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
      self, target: Node, **kwargs
  ):
    super(BPFF, self).__init__(target=target, **kwargs)

  def predict(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      initial_states: Union[Tensor, Dict[str, Tensor]] = None,
      initial_feedbacks: Dict[str, Tensor] = None,
      reset: bool = True,
      shared_kwargs: Dict = None,
      forced_states: Dict[str, Tensor] = None,
      forced_feedbacks: Dict[str, Tensor] = None,
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
    forced_states: None
      The fixed node states.
    forced_feedbacks: None
      The fixed feedback states.
    initial_states: JaxArray, ndarray, dict
      The initial states. Each tensor in ``initial_states`` must be a
      tensor with the shape of `(num_sample, num_feature)`.
    initial_feedbacks: dict
      The initial feedbacks for the node in the network model.
      Each tensor in ``initial_feedbacks`` must be a
      tensor with the shape of `(num_sample, num_feature)`.
    reset: bool
      Whether reset the model states.
    shared_kwargs: optional, dict
      The shared arguments across different layers.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # format input data
    xs = self._format_ys(xs)
    num_batch = self._get_xs_info(xs)
    # get forced data
    forced_states = self._check_forced_states(forced_states, num_batch)
    forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks, num_batch)
    # set initial states
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)
    # reset the model states
    if reset:
      self.target.initialize(num_batch)
    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # prediction
    outputs, hists = self._predict(xs=xs,
                                   forced_states=forced_states,
                                   forced_feedbacks=forced_feedbacks,
                                   shared_kwargs=shared_kwargs)
    # post-running for monitors
    for key in self.mon.item_names:
      self.mon.item_contents[key] = hists[key]
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return outputs

  def _check_forced_states(self, forced_states, num_batch):
    iter_forced_states = dict()
    if forced_states is not None:
      if isinstance(self.target, Network):
        nodes = [node.name for node in self.target.lnodes]
        if not isinstance(forced_states, dict):
          raise ValueError('"forced_states" must be a dict of (str, Tensor)')
        for key, tensor in forced_states.items():
          if not isinstance(key, str):
            raise ValueError(f'"forced_states" must be a dict of (str, tensor). '
                             f'But got a dict of ({type(key)}, {type(tensor)})')
          if key not in nodes:
            raise ValueError(f'Node "{key}" is not defined in the target model. '
                             f'We only detect: \n{self.target.lnodes}')
          if not isinstance(tensor, (bm.ndarray, jnp.ndarray)):
            raise ValueError(f'"forced_states" must a dict of (str, tensor), '
                             f'while we got ({type(key)}, {type(tensor)})')
          if bm.ndim(tensor) != self.target[key].state.ndim:
            raise ValueError(f'Must be a tensor with shape of (num_batch, '
                             f'{str(self.target[key].state.shape)[1:-1]}), '
                             f'but we got {tensor.shape}')
          if tensor.shape[0] != num_batch:
            raise ValueError(f'The number of the batch size ({tensor.shape[0]}) '
                             f'of the forced state of {key} does not '
                             f'match with the batch size in inputs {num_batch}.')
          if self.target[key].output_shape[1:] != tensor.shape[2:]:
            raise UnsupportedError(f'The forced state of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}. '
                                   f'Each tensor in forced state should have the shape '
                                   f'of (num_sample, num_time, num_feature) or '
                                   f'(num_sample, num_feature).')
          iter_forced_states[key] = bm.moveaxis(tensor, 0, 1)  # shape of (num_time, num_sample, num_feature)
      else:
        raise UnsupportedError('We do not support forced feedback state '
                               'for a single brainpy.nn.Node instance')
    return iter_forced_states

  def _check_forced_feedbacks(self, forced_feedbacks, num_batch):
    iter_forced_feedbacks = dict()
    if forced_feedbacks is not None:
      if isinstance(self.target, Network):
        if not isinstance(forced_feedbacks, dict):
          raise ValueError('"forced_feedbacks" must be a dict of (str, Tensor)')
        feedback_node_names = [node.name for node in self.target.feedback_nodes]
        for key, tensor in forced_feedbacks.items():
          if not isinstance(key, str):
            raise ValueError(f'"forced_feedbacks" must be a dict of (str, tensor). '
                             f'But got a dict of ({type(key)}, {type(tensor)})')
          if key not in feedback_node_names:
            raise ValueError(f'{self.target} has no feedback node {key}, '
                             f'it only has {feedback_node_names}')
          if not isinstance(tensor, (bm.ndarray, jnp.ndarray)):
            raise ValueError('"forced_feedbacks" must a dict of (str, tensor), '
                             'while we got ({type(key)}, {type(tensor)})')
          if bm.ndim(tensor) != self.target[key].fb_output.ndim:
            raise ValueError(f'Must be a tensor with shape of (num_batch, '
                             f'{str(self.target[key].fb_output.shape)[1:-1]}), '
                             f'but we got {tensor.shape}')
          if tensor.shape[0] != num_batch:
            raise ValueError(f'The number of the batch size ({tensor.shape[0]}) '
                             f'of the forced feedback of {key} does not '
                             f'match with the batch size in inputs {num_batch}.')
          if self.target[key].output_shape[1:] != tensor.shape[2:]:
            raise UnsupportedError(f'The forced feedback of {key} has the shape of '
                                   f'{tensor.shape}, which is not consistent with '
                                   f'its output shape {self.target[key].output_shape}. '
                                   f'Each tensor in forced feedback should have the shape '
                                   f'of (num_sample, num_time, num_feature) or '
                                   f'(num_sample, num_feature).')
          iter_forced_feedbacks[key] = bm.moveaxis(tensor, 0, 1)  # shape of (num_time, num_sample, num_feature)
      else:
        raise UnsupportedError('We do not support forced states for '
                               'a single brainpy.nn.Node instance')
    return iter_forced_feedbacks

  def _predict(
      self,
      xs: Dict[str, Tensor],
      shared_kwargs: Dict = None,
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
    shared_kwargs: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self._get_predict_func(shared_kwargs)
    # rune the model
    forced_states = dict() if forced_states is None else forced_states
    forced_feedbacks = dict() if forced_feedbacks is None else forced_feedbacks
    return _predict_func(xs, forced_states, forced_feedbacks)

  def _make_f_loss(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'Only supports dict for "shared_kwargs". '
                       f'But got {type(shared_kwargs)}: {shared_kwargs}')

    def loss_fun(inputs, targets, forced_states=None, forced_feedbacks=None):
      inputs = self._format_xs(inputs)
      targets = self._format_ys(targets)
      num_batch, num_step = list(inputs.values())[0].shape[:2]
      forced_states = self._check_forced_states(forced_states, num_batch)
      forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks, num_batch)
      outputs, _ = self._predict(xs=inputs,
                                 shared_kwargs=shared_kwargs,
                                 forced_states=forced_states,
                                 forced_feedbacks=forced_feedbacks)
      outputs = self._format_ys(outputs)
      loss = 0.
      for key, output in outputs.items():
        loss += self.loss_fun(output, targets[key])
      return loss

    return loss_fun

  def _get_predict_func(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._predict_func:
      self._predict_func[shared_kwargs_str] = self._make_predict_func(shared_kwargs)
    return self._predict_func[shared_kwargs_str]

  def _make_predict_func(self, shared_kwargs: Dict):
    if not isinstance(shared_kwargs, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, '
                       f'but got {type(shared_kwargs)}')

    def run_func(xs, forced_states, forced_feedbacks):
      monitors = self.mon.item_contents.keys()
      return self.target(xs,
                         forced_states=forced_states,
                         forced_feedbacks=forced_feedbacks,
                         monitors=monitors,
                         **shared_kwargs)

    if self.jit['predict']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      run_func = bm.jit(run_func, dyn_vars=dyn_vars.unique())
    return run_func

  def _get_xs_info(self, xs):
    input_shapes = {}
    if isinstance(self.target, Network):
      for node in self.target.entry_nodes:
        name = self.target.entry_nodes[0].name
        input_shapes[name] = node._feedforward_shapes[name]
    else:
      name = self.target.name
      input_shapes[name] = self.target._feedforward_shapes[name]
    num_batch_sizes = []
    for key, val in xs.items():
      if key not in input_shapes:
        raise ValueError(f'Cannot find {key} in the required inputs. Please check!')
      shape = input_shapes[key]
      if bm.ndim(val) != len(shape):
        raise ValueError(f'Each tensor in "xs" must be a tensor of shape '
                         f'(num_sample, {str(shape[1:])[1:-1]}). '
                         f'But we got {val.shape}.')
      num_batch_sizes.append(val.shape[0])
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across tensors in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    return num_batch_sizes[0]
