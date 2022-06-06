# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable

import tqdm.auto
from jax.experimental.host_callback import id_tap

import numpy as np
from brainpy.base import Base
import brainpy.math as bm
from brainpy.errors import NoImplementationError
from brainpy.compat.nn.algorithms.offline import get, RidgeRegression, OfflineAlgorithm
from brainpy.compat.nn.base import Node, Network
from brainpy.compat.nn.utils import serialize_kwargs
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'OfflineTrainer',
  'RidgeTrainer',
]


class OfflineTrainer(RNNTrainer):
  """Offline trainer for models with recurrent dynamics.

  Parameters
  ----------
  target: Node
    The target model to train.
  fit_method: OfflineAlgorithm, Callable, dict, str
    The fitting method applied to the target model.
    - It can be a string, which specify the shortcut name of the training algorithm.
      Like, ``fit_method='ridge'`` means using the Ridge regression method.
      All supported fitting methods can be obtained through
      :py:func:`brainpy.nn.runners.get_supported_offline_methods`
    - It can be a dict, whose "name" item specifies the name of the training algorithm,
      and the others parameters specify the initialization parameters of the algorithm.
      For example, ``fit_method={'name': 'ridge', 'beta': 1e-4}``.
    - It can be an instance of :py:class:`brainpy.nn.runners.OfflineAlgorithm`.
      For example, ``fit_meth=bp.nn.runners.RidgeRegression(beta=1e-5)``.
    - It can also be a callable function, which receives three arguments "targets", "x" and "y".
      For example, ``fit_method=lambda targets, x, y: numpy.linalg.lstsq(x, targets)[0]``.
  **kwargs
    The other general parameters for RNN running initialization.
  """

  def __init__(
      self,
      target: Node,
      fit_method: Union[OfflineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    self.true_numpy_mon_after_run = kwargs.get('numpy_mon_after_run', True)
    kwargs['numpy_mon_after_run'] = False
    super(OfflineTrainer, self).__init__(target=target, **kwargs)

    # training method
    if fit_method is None:
      fit_method = RidgeRegression(beta=1e-7)
    elif isinstance(fit_method, str):
      fit_method = get(fit_method)()
    elif isinstance(fit_method, dict):
      name = fit_method.pop('name')
      fit_method = get(name)(**fit_method)
    if not callable(fit_method):
      raise ValueError(f'"train_method" must be an instance of callable function, '
                       f'but we got {type(fit_method)}.')
    self.fit_method = fit_method
    # check the required interface in the trainable nodes
    self._check_interface()

    # set the training method
    for node in self.train_nodes:
      node.offline_fit_by = fit_method

    # update dynamical variables
    if isinstance(self.fit_method, Base):
      self.dyn_vars.update(self.fit_method.vars().unique())

    # add the monitor items which are needed for the training process
    self._added_items = self._add_monitor_items()

    # training function
    self._f_train = dict()

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}jit={self.jit}, \n\t'
            f'{prefix}fit_method={self.fit_method})')

  def fit(
      self,
      train_data: Sequence,
      test_data=None,
      reset: bool = False,
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
    train_data: sequence of data
      It should be a pair of `(X, Y)` train set.
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
      or a tuple/list representing `(X, Y)` data. But this argument
      is supported in offline trainers.
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

    # set initial states
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)

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
          self.mon[f'{node.name}.inputs'] = inputs
          self._added_items.add(f'{node.name}.inputs')
    elif isinstance(self.target, Node):
      if self.target in self.train_nodes:
        inputs = self.target.data_pass_func({self.target.name: xs[self.target.name]})
        self.mon[f'{self.target.name}.inputs'] = inputs
        self._added_items.add(f'{self.target.name}.inputs')

    # format target data
    ys = self._check_ys(ys, num_batch=num_batch, num_step=num_step, move_axis=False)

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=len(self.train_nodes))
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # training
    monitor_data = dict()
    for node in self.train_nodes:
      monitor_data[f'{node.name}.inputs'] = self.mon.get(f'{node.name}.inputs', None)
      monitor_data[f'{node.name}.feedbacks'] = self.mon.get(f'{node.name}.feedbacks', None)
    self.f_train(shared_kwargs)(monitor_data, ys)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for key in self._added_items:
      self.mon.pop(key)
    if self.true_numpy_mon_after_run:
      for key in self.mon.keys():
        if key != 'var_names':
          self.mon[key] = np.asarray(self.mon[key])

  def f_train(self, shared_kwargs: Dict = None) -> Callable:
    """Get training function."""
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
          node.offline_fit(targets, ff, **shared_kwargs)
        else:
          node.offline_fit(targets, ff, fb, **shared_kwargs)
        if self.progress_bar:
          id_tap(lambda *args: self._pbar.update(), ())

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      train_func = bm.jit(train_func, dyn_vars=dyn_vars.unique())
    return train_func

  def _add_monitor_items(self):
    added_items = set()
    if isinstance(self.target, Network):
      for node in self.train_nodes:
        if node not in self.target.entry_nodes:
          if f'{node.name}.inputs' not in self.mon.var_names:
            self.mon.var_names += (f'{node.name}.inputs', )
            self.mon[f'{node.name}.inputs'] = []
            added_items.add(f'{node.name}.inputs')
        if node in self.target.fb_senders:
          if f'{node.name}.feedbacks' not in self.mon.var_names:
            self.mon.var_names += (f'{node.name}.feedbacks',)
            self.mon[f'{node.name}.feedbacks'] = []
            added_items.add(f'{node.name}.feedbacks')
    else:
      # brainpy.nn.Node instance does not need to monitor its inputs
      pass
    return added_items

  def _check_interface(self):
    for node in self.train_nodes:
      if hasattr(node.offline_fit, 'not_implemented'):
        if node.offline_fit.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "offline_fit()" function. '
          )


class RidgeTrainer(OfflineTrainer):
  """
  Trainer of ridge regression, also known as regression with Tikhonov regularization.

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
    super(RidgeTrainer, self).__init__(target=target,
                                       fit_method=dict(name='ridge', beta=beta),
                                       **kwargs)
