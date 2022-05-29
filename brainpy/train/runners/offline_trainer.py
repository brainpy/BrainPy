# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable

import tqdm.auto
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy.base import Base
from brainpy.errors import NoImplementationError
from brainpy.train.algorithms.offline import get, RidgeRegression, OfflineAlgorithm
from brainpy.train.base import TrainingSystem
from brainpy.train.utils import serialize_kwargs
from brainpy.types import Tensor
from .base_runner import DSTrainer

__all__ = [
  'OfflineTrainer',
  'RidgeTrainer',
]


class OfflineTrainer(DSTrainer):
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
      target: TrainingSystem,
      fit_method: Union[OfflineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    self.true_numpy_mon_after_run = kwargs.get('numpy_mon_after_run', True)
    kwargs['numpy_mon_after_run'] = False
    super(OfflineTrainer, self).__init__(target=target, **kwargs)

    # get all trainable nodes
    self.train_nodes = self._get_trainable_nodes()

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

    # initialize the fitting method
    for node in self.train_nodes:
      node.offline_init()

    # update dynamical variables
    if isinstance(self.fit_method, Base):
      self.dyn_vars.update(self.fit_method.vars().unique())

    # training function
    self._f_train = dict()

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}fit_method={self.fit_method})')

  def fit(
      self,
      train_data: Sequence,
      reset_state: bool = False,
      shared_kwargs: Dict = None,
  ):
    """Fit the target model according to the given training and testing data.

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
    reset_state: bool
      Whether reset the initial states of the target model.
    shared_kwargs: dict
      The shared keyword arguments for the target models.
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
    xs, ys = train_data

    # prediction, get all needed data
    _ = self.predict(xs=xs, reset_state=reset_state)

    # get all input data
    xs, num_step, num_batch = self._check_xs(xs, move_axis=False)

    # check target data
    ys = self._check_ys(ys, num_batch=num_batch, num_step=num_step, move_axis=False)

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=len(self.train_nodes))
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # training
    monitor_data = dict()
    for node in self.train_nodes:
      key = f'{node.name}-fit_record'
      monitor_data[key] = self.mon.get(key)
    self.f_train(shared_kwargs)(monitor_data, ys)
    del monitor_data

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for node in self.train_nodes:
      self.mon.item_contents.pop(f'{node.name}-fit_record')
    if self.true_numpy_mon_after_run:
      self.mon.numpy()

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
        fit_record = monitor_data[f'{node.name}-fit_record']
        targets = target_data[node.name]
        node.offline_fit(targets, fit_record, shared_kwargs)
        if self.progress_bar:
          id_tap(lambda *args: self._pbar.update(), ())

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      train_func = bm.jit(train_func, dyn_vars=dyn_vars.unique())
    return train_func

  def build_monitors(self, return_without_idx, return_with_idx, flatten=False):
    def func(_t, _dt):
        res = {k: v.value for k, v in return_without_idx.items()}
        res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
        res.update({k: f(_t, _dt) for k, f in self.fun_monitors.items()})
        res.update({f'{node.name}-fit_record': node.fit_record for node in self.train_nodes})
        # for node in self.train_nodes:
        #   node.fit_record.clear()
        return res

    return func

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
      if hasattr(node.offline_init, 'not_implemented'):
        if node.offline_init.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "offline_init()" function. '
          )


class RidgeTrainer(OfflineTrainer):
  """
  Trainer of ridge regression, also known as regression with Tikhonov regularization.

  Parameters
  ----------
  target: TrainingSystem
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
