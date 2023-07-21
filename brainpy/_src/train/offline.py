# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable, Any

import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy import tools
from brainpy._src.context import share
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.runners import _call_fun_with_share
from brainpy.algorithms.offline import get, RidgeRegression, OfflineAlgorithm
from brainpy.errors import NoImplementationError
from brainpy.types import ArrayType, Output
from ._utils import format_ys
from .base import DSTrainer

__all__ = [
  'OfflineTrainer',
  'RidgeTrainer',
]


class OfflineTrainer(DSTrainer):
  """Offline trainer for models with recurrent dynamics.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: DynamicalSystem
    The target model to train.
  fit_method: OfflineAlgorithm, Callable, dict, str
    The fitting method applied to the target model.
    - It can be a string, which specify the shortcut name of the training algorithm.
      Like, ``fit_method='ridge'`` means using the Ridge regression method.
      All supported fitting methods can be obtained through
      :py:func:`~get_supported_offline_methods`.
    - It can be a dict, whose "name" item specifies the name of the training algorithm,
      and the others parameters specify the initialization parameters of the algorithm.
      For example, ``fit_method={'name': 'ridge', 'alpha': 0.1}``.
    - It can be an instance of :py:class:`brainpy.algorithms.OfflineAlgorithm`.
      For example, ``fit_meth=bp.algorithms.RidgeRegression(alpha=0.1)``.
    - It can also be a callable function, which receives three arguments "targets", "x" and "y".
      For example, ``fit_method=lambda targets, x, y: numpy.linalg.lstsq(x, targets)[0]``.

  kwargs: Any
    Other general parameters please see :py:class:`~.DSRunner`.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      fit_method: Union[OfflineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    self._true_numpy_mon_after_run = kwargs.get('numpy_mon_after_run', True)
    kwargs['numpy_mon_after_run'] = False
    super().__init__(target=target, **kwargs)

    # get all trainable nodes
    nodes = self.target.nodes(level=-1, include_self=True).subset(DynamicalSystem).unique()
    self.train_nodes = tuple([node for node in nodes.values() if isinstance(node.mode, bm.TrainingMode)])
    if len(self.train_nodes) == 0:
      raise ValueError('Found no trainable nodes.')

    # check the required interface in the trainable nodes
    self._check_interface()

    # training method
    if fit_method is None:
      fit_method = RidgeRegression(alpha=1e-7)
    elif isinstance(fit_method, str):
      fit_method = get(fit_method)()
    elif isinstance(fit_method, dict):
      name = fit_method.pop('name')
      fit_method = get(name)(**fit_method)
    if not callable(fit_method):
      raise ValueError(f'"train_method" must be an instance of callable function, '
                       f'but we got {type(fit_method)}.')
    self.fit_method = fit_method

    # set the training method
    for node in self.train_nodes:
      node.offline_fit_by = fit_method
    # training function
    self._jit_fun_train = bm.jit(self._fun_train, static_argnames=['shared_args'])

  def __repr__(self):
    name = self.__class__.__name__
    prefix = ' ' * len(name)
    return (f'{name}(target={self.target}, \n\t'
            f'{prefix}fit_method={self.fit_method})')

  def predict(
      self,
      inputs: Any,
      reset_state: bool = False,
      shared_args: Dict = None,
      eval_time: bool = False
  ) -> Output:
    """Prediction function.

    What's different from `predict()` function in :py:class:`~.DynamicalSystem` is that
    the `inputs_are_batching` is default `True`.

    Parameters
    ----------
    inputs: ArrayType
      The input values.
    reset_state: bool
      Reset the target state before running.
    eval_time: bool
      Whether we evaluate the running time or not?
    shared_args: dict
      The shared arguments across nodes.

    Returns
    -------
    output: ArrayType
      The running output.
    """
    outs = super().predict(inputs=inputs, reset_state=reset_state,
                           eval_time=eval_time, shared_args=shared_args)
    for node in self.train_nodes:
      node.fit_record.clear()
    return outs

  def fit(
      self,
      train_data: Sequence,
      reset_state: bool = False,
      shared_args: Dict = None,
  ) -> Output:
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
    shared_args: dict
      The shared keyword arguments for the target models.
    """
    if shared_args is None:
      shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', True)
    shared_args = tools.DotDict(shared_args)

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
    shared_args['fit'] = shared_args.get('fit', False)
    outs = self.predict(inputs=xs, reset_state=reset_state, shared_args=shared_args)

    # check target data
    ys = format_ys(self, ys)

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=len(self.train_nodes))
      self._pbar.set_description(f"Train {len(self.train_nodes)} nodes: ", refresh=True)

    # training
    monitor_data = dict()
    for node in self.train_nodes:
      key = f'{node.name}-fit_record'
      monitor_data[key] = self.mon.get(key)
    run_fun = self._jit_fun_train if self.jit['fit'] else self._fun_train
    shared_args['fit'] = True
    run_fun(monitor_data, ys, shared_args=shared_args)
    del monitor_data

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # final things
    for node in self.train_nodes:
      self.mon.pop(f'{node.name}-fit_record')
      node.fit_record.clear()  # clear fit records
    if self._true_numpy_mon_after_run:
      for key in self.mon.keys():
        self.mon[key] = np.asarray(self.mon[key])

    return outs

  def _fun_train(self,
                 monitor_data: Dict[str, ArrayType],
                 target_data: Dict[str, ArrayType],
                 shared_args: Dict = None):
    if shared_args is None:
      shared_args = dict()
    share.save(**shared_args)

    for node in self.train_nodes:
      fit_record = monitor_data[f'{node.name}-fit_record']
      targets = target_data[node.name]
      node.offline_fit(targets, fit_record)
      if self.progress_bar:
        id_tap(lambda *args: self._pbar.update(), ())

  def _step_func_monitor(self):
    res = dict()
    for key, val in self._monitors.items():
      if callable(val):
        res[key] = _call_fun_with_share(val)
      else:
        (variable, idx) = val
        if idx is None:
          res[key] = variable.value
        else:
          res[key] = variable[bm.asarray(idx)]
    if share.load('fit'):
      for node in self.train_nodes:
        res[f'{node.name}-fit_record'] = node.fit_record
    return res

  def _check_interface(self):
    for node in self.train_nodes:
      if not hasattr(node, 'offline_fit'):
        raise NoImplementationError(
          f'''
            The node
            
            {node}
            
            is set to be computing mode of {bm.training_mode} with {self.__class__.__name__}. 
            However, it does not implement the required training 
            interface "offline_fit()" function. 
            '''
        )


class RidgeTrainer(OfflineTrainer):
  """Trainer of ridge regression, also known as regression with Tikhonov regularization.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: TrainingSystem, DynamicalSystem
    The target model.
  alpha: float
    The regularization coefficient.
  """

  def __init__(self, target, alpha=1e-7, **kwargs):
    super().__init__(target=target, fit_method=dict(name='ridge', alpha=alpha), **kwargs)
