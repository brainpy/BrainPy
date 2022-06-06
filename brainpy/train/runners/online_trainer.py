# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable

import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map

import brainpy.math as bm
from brainpy.base import Base
from brainpy.errors import NoImplementationError
from brainpy.train.algorithms.online import get, OnlineAlgorithm, RLS
from brainpy.train.base import TrainingSystem
from brainpy.train.utils import (serialize_kwargs, check_data_batch_size)
from brainpy.types import Tensor
from .base_runner import DSTrainer

__all__ = [
  'OnlineTrainer',
  'ForceTrainer',
]


class OnlineTrainer(DSTrainer):
  """Online trainer for models with recurrent dynamics.

  Parameters
  ----------
  target: Node
    The target model to train.
  fit_method: OnlineAlgorithm, Callable, dict, str
    The fitting method applied to the target model.
    - It can be a string, which specify the shortcut name of the training algorithm.
      Like, ``fit_method='ridge'`` means using the RLS method.
      All supported fitting methods can be obtained through
      :py:func:`brainpy.nn.runners.get_supported_online_methods`
    - It can be a dict, whose "name" item specifies the name of the training algorithm,
      and the others parameters specify the initialization parameters of the algorithm.
      For example, ``fit_method={'name': 'ridge', 'beta': 1e-4}``.
    - It can be an instance of :py:class:`brainpy.nn.runners.OnlineAlgorithm`.
      For example, ``fit_meth=bp.nn.runners.RLS(alpha=1e-5)``.
    - It can also be a callable function.
  **kwargs
    The other general parameters for RNN running initialization.
  """

  def __init__(
      self,
      target: TrainingSystem,
      fit_method: Union[OnlineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    super(OnlineTrainer, self).__init__(target=target, **kwargs)

    # get all trainable nodes
    self.train_nodes = self._get_trainable_nodes()

    # training method
    if fit_method is None:
      fit_method = RLS(alpha=1e-7)
    elif isinstance(fit_method, str):
      fit_method = get(fit_method)()
    elif isinstance(fit_method, dict):
      name = fit_method.pop('name')
      fit_method = get(name)(**fit_method)
    self.fit_method = fit_method
    if not callable(fit_method):
      raise ValueError(f'"train_method" must be an instance of callable function, '
                       f'but we got {type(fit_method)}.')

    # check the required interface in the trainable nodes
    self._check_interface()

    # set the training method
    for node in self.train_nodes:
      node.online_fit_by = fit_method

    # initialize the fitting method
    for node in self.train_nodes:
      node.online_init()

    # update dynamical variables
    if isinstance(self.fit_method, Base):
      self.dyn_vars.update(self.fit_method.vars().unique())

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
      reset_state: bool = False,
      shared_args: Dict = None,
  ):
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

    # format input data
    xs, num_step, num_batch = self._check_xs(xs, move_axis=True)

    # format target data
    ys = self._check_ys(ys, num_batch=num_batch, num_step=num_step, move_axis=True)

    # reset the model states
    if reset_state:
      self.target.reset_state(num_batch)

    # init monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Train {num_step} steps: ", refresh=True)

    # prediction
    hists = self._fit(xs=xs, ys=ys, shared_args=shared_args)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # post-running for monitors
    for key in hists.keys():
      self.mon[key] = hists[key]
    if self.numpy_mon_after_run:
      self.mon.ts = np.asarray(self.mon.ts)
      for key in hists.keys():
        self.mon[key] = np.asarray(self.mon[key])

  def _fit(
      self,
      xs: Dict[str, Tensor],
      ys: Dict[str, Tensor],
      shared_args: Dict = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    ys: dict
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self._get_fit_func(shared_args)
    hists = _predict_func([xs, ys])
    hists = tree_map(lambda x: bm.moveaxis(x, 0, 1), hists,
                     is_leaf=lambda x: isinstance(x, bm.JaxArray))
    return hists

  def _get_fit_func(self, shared_kwargs: Dict = None):
    if shared_kwargs is None: shared_kwargs = dict()
    shared_kwargs_str = serialize_kwargs(shared_kwargs)
    if shared_kwargs_str not in self._f_train:
      self._f_train[shared_kwargs_str] = self._make_fit_func(shared_kwargs)
    return self._f_train[shared_kwargs_str]

  def _make_fit_func(self, shared_args: Dict):
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, but got {type(shared_args)}')

    def _step_func(all_inputs):
      xs, ys = all_inputs
      t = 0.
      self._input_step(t, self.dt)
      if xs is None:
        args = (t, self.dt)
      else:
        args = (t, self.dt, xs)
      kwargs = dict()
      if len(shared_args):
        kwargs['shared_args'] = shared_args
      out = self.target.update(*args, **kwargs)
      monitors = self._monitor_step(t, self.dt)
      for node in self.train_nodes:
        fit_record = monitors.pop(f'{node.name}-fit_record')
        target = ys[node.name]
        node.online_fit(target, fit_record, shared_args)
      if self.progress_bar:
        id_tap(lambda *arg: self._pbar.update(), ())
      return out, monitors

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      f = bm.make_loop(_step_func, dyn_vars=dyn_vars.unique(), has_return=True)
      return lambda all_inputs: f(all_inputs)[1]

    else:
      def run_func(all_inputs):
        xs, ys = all_inputs
        outputs = []
        monitors = {key: [] for key in
                    set(self.mon.item_contents.keys()) |
                    set(self.fun_monitors.keys())}
        for i in range(check_data_batch_size(xs)):
          x = tree_map(lambda x: x[i], xs)
          y = tree_map(lambda x: x[i], ys)
          output, mon = _step_func((x, y))
          outputs.append(output)
          for key, value in mon.items():
            monitors[key].append(value)
        if outputs[0] is None:
          outputs = None
        else:
          outputs = bm.asarray(outputs)
        for key, value in monitors.items():
          monitors[key] = bm.asarray(value)
        return outputs, monitors
    return run_func

  def _check_interface(self):
    for node in self.train_nodes:
      if hasattr(node.online_fit, 'not_implemented'):
        if node.online_fit.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_fit()" function. '
          )
      if hasattr(node.online_init, 'not_implemented'):
        if node.online_init.not_implemented:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_init()" function. '
          )

  def build_monitors(self, return_without_idx, return_with_idx, flatten=False):
    def func(t, dt):
      res = {k: v.value for k, v in return_without_idx.items()}
      res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
      res.update({k: f(t, dt) for k, f in self.fun_monitors.items()})
      res.update({f'{node.name}-fit_record': {k: node.fit_record.pop(k)
                                              for k in node.fit_record.keys()}
                  for node in self.train_nodes})
      return res

    return func


class ForceTrainer(OnlineTrainer):
  """Force learning."""

  def __init__(self, target, alpha=1., **kwargs):
    fit_method = RLS(alpha=alpha)
    super(ForceTrainer, self).__init__(target=target,
                                       fit_method=fit_method,
                                       **kwargs)
