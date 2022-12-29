# -*- coding: utf-8 -*-

from functools import partial
from typing import Dict, Sequence, Union, Callable, Tuple

import jax
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map

from brainpy import math as bm, tools
from brainpy.algorithms.online import get, OnlineAlgorithm, RLS
from brainpy.check import serialize_kwargs
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import NoImplementationError
from brainpy.types import ArrayType, Output
from ._utils import format_ys
from .base import DSTrainer

__all__ = [
  'OnlineTrainer',
  'ForceTrainer',
]


class OnlineTrainer(DSTrainer):
  """Online trainer for models with recurrent dynamics.

  For more parameters, users should refer to :py:class:`~.DSRunner`.

  Parameters
  ----------
  target: DynamicalSystem
    The target model to train.

  fit_method: OnlineAlgorithm, Callable, dict, str
    The fitting method applied to the target model.

    - It can be a string, which specify the shortcut name of the training algorithm.
      Like, ``fit_method='rls'`` means using the RLS method.
      All supported fitting methods can be obtained through
      :py:func:`~.get_supported_online_methods`.
    - It can be a dict, whose "name" item specifies the name of the training algorithm,
      and the others parameters specify the initialization parameters of the algorithm.
      For example, ``fit_method={'name': 'rls', 'alpha': 0.1}``.
    - It can be an instance of :py:class:`brainpy.algorithms.OnlineAlgorithm`.
      For example, ``fit_meth=bp.algorithms.RLS(alpha=1e-5)``.
    - It can also be a callable function.

  kwargs: Any
    Other general parameters please see :py:class:`~.DSRunner`.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      fit_method: Union[OnlineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    super(OnlineTrainer, self).__init__(target=target, **kwargs)

    # get all trainable nodes
    nodes = self.target.nodes(level=-1, include_self=True).subset(DynamicalSystem).unique()
    self.train_nodes = tuple([node for node in nodes.values() if isinstance(node.mode, bm.TrainingMode)])
    if len(self.train_nodes) == 0:
      raise ValueError('Found no trainable nodes.')

    # check the required interface in the trainable nodes
    self._check_interface()

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

    # set the training method
    for node in self.train_nodes:
      node.online_fit_by = fit_method

    # initialize the fitting method
    for node in self.train_nodes:
      node.online_init()

    # training function
    self._f_fit_compiled = dict()

  def __repr__(self):
    name = self.__class__.__name__
    indent = ' ' * len(name)
    indent2 = indent + " " * len("target")
    return (f'{name}(target={tools.repr_context(str(self.target), indent2)}, \n'
            f'{indent}jit={self.jit}, \n'
            f'{indent}fit_method={self.fit_method})')

  def predict(
      self,
      inputs: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]],
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
    shared_args: dict
      The shared arguments across nodes.
    eval_time: bool
      Whether we evaluate the running time or not?

    Returns
    -------
    output: ArrayType
      The running output.
    """
    outs = super().predict(inputs=inputs,
                           reset_state=reset_state,
                           shared_args=shared_args,
                           eval_time=eval_time)
    for node in self.train_nodes:
      node.fit_record.clear()
    return outs

  def fit(
      self,
      train_data: Sequence,
      reset_state: bool = False,
      shared_args: Dict = None,
  ) -> Output:
    if shared_args is None: shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', True)

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

    # reset the model states
    if reset_state:
      num_batch = self._get_input_batch_size(xs)
      self.target.reset_state(num_batch)
      self.reset_state()

    # format input/target data
    ys = format_ys(self, ys)
    num_step = self._get_input_time_step(xs=xs)
    shared = tools.DotDict(i=bm.arange(num_step, dtype=bm.int_).value)
    shared['t'] = shared['i'] * self.dt
    shared['t'] += self.t0
    shared['i'] += self.i0

    if self.data_first_axis == 'B':
      xs = tree_map(lambda x: bm.moveaxis(x, 0, 1),
                    xs,
                    is_leaf=lambda x: isinstance(x, bm.Array))
      ys = tree_map(lambda y: bm.moveaxis(y, 0, 1),
                    ys,
                    is_leaf=lambda y: isinstance(y, bm.Array))

    # init monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Train {num_step} steps: ", refresh=True)

    # prediction
    outs, hists = self._fit(tix=(shared['t'], shared['i'], xs), ys=ys, shared_args=shared_args)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # post-running for monitors
    hists['ts'] = shared['t'] + self.dt
    if self.numpy_mon_after_run:
      hists = tree_map(lambda a: np.asarray(a), hists, is_leaf=lambda a: isinstance(a, bm.Array))
    for key in hists.keys():
      self.mon[key] = hists[key]
    self.i0 += shared['t'].shape[0]
    self.t0 += num_step * self.dt
    return outs

  def _fit(
      self,
      tix: Tuple,
      ys: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]],
      shared_args: Dict = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    tix: tuple
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    ys: ArrayType
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _fit_func = self._get_fit_func(shared_args)
    hists = _fit_func(tix + (ys,))
    hists = tree_map(lambda x: bm.moveaxis(x, 0, 1),
                     hists,
                     is_leaf=lambda x: isinstance(x, bm.Array))
    return hists

  def _get_fit_func(self, shared_args: Dict = None):
    if shared_args is None: shared_args = dict()
    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._f_fit_compiled:
      dyn_vars = self.vars().unique()
      dyn_vars = dyn_vars - dyn_vars.subset(bm.VariableView)

      def run_func(all_inputs):
        with jax.disable_jit(not self.jit['fit']):
          return bm.for_loop(partial(self._step_func_fit, shared_args),
                             all_inputs,
                             dyn_vars=dyn_vars)

      self._f_fit_compiled[shared_kwargs_str] = run_func
    return self._f_fit_compiled[shared_kwargs_str]

  def _step_func_fit(self, shared_args, t, i, x, ys):
    shared = tools.DotDict(t=t, dt=self.dt, i=i)

    # input step
    self.target.clear_input()
    self._step_func_input(shared)

    # update step
    shared.update(shared_args)
    args = (shared,) if x is None else (shared, x)
    out = self.target(*args)

    # monitor step
    monitors = self._step_func_monitor(shared)
    for node in self.train_nodes:
      fit_record = monitors.pop(f'{node.name}-fit_record')
      target = ys[node.name]
      node.online_fit(target, fit_record)

    # finally
    if self.progress_bar:
      id_tap(lambda *arg: self._pbar.update(), ())
    return out, monitors

  def _check_interface(self):
    for node in self.train_nodes:
      if hasattr(node.online_fit, 'not_customized'):
        if node.online_fit.not_customized:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_fit()" function. '
          )
      if hasattr(node.online_init, 'not_customized'):
        if node.online_init.not_customized:
          raise NoImplementationError(
            f'The node \n\n{node}\n\n'
            f'is set to be trainable with {self.__class__.__name__} method. '
            f'However, it does not implement the required training '
            f'interface "online_init()" function. '
          )

  def _step_func_monitor(self, shared):
    res = dict()
    for key, val in self._monitors.items():
      if callable(val):
        res[key] = val(shared)
      else:
        (variable, idx) = val
        if idx is None:
          res[key] = variable.value
        else:
          res[key] = variable[bm.asarray(idx)]
    if shared.get('fit', False):
      for node in self.train_nodes:
        res[f'{node.name}-fit_record'] = node.fit_record
    return res


class ForceTrainer(OnlineTrainer):
  """FORCE learning."""

  def __init__(self, target, alpha=1., **kwargs):
    super(ForceTrainer, self).__init__(target=target,
                                       fit_method=RLS(alpha=alpha),
                                       **kwargs)
