# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Union, Callable, Tuple

import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map

import brainpy.math as bm
from brainpy.algorithms.online import get, OnlineAlgorithm, RLS
from brainpy.base import Base
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import NoImplementationError
from brainpy.modes import TrainingMode
from brainpy.tools.checking import serialize_kwargs
from brainpy.tools.others.dicts import DotDict
from brainpy.types import Array, Output
from .base import DSTrainer

__all__ = [
  'OnlineTrainer',
  'ForceTrainer',
]


class OnlineTrainer(DSTrainer):
  """Online trainer for models with recurrent dynamics.

  Parameters
  ----------
  target: DynamicalSystem
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
      target: DynamicalSystem,
      fit_method: Union[OnlineAlgorithm, Callable, Dict, str] = None,
      **kwargs
  ):
    super(OnlineTrainer, self).__init__(target=target, **kwargs)

    # get all trainable nodes
    nodes = self.target.nodes(level=-1, include_self=True).subset(DynamicalSystem).unique()
    self.train_nodes = tuple([node for node in nodes.values() if isinstance(node.mode, TrainingMode)])
    if len(self.train_nodes) == 0:
        raise ValueError('Found no trainable nodes.')

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

  def predict(
      self,
      inputs: Union[Array, Sequence[Array], Dict[str, Array]],
      reset_state: bool = False,
      shared_args: Dict = None,
      eval_time: bool = False
  ) -> Output:
    """Prediction function.

    What's different from `predict()` function in :py:class:`~.DynamicalSystem` is that
    the `inputs_are_batching` is default `True`.

    Parameters
    ----------
    inputs: Array, sequence of Array, dict of Array
      The input values.
    reset_state: bool
      Reset the target state before running.
    shared_args: dict
      The shared arguments across nodes.
    eval_time: bool
      Whether we evaluate the running time or not?

    Returns
    -------
    output: Array, sequence of Array, dict of Array
      The running output.
    """
    outs = super(OnlineTrainer, self).predict(inputs=inputs,
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

    # format input data
    times, indices, xs, num_step, num_batch, duration, _ = self._format_xs(
      None, inputs=xs, inputs_are_batching=True)

    # format target data
    ys = self._check_ys(ys, num_batch=num_batch, num_step=num_step, move_axis=True)

    # reset the model states
    if reset_state:
      self.target.reset_state(num_batch)
      self.reset_state()

    # init monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Train {num_step} steps: ", refresh=True)

    # prediction
    outs, hists = self._fit(xs=(times, indices, xs), ys=ys, shared_args=shared_args)

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # post-running for monitors
    hists['ts'] = times + self.dt
    if self.numpy_mon_after_run:
      hists = tree_map(lambda a: np.asarray(a), hists, is_leaf=lambda a: isinstance(a, bm.JaxArray))
    for key in hists.keys():
      self.mon[key] = hists[key]
    self.i0 += times.shape[0]
    self.t0 += duration
    return outs

  def _fit(
      self,
      xs: Tuple,
      ys: Union[Array, Sequence[Array], Dict[str, Array]],
      shared_args: Dict = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: tuple
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    ys: Array, sequence of Array, dict of Array
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _fit_func = self._get_fit_func(shared_args)
    hists = _fit_func(xs + (ys,))
    hists = tree_map(lambda x: bm.moveaxis(x, 0, 1), hists,
                     is_leaf=lambda x: isinstance(x, bm.JaxArray))
    return hists

  def _get_fit_func(self, shared_args: Dict = None):
    if shared_args is None: shared_args = dict()
    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._f_train:
      self._f_train[shared_kwargs_str] = self._make_fit_func(shared_args)
    return self._f_train[shared_kwargs_str]

  def _make_fit_func(self, shared_args: Dict):
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, but got {type(shared_args)}')

    monitor_func = self.build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

    def _step_func(t, i, x, ys):
      shared = DotDict(t=t, dt=self.dt, i=i)

      # input step
      self.target.clear_input()
      self._input_step(shared)

      # update step
      shared.update(shared_args)
      args = (shared,) if x is None else (shared, x)
      out = self.target(*args)

      # monitor step
      monitors = monitor_func(shared)
      for node in self.train_nodes:
        fit_record = monitors.pop(f'{node.name}-fit_record')
        target = ys[node.name]
        node.online_fit(target, fit_record)

      # finally
      if self.progress_bar:
        id_tap(lambda *arg: self._pbar.update(), ())
      return out, monitors

    if self.jit['fit']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      return lambda all_inputs: bm.for_loop(_step_func, dyn_vars.unique(), all_inputs)

    else:
      def run_func(all_inputs):
        times, indices, xs, ys = all_inputs
        outputs = []
        monitors = {key: [] for key in ((set(self.mon.keys()) - {'ts'}) | set(self.fun_monitors.keys()))}
        for i in range(times.shape[0]):
          x = tree_map(lambda x: x[i], xs)
          y = tree_map(lambda x: x[i], ys)
          output, mon = _step_func(times[i], indices[i], x, y)
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

  def build_monitors(self, return_without_idx, return_with_idx, shared_args: dict):
    if shared_args.get('fit', False):
      def func(tdi):
        res = {k: v.value for k, v in return_without_idx.items()}
        res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
        res.update({k: f(tdi) for k, f in self.fun_monitors.items()})
        res.update({f'{node.name}-fit_record': node.fit_record for node in self.train_nodes})
        return res
    else:
      def func(tdi):
        res = {k: v.value for k, v in return_without_idx.items()}
        res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
        res.update({k: f(tdi) for k, f in self.fun_monitors.items()})
        return res

    return func


class ForceTrainer(OnlineTrainer):
  """FORCE learning."""

  def __init__(self, target, alpha=1., **kwargs):
    super(ForceTrainer, self).__init__(target=target,
                                       fit_method=RLS(alpha=alpha),
                                       **kwargs)
