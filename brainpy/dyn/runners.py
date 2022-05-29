# -*- coding: utf-8 -*-

import time
from collections.abc import Iterable
from typing import Dict, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map, tree_flatten

from brainpy import math as bm
from brainpy.base.collector import TensorCollector
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import RunningError
from brainpy.running.runner import Runner
from brainpy.types import Tensor
from .utils import serialize_kwargs, check_data_batch_size

__all__ = [
  'DSRunner', 'ReportRunner',
]

SUPPORTED_INPUT_OPS = ['-', '+', '*', '/', '=']
SUPPORTED_INPUT_TYPE = ['fix', 'iter', 'func']


def check_and_format_inputs(host, inputs):
  """Check inputs and get the formatted inputs for the given population.

  Parameters
  ----------
  host : DynamicalSystem
      The host which contains all data.
  inputs : tuple, list
      The inputs of the population.

  Returns
  -------
  formatted_inputs : tuple, list
      The formatted inputs of the population.
  """

  # 1. check inputs
  # ---------
  if inputs is None:
    inputs = []
  if not isinstance(inputs, (tuple, list)):
    raise RunningError('"inputs" must be a tuple/list.')
  if len(inputs) > 0 and not isinstance(inputs[0], (list, tuple)):
    if isinstance(inputs[0], str):
      inputs = [inputs]
    else:
      raise RunningError('Unknown input structure, only support inputs '
                         'with format of "(target, value, [type, operation])".')
  for one_input in inputs:
    if not 2 <= len(one_input) <= 4:
      raise RunningError('For each target, you must specify '
                         '"(target, value, [type, operation])".')
    if len(one_input) == 3 and one_input[2] not in SUPPORTED_INPUT_TYPE:
      raise RunningError(f'Input type only supports '
                         f'"{SUPPORTED_INPUT_TYPE}", '
                         f'not "{one_input[2]}".')
    if len(one_input) == 4 and one_input[3] not in SUPPORTED_INPUT_OPS:
      raise RunningError(f'Input operation only supports '
                         f'"{SUPPORTED_INPUT_OPS}", '
                         f'not "{one_input[3]}".')

  # 2. get targets and attributes
  # ---------
  inputs_which_found_target = []
  inputs_not_found_target = []

  # checking 1: absolute access
  #    Check whether the input target node is accessible,
  #    and check whether the target node has the attribute
  nodes = host.nodes(method='absolute')
  nodes[host.name] = host
  for one_input in inputs:
    key = one_input[0]
    if not isinstance(key, str):
      raise RunningError(f'For each input, input[0] must be a string  to '
                         f'specify variable of the target, but we got {key}.')
    splits = key.split('.')
    target = '.'.join(splits[:-1])
    key = splits[-1]
    if target == '':
      real_target = host
    else:
      if target not in nodes:
        inputs_not_found_target.append(one_input)
        continue
      real_target = nodes[target]
    if not hasattr(real_target, key):
      raise RunningError(f'Input target key "{key}" is not defined in {real_target}.')
    inputs_which_found_target.append((real_target, key) + tuple(one_input[1:]))

  # checking 2: relative access
  #    Check whether the input target node is accessible
  #    and check whether the target node has the attribute
  if len(inputs_not_found_target):
    nodes = host.nodes(method='relative')
    for one_input in inputs_not_found_target:
      splits = one_input[0].split('.')
      target, key = '.'.join(splits[:-1]), splits[-1]
      if target not in nodes:
        raise RunningError(f'Input target "{target}" is not defined in {host}.')
      real_target = nodes[target]
      if not hasattr(real_target, key):
        raise RunningError(f'Input target key "{key}" is not defined in {real_target}.')
      inputs_which_found_target.append((real_target, key) + tuple(one_input[1:]))

  # 3. format inputs
  # ---------
  formatted_inputs = []
  for one_input in inputs_which_found_target:
    # input value
    data_value = one_input[2]

    # input type
    if len(one_input) >= 4:
      if one_input[3] == 'iter':
        if not isinstance(data_value, Iterable):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}.{one_input[1]}" '
                           f'is set to be "iter" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[3] == 'func':
        if not callable(data_value):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}.{one_input[1]}" '
                           f'is set to be "func" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[3] != 'fix':
        raise RunningError(f'Only support {SUPPORTED_INPUT_TYPE} input type, but '
                           f'we got "{one_input[3]}" in {one_input}')

      data_type = one_input[3]
    else:
      data_type = 'fix'

    # operation
    if len(one_input) == 5:
      data_op = one_input[4]
    else:
      data_op = '+'
    if data_op not in SUPPORTED_INPUT_OPS:
      raise RunningError(f'Only support {SUPPORTED_INPUT_OPS}, while we got '
                         f'{data_op} in {one_input}')

    # final
    format_inp = one_input[:2] + (data_value, data_type, data_op)
    formatted_inputs.append(format_inp)

  return formatted_inputs


class DSRunner(Runner):
  """The runner for dynamical systems.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  inputs : list, tuple
    The inputs for the target DynamicalSystem. It should be the format
    of `[(target, value, [type, operation])]`, where `target` is the
    input target, `value` is the input value, `type` is the input type
    (such as "fix", "iter", "func"), `operation` is the operation for inputs
    (such as "+", "-", "*", "/", "=").

    - ``target``: should be a string. Can be specified by the *absolute access* or *relative access*.
    - ``value``: should be a scalar, vector, matrix, iterable function or objects.
    - ``type``: should be a string. "fix" means the input `value` is a constant. "iter" means the
      input `value` can be changed over time. "func" mean the input is obtained through the functional call.
    - ``operation``: should be a string, support `+`, `-`, `*`, `/`, `=`.
    - Also, if you want to specify multiple inputs, just give multiple ``(target, value, [type, operation])``,
      for example ``[(target1, value1), (target2, value2)]``.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      inputs: Sequence = (),
      dt: float = None,
      **kwargs
  ):
    if not isinstance(target, DynamicalSystem):
      raise RunningError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                         f'but we got {type(target)}: {target}')
    super(DSRunner, self).__init__(target=target, **kwargs)

    # parameters
    dt = bm.get_dt() if dt is None else dt
    if not isinstance(dt, (int, float)):
      raise RunningError(f'"dt" must be scalar, but got {dt}')
    self.dt = dt

    # Build the monitor function
    self._monitor_step = self.build_monitors(*self.format_monitors())

    # whether it has iterable input data
    self._has_iter_array = False  # default do not have iterable input array
    self._i = bm.Variable(bm.asarray([0]))

    # Build input function
    inputs = check_and_format_inputs(host=target, inputs=inputs)
    self._input_step = self.build_inputs(inputs)

    # start simulation time
    self._start_t = None

    # JAX does not support iterator in fori_loop, scan, etc.
    #   https://github.com/google/jax/issues/3567
    # We use Variable i to index the current input data.
    if self._has_iter_array:  # must behind of "self.build_input()"
      self.dyn_vars.update({'_i': self._i})
    else:
      self._i = None

    # run function
    self._predict_func = dict()
    # self._run_func = self.build_run_function()

  def build_inputs(self, inputs):
    fix_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    next_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    func_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
    array_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}

    for target, key, value, type_, op in inputs:
      # variable
      variable = getattr(target, key)
      if not isinstance(variable, bm.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'give its input.')

      # input data
      if type_ == 'iter':
        if isinstance(value, (bm.ndarray, np.ndarray, jnp.ndarray)):
          array_inputs[op].append([variable, bm.asarray(value)])
          self._has_iter_array = True
        else:
          next_inputs[op].append([variable, iter(value)])
      elif type_ == 'func':
        func_inputs[op].append([variable, value])
      else:
        fix_inputs[op].append([variable, value])

    def _f_ops(ops, var, data):
      if ops == '=':
        var[:] = data
      elif ops == '+':
        var += data
      elif ops == '-':
        var -= data
      elif ops == '*':
        var *= data
      elif ops == '/':
        var /= data
      else:
        raise ValueError

    def func(_t, _dt):
      for ops, values in fix_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data)
      for ops, values in array_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data[self._i[0]])
      for ops, values in func_inputs.items():
        for var, data in values:
          _f_ops(ops, var, data(_t, _dt))
      for ops, values in next_inputs.items():
        for var, data in values:
          _f_ops(ops, var, next(data))
      if self._has_iter_array:
        self._i += 1

    return func

  def build_monitors(self, return_without_idx, return_with_idx, flatten=False):
    if flatten:
      def func(_t, _dt):
        res = {k: (v.flatten() if bm.ndim(v) > 1 else v.value) for k, v in return_without_idx.items()}
        res.update({k: (v.flatten()[idx] if bm.ndim(v) > 1 else v[idx]) for k, (v, idx) in return_with_idx.items()})
        res.update({k: f(_t, _dt) for k, f in self.fun_monitors.items()})
        return res
    else:
      def func(_t, _dt):
        res = {k: v.value for k, v in return_without_idx.items()}
        res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
        res.update({k: f(_t, _dt) for k, f in self.fun_monitors.items()})
        return res

    return func

  def predict(
      self,
      xs: Union[Tensor, Dict[str, Tensor]],
      reset_state: bool = False,
      shared_args: Dict = None,
      progress_bar: bool = True,
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
    progress_bar: bool
      Whether report the progress of the simulation using progress bar.

    Returns
    -------
    output: Tensor, dict
      The model output.
    """
    # format input data
    xs, num_step, num_batch = self._check_xs(xs)
    times = jax.device_put(jnp.linspace(0., self.dt * (num_step - 1), num_step))
    xs = (times, xs,)
    # reset the model states
    if reset_state:
      self.target.reset_batch_state(num_batch)
    # init monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # init progress bar
    if self.progress_bar and progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(f"Predict {num_step} steps: ", refresh=True)
    # prediction
    outputs, hists = self._predict(xs=xs, shared_args=shared_args)
    outputs = tree_map(lambda x: bm.moveaxis(x, 0, 1), outputs, is_leaf=lambda x: isinstance(x, bm.JaxArray))
    hists = tree_map(lambda x: bm.moveaxis(x, 0, 1), hists, is_leaf=lambda x: isinstance(x, bm.JaxArray))
    # close the progress bar
    if self.progress_bar and progress_bar:
      self._pbar.close()
    # post-running for monitors
    for key, val in hists.items():
      self.mon.item_contents[key] = val
    if self.numpy_mon_after_run:
      self.mon.numpy()
    return outputs

  def _predict(
      self,
      xs: Sequence[Tensor],
      shared_args: Dict = None,
  ):
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: sequence
      Each tensor should have the shape of `(num_time, num_batch, num_feature)`.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self._get_predict_func(shared_args)
    outputs, hists = _predict_func(xs)
    return outputs, hists

  def run(self, duration, start_t=None, shared_args: Dict = None, eval_time=False):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
      The running duration.
    start_t : float, optional
      The start time.
    shared_args: dict
      The shared arguments across nodes.
    eval_time: bool
      Whether we record the running time?
    """
    # time step
    if start_t is None:
      if self._start_t is None:
        start_t = 0.
      else:
        start_t = float(self._start_t)
    end_t = float(start_t + duration)
    # times
    times = jax.device_put(jnp.arange(start_t, end_t, self.dt))
    # build monitor
    for key in self.mon.item_contents.keys():
      self.mon.item_contents[key] = []  # reshape the monitor items
    # running
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=times.size)
      self._pbar.set_description(f"Running a duration of {round(float(duration), 3)} ({times.size} steps)",
                                 refresh=True)
    if eval_time:
      t0 = time.time()
    outputs, hists = self._predict((times, None), shared_args=shared_args)
    if eval_time:
      running_time = time.time() - t0
    if self.progress_bar:
      self._pbar.close()
    # post-running
    if self.jit:
      self.mon.ts = times + self.dt
      for key in self.mon.item_names:
        self.mon.item_contents[key] = bm.asarray(hists[key])
    else:
      self.mon.ts = times + self.dt
      for key in self.mon.item_names:
        self.mon.item_contents[key] = bm.asarray(self.mon.item_contents[key])
    self._start_t = end_t
    if self.numpy_mon_after_run:
      self.mon.numpy()
    if eval_time:
      return running_time, outputs
    else:
      return outputs

  def _check_xs(self, xs, move_axis=True):
    leaves, tree = tree_flatten(xs, is_leaf=lambda x: isinstance(x, bm.JaxArray))
    # get information of time step and batch size
    num_times, num_batch_sizes = [], []
    for val in leaves:
      num_batch_sizes.append(val.shape[0])
      num_times.append(val.shape[1])
    if len(set(num_times)) != 1:
      raise ValueError(f'Number of time step is different across tensors in '
                       f'the provided "xs". We got {set(num_times)}.')
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across tensors in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    num_step = num_times[0]
    num_batch = num_batch_sizes[0]
    # change shape to (num_time, num_sample, num_feature)
    if move_axis:
      xs = tree_map(lambda x: bm.moveaxis(x, 0, 1), xs)
    return xs, num_step, num_batch

  def _get_predict_func(self, shared_args: Dict = None):
    if shared_args is None: shared_args = dict()
    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._predict_func:
      self._predict_func[shared_kwargs_str] = self._make_predict_func(shared_args)
    return self._predict_func[shared_kwargs_str]

  def _make_predict_func(self, shared_args: Dict):
    if not isinstance(shared_args, dict):
      raise ValueError(f'"shared_kwargs" must be a dict, but got {type(shared_args)}')

    def _step_func(inputs):
      t, x = inputs
      self._input_step(t, self.dt)
      if x is None:
        args = (t, self.dt)
      else:
        args = (t, self.dt, x)
      kwargs = dict()
      if len(shared_args):
        kwargs['shared_args'] = shared_args
      out = self.target.update(*args, **kwargs)
      if self.progress_bar:
        id_tap(lambda *arg: self._pbar.update(), ())
      return out, self._monitor_step(t, self.dt)

    if self.jit['predict']:
      dyn_vars = self.target.vars()
      dyn_vars.update(self.dyn_vars)
      f = bm.make_loop(_step_func, dyn_vars=dyn_vars.unique(), has_return=True)
      return lambda all_inputs: f(all_inputs)[1]

    else:
      def run_func(xs):
        outputs = []
        monitors = {key: [] for key in set(self.mon.item_contents.keys()) | set(self.fun_monitors.keys())}
        for i in range(check_data_batch_size(xs)):
          x = tree_map(lambda x: x[i], xs)
          output, mon = _step_func(x)
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


class ReportRunner(DSRunner):
  """The runner provides convenient interface for debugging.
  It is also able to report the running progress.

  .. deprecated:: 2.0.3
     Prefer the use of :py:class:`brainpy.dyn.DSRunner` for dynamical system running.
     This runner is deprecated since 2.0.3.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.
  monitors : None, list of str, tuple of str, Monitor
    Variables to monitor.
  inputs : list, tuple
    The input settings.
  """

  def __init__(self, target, inputs=(), jit=False, dt=None, **kwargs):
    super(ReportRunner, self).__init__(target=target, inputs=inputs, dt=dt, jit=False, **kwargs)

    # Build the update function
    if jit:
      dyn_vars = TensorCollector()
      dyn_vars.update(self.dyn_vars)
      dyn_vars.update(self.target.vars().unique())
      self._update_step = bm.jit(self.target.update, dyn_vars=dyn_vars)
    else:
      self._update_step = self.target.update

  def _run_one_step(self, _t):
    self._input_step(_t, self.dt)
    self._update_step(_t, self.dt)
    if self.progress_bar:
      self._pbar.update()
    return self._monitor_step(_t, self.dt)

  def build_run_function(self):
    def f_run(all_t):
      for i in range(all_t.shape[0]):
        mon = self._run_one_step(all_t[i])
        for k, v in mon.items():
          self.mon.item_contents[k].append(v)
      return None, {}

    return f_run
