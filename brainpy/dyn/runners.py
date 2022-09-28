# -*- coding: utf-8 -*-

import time
from collections.abc import Iterable
from typing import Dict, Union, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map, tree_flatten

from brainpy import math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import RunningError
from brainpy.running.runner import Runner
from brainpy.tools.checking import check_float, serialize_kwargs
from brainpy.tools.others.dicts import DotDict
from brainpy.types import Array, Output, Monitor

__all__ = [
  'DSRunner',
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
    if isinstance(inputs[0], (str, bm.Variable)):
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
  nodes = None
  for one_input in inputs:
    key = one_input[0]
    if isinstance(key, bm.Variable):
      real_target = key
    elif isinstance(key, str):
      if nodes is None:
        nodes = host.nodes(method='absolute', level=-1, include_self=True)
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
      real_target = getattr(real_target, key)
    else:
      raise RunningError(f'For each input, input[0] must be a string  to '
                         f'specify variable of the target, but we got {key}.')
    inputs_which_found_target.append((real_target,) + tuple(one_input[1:]))

  # checking 2: relative access
  #    Check whether the input target node is accessible
  #    and check whether the target node has the attribute
  if len(inputs_not_found_target):
    nodes = host.nodes(method='relative', level=-1, include_self=True)
    for one_input in inputs_not_found_target:
      splits = one_input[0].split('.')
      target, key = '.'.join(splits[:-1]), splits[-1]
      if target not in nodes:
        raise RunningError(f'Input target "{target}" is not defined in {host}.')
      real_target = nodes[target]
      if not hasattr(real_target, key):
        raise RunningError(f'Input target key "{key}" is not defined in {real_target}.')
      real_target = getattr(real_target, key)
      inputs_which_found_target.append((real_target,) + tuple(one_input[1:]))

  # 3. format inputs
  # ---------
  formatted_inputs = []
  for one_input in inputs_which_found_target:
    # input value
    data_value = one_input[1]

    # input type
    if len(one_input) >= 3:
      if one_input[2] == 'iter':
        if not isinstance(data_value, Iterable):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}" \n'
                           f'is set to be "iter" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[2] == 'func':
        if not callable(data_value):
          raise ValueError(f'Input "{data_value}" for "{one_input[0]}" \n'
                           f'is set to be "func" type, however we got the value with '
                           f'the type of {type(data_value)}')
      elif one_input[2] != 'fix':
        raise RunningError(f'Only support {SUPPORTED_INPUT_TYPE} input type, but '
                           f'we got "{one_input[2]}"')

      data_type = one_input[2]
    else:
      data_type = 'fix'

    # operation
    if len(one_input) == 4:
      data_op = one_input[3]
    else:
      data_op = '+'
    if data_op not in SUPPORTED_INPUT_OPS:
      raise RunningError(f'Only support {SUPPORTED_INPUT_OPS}, while we got '
                         f'{data_op} in {one_input}')

    # final
    format_inp = (one_input[0], data_value, data_type, data_op)
    formatted_inputs.append(format_inp)

  return formatted_inputs


def build_inputs(inputs, fun_inputs):
  """Build input function.

  Parameters
  ----------
  inputs : tuple, list
    The inputs of the population.
  fun_inputs: optional, callable
    The input function customized by users.

  Returns
  -------
  func: callable
    The input function.
  """

  fix_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  next_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  func_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  array_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}

  if not (fun_inputs is None or callable(fun_inputs)):
    raise ValueError

  _has_iter_array = False
  for variable, value, type_, op in inputs:
    # variable
    if not isinstance(variable, bm.Variable):
      raise RunningError(f'{variable}\n is not a dynamically changed Variable, '
                         f'its value will not change, we think there is no need to '
                         f'give its input.')

    # input data
    if type_ == 'iter':
      if isinstance(value, (bm.ndarray, np.ndarray, jnp.ndarray)):
        array_inputs[op].append([variable, bm.asarray(value)])
        _has_iter_array = True
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
      raise ValueError(f'Unknown input operation: {ops}')

  def func(tdi):
    if fun_inputs is not None:
      fun_inputs(tdi)
    for ops, values in fix_inputs.items():
      for var, data in values:
        _f_ops(ops, var, data)
    for ops, values in array_inputs.items():
      for var, data in values:
        _f_ops(ops, var, data[tdi['i']])
    for ops, values in func_inputs.items():
      for var, data in values:
        _f_ops(ops, var, data(tdi['t'], tdi['dt']))
    for ops, values in next_inputs.items():
      for var, data in values:
        _f_ops(ops, var, next(data))

  return func, _has_iter_array


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

  fun_inputs: callable
    The functional inputs. Manually specify the inputs for the target variables.
    This input function should receive one argument `shared` which contains the shared arguments like
    time `t`, time step `dt`, and index `i`.

  monitors: None, sequence of str, dict, Monitor
    Variables to monitor.

    - A list of string. Like `monitors=['a', 'b', 'c']`
    - A list of string with index specification. Like `monitors=[('a', 1), ('b', [1,3,5]), 'c']`
    - A dict with the explicit monitor target, like: `monitors={'a': model.spike, 'b': model.V}`
    - A dict with the index specification, like: `monitors={'a': (model.spike, 0), 'b': (model.V, [1,2])}`

  fun_monitors: dict
    Monitoring variables by callable functions. Should be a dict.
    The `key` should be a string for the later retrieval by `runner.mon[key]`.
    The `value` should be a callable function which receives two arguments: `t` and `dt`.

  jit: bool, dict
    The JIT settings.

  progress_bar: bool
    Use progress bar to report the running progress or not?

  dyn_vars: Optional, dict
    The dynamically changed variables. Instance of :py:class:`~.Variable`.

  numpy_mon_after_run : bool
    When finishing the network running, transform the JAX arrays into numpy ndarray or not?

  """

  target: DynamicalSystem

  def __init__(
      self,
      target: DynamicalSystem,

      # inputs for target variables
      inputs: Sequence = (),
      fun_inputs: Callable = None,

      # extra info
      dt: float = None,
      t0: Union[float, int] = 0.,
      **kwargs
  ):
    if not isinstance(target, DynamicalSystem):
      raise RunningError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                         f'but we got {type(target)}: {target}')
    super(DSRunner, self).__init__(target=target, **kwargs)

    # t0 and i0
    self._t0 = t0
    self.i0 = 0
    self.t0 = check_float(t0, 't0', allow_none=False, allow_int=True)

    # parameters
    dt = bm.get_dt() if dt is None else dt
    if not isinstance(dt, (int, float)):
      raise RunningError(f'"dt" must be scalar, but got {dt}')
    self.dt = dt

    # Build the monitor function
    self._mon_info = self.format_monitors()

    # Build input function
    self._input_step, _ = build_inputs(check_and_format_inputs(host=target, inputs=inputs),
                                       fun_inputs=fun_inputs)

    # run function
    self._f_predict_compiled = dict()

  def build_monitors(self, return_without_idx, return_with_idx, shared_args: dict):
    def func(tdi):
      res = {k: v.value for k, v in return_without_idx.items()}
      res.update({k: v[idx] for k, (v, idx) in return_with_idx.items()})
      res.update({k: f(tdi) for k, f in self.fun_monitors.items()})
      return res

    return func

  def reset_state(self):
    self.i0 = 0
    self.t0 = check_float(self._t0, 't0', allow_none=False, allow_int=True)

  def predict(
      self,
      duration: Union[float, int] = None,
      inputs: Union[Array, Sequence[Array], Dict[str, Array]] = None,
      inputs_are_batching: bool = False,
      reset_state: bool = False,
      shared_args: Dict = None,
      progress_bar: bool = True,
      eval_time: bool = False
  ) -> Output:
    """Running a duration with the given target model. See `.predict()` function
    for more details.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    duration: int, float
      The simulation time length.
    inputs: Array, dict of Array, sequence of Array
      The input data. If ``inputs_are_batching=True``, ``inputs`` must be a
      PyTree of data with two dimensions: `(num_sample, num_time, ...)`.
      Otherwise, the ``inputs`` should be a PyTree of data with one dimension:
      `(num_time, ...)`.
    inputs_are_batching: bool
      Whether the ``inputs`` are batching. If `True`, the batching axis is the
      first dimension.
    reset_state: bool
      Whether reset the model states.
    shared_args: optional, dict
      The shared arguments across different layers.
    progress_bar: bool
      Whether report the progress of the simulation using progress bar.
    eval_time: bool
      Whether ro evaluate the running time.

    Returns
    -------
    output: Array, dict, sequence
      The model output.
    """

    # shared arguments
    if shared_args is None: shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', False)

    # times and inputs
    times, indices, xs, num_step, num_batch, duration, description = self._format_xs(
      duration, inputs, inputs_are_batching)

    # reset the states of the model and the runner
    if reset_state:
      self.target.reset_state(num_batch)
      self.reset_state()
    indices += self.i0
    times += self.t0

    # build monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar and progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(description, refresh=True)

    # running
    if eval_time: t0 = time.time()
    outputs, hists = self._predict(xs=(times, indices, xs), shared_args=shared_args)
    if eval_time: running_time = time.time() - t0

    # format
    if inputs_are_batching:
      outputs = tree_map(lambda x: bm.moveaxis(x, 0, 1), outputs, is_leaf=lambda x: isinstance(x, bm.JaxArray))
      hists = tree_map(lambda x: bm.moveaxis(x, 0, 1), hists, is_leaf=lambda x: isinstance(x, bm.JaxArray))

    # close the progress bar
    if self.progress_bar and progress_bar:
      self._pbar.close()

    # post-running for monitors
    hists['ts'] = times + self.dt
    if self.numpy_mon_after_run:
      hists = tree_map(lambda a: np.asarray(a), hists, is_leaf=lambda a: isinstance(a, bm.JaxArray))
    for key in hists.keys():
      self.mon[key] = hists[key]
    self.i0 += times.shape[0]
    self.t0 += duration
    return outputs if not eval_time else (running_time, outputs)

  def _predict(
      self,
      xs: Sequence,
      shared_args: Dict = None,
  ) -> Union[Output, Monitor]:
    """Predict the output according to the inputs.

    Parameters
    ----------
    xs: sequence
      Must be a tuple/list of data, including `(times, indices, inputs)`.
      If `inputs` is not None, it should be a tensor with the shape of
      :math:`(num_time, ...)`.
    shared_args: optional, dict
      The shared keyword arguments.

    Returns
    -------
    outputs, hists
      A tuple of pair of (outputs, hists).
    """
    _predict_func = self.f_predict(shared_args)
    outputs, hists = _predict_func(xs)
    return outputs, hists

  def run(self, *args, **kwargs) -> Output:
    """Predict a series of input data with the given target model.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    feedbacks and its output.

    Parameters
    ----------
    duration: int, float
      The simulation time length.
    inputs: Array, dict of Array, sequence of Array
      The input data. If ``inputs_are_batching=True``, ``inputs`` must be a
      PyTree of data with two dimensions: `(num_sample, num_time, ...)`.
      Otherwise, the ``inputs`` should be a PyTree of data with one dimension:
      `(num_time, ...)`.
    inputs_are_batching: bool
      Whether the ``inputs`` are batching. If `True`, the batching axis is the
      first dimension.
    reset_state: bool
      Whether reset the model states.
    shared_args: optional, dict
      The shared arguments across different layers.
    progress_bar: bool
      Whether report the progress of the simulation using progress bar.
    eval_time: bool
      Whether to evaluate the running time.

    Returns
    -------
    output: Array, dict, sequence
      The model output.
    """
    return self.predict(*args, **kwargs)

  def __call__(self, *args, **kwargs) -> Output:
    return self.predict(*args, **kwargs)

  def _format_xs(self, duration, inputs, inputs_are_batching=True, move_axis=True):
    if duration is None:
      if inputs is None:
        raise ValueError('"duration" and "inputs" can not both be None.')
      xs, num_step, num_batch = self._check_xs(inputs,
                                               move_axis=move_axis,
                                               inputs_are_batching=inputs_are_batching)
      indices = jax.device_put(jnp.arange(num_step))
      times = jax.device_put(indices * self.dt)
      description = f'Predict {num_step} steps: '
      duration = num_step * self.dt
    else:
      times = jax.device_put(jnp.arange(0, duration, self.dt))
      num_step = times.shape[0]
      indices = jax.device_put(jnp.arange(num_step))
      description = f'Running a duration of {round(float(duration), 3)} ({times.shape[0]} steps)'
      if inputs is None:
        xs, num_batch = None, None
      else:
        xs, num_step_, num_batch = self._check_xs(inputs,
                                                  move_axis=move_axis,
                                                  inputs_are_batching=inputs_are_batching)
        if num_step != num_step:
          raise ValueError('The step numbers of "time" and "inputs" '
                           f'do not match: {num_step_} != {num_step}.')
    return times, indices, xs, num_step, num_batch, duration, description

  def _check_xs(self, xs, move_axis=True, inputs_are_batching=True):
    leaves, tree = tree_flatten(xs, is_leaf=lambda x: isinstance(x, bm.JaxArray))

    # get information of time step and batch size
    if inputs_are_batching:
      num_times, num_batch_sizes = [], []
      for val in leaves:
        num_batch_sizes.append(val.shape[0])
        num_times.append(val.shape[1])
    else:
      num_times = [val.shape[0] for val in leaves]
    if len(set(num_times)) != 1:
      raise ValueError(f'Number of time step is different across tensors in '
                       f'the provided "xs". We got {set(num_times)}.')
    num_step = num_times[0]
    if inputs_are_batching:
      if len(set(num_batch_sizes)) != 1:
        raise ValueError(f'Number of batch size is different across tensors in '
                         f'the provided "xs". We got {set(num_batch_sizes)}.')
      num_batch = num_batch_sizes[0]
    else:
      num_batch = None

    # change shape to (num_time, num_sample, num_feature)
    if move_axis and inputs_are_batching:
      xs = tree_map(lambda x: bm.moveaxis(x, 0, 1), xs,
                    is_leaf=lambda x: isinstance(x, bm.JaxArray))
    return xs, num_step, num_batch

  def f_predict(self, shared_args: Dict = None):
    if shared_args is None: shared_args = dict()

    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._f_predict_compiled:

      monitor_func = self.build_monitors(self._mon_info[0], self._mon_info[1], shared_args)

      def _step_func(t, i, x):
        self.target.clear_input()
        # input step
        shared = DotDict(t=t, i=i, dt=self.dt)
        self._input_step(shared)
        # dynamics update step
        shared.update(shared_args)
        args = (shared,) if x is None else (shared, x)
        out = self.target(*args)
        # monitor step
        mon = monitor_func(shared)
        # finally
        if self.progress_bar:
          id_tap(lambda *arg: self._pbar.update(), ())
        return out, mon

      if self.jit['predict']:
        dyn_vars = self.target.vars()
        dyn_vars.update(self.dyn_vars)
        run_func = lambda all_inputs: bm.for_loop(_step_func, dyn_vars.unique(), all_inputs)

      else:
        def run_func(xs):
          # total data
          times, indices, xs = xs

          outputs = []
          monitors = {key: [] for key in (set(self.mon.var_names) | set(self.fun_monitors.keys()))}
          for i in range(times.shape[0]):
            # data at time i
            x = tree_map(lambda x: x[i], xs, is_leaf=lambda x: isinstance(x, bm.JaxArray))

            # step at the i
            output, mon = _step_func(times[i], indices[i], x)

            # append output and monitor
            outputs.append(output)
            for key, value in mon.items():
              monitors[key].append(value)

          # final work
          if outputs[0] is None:
            outputs = None
          else:
            outputs = bm.asarray(outputs)
          for key, value in monitors.items():
            monitors[key] = bm.asarray(value)
          return outputs, monitors
      self._f_predict_compiled[shared_kwargs_str] = run_func
    return self._f_predict_compiled[shared_kwargs_str]

  def __del__(self):
    if hasattr(self, '_predict_func'):
      for key in tuple(self._f_predict_compiled.keys()):
        del self._f_predict_compiled[key]
    super(DSRunner, self).__del__()
