# -*- coding: utf-8 -*-

import time
import warnings
from collections.abc import Iterable
from functools import partial
from typing import Dict, Union, Sequence, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm.auto
from jax.experimental.host_callback import id_tap
from jax.tree_util import tree_map, tree_flatten

from brainpy import math as bm, tools
from brainpy.check import is_float, serialize_kwargs
from brainpy.dyn.base import DynamicalSystem
from brainpy.errors import RunningError, NoLongerSupportError
from brainpy.running.runner import Runner
from brainpy.types import ArrayType, Output, Monitor

__all__ = [
  'DSRunner',
]

SUPPORTED_INPUT_OPS = ['-', '+', '*', '/', '=']
SUPPORTED_INPUT_TYPE = ['fix', 'iter', 'func']


def _is_brainpy_array(x):
  return isinstance(x, bm.Array)


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

  fix_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  next_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  func_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}
  array_inputs = {'=': [], '+': [], '-': [], '*': [], '/': []}

  for variable, value, type_, op in formatted_inputs:
    # variable
    if not isinstance(variable, bm.Variable):
      raise RunningError(f'{variable}\n is not a dynamically changed Variable, '
                         f'its value will not change, we think there is no need to '
                         f'give its input.')

    # input data
    if type_ == 'iter':
      if isinstance(value, (bm.ndarray, np.ndarray, jnp.ndarray)):
        array_inputs[op].append([variable, bm.asarray(value)])
      else:
        next_inputs[op].append([variable, iter(value)])
    elif type_ == 'func':
      func_inputs[op].append([variable, value])
    else:
      fix_inputs[op].append([variable, value])

  return {'fixed': fix_inputs,
          'iterated': next_inputs,
          'functional': func_inputs,
          'array': array_inputs}


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


class DSRunner(Runner):
  """The runner for :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  target : DynamicalSystem
    The target model to run.

  inputs : list, tuple, callable
    The inputs for variables in the target model.

    .. note::

       This argument can be used to set the inputs to the
       :py:class:`~.Variable` instances in the ``target``.
       If you peruse to give time-dependent inputs, please use
       ``DSRunner.predict()`` or ``DSRunner.run()`` function.

    - It can be a list/tuple with the format
      of `[(target, value, [type, operation])]`, where `target` is the
      input target, `value` is the input value, `type` is the input type
      (such as "fix", "iter", "func"), `operation` is the operation for inputs
      (such as "+", "-", "*", "/", "=").

      - ``target``: should be a string or :py:class:`~.Variable`. Can be specified by the
        *absolute access* or *relative access*.
      - ``value``: should be a scalar, vector, matrix.
      - ``type``: should be a string. "fix" means the input `value`
        is a constant. "iter" means the input `value` can be changed
        over time. "func" mean the input is obtained through the functional call.
      - ``operation``: should be a string, support `+`, `-`, `*`, `/`, `=`.
      - Also, if you want to specify multiple inputs, just give multiple
        ``(target, value, [type, operation])``,
        for example ``[(target1, value1), (target2, value2)]``.

    - It can also be a callable function which receives the shared argument.
      In this functional input, users can manually specify the inputs for the target variables.
      This input function should receive one argument ``shared`` which contains the
      shared arguments like time ``t``, time step ``dt``, and index ``i``.

      .. versionchanged:: 2.3.1
         ``fun_inputs`` are merged into ``inputs``.
  fun_inputs: callable
    The functional inputs. Manually specify the inputs for the target variables.
    This input function should receive one argument ``shared`` which contains the
    shared arguments like time ``t``, time step ``dt``, and index ``i``.

    .. deprecated:: 2.3.1
       Will be removed since version 2.4.0.
  monitors: Optional, sequence of str, dict, Monitor
    Variables to monitor.

    - A list of string. Like ``monitors=['a', 'b', 'c']``.
    - A list of string with index specification. Like ``monitors=[('a', 1), ('b', [1,3,5]), 'c']``
    - A dict with the explicit monitor target, like: ``monitors={'a': model.spike, 'b': model.V}``
    - A dict with the index specification, like: ``monitors={'a': (model.spike, 0), 'b': (model.V, [1,2])}``
    - A dict with the callable function, like ``monitors={'a': lambda tdi: model.spike[:5]}``

    .. versionchanged:: 2.3.1
       ``fun_monitors`` are merged into ``monitors``.
  fun_monitors: dict
    Monitoring variables by a dict of callable functions.
    The dict ``key`` should be a string for the later retrieval by ``runner.mon[key]``.
    The dict ``value`` should be a callable function which receives two arguments: ``t`` and ``dt``.
    .. code-block::
       fun_monitors = {'spike': lambda tdi: model.spike[:10],
                       'V10': lambda tdi: model.V[10]}

    .. deprecated:: 2.3.1
       Will be removed since version 2.4.0.
  jit: bool, dict
    The JIT settings.
    Using dict is able to set the jit mode at different phase,
    for instance, ``jit={'predict': True, 'fit': False}``.

  progress_bar: bool
    Use progress bar to report the running progress or not?

  dyn_vars: Optional, dict
    The dynamically changed variables. Instance of :py:class:`~.Variable`.
    These variables together with variable retrieved from the ``target``
    constitute all dynamical variables in this runner.

  numpy_mon_after_run : bool
    When finishing the network running, transform the JAX arrays into numpy ndarray or not?

  data_first_axis: str
    Set the default data dimension arrangement.
    To indicate whether the first axis is the batch size (``data_first_axis='B'``) or the
    time length (``data_first_axis='T'``).
    In order to be compatible with previous API, default is set to be ``False``.

    .. versionadded:: 2.3.1
  """

  target: DynamicalSystem

  def __init__(
      self,
      target: DynamicalSystem,

      # inputs for target variables
      inputs: Union[Sequence, Callable] = (),

      # monitors
      monitors: Optional[Union[Sequence, Dict]] = None,
      numpy_mon_after_run: bool = True,

      # jit
      jit: Union[bool, Dict[str, bool]] = True,
      dyn_vars: Optional[Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]]] = None,

      # extra info
      dt: Optional[float] = None,
      t0: Union[float, int] = 0.,
      progress_bar: bool = True,
      data_first_axis: Optional[str] = None,

      # deprecated
      fun_inputs: Optional[Callable] = None,
      fun_monitors: Optional[Dict[str, Callable]] = None,
  ):
    if not isinstance(target, DynamicalSystem):
      raise RunningError(f'"target" must be an instance of {DynamicalSystem.__name__}, '
                         f'but we got {type(target)}: {target}')
    super(DSRunner, self).__init__(target=target,
                                   monitors=monitors,
                                   fun_monitors=fun_monitors,
                                   jit=jit,
                                   progress_bar=progress_bar,
                                   dyn_vars=dyn_vars,
                                   numpy_mon_after_run=numpy_mon_after_run)

    # t0 and i0
    is_float(t0, 't0', allow_none=False, allow_int=True)
    self._t0 = t0
    self.i0 = bm.Variable(bm.asarray([1], dtype=bm.int_))
    self.t0 = bm.Variable(bm.asarray([t0], dtype=bm.float_))
    if data_first_axis is None:
      data_first_axis = 'B' if isinstance(self.target.mode, bm.BatchingMode) else 'T'
    assert data_first_axis in ['B', 'T']
    self.data_first_axis = data_first_axis

    # parameters
    dt = bm.get_dt() if dt is None else dt
    if not isinstance(dt, float):
      raise RunningError(f'"dt" must be float, but got {dt}')
    self.dt = dt

    # Build input function
    if fun_inputs is not None:
      warnings.warn('`fun_inputs` is deprecated since version 2.3.1. '
                    'Define `fun_inputs` as `inputs` instead.',
                    UserWarning)
    if callable(inputs):
      self._inputs = inputs
    else:
      self._inputs = check_and_format_inputs(host=target, inputs=inputs)

    # run function
    self._f_predict_compiled = dict()

  def __repr__(self):
    name = self.__class__.__name__
    indent = " " * len(name) + ' '
    indent2 = indent + " " * len("target")
    return (f'{name}(target={tools.repr_context(str(self.target), indent2)}, \n'
            f'{indent}jit={self.jit},\n'
            f'{indent}dt={self.dt},\n'
            f'{indent}data_first_axis={self.data_first_axis})')

  def reset_state(self):
    """Reset state of the ``DSRunner``."""
    self.i0.value = bm.zeros_like(self.i0)
    self.t0.value = bm.ones_like(self.t0) * self._t0

  def predict(
      self,
      duration: float = None,
      inputs: Union[ArrayType, Sequence[ArrayType], Dict[str, ArrayType]] = None,
      reset_state: bool = False,
      shared_args: Dict = None,
      eval_time: bool = False,

      # deprecated
      inputs_are_batching: bool = None,
  ) -> Union[Output, Tuple[float, Output]]:
    """Running a duration with the given target model. See `.predict()` function
    for more details.

    This function use the JIT compilation to accelerate the model simulation.
    Moreover, it can automatically monitor the node variables, states, inputs,
    and its output.

    Parameters
    ----------
    duration: float
      The simulation time length.
      If you have provided ``inputs``, there is no longer need to provide ``duration``.
    inputs: ArrayType, dict of ArrayType, sequence of ArrayType
      The input data.

      - If the mode of ``target`` is instance of :py:class:`~.BatchingMode`,
        ``inputs`` must be a PyTree of data with two dimensions:
        ``(batch, time, ...)`` when ``data_first_axis='B'``,
        or ``(time, batch, ...)`` when ``data_first_axis='T'``.
      - If the mode of ``target`` is instance of :py:class:`~.NonBatchingMode`,
        the ``inputs`` should be a PyTree of data with one dimension:
        ``(time, ...)``.
    inputs_are_batching: bool
      Whether the ``inputs`` are batching. If `True`, the batching axis is the
      first dimension.

      .. deprecated:: 2.3.1
         Will be removed after version 2.4.0.
    reset_state: bool
      Whether reset the model states.
    shared_args: optional, dict
      The shared arguments across different layers.
    eval_time: bool
      Whether ro evaluate the running time.

    Returns
    -------
    output: ArrayType, dict, sequence
      The model output.
    """

    if inputs_are_batching is not None:
      raise NoLongerSupportError(
        f'''
        `inputs_are_batching` is no longer supported. 
        The target mode of {self.target.mode} has already indicated the input should be batching.
        '''
      )
    if duration is None:
      if inputs is None:
        raise ValueError('Please provide "duration" or "inputs".')
    else:
      if inputs is not None:
        raise ValueError('Please provide "duration" or "inputs".')

    num_step = self._get_input_time_step(duration, inputs)
    description = f'Predict {num_step} steps: '

    # reset the states of the model and the runner
    if reset_state:
      self.target.reset_state(self._get_input_batch_size(inputs))
      self.reset_state()

    # shared arguments and inputs
    if shared_args is None:
      shared_args = dict()
    shared_args['fit'] = shared_args.get('fit', False)
    shared = tools.DotDict(i=jnp.arange(num_step, dtype=bm.int_))
    shared['t'] = shared['i'] * self.dt
    shared['i'] += self.i0
    shared['t'] += self.t0

    if isinstance(self.target.mode, bm.BatchingMode) and self.data_first_axis == 'B':
      inputs = tree_map(lambda x: bm.moveaxis(x, 0, 1),
                        inputs,
                        is_leaf=lambda x: isinstance(x, bm.Array))

    # build monitor
    for key in self.mon.var_names:
      self.mon[key] = []  # reshape the monitor items

    # init progress bar
    if self.progress_bar:
      self._pbar = tqdm.auto.tqdm(total=num_step)
      self._pbar.set_description(description, refresh=True)

    # running
    if eval_time:
      t0 = time.time()
    outputs, hists = self._predict(xs=(shared['t'], shared['i'], inputs), shared_args=shared_args)
    if eval_time:
      running_time = time.time() - t0

    # close the progress bar
    if self.progress_bar:
      self._pbar.close()

    # post-running for monitors
    hists['ts'] = shared['t'] + self.dt
    if self.numpy_mon_after_run:
      hists = tree_map(lambda a: np.asarray(a), hists, is_leaf=lambda a: isinstance(a, bm.Array))
    for key in hists.keys():
      self.mon[key] = hists[key]
    self.i0 += num_step
    self.t0 += (num_step * self.dt if duration is None else duration)
    return outputs if not eval_time else (running_time, outputs)

  def run(self, *args, **kwargs) -> Union[Output, Tuple[float, Output]]:
    """Same as :py:func:`~.DSRunner.predict`.
    """
    return self.predict(*args, **kwargs)

  def __call__(self, *args, **kwargs) -> Union[Output, Tuple[float, Output]]:
    """Same as :py:func:`~.DSRunner.predict`.
    """
    return self.predict(*args, **kwargs)

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
    _predict_func = self._get_f_predict(shared_args)
    outs_and_mons = _predict_func(xs)
    if isinstance(self.target.mode, bm.BatchingMode) and self.data_first_axis == 'B':
      outs_and_mons = tree_map(lambda x: bm.moveaxis(x, 0, 1),
                               outs_and_mons,
                               is_leaf=lambda x: isinstance(x, bm.Array))
    return outs_and_mons

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
    return res

  def _step_func_input(self, shared):
    if callable(self._inputs):
      self._inputs(shared)
    else:
      for ops, values in self._inputs['fixed'].items():
        for var, data in values:
          _f_ops(ops, var, data)
      for ops, values in self._inputs['array'].items():
        for var, data in values:
          _f_ops(ops, var, data[shared['i']])
      for ops, values in self._inputs['functional'].items():
        for var, data in values:
          _f_ops(ops, var, data(shared))
      for ops, values in self._inputs['iterated'].items():
        for var, data in values:
          _f_ops(ops, var, next(data))

  def _get_input_batch_size(self, xs=None) -> Optional[int]:
    """Get the batch size in the given input data."""
    if xs is None:
      return None
    if isinstance(self.target.mode, bm.NonBatchingMode):
      return None
    if isinstance(xs, (bm.Array, jax.Array, np.ndarray)):
      return xs.shape[1] if self.data_first_axis == 'T' else xs.shape[0]
    leaves, _ = tree_flatten(xs, is_leaf=_is_brainpy_array)
    if self.data_first_axis == 'T':
      num_batch_sizes = [x.shape[1] for x in leaves]
    else:
      num_batch_sizes = [x.shape[0] for x in leaves]
    if len(set(num_batch_sizes)) != 1:
      raise ValueError(f'Number of batch size is different across arrays in '
                       f'the provided "xs". We got {set(num_batch_sizes)}.')
    return num_batch_sizes[0]

  def _get_input_time_step(self, duration=None, xs=None) -> int:
    """Get the length of time step in the given ``duration`` and ``xs``."""
    if duration is not None:
      return int(duration / self.dt)
    if xs is not None:
      if isinstance(xs, (bm.Array, jnp.ndarray)):
        return xs.shape[0] if self.data_first_axis == 'T' else xs.shape[1]
      else:
        leaves, _ = tree_flatten(xs, is_leaf=lambda x: isinstance(x, bm.Array))
        if self.data_first_axis == 'T':
          num_steps = [x.shape[0] for x in leaves]
        else:
          num_steps = [x.shape[1] for x in leaves]
        if len(set(num_steps)) != 1:
          raise ValueError(f'Number of time step is different across arrays in '
                           f'the provided "xs". We got {set(num_steps)}.')
        return num_steps[0]

    else:
      raise ValueError

  def _step_func_predict(self, shared_args, t, i, x):
    # input step
    shared = tools.DotDict(t=t, i=i, dt=self.dt)
    shared.update(shared_args)
    self.target.clear_input()
    self._step_func_input(shared)

    # dynamics update step
    args = (shared,) if x is None else (shared, x)
    out = self.target(*args)

    # monitor step
    shared['t'] += self.dt
    mon = self._step_func_monitor(shared)

    # finally
    if self.progress_bar:
      id_tap(lambda *arg: self._pbar.update(), ())
    return out, mon

  def _get_f_predict(self, shared_args: Dict = None):
    if shared_args is None:
      shared_args = dict()

    shared_kwargs_str = serialize_kwargs(shared_args)
    if shared_kwargs_str not in self._f_predict_compiled:
      dyn_vars = self.target.vars()
      dyn_vars.update(self._dyn_vars)
      dyn_vars.update(self.vars(level=0))
      dyn_vars = dyn_vars.unique()

      def run_func(all_inputs):
        with jax.disable_jit(not self.jit['predict']):
          return bm.for_loop(partial(self._step_func_predict, shared_args),
                             all_inputs,
                             dyn_vars=dyn_vars)

      if self.jit['predict']:
        self._f_predict_compiled[shared_kwargs_str] = bm.jit(run_func, dyn_vars=dyn_vars)
      else:
        self._f_predict_compiled[shared_kwargs_str] = run_func
    return self._f_predict_compiled[shared_kwargs_str]

  def __del__(self):
    if hasattr(self, '_f_predict_compiled'):
      for key in tuple(self._f_predict_compiled.keys()):
        self._f_predict_compiled.pop(key)
    super(DSRunner, self).__del__()
