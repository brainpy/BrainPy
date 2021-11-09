# -*- coding: utf-8 -*-

import _thread as thread
import inspect
import threading

import numpy as np

from brainpy import errors, math, tools
from brainpy.integrators import analysis_by_ast, constants
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.integrators.ode.base import ODEIntegrator
from pprint import pprint
from brainpy.simulation.utils import run_model


try:
  import numba
  from numba.core.dispatcher import Dispatcher
  import numba.misc.help.inspector as inspector
  from brainpy.math.numpy.ast2numba import _jit_cls_func
except ModuleNotFoundError:
  numba = None
  Dispatcher = None
  inspector = None
  _jit_cls_func = None

__all__ = [
  'transform_integrals_to_model',
  'DynamicModel',
  'rescale',
  'timeout',
  'jit_compile',
  'add_arrow',
  'contain_unknown_symbol',
  'find_indexes_of_limit_cycle_max',
  'find_indexes_of_limit_cycle_max2',
  'find_indexes_of_limit_cycle_min',
  'find_indexes_of_limit_cycle_min2',
]


def _static_self_data(f):
  if _jit_cls_func is None:
    raise errors.PackageMissingError('Must install numba when using symbolic analysis.')

  # function source code
  if tools.is_lambda_function(f):
    raise errors.AnalyzerError(f'Cannot analyze lambda function: {f}.')
  code = tools.deindent(inspect.getsource(f))
  code_scope = dict()

  # static parameters in the class
  if hasattr(f, '__self__'):
    r = _jit_cls_func(f=f, code=code, host=f.__self__, nopython=True, fastmath=True)
    code = r['code']
    code_scope.update(r['nodes'])
    code_scope.update(r['code_scope'])
    if len(r['arguments']) > 0:
      raise errors.UnsupportedError(f'Symbolic analysis does not support variable data '
                                    f'in the derivative function: {f}')
  return code, code_scope


def transform_integrals_to_model(integrals):
  from brainpy.integrators import analysis_by_sympy

  # check integrals
  if isinstance(integrals, ODEIntegrator):
    integrals = [integrals]
  if isinstance(integrals, (list, tuple)):
    assert len(integrals), f'Found no integrators: {integrals}'
    integrals = tuple(integrals)
    for intg in integrals:
      assert isinstance(intg, ODEIntegrator), f'Must be the instance of {ODEIntegrator}, but got {intg}.'
  elif isinstance(integrals, dict):
    assert len(integrals), f'Found no integrators: {integrals}'
    integrals = tuple(integrals.values())
    for intg in integrals:
      assert isinstance(intg, ODEIntegrator), f'Must be the instance of {ODEIntegrator}, but got {intg}'
  elif isinstance(integrals, DynamicalSystem):
    integrals = tuple(integrals.ints().unique().values())
  else:
    raise errors.UnsupportedError(f'Dynamics analysis by symbolic approach only supports '
                                  f'{ODEIntegrator} or {DynamicalSystem}, but we got: '
                                  f'{type(integrals)}: {str(integrals)}')

  # pars to update
  pars_update = set()
  for intg in integrals:
    pars_update.update(intg.parameters[1:])

  all_scope = dict(math=math)
  all_variables = set()
  all_parameters = set()
  analyzers = []
  for integral in integrals:
    assert isinstance(integral, ODEIntegrator)

    # separate variables
    f = integral.derivative[constants.F]
    f_code, code_scope = _static_self_data(f)
    closure_vars = inspect.getclosurevars(f)
    code_scope.update(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)
    if hasattr(integral.derivative[constants.F], '__self__'):
      code_scope['self'] = integral.derivative[constants.F].__self__

    analysis = analysis_by_ast.separate_variables(f_code)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, vars) in enumerate(variables_for_returns.items()):
      variables = []
      for v in vars:
        if len(v) > 1:
          raise ValueError(f'Cannot analyze multi-assignment code line: {vars}.')
        variables.append(v[0])
      expressions = expressions_for_returns[key]
      var_name = integral.variables[vi]
      DE = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                          variables=variables,
                                          expressions=expressions,
                                          derivative_expr=key,
                                          scope={k: v for k, v in code_scope.items()},
                                          func_name=integral.func_name)
      analyzers.append(DE)

    # others
    for var in integral.variables:
      if var in all_variables:
        raise errors.BrainPyError(f'Variable {var} has been defined before. Cannot '
                                  f'group this integral as a dynamic system.')
      all_variables.add(var)

    # final
    all_parameters.update(integral.parameters)
    all_scope.update(code_scope)

  # form a dynamic model
  return DynamicModel(integrals=integrals,
                      analyzers=analyzers,
                      variables=list(all_variables),
                      parameters=list(all_parameters),
                      pars_update=pars_update,
                      scopes=all_scope)


class DynamicModel(object):
  """The wrapper of a dynamical model."""
  def __init__(self,
               integrals,
               analyzers,
               variables,
               parameters,
               scopes,
               pars_update=None):
    self.integrals = integrals
    self.analyzers = analyzers
    self.variables = variables
    self.parameters = parameters
    self.scopes = scopes
    self.pars_update = pars_update


class Trajectory(object):
  """Trajectory Class.

  Parameters
  ----------
  model : DynamicModel
    The instance of DynamicModel.
  size : int, tuple, list
    The network size.
  target_vars : dict
    The target variables, with the format of "{key: initial_v}".
  fixed_vars : dict
    The fixed variables, with the format of "{key: fixed_v}".
  pars_update : dict
    The parameters to update.
  """
  def __init__(self, model, size, target_vars, fixed_vars, pars_update, show_code=False):
    assert isinstance(model, DynamicModel), f'"model" must be an instance of {DynamicModel}, ' \
                                            f'while we got {model}'
    self.model = model
    self.target_vars = target_vars
    self.fixed_vars = fixed_vars
    self.pars_update = pars_update
    self.show_code = show_code
    self.scope = {k: v for k, v in model.scopes.items()}

    # check network size
    if isinstance(size, int):
      size = (size,)
    elif isinstance(size, (tuple, list)):
      assert isinstance(size[0], int)
      size = tuple(size)
    else:
      raise ValueError

    # monitors, variables, parameters
    self.mon = tools.DictPlus()
    self.vars_and_pars = tools.DictPlus()
    for key, val in target_vars.items():
      self.vars_and_pars[key] = math.ones(size) * val
      self.mon[key] = []
    for key, val in fixed_vars.items():
      self.vars_and_pars[key] = math.ones(size) * val
    for key, val in pars_update.items():
      self.vars_and_pars[key] = val
    self.scope['VP'] = self.vars_and_pars
    self.scope['MON'] = self.mon
    self.scope['_fixed_vars'] = fixed_vars

    code_lines = ['def run_func(t_and_dt):']
    code_lines.append('  _t, _dt = t_and_dt')
    for integral in self.model.integrals:
      assert isinstance(integral, ODEIntegrator)
      func_name = integral.func_name
      self.scope[func_name] = integral
      # update the step function
      assigns = [f'VP["{var}"]' for var in integral.variables]
      calls = [f'VP["{var}"]' for var in integral.variables]
      calls.append('_t')
      calls.extend([f'VP["{var}"]' for var in integral.parameters[1:]])
      code_lines.append(f'  {", ".join(assigns)} = {func_name}({", ".join(calls)})')
      # reassign the fixed variables
      for key, val in fixed_vars.items():
        code_lines.append(f'  VP["{key}"][:] = _fixed_vars["{key}"]')
    # monitor the target variables
    for key in target_vars.keys():
      code_lines.append(f'  MON["{key}"].append(VP["{key}"])')
    # compile
    code = '\n'.join(code_lines)
    if show_code:
      print(code)
      print()
      pprint(self.scope)
      print()

    # recompile
    exec(compile(code, '', 'exec'), self.scope)
    self.run_func = self.scope['run_func']

  def run(self, duration, report=0.1):
    if isinstance(duration, (int, float)):
      duration = [0, duration]
    elif isinstance(duration, (tuple, list)):
      assert len(duration) == 2
      duration = tuple(duration)
    else:
      raise ValueError

    # get the times
    times = math.arange(duration[0], duration[1], math.get_dt())
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = []
    # run the model
    run_model(run_func=self.run_func, times=times, report=report)
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = math.asarray(self.mon[key])


def rescale(min_max, scale=0.01):
  """Rescale lim."""
  min_, max_ = min_max
  length = max_ - min_
  min_ -= scale * length
  max_ += scale * length
  return min_, max_


def timeout(s):
  """Add a timeout parameter to a function and return it.

  Parameters
  ----------
  s : float
      Time limit in seconds.

  Returns
  -------
  func : callable
      Functional results. Or, raise an error of KeyboardInterrupt.
  """

  def outer(fn):
    def inner(*args, **kwargs):
      timer = threading.Timer(s, thread.interrupt_main)
      timer.start()
      try:
        result = fn(*args, **kwargs)
      finally:
        timer.cancel()
      return result
    return inner
  return outer


def _check_numpy_bk():
  bk_name = math.get_backend_name()
  if bk_name != 'numpy':
    raise errors.UnsupportedError('Only support "numpy" backend in symbolic analysis.')


def _jit(func):
  if inspector.inspect_function(func)['numba_type'] is not None:
    return func
  if isinstance(func, Dispatcher):
    return func
  vars = inspect.getclosurevars(func)
  code_scope = dict(vars.nonlocals)
  code_scope.update(vars.globals)

  modified = False
  # check scope variables
  for k, v in code_scope.items():
    # function
    if callable(v):
      if (inspector.inspect_function(v)['numba_type'] is None) and (not isinstance(v, Dispatcher)):
        code_scope[k] = _jit(v)
        modified = True

  if modified:
    func_code = tools.deindent(tools.get_func_source(func))
    exec(compile(func_code, '', "exec"), code_scope)
    func = code_scope[func.__name__]
    return numba.njit(func)
  else:
    return numba.njit(func)


def jit_compile(scope, func_code, func_name):
  _check_numpy_bk()
  assert Dispatcher is not None, 'Please install Numba first when using symbolic analysis.'
  # get function scope
  func_scope = dict()
  for key, val in scope.items():
    if callable(val) and (inspector.inspect_function(val)['numba_type'] is None):
      val = _jit(val)
    func_scope[key] = val
  # compile function
  exec(compile(func_code, '', 'exec'), func_scope)
  return numba.njit(func_scope[func_name])


def contain_unknown_symbol(expr, scope):
  """Examine where the given expression ``expr`` has the unknown symbol in ``scope``.

  Returns
  -------
  res : bool
      True or False.
  """
  ids = tools.get_identifiers(expr)
  for id_ in ids:
    if '.' in id_:
      prefix = id_.split('.')[0].strip()
      if prefix not in scope:
        return True
    if id_ not in scope:
      return True
  return False


def add_arrow(line, position=None, direction='right', size=15, color=None):
  """
  add an arrow to a line.

  line:       Line2D object
  position:   x-position of the arrow. If None, mean of xdata is taken
  direction:  'left' or 'right'
  size:       size of the arrow in fontsize points
  color:      if None, line color is taken.
  """
  if color is None:
    color = line.get_color()

  xdata = line.get_xdata()
  ydata = line.get_ydata()

  if position is None:
    position = xdata.mean()
  # find closest index
  start_ind = np.argmin(np.absolute(xdata - position))
  if direction == 'right':
    end_ind = start_ind + 1
  else:
    end_ind = start_ind - 1

  line.axes.annotate(text='',
                     xytext=(xdata[start_ind], ydata[start_ind]),
                     xy=(xdata[end_ind], ydata[end_ind]),
                     arrowprops=dict(arrowstyle="->", color=color),
                     size=size)


@tools.numba_jit
def _f1(arr, grad, tol):
  condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
  indexes = np.where(condition)[0]
  if len(indexes) >= 2:
    data = arr[indexes[-2]: indexes[-1]]
    length = np.max(data) - np.min(data)
    a = arr[indexes[-2]]
    b = arr[indexes[-1]]
    if np.abs(a - b) < tol * length:
      return indexes[-2:]
  return np.array([-1, -1])


@tools.numba_jit
def _f2(arr, grad, tol):
  condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] <= 0)
  indexes = np.where(condition)[0]
  if len(indexes) >= 2:
    data = arr[indexes[-2]: indexes[-1]]
    length = np.max(data) - np.min(data)
    a = arr[indexes[-2]]
    b = arr[indexes[-1]]
    if np.abs(a - b) < tol * length:
      return indexes[-2:]
  return np.array([-1, -1])


def find_indexes_of_limit_cycle_max(arr, tol=0.001):
  grad = np.gradient(arr)
  return _f1(arr, grad, tol)


def find_indexes_of_limit_cycle_min(arr, tol=0.001):
  grad = np.gradient(arr)
  return _f2(arr, grad, tol)


@tools.numba_jit
def _identity(a, b, tol=0.01):
  if np.abs(a - b) < tol:
    return True
  else:
    return False


def find_indexes_of_limit_cycle_max2(arr, tol=0.001):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
    indexes = np.where(condition)[0]
    if len(indexes) >= 2:
      data = arr[indexes[-2]: indexes[-1]]
      length = np.max(data) - np.min(data)
      if _identity(arr[indexes[-2]], arr[indexes[-1]], tol * length):
        return indexes[-2:]
    return np.array([-1, -1])

  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] >= 0)
    indexes = -np.ones((len(conditions), 2), dtype=int)
    for i, condition in enumerate(conditions):
      idx = np.where(condition)[0]
      if len(idx) >= 2:
        if _identity(arr[idx[-2]], arr[idx[-1]], tol):
          indexes[i] = idx[-2:]
    return indexes

  else:
    raise ValueError


def find_indexes_of_limit_cycle_min2(arr, tol=0.01):
  if np.ndim(arr) == 1:
    grad = np.gradient(arr)
    condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] <= 0)
    indexes = np.where(condition)[0]
    if len(indexes) >= 2:
      indexes += 1
      if _identity(arr[indexes[-2]], arr[indexes[-1]], tol):
        return indexes[-2:]
    return np.array([-1, -1])

  elif np.ndim(arr) == 2:
    # The data with the shape of (axis_along_time, axis_along_neuron)
    grads = np.gradient(arr, axis=0)
    conditions = np.logical_and(grads[:-1] * grads[1:] <= 0, grads[:-1] <= 0)
    indexes = -np.ones((len(conditions), 2), dtype=int)
    for i, condition in enumerate(conditions):
      idx = np.where(condition)[0]
      if len(idx) >= 2:
        idx += 1
        if _identity(arr[idx[-2]], arr[idx[-1]], tol):
          indexes[i] = idx[-2:]
    return indexes

  else:
    raise ValueError

