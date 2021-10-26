# -*- coding: utf-8 -*-

import _thread as thread
import ast
import inspect
import re
import threading

import numpy as np

from brainpy import errors, math, tools
from brainpy.integrators import analysis_by_ast, utils, DE_INT

try:
  import numba
  from numba.core.dispatcher import Dispatcher
except ModuleNotFoundError:
  numba = None
  Dispatcher = None

DynamicalSystem = None

__all__ = [
  'transform_integrals_to_model',
  'DynamicModel',
  'rescale',
  'timeout',
  'jit_compile',
  'add_arrow',
  'contain_unknown_symbol',
]

# Get functions in backend
_functions_in_math = []
for key in dir(math):
  if not key.startswith('__'):
    _functions_in_math.append(getattr(math, key))

# Get functions in NumPy
_functions_in_numpy = []
for key in dir(np):
  if not key.startswith('__'):
    _functions_in_numpy.append(getattr(np, key))
for key in dir(np.random):
  if not key.startswith('__'):
    _functions_in_numpy.append(getattr(np.random, key))
for key in dir(np.linalg):
  if not key.startswith('__'):
    _functions_in_numpy.append(getattr(np.linalg, key))


def func_in_numpy_or_math(func):
  return func in _functions_in_math or func in _functions_in_numpy


def transform_integrals(integrals, method='euler'):
  global DynamicalSystem
  if DynamicalSystem is None: from brainpy.simulation.brainobjects.base import DynamicalSystem

  pars_update = {}
  new_integrals = []
  for integral in integrals:
    # integral function
    if Dispatcher is not None and isinstance(integral, Dispatcher):
      integral = integral.py_func
    else:
      integral = integral

    # derivative function
    func = integral.brainpy_data['raw_func']['f']
    if Dispatcher is not None and isinstance(func, Dispatcher):
      func = func.py_func
    func_name = func.__name__

    # arguments
    class_kw, variables, parameters, _ = utils.get_args(func)

    if len(class_kw) == 0:
      pars_update.update({p: None for p in parameters[1:]})
      new_integrals.append(integral)

    else:
      assert len(class_kw) == 1
      code = tools.deindent(inspect.getsource(func)).strip()
      tree = ast.parse(code)
      tree.body[0].args.args.pop(0)  # remove "self" arg
      self_data = re.findall('\\b' + class_kw[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
      self_data = list(set(self_data))

      # node of integral
      f_node = None
      if hasattr(integral, '__self__') and isinstance(integral.__self__, DynamicalSystem):
        f_node = integral.__self__

      # node of derivative function
      func_node = None
      if f_node:
        func_node = f_node
      elif hasattr(func, '__self__') and isinstance(func.__self__, DynamicalSystem):
        func_node = func.__self__

      # code scope
      closure_vars = inspect.getclosurevars(func)
      code_scope = dict(closure_vars.nonlocals)
      code_scope.update(closure_vars.globals)

      # analyze variables and functions accessed by the self.xx
      if func_node:
        arguments = set()
        data_to_replace = {}
        for key in self_data:
          split_keys = key.split('.')
          if len(split_keys) < 2:
            raise errors.BrainPyError

          # get target and data
          target = func_node
          for i in range(1, len(split_keys)):
            next_target = getattr(target, split_keys[i])
            if not isinstance(next_target, DynamicalSystem):
              break
            target = next_target
          else:
            raise errors.BrainPyError
          data = getattr(target, split_keys[i])

          key = '.'.join(split_keys[:i + 1])
          if isinstance(data, np.random.RandomState):
            data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
            code_scope[f'{target.name}_{split_keys[i]}'] = np.random  # replace RandomState

          elif callable(data):
            data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
            code_scope[f'{target.name}_{split_keys[i]}'] = data

          elif isinstance(data, (math.ndarray, int, float)):
            assert len(split_keys) == i + 1
            arguments.add(split_keys[i])
            pars_update[split_keys[i]] = data
            data_to_replace[key] = split_keys[i]  # replace the data

          else:  # parameters
            data_to_replace[key] = f'{target.name}_{split_keys[i]}'  # replace the data
            code_scope[f'{target.name}_{split_keys[i]}'] = data

        # final code
        tree.body[0].decorator_list.clear()
        tree.body[0].args.args.extend([ast.Name(id=a) for a in sorted(arguments)])
        tree.body[0].args.defaults.extend([ast.Constant(None) for _ in sorted(arguments)])
        code = tools.ast2code(tree)
        code = tools.word_replace(code, data_to_replace, exclude_dot=False)

        # compile new function
        # if show_code:
        #   print(code)
        #   print()
        #   pprint(code_scope)
        #   print()
        exec(compile(code, '', 'exec'), code_scope)
        func = code_scope[func_name]
        func.code = code

        new_integrals.append(odeint(f=func, method=method))
  return new_integrals, pars_update


def transform_integrals_to_model(integrals, method='euler'):
  from brainpy.integrators import analysis_by_sympy
  global DynamicalSystem
  if DynamicalSystem is None: from brainpy.simulation.brainobjects.base import DynamicalSystem

  # check integrals
  if callable(integrals):
    integrals = [integrals]
  if isinstance(integrals, (list, tuple)):
    integrals = tuple(integrals)
  elif isinstance(integrals, dict):
    integrals = tuple(integrals.values())
  elif isinstance(integrals, DynamicalSystem):
    integrals = tuple(integrals.ints().unique().values())
  else:
    raise errors.UnsupportedError(f'Dynamics analysis by symbolic approach only supports '
                                  f'integrators, but we got: {type(integrals)}: {str(integrals)}')
  for intg in integrals:
    assert callable(intg) and intg.__name__.startswith(DE_INT)

  integrals, pars_update = transform_integrals(integrals, method=method)

  all_scope = dict(math=math)
  all_variables = set()
  all_parameters = set()
  analyzers = []
  for integral in integrals:
    # integral function
    if Dispatcher is not None and isinstance(integral, Dispatcher):
      integral = integral.py_func
    else:
      integral = integral

    # original function
    f = integral.brainpy_data['raw_func']['f']
    if Dispatcher is not None and isinstance(f, Dispatcher):
      f = f.py_func
    func_name = f.__name__

    # code scope
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(dict(closure_vars.globals))

    # separate variables
    analysis = analysis_by_ast.separate_variables(f.code if hasattr(f, 'code') else f)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, vars) in enumerate(variables_for_returns.items()):
      variables = []
      for v in vars:
        if len(v) > 1:
          raise ValueError('Cannot analyze multi-assignment code line.')
        variables.append(v[0])
      expressions = expressions_for_returns[key]
      var_name = integral.brainpy_data['variables'][vi]
      DE = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                          variables=variables,
                                          expressions=expressions,
                                          derivative_expr=key,
                                          scope=code_scope,
                                          func_name=func_name)
      analyzers.append(DE)

    # others
    for var in integral.brainpy_data['variables']:
      if var in all_variables:
        raise errors.BrainPyError(f'Variable {var} has been defined before. Cannot group '
                                  f'this integral as a dynamic system.')
      all_variables.add(var)
    all_parameters.update(integral.brainpy_data['parameters'])
    all_scope.update(code_scope)

  # form a dynamic model
  return DynamicModel(integrals=integrals,
                      analyzers=analyzers,
                      variables=list(all_variables),
                      parameters=list(all_parameters),
                      pars_update=pars_update,
                      scopes=all_scope)


class DynamicModel(object):
  def __init__(self, integrals, analyzers, variables, parameters, scopes,
               pars_update=None):
    self.integrals = integrals
    self.analyzers = analyzers
    self.variables = variables
    self.parameters = parameters
    self.scopes = scopes
    self.pars_update = pars_update


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


def _jit(func):
  if func_in_numpy_or_math(func):
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
      if (not func_in_numpy_or_math(v)) and (not isinstance(v, Dispatcher)):
        code_scope[k] = _jit(v)
        modified = True

  if modified:
    func_code = tools.deindent(tools.get_func_source(func))
    exec(compile(func_code, '', "exec"), code_scope)
    func = code_scope[func.__name__]
    return numba.njit(func)
  else:
    return numba.njit(func)


def _is_numpy_bk():
  bk_name = math.get_backend_name()
  return bk_name.startswith('numba') or bk_name == 'numpy'


def jit_compile(scope, func_code, func_name):
  if (numba is None) or (not _is_numpy_bk()):
    func_scope = scope
  else:
    assert Dispatcher is not None
    # get function scope
    func_scope = dict()
    for key, val in scope.items():
      if callable(val):
        if func_in_numpy_or_math(val):
          pass
        elif isinstance(val, Dispatcher):
          pass
        else:
          val = _jit(val)
      func_scope[key] = val

  # compile function
  exec(compile(func_code, '', 'exec'), func_scope)
  if numba is None:
    return func_scope[func_name]
  else:
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
def f1(arr, grad, tol):
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
def f2(arr, grad, tol):
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
  return f1(arr, grad, tol)


def find_indexes_of_limit_cycle_min(arr, tol=0.001):
  grad = np.gradient(arr)
  return f2(arr, grad, tol)


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
