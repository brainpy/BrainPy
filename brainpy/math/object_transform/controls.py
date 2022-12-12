# -*- coding: utf-8 -*-


from typing import Union, Sequence, Any, Dict, Callable, Optional

import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten

try:
  from jax.errors import UnexpectedTracerError
except ImportError:
  from jax.core import UnexpectedTracerError

from brainpy import errors
from brainpy.base.naming import get_unique_name
from brainpy.math.jaxarray import (JaxArray, Variable,
                                   add_context,
                                   del_context)
from brainpy.math.numpy_ops import as_device_array

__all__ = [
  'make_loop',
  'make_while',
  'make_cond',

  'cond',
  'ifelse',
  'for_loop',
  'while_loop',
]


def _get_scan_info(f, dyn_vars, out_vars=None, has_return=False):
  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(
      f'"dyn_vars" does not support {type(dyn_vars)}, '
      f'only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(
        f'brainpy.math.jax.make_loop only support '
        f'{JaxArray.__name__}, but got {type(v)}')

  # outputs
  if out_vars is None:
    out_vars = ()
    _, tree = tree_flatten(out_vars)
  elif isinstance(out_vars, JaxArray):
    _, tree = tree_flatten(out_vars)
    out_vars = (out_vars,)
  elif isinstance(out_vars, dict):
    _, tree = tree_flatten(out_vars)
    out_vars = tuple(out_vars.values())
  elif isinstance(out_vars, (tuple, list)):
    _, tree = tree_flatten(out_vars)
    out_vars = tuple(out_vars)
  else:
    raise ValueError(
      f'"out_vars" does not support {type(out_vars)}, '
      f'only support dict/list/tuple of {JaxArray.__name__}')

  # functions
  if has_return:
    def fun2scan(dyn_values, x):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      results = f(x)
      dyn_values = [v.value for v in dyn_vars]
      out_values = [v.value for v in out_vars]
      return dyn_values, (out_values, results)

  else:
    def fun2scan(dyn_values, x):
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      f(x)
      dyn_values = [v.value for v in dyn_vars]
      out_values = [v.value for v in out_vars]
      return dyn_values, out_values

  return fun2scan, dyn_vars, tree


def make_loop(body_fun, dyn_vars, out_vars=None, has_return=False):
  """Make a for-loop function, which iterate over inputs.

  Examples
  --------

  >>> import brainpy.math as bm
  >>>
  >>> a = bm.Variable(bm.zeros(1))
  >>> def f(x): a.value += 1.
  >>> loop = bm.make_loop(f, dyn_vars=[a], out_vars=a)
  >>> loop(bm.arange(10))
  Variable([[ 1.],
            [ 2.],
            [ 3.],
            [ 4.],
            [ 5.],
            [ 6.],
            [ 7.],
            [ 8.],
            [ 9.],
            [10.]], dtype=float32)
  >>> b = bm.Variable(bm.zeros(1))
  >>> def f(x):
  >>>   b.value += 1
  >>>   return b + 1
  >>> loop = bm.make_loop(f, dyn_vars=[b], out_vars=b, has_return=True)
  >>> hist_b, hist_b_plus = loop(bm.arange(10))
  >>> hist_b
  Variable([[ 1.],
            [ 2.],
            [ 3.],
            [ 4.],
            [ 5.],
            [ 6.],
            [ 7.],
            [ 8.],
            [ 9.],
            [10.]], dtype=float32)
  >>> hist_b_plus
  JaxArray([[ 2.],
            [ 3.],
            [ 4.],
            [ 5.],
            [ 6.],
            [ 7.],
            [ 8.],
            [ 9.],
            [10.],
            [11.]], dtype=float32)

  Parameters
  ----------
  body_fun : callable, function
    A function receive one argument. This argument refers to the iterable input ``x``.
  dyn_vars : dict of JaxArray, sequence of JaxArray
    The dynamically changed variables, while iterate between trials.
  out_vars : optional, JaxArray, dict of JaxArray, sequence of JaxArray
    The variables to output their values.
  has_return : bool
    The function has the return values.

  Returns
  -------
  loop_func : callable, function
    The function for loop iteration. This function receives one argument ``xs``, denoting
    the input tensor which interate over the time (note ``body_fun`` receive ``x``).
  """

  fun2scan, dyn_vars, tree = _get_scan_info(f=body_fun,
                                            dyn_vars=dyn_vars,
                                            out_vars=out_vars,
                                            has_return=has_return)

  name = get_unique_name('_brainpy_object_oriented_make_loop_')

  # functions
  if has_return:
    def call(xs=None, length=None):
      init_values = [v.value for v in dyn_vars]
      try:
        add_context(name)
        dyn_values, (out_values, results) = lax.scan(
          f=fun2scan, init=init_values, xs=xs, length=length)
        del_context(name)
      except UnexpectedTracerError as e:
        del_context(name)
        for v, d in zip(dyn_vars, init_values): v._value = d
        raise errors.JaxTracerError(variables=dyn_vars) from e
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      return tree_unflatten(tree, out_values), results

  else:
    def call(xs):
      init_values = [v.value for v in dyn_vars]
      try:
        add_context(name)
        dyn_values, out_values = lax.scan(f=fun2scan, init=init_values, xs=xs)
        del_context(name)
      except UnexpectedTracerError as e:
        del_context(name)
        for v, d in zip(dyn_vars, init_values): v._value = d
        raise errors.JaxTracerError(variables=dyn_vars) from e
      except Exception as e:
        del_context(name)
        for v, d in zip(dyn_vars, init_values): v._value = d
        raise e
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      return tree_unflatten(tree, out_values)

  return call


def make_while(cond_fun, body_fun, dyn_vars):
  """Make a while-loop function.

  This function is similar to the ``jax.lax.while_loop``. The difference is that,
  if you are using ``JaxArray`` in your while loop codes, this function will help
  you make a easy while loop function. Note: ``cond_fun`` and ``body_fun`` do no
  receive any arguments. ``cond_fun`` shoule return a boolean value. ``body_fun``
  does not support return values.

  Examples
  --------
  >>> import brainpy.math as bm
  >>>
  >>> a = bm.zeros(1)
  >>>
  >>> def cond_f(x): return a[0] < 10
  >>> def body_f(x): a.value += 1.
  >>>
  >>> loop = bm.make_while(cond_f, body_f, dyn_vars=[a])
  >>> loop()
  >>> a
  JaxArray([10.], dtype=float32)

  Parameters
  ----------
  cond_fun : function, callable
    A function receives one argument, but return a boolean value.
  body_fun : function, callable
    A function receives one argument, without any returns.
  dyn_vars : dict of JaxArray, sequence of JaxArray
    The dynamically changed variables, while iterate between trials.

  Returns
  -------
  loop_func : callable, function
      The function for loop iteration, which receive one argument ``x`` for external input.
  """
  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, '
                     f'only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'Only support {JaxArray.__name__}, but got {type(v)}')

  def _body_fun(op):
    dyn_values, static_values = op
    for v, d in zip(dyn_vars, dyn_values): v._value = d
    body_fun(static_values)
    return [v.value for v in dyn_vars], static_values

  def _cond_fun(op):
    dyn_values, static_values = op
    for v, d in zip(dyn_vars, dyn_values): v._value = d
    return as_device_array(cond_fun(static_values))

  name = get_unique_name('_brainpy_object_oriented_make_while_')

  def call(x=None):
    dyn_init = [v.value for v in dyn_vars]
    try:
      add_context(name)
      dyn_values, _ = lax.while_loop(cond_fun=_cond_fun,
                                     body_fun=_body_fun,
                                     init_val=(dyn_init, x))
      del_context(name)
    except UnexpectedTracerError as e:
      del_context(name)
      for v, d in zip(dyn_vars, dyn_init): v._value = d
      raise errors.JaxTracerError(variables=dyn_vars) from e
    except Exception as e:
      del_context(name)
      for v, d in zip(dyn_vars, dyn_init): v._value = d
      raise e
    for v, d in zip(dyn_vars, dyn_values): v._value = d

  return call


def make_cond(true_fun, false_fun, dyn_vars=None):
  """Make a condition (if-else) function.

  Examples
  --------

  >>> import brainpy.math as bm
  >>> a = bm.zeros(2)
  >>> b = bm.ones(2)
  >>>
  >>> def true_f(x):  a.value += 1
  >>> def false_f(x): b.value -= 1
  >>>
  >>> cond = bm.make_cond(true_f, false_f, dyn_vars=[a, b])
  >>> cond(True)
  >>> a, b
  (JaxArray([1., 1.], dtype=float32),
   JaxArray([1., 1.], dtype=float32))
  >>> cond(False)
  >>> a, b
  (JaxArray([1., 1.], dtype=float32),
   JaxArray([0., 0.], dtype=float32))

  Parameters
  ----------
  true_fun : callable, function
    A function receives one argument, without any returns.
  false_fun : callable, function
    A function receives one argument, without any returns.
  dyn_vars : dict of JaxArray, sequence of JaxArray
    The dynamically changed variables.

  Returns
  -------
  cond_func : callable, function
      The condictional function receives two arguments: ``pred`` for true/false judgement
      and ``x`` for external input.
  """
  # iterable variables
  if dyn_vars is None:
    dyn_vars = []
  if isinstance(dyn_vars, JaxArray):
    dyn_vars = (dyn_vars,)
  elif isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, '
                     f'only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'Only support {JaxArray.__name__}, but got {type(v)}')

  name = get_unique_name('_brainpy_object_oriented_make_cond_')

  if len(dyn_vars) > 0:
    def _true_fun(op):
      dyn_vals, static_vals = op
      for v, d in zip(dyn_vars, dyn_vals): v._value = d
      res = true_fun(static_vals)
      dyn_vals = [v.value for v in dyn_vars]
      return dyn_vals, res

    def _false_fun(op):
      dyn_vals, static_vals = op
      for v, d in zip(dyn_vars, dyn_vals): v._value = d
      res = false_fun(static_vals)
      dyn_vals = [v.value for v in dyn_vars]
      return dyn_vals, res

    def call(pred, x=None):
      old_values = [v.value for v in dyn_vars]
      try:
        add_context(name)
        dyn_values, res = lax.cond(pred, _true_fun, _false_fun, (old_values, x))
        del_context(name)
      except UnexpectedTracerError as e:
        del_context(name)
        for v, d in zip(dyn_vars, old_values): v._value = d
        raise errors.JaxTracerError(variables=dyn_vars) from e
      except Exception as e:
        del_context(name)
        for v, d in zip(dyn_vars, old_values): v._value = d
        raise e
      for v, d in zip(dyn_vars, dyn_values): v._value = d
      return res

  else:
    def call(pred, x=None):
      add_context(name)
      res = lax.cond(pred, true_fun, false_fun, x)
      del_context(name)
      return res

  return call


def _check_f(f):
  if callable(f):
    return f
  else:
    return (lambda _: f)


def _check_sequence(a):
  return isinstance(a, (list, tuple))


def cond(
    pred: bool,
    true_fun: Union[Callable, jnp.ndarray, JaxArray, float, int, bool],
    false_fun: Union[Callable, jnp.ndarray, JaxArray, float, int, bool],
    operands: Any,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None
):
  """Simple conditional statement (if-else) with instance of :py:class:`~.Variable`.

  >>> import brainpy.math as bm
  >>> a = bm.Variable(bm.zeros(2))
  >>> b = bm.Variable(bm.ones(2))
  >>> def true_f(_):  a.value += 1
  >>> def false_f(_): b.value -= 1
  >>>
  >>> bm.cond(True, true_f, false_f, dyn_vars=[a, b])
  >>> a, b
  Variable([1., 1.], dtype=float32), Variable([1., 1.], dtype=float32)
  >>>
  >>> bm.cond(False, true_f, false_f, dyn_vars=[a, b])
  >>> a, b
  Variable([1., 1.], dtype=float32), Variable([0., 0.], dtype=float32)

  Parameters
  ----------
  pred: bool
    Boolean scalar type, indicating which branch function to apply.
  true_fun: callable, jnp.ndarray, JaxArray, float, int, bool
    Function to be applied if ``pred`` is True.
    This function must receive one arguement for ``operands``.
  false_fun: callable, jnp.ndarray, JaxArray, float, int, bool
    Function to be applied if ``pred`` is False.
    This function must receive one arguement for ``operands``.
  operands: Any
    Operands (A) input to branching function depending on ``pred``. The type
    can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof.
  dyn_vars: optional, Variable, sequence of Variable, dict
    The dynamically changed variables.

  Returns
  -------
  res: Any
    The conditional results.
  """
  # checking
  true_fun = _check_f(true_fun)
  false_fun = _check_f(false_fun)
  if dyn_vars is None:
    dyn_vars = tuple()
  elif isinstance(dyn_vars, Variable):
    dyn_vars = (dyn_vars,)
  elif isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, '
                     f'only support dict/list/tuple of {Variable.__name__}')
  for v in dyn_vars:
    if not isinstance(v, Variable):
      raise ValueError(f'Only support {Variable.__name__}, but got {type(v)}')

  name = get_unique_name('_brainpy_object_oriented_cond_')

  # calling the model
  if len(dyn_vars) > 0:
    def _true_fun(op):
      dyn_vals, static_vals = op
      for v, d in zip(dyn_vars, dyn_vals): v._value = d
      res = true_fun(static_vals)
      dyn_vals = [v.value for v in dyn_vars]
      return dyn_vals, res

    def _false_fun(op):
      dyn_vals, static_vals = op
      for v, d in zip(dyn_vars, dyn_vals): v._value = d
      res = false_fun(static_vals)
      dyn_vals = [v.value for v in dyn_vars]
      return dyn_vals, res

    old_values = [v.value for v in dyn_vars]
    try:
      add_context(name)
      dyn_values, res = lax.cond(pred=pred,
                                 true_fun=_true_fun,
                                 false_fun=_false_fun,
                                 operand=(old_values, operands))
      del_context(name)
    except UnexpectedTracerError as e:
      del_context(name)
      for v, d in zip(dyn_vars, old_values): v._value = d
      raise errors.JaxTracerError(variables=dyn_vars) from e
    except Exception as e:
      del_context(name)
      for v, d in zip(dyn_vars, old_values): v._value = d
      raise e
    for v, d in zip(dyn_vars, dyn_values): v._value = d
  else:
    add_context(name)
    res = lax.cond(pred, true_fun, false_fun, operands)
    del_context(name)
  return res


def ifelse(
    conditions: Union[bool, Sequence[bool]],
    branches: Sequence,
    operands: Any = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    show_code: bool = False,
):
  """``If-else`` control flows looks like native Pythonic programming.

  Examples
  --------

  >>> import brainpy.math as bm
  >>> def f(a):
  >>>    return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                     branches=[lambda _: 1,
  >>>                               lambda _: 2,
  >>>                               lambda _: 3,
  >>>                               lambda _: 4,
  >>>                               lambda _: 5])
  >>> f(1)
  4
  >>> # or, it can be expressed as:
  >>> def f(a):
  >>>   return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
  >>>                    branches=[1, 2, 3, 4, 5])
  >>> f(3)
  3

  Parameters
  ----------
  conditions: bool, sequence of bool
    The boolean conditions.
  branches: Sequence
    The branches, at least has two elements. Elements can be functions,
    arrays, or numbers. The number of ``branches`` and ``conditions`` has
    the relationship of `len(branches) == len(conditions) + 1`.
    Each branch should receive one arguement for ``operands``.
  operands: optional, Any
    The operands for each branch.
  dyn_vars: Variable, sequence of Variable, dict
    The dynamically changed variables.
  show_code: bool
    Whether show the formatted code.

  Returns
  -------
  res: Any
    The results of the control flow.
  """
  # checking
  if not isinstance(conditions, (tuple, list)):
    conditions = [conditions]
  if not isinstance(conditions, (tuple, list)):
    raise ValueError(f'"conditions" must be a tuple/list of boolean values. '
                     f'But we got {type(conditions)}: {conditions}')
  if not isinstance(branches, (tuple, list)):
    raise ValueError(f'"branches" must be a tuple/list. '
                     f'But we got {type(branches)}.')
  branches = [_check_f(b) for b in branches]
  if len(branches) != len(conditions) + 1:
    raise ValueError(f'The numbers of branches and conditions do not match. '
                     f'Got len(conditions)={len(conditions)} and len(branches)={len(branches)}. '
                     f'We expect len(conditions) + 1 == len(branches). ')
  if dyn_vars is None:
    dyn_vars = []
  if isinstance(dyn_vars, Variable):
    dyn_vars = (dyn_vars,)
  elif isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, only '
                     f'support dict/list/tuple of brainpy.math.Variable')
  for v in dyn_vars:
    if not isinstance(v, Variable):
      raise ValueError(f'Only support brainpy.math.Variable, but we got {type(v)}')

  # format new codes
  if len(conditions) == 1:
    if len(dyn_vars) > 0:
      return cond(conditions[0], branches[0], branches[1], operands, dyn_vars)
    else:
      return lax.cond(conditions[0], branches[0], branches[1], operands)
  else:
    code_scope = {'conditions': conditions, 'branches': branches}
    codes = ['def f(operands):',
             f'  f0 = branches[{len(conditions)}]']
    num_cond = len(conditions) - 1
    if len(dyn_vars) > 0:
      code_scope['_cond'] = cond
      code_scope['dyn_vars'] = dyn_vars
      for i in range(len(conditions) - 1):
        codes.append(f'  f{i + 1} = lambda r: _cond(conditions[{num_cond - i}], '
                     f'branches[{num_cond - i}], f{i}, r, dyn_vars)')
      codes.append(f'  return _cond(conditions[0], branches[0], f{len(conditions) - 1}, operands, dyn_vars)')
    else:
      code_scope['_cond'] = lax.cond
      for i in range(len(conditions) - 1):
        codes.append(f'  f{i + 1} = lambda r: _cond(conditions[{num_cond - i}], branches[{num_cond - i}], f{i}, r)')
      codes.append(f'  return _cond(conditions[0], branches[0], f{len(conditions) - 1}, operands)')
    codes = '\n'.join(codes)
    if show_code: print(codes)
    exec(compile(codes.strip(), '', 'exec'), code_scope)
    f = code_scope['f']
    name = get_unique_name('_brainpy_object_oriented_ifelse_')
    add_context(name)
    r = f(operands)
    del_context(name)
    return r


def for_loop(body_fun: Callable,
             dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]],
             operands: Any,
             reverse: bool = False,
             unroll: int = 1):
  """``for-loop`` control flow with :py:class:`~.Variable`.

  Simply speaking, all dynamically changed variables used in the body function should
  be labeld in ``dyn_vars`` argument. All returns in body function will be gathered
  as the return of the whole loop.

  >>> import brainpy.math as bm
  >>> a = bm.Variable(bm.zeros(1))
  >>> b = bm.Variable(bm.ones(1))
  >>> # first example
  >>> def body(x):
  >>>    a.value += x
  >>>    b.value *= x
  >>>    return a.value
  >>> a_hist = bm.for_loop(body, dyn_vars=[a, b], operands=bm.arange(1, 5))
  >>> a_hist
  DeviceArray([[ 1.],
               [ 3.],
               [ 6.],
               [10.]], dtype=float32)
  >>> a
  Variable([10.], dtype=float32)
  >>> b
  Variable([24.], dtype=float32)
  >>>
  >>> # another example
  >>> def body(x, y):
  >>>   a.value += x
  >>>   b.value *= y
  >>>   return a.value
  >>> a_hist = bm.for_loop(body,
  >>>                      dyn_vars=[a, b],
  >>>                      operands=(bm.arange(1, 5), bm.arange(2, 6)))
  >>> a_hist
  [[11.]
   [13.]
   [16.]
   [20.]]

  .. versionadded:: 2.1.11

  Parameters
  ----------
  body_fun: callable
    A Python function to be scanned. This function accepts one argument and returns one output.
    The argument denotes a slice of ``operands`` along its leading axis, and that
    output represents a slice of the return value.
  dyn_vars: Variable, sequence of Variable, dict
    The instances of :py:class:`~.Variable`.
  operands: Any
    The value over which to scan along the leading axis,
    where ``operands`` can be an array or any pytree (nested Python
    tuple/list/dict) thereof with consistent leading axis sizes.
    If body function `body_func` receives multiple arguments,
    `operands` should be a tuple/list whose length is equal to the
    number of arguments.
  reverse: bool
    Optional boolean specifying whether to run the scan iteration
    forward (the default) or in reverse, equivalent to reversing the leading
    axes of the arrays in both ``xs`` and in ``ys``.
  unroll: int
    Optional positive int specifying, in the underlying operation of the
    scan primitive, how many scan iterations to unroll within a single
    iteration of a loop.

  Returns
  -------
  outs: Any
    The stacked outputs of ``body_fun`` when scanned over the leading axis of the inputs.
  """
  # check variables
  if dyn_vars is None:
    dyn_vars = ()
  if isinstance(dyn_vars, Variable):
    dyn_vars = (dyn_vars,)
  elif isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, '
                     f'only support dict/list/tuple of {Variable.__name__}')
  for v in dyn_vars:
    if not isinstance(v, Variable):
      raise ValueError(f'brainpy.math.for_loop only support {Variable.__name__} '
                       f'in "dyn_vars", but got {type(v)}')

  # functions
  def fun2scan(dyn_vals, x):
    for v, d in zip(dyn_vars, dyn_vals): v._value = d
    if not isinstance(x, (tuple, list)):
      x = (x,)
    results = body_fun(*x)
    return [v.value for v in dyn_vars], results

  name = get_unique_name('_brainpy_object_oriented_for_loop_')

  # functions
  init_vals = [v.value for v in dyn_vars]
  try:
    add_context(name)
    dyn_vals, out_vals = lax.scan(f=fun2scan,
                                  init=init_vals,
                                  xs=operands,
                                  reverse=reverse,
                                  unroll=unroll)
    del_context(name)
  except UnexpectedTracerError as e:
    del_context(name)
    for v, d in zip(dyn_vars, init_vals): v._value = d
    raise errors.JaxTracerError(variables=dyn_vars) from e
  except Exception as e:
    del_context(name)
    for v, d in zip(dyn_vars, init_vals): v._value = d
    raise e
  for v, d in zip(dyn_vars, dyn_vals): v._value = d
  return out_vals


def while_loop(
    body_fun: Callable,
    cond_fun: Callable,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]],
    operands: Any,
):
  """``while-loop`` control flow with :py:class:`~.Variable`.

  Note the diference between ``for_loop`` and ``while_loop``:

  1. ``while_loop`` does not support accumulating history values.
  2. The returns on the body function of ``for_loop`` represent the values to stack at one moment.
     However, the functional returns of body function in ``while_loop`` represent the operands'
     values at the next moment, meaning that the body function of ``while_loop`` defines the
     updating rule of how the operands are updated.

  >>> import brainpy.math as bm
  >>>
  >>> a = bm.Variable(bm.zeros(1))
  >>> b = bm.Variable(bm.ones(1))
  >>>
  >>> def cond(x, y):
  >>>    return x < 6.
  >>>
  >>> def body(x, y):
  >>>    a.value += x
  >>>    b.value *= y
  >>>    return x + b[0], y + 1.
  >>>
  >>> res = bm.while_loop(body, cond, dyn_vars=[a, b], operands=(1., 1.))
  >>> res
  (10.0, 4.0)

  .. versionadded:: 2.1.11

  Parameters
  ----------
  body_fun: callable
    A function which define the updating logic. It receives one argument for ``operands``, without returns.
  cond_fun: callable
    A function which define the stop condition. It receives one argument for ``operands``,
    with one boolean value return.
  dyn_vars: Variable, sequence of Variable, dict
    The dynamically changed variables.
  operands: Any
    The operands for ``body_fun`` and ``cond_fun`` functions.
  """
  # iterable variables
  if isinstance(dyn_vars, Variable):
    dyn_vars = (dyn_vars,)
  elif isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'"dyn_vars" does not support {type(dyn_vars)}, '
                     f'only support dict/list/tuple of {Variable.__name__}')
  for v in dyn_vars:
    if not isinstance(v, Variable):
      raise ValueError(f'Only support {Variable.__name__}, but got {type(v)}')
  if not isinstance(operands, (list, tuple)):
    operands = (operands, )

  def _body_fun(op):
    dyn_vals, static_vals = op
    for v, d in zip(dyn_vars, dyn_vals): v._value = d
    if not isinstance(static_vals, (tuple, list)):
      static_vals = (static_vals, )
    new_vals = body_fun(*static_vals)
    if new_vals is None:
      new_vals = tuple()
    if not isinstance(new_vals, tuple):
      new_vals = (new_vals, )
    return [v.value for v in dyn_vars], new_vals

  def _cond_fun(op):
    dyn_vals, static_vals = op
    for v, d in zip(dyn_vars, dyn_vals): v._value = d
    r = cond_fun(*static_vals)
    return r if isinstance(r, JaxArray) else r

  name = get_unique_name('_brainpy_object_oriented_while_loop_')
  dyn_init = [v.value for v in dyn_vars]
  try:
    add_context(name)
    dyn_values, out = lax.while_loop(cond_fun=_cond_fun,
                                     body_fun=_body_fun,
                                     init_val=(dyn_init, operands))
    del_context(name)
  except UnexpectedTracerError as e:
    del_context(name)
    for v, d in zip(dyn_vars, dyn_init): v._value = d
    raise errors.JaxTracerError(variables=dyn_vars) from e
  except Exception as e:
    del_context(name)
    for v, d in zip(dyn_vars, dyn_init): v._value = d
    raise e
  for v, d in zip(dyn_vars, dyn_values): v._value = d
  return out
