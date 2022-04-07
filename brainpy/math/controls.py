# -*- coding: utf-8 -*-


from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten
try:
  from jax.errors import UnexpectedTracerError
except ImportError:
  from jax.core import UnexpectedTracerError

from brainpy import errors
from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import as_device_array

__all__ = [
  'make_loop',
  'make_while',
  'make_cond',
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
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      results = f(x)
      dyn_values = [v.value for v in dyn_vars]
      out_values = [v.value for v in out_vars]
      return dyn_values, (out_values, results)

  else:
    def fun2scan(dyn_values, x):
      for v, d in zip(dyn_vars, dyn_values): v.value = d
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
  >>> a = bm.zeros(1)
  >>> def f(x): a.value += 1.
  >>> loop = bm.make_loop(f, dyn_vars=[a], out_vars=a)
  >>> loop(length=10)
  JaxArray(DeviceArray([[ 1.],
                        [ 2.],
                        [ 3.],
                        [ 4.],
                        [ 5.],
                        [ 6.],
                        [ 7.],
                        [ 8.],
                        [ 9.],
                        [10.]], dtype=float32))
  >>> b = bm.zeros(1)
  >>> def f(x):
  >>>   b.value += 1
  >>>   return b + 1
  >>> loop = bm.make_loop(f, dyn_vars=[b], out_vars=b, has_return=True)
  >>> hist_b, hist_b_plus = loop(length=10)
  >>> hist_b
  JaxArray(DeviceArray([[ 1.],
                        [ 2.],
                        [ 3.],
                        [ 4.],
                        [ 5.],
                        [ 6.],
                        [ 7.],
                        [ 8.],
                        [ 9.],
                        [10.]], dtype=float32))
  >>> hist_b_plus
  JaxArray(DeviceArray([[ 2.],
                        [ 3.],
                        [ 4.],
                        [ 5.],
                        [ 6.],
                        [ 7.],
                        [ 8.],
                        [ 9.],
                        [10.],
                        [11.]], dtype=float32))

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

  # functions
  if has_return:
    def call(xs=None, length=None):
      init_values = [v.value for v in dyn_vars]
      try:
        dyn_values, (out_values, results) = lax.scan(
          f=fun2scan, init=init_values, xs=xs, length=length)
      except UnexpectedTracerError as e:
        for v, d in zip(dyn_vars, init_values): v.value = d
        raise errors.JaxTracerError(variables=dyn_vars) from e
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return tree_unflatten(tree, out_values), results

  else:
    def call(xs):
      init_values = [v.value for v in dyn_vars]
      try:
        dyn_values, out_values = lax.scan(f=fun2scan, init=init_values, xs=xs)
      except UnexpectedTracerError as e:
        for v, d in zip(dyn_vars, init_values): v.value = d
        raise errors.JaxTracerError(variables=dyn_vars) from e
      for v, d in zip(dyn_vars, dyn_values): v.value = d
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
  JaxArray(DeviceArray([10.], dtype=float32))

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
    raise ValueError(
      f'"dyn_vars" does not support {type(dyn_vars)}, '
      f'only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'brainpy.math.jax.loops only support {JaxArray.__name__}, but got {type(v)}')

  def _body_fun(op):
    dyn_values, static_values = op
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    body_fun(static_values)
    return [v.value for v in dyn_vars], static_values

  def _cond_fun(op):
    dyn_values, static_values = op
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    return as_device_array(cond_fun(static_values))

  def call(x=None):
    dyn_init = [v.value for v in dyn_vars]
    try:
      dyn_values, _ = lax.while_loop(cond_fun=_cond_fun,
                                     body_fun=_body_fun,
                                     init_val=(dyn_init, x))
    except UnexpectedTracerError as e:
      for v, d in zip(dyn_vars, dyn_init): v.value = d
      raise errors.JaxTracerError(variables=dyn_vars) from e
    for v, d in zip(dyn_vars, dyn_values): v.value = d

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
  (JaxArray(DeviceArray([1., 1.], dtype=float32)),
   JaxArray(DeviceArray([1., 1.], dtype=float32)))
  >>> cond(False)
  >>> a, b
  (JaxArray(DeviceArray([1., 1.], dtype=float32)),
   JaxArray(DeviceArray([0., 0.], dtype=float32)))

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
    dyn_vars = (dyn_vars, )
  elif isinstance(dyn_vars, dict):
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
        f'brainpy.math.jax.loops only support '
        f'{JaxArray.__name__}, but got {type(v)}')

  def _true_fun(op):
    dyn_vals, static_vals = op
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    res = true_fun(static_vals)
    dyn_vals = [v.value for v in dyn_vars]
    return dyn_vals, res

  def _false_fun(op):
    dyn_vals, static_vals = op
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    res = false_fun(static_vals)
    dyn_vals = [v.value for v in dyn_vars]
    return dyn_vals, res

  def call(pred, x=None):
    old_values = [v.value for v in dyn_vars]
    try:
      dyn_values, res = lax.cond(pred=pred,
                                 true_fun=_true_fun,
                                 false_fun=_false_fun,
                                 operand=(old_values, x))
    except UnexpectedTracerError as e:
      for v, d in zip(dyn_vars, old_values): v.value = d
      raise errors.JaxTracerError(variables=dyn_vars) from e
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    return res

  return call
