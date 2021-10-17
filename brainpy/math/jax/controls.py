# -*- coding: utf-8 -*-


from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten

from brainpy.math.jax.jaxarray import JaxArray
from brainpy.math.jax.ops import arange

__all__ = [
  'easy_scan',
  'easy_loop',
  'easy_while',
  'easy_cond',
]


def _get_scan_info(f, dyn_vars, out_vars=None, has_return=False):
  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'Do not support {type(dyn_vars)}, only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'brainpy.math.jax.loops only support {JaxArray.__name__}, but got {type(v)}')

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
    raise ValueError(f'Do not support {type(out_vars)}, only support dict/list/tuple of {JaxArray.__name__}')

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

  return fun2scan, tree


def easy_scan(f, dyn_vars, out_vars=None, has_return=False):
  """Make a scan function.

  Parameters
  ----------
  f : callable, function
  dyn_vars : dict of JaxArray, sequence of JaxArray
    The dynamically changed variables, while iterate between trials.
  out_vars : Optional, JaxArray, dict of JaxArray, sequence of JaxArray
    The variables to output their values.
  has_return : bool
    The function has the return values.

  Returns
  -------
  scan_func : callable, function
    The function for scan iteration.
  """

  fun2scan, tree = _get_scan_info(f=f,
                                  dyn_vars=dyn_vars,
                                  out_vars=out_vars,
                                  has_return=has_return)

  # functions
  if has_return:
    def call(xs=None, length=None):
      dyn_values, (out_values, results) = lax.scan(f=fun2scan,
                                                   init=[v.value for v in dyn_vars],
                                                   xs=xs,
                                                   length=length)
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return (tree_unflatten(tree, out_values), results)

  else:
    def call(xs=None, length=None):
      dyn_values, out_values = lax.scan(f=fun2scan,
                                        init=[v.value for v in dyn_vars],
                                        xs=xs,
                                        length=length)
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return tree_unflatten(tree, out_values)

  return call


def easy_loop(f, dyn_vars, out_vars, has_return=False):
  fun2scan, tree = _get_scan_info(f=f,
                                  dyn_vars=dyn_vars,
                                  out_vars=out_vars,
                                  has_return=has_return)

  # functions
  if has_return:
    def call(lower, upper):
      dyn_values, (out_values, results) = lax.scan(f=fun2scan,
                                                   init=[v.value for v in dyn_vars],
                                                   xs=arange(lower, upper))
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return tree_unflatten(tree, out_values), results

  else:
    def call(lower, upper):
      dyn_values, out_values = lax.scan(f=fun2scan,
                                        init=[v.value for v in dyn_vars],
                                        xs=arange(lower, upper))
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return tree_unflatten(tree, out_values)

  return call


def easy_while(cond_fun, body_fun, dyn_vars):
  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'Do not support {type(dyn_vars)}, only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'brainpy.math.jax.loops only support {JaxArray.__name__}, but got {type(v)}')

  def _body_fun(init_val):
    dyn_vals, old_static_vals = init_val
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    new_static_vals = body_fun(old_static_vals)
    dyn_vals = [v.value for v in dyn_vars]
    return (dyn_vals, new_static_vals)

  def _cond_fun(init_val):
    dyn_vals, static_vals = init_val
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    return cond_fun(static_vals)

  def call(init_val):
    init_val = ([v.value for v in dyn_vars], init_val)
    dyn_values, new_static_val = lax.while_loop(cond_fun=_cond_fun,
                                                body_fun=_body_fun,
                                                init_val=init_val)
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    return new_static_val

  return call


def easy_cond(true_fun, false_fun, dyn_vars):
  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'Do not support {type(dyn_vars)}, only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'brainpy.math.jax.loops only support {JaxArray.__name__}, but got {type(v)}')

  def _true_fun(dyn_vals):
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    true_fun()
    dyn_vals = [v.value for v in dyn_vars]
    return dyn_vals

  def _false_fun(dyn_vals):
    for v, d in zip(dyn_vars, dyn_vals): v.value = d
    false_fun()
    dyn_vals = [v.value for v in dyn_vars]
    return dyn_vals

  def call(pred):
    dyn_values = lax.cond(pred=pred,
                          true_fun=_true_fun,
                          false_fun=_false_fun,
                          operand=[v.value for v in dyn_vars])
    for v, d in zip(dyn_vars, dyn_values): v.value = d

  return call
