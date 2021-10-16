# -*- coding: utf-8 -*-


from jax import lax
from jax.tree_util import tree_flatten, tree_unflatten

from brainpy.math.jax.jaxarray import JaxArray

__all__ = [
  'easy_scan',
  'easy_loop',
  'easy_while',
  'easy_cond',
]


def easy_scan(f, dyn_vars, out_vars=None, has_return=False):
  """Make a scan function.

  Parameters
  ----------
  f : callable, function
  dyn_vars : dict of JaxArray, sequence of JaxArray
  out_vars : Optional, JaxArray, dict of JaxArray, sequence of JaxArray
  has_return : bool

  Returns
  -------
  scan_func : callable, function
    The function for scan iteration.
  """

  # iterable variables
  if isinstance(dyn_vars, dict):
    dyn_vars = tuple(dyn_vars.values())
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = tuple(dyn_vars)
  else:
    raise ValueError(f'Do not support {type(dyn_vars)}, only support dict/list/tuple of {JaxArray.__name__}')
  for v in dyn_vars:
    if not isinstance(v, JaxArray):
      raise ValueError(f'easy_scan only support {JaxArray.__name__}, but got {type(v)}')

  # outputs
  if out_vars is None:
    out_vars = ()
    _, tree = tree_flatten(out_vars)
  elif isinstance(out_vars, JaxArray):
    _, tree = tree_flatten(out_vars)
    out_vars = (out_vars, )
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

    def call(xs=None, length=None, reverse=False, unroll=1):
      dyn_values, (out_values, results) = lax.scan(
        f=fun2scan, init=[v.value for v in dyn_vars],
        xs=xs, length=length, reverse=reverse, unroll=unroll)
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return (tree_unflatten(tree, out_values), results)

  else:
    def fun2scan(dyn_values, x):
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      f(x)
      dyn_values = [v.value for v in dyn_vars]
      out_values = [v.value for v in out_vars]
      return dyn_values, out_values

    def call(xs=None, length=None, reverse=False, unroll=1):
      dyn_values, out_values = lax.scan(
        f=fun2scan, init=[v.value for v in dyn_vars],
        xs=xs, length=length, reverse=reverse, unroll=unroll)
      for v, d in zip(dyn_vars, dyn_values): v.value = d
      return tree_unflatten(tree, out_values)

  return call


def easy_loop(lower, upper, body_fun, init_val):
  return lax.fori_loop(lower=lower, upper=upper, body_fun=body_fun, init_val=init_val)


def easy_while(cond_fun, body_fun, init_val):
  return lax.while_loop(cond_fun, body_fun, init_val)


def easy_cond(pred, true_fun, false_fun, operand):
  return lax.cond(pred, true_fun, false_fun, operand)
