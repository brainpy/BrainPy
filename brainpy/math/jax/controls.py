# -*- coding: utf-8 -*-


from jax import lax

from brainpy.math.jax.jaxarray import JaxArray

__all__ = [
  'easy_scan',
  'easy_loop',
  'easy_while',
  'easy_cond',
]


def easy_scan(f, dyn_vars, out_vars):
  """Make a scan function.

  Parameters
  ----------
  f
  dyn_vars
  out_vars

  Returns
  -------

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
  if isinstance(out_vars, dict):
    out_keys = list(out_vars.keys())
    out_vars = tuple(out_vars.values())
    dict_return = True
  elif isinstance(out_vars, (tuple, list)):
    out_keys = None
    out_vars = tuple(out_vars)
    dict_return = False
  else:
    raise ValueError(f'Do not support {type(out_vars)}, only support dict/list/tuple of {JaxArray.__name__}')

  # base function
  def fun2scan(dyn_values, x):
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    f(x)
    return [v.value for v in dyn_vars], [v.value for v in out_vars]

  # return function
  def call(xs=None, length=None, reverse=False, unroll=1):
    dyn_values, out_values = lax.scan(f=fun2scan, init=[v.value for v in dyn_vars],
                                      xs=xs, length=length, reverse=reverse, unroll=unroll)
    for v, d in zip(dyn_vars, dyn_values): v.value = d
    if dict_return:
      return {k: d for k, d in zip(out_keys, out_values)}
    else:
      return out_values

  return call


def easy_loop(lower, upper, body_fun, init_val):
  return lax.fori_loop(lower=lower, upper=upper, body_fun=body_fun, init_val=init_val)


def easy_while(cond_fun, body_fun, init_val):
  return lax.while_loop(cond_fun, body_fun, init_val)


def easy_cond(pred, true_fun, false_fun, operand):
  return lax.cond(pred, true_fun, false_fun, operand)
