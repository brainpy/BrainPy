# -*- coding: utf-8 -*-


from jax import lax
from brainpy.base import ArrayCollector

__all__ = [
  'easy_scan',
  'fori_loop',
  'while_loop',
  'cond',
]


def easy_scan(f, dyn_vars, out_vars):
  """

  Parameters
  ----------
  f
  dyn_vars
  out_vars

  Returns
  -------

  """
  # outputs
  if isinstance(out_vars, dict):
    collector_outs = ArrayCollector(out_vars).unique()
    dict_return = True
  elif isinstance(out_vars, (tuple, list)):
    collector_outs = ArrayCollector({f'_o{i}': v for i, v in enumerate(out_vars)}).unique()
    dict_return = False
  else:
    raise ValueError

  # inits
  if isinstance(dyn_vars, dict):
    dyn_vars = ArrayCollector(dyn_vars).unique()
  elif isinstance(dyn_vars, (tuple, list)):
    dyn_vars = ArrayCollector({f'_v{i}': v for i, v in enumerate(dyn_vars)}).unique()
  else:
    raise ValueError

  def fun2scan(init_dicts, x):
    dyn_vars.assign(init_dicts)
    f(x)
    return dyn_vars.dict(), collector_outs.dict()

  if dict_return:
    def call(xs=None, length=None, reverse=False, unroll=1):
      init_dicts, all_outs = lax.scan(f=fun2scan, init=dyn_vars.dict(), xs=xs, length=length, reverse=reverse, unroll=unroll)
      dyn_vars.assign(init_dicts)
      return all_outs
  else:
    def call(xs=None, length=None, reverse=False, unroll=1):
      init_dicts, all_outs = lax.scan(f=fun2scan, init=dyn_vars.dict(), xs=xs, length=length, reverse=reverse, unroll=unroll)
      dyn_vars.assign(init_dicts)
      return tuple(all_outs.values())
  return call


def fori_loop(lower, upper, body_fun, init_val):
  return lax.fori_loop(lower=lower, upper=upper, body_fun=body_fun, init_val=init_val)


def while_loop(cond_fun, body_fun, init_val):
  return lax.while_loop(cond_fun, body_fun, init_val)


def cond(pred, true_fun, false_fun, operand):
  return lax.cond(pred, true_fun, false_fun, operand)
