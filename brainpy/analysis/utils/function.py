# -*- coding: utf-8 -*-

import inspect

import brainpy.math as bm
from brainpy import errors

__all__ = [
  'f_without_jaxarray_return',
  'remove_return_shape',
  'get_args',
  'std_derivative',
  'std_func',
]


def f_without_jaxarray_return(f):
  def f2(*args, **kwargs):
    r = f(*args, **kwargs)
    return r.value if isinstance(r, bm.Array) else r

  return f2


def remove_return_shape(f):
  def f2(*args, **kwargs):
    r = f(*args, **kwargs)
    if r.shape == (1,):r = r[0]
    return r
  return f2


def get_args(f, gather_var=True):
  reduced_args = []
  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.DiffEqError(f'Don not support dict of keyword arguments: {str(par)}')
    else:
      raise errors.DiffEqError(f'Unknown argument type: {par.kind}')

  if gather_var:
    var_names = []
    for a in reduced_args:
      if a == 't': break
      var_names.append(a)
    else:
      raise ValueError('Do not find time variable "t".')
    return var_names, reduced_args
  else:
    return reduced_args


def std_derivative(original_fargs, target_vars, target_pars):
  var = original_fargs[0]
  num_vars = len(target_vars)

  def inner(f):
    def call(*dyn_vars_and_pars, **fixed_vars_and_pars):
      params = dict()
      for i, v in enumerate(target_vars):
        if (v != var) and (v in original_fargs):
          params[v] = dyn_vars_and_pars[i]
      for j, p in enumerate(target_pars):
        if p in original_fargs:
          params[p] = dyn_vars_and_pars[num_vars + j]
      for k, v in fixed_vars_and_pars.items():
        if k in original_fargs:
          params[k] = v
      return f(dyn_vars_and_pars[target_vars.index(var)], 0., **params)

    return call

  return inner


def std_func(original_fargs, target_vars, target_pars):
  num_vars = len(target_vars)

  def inner(f):
    def call(*dyn_vars_and_pars, **fixed_vars_and_pars):
      params = dict()
      for i, v in enumerate(target_vars):
        if v in original_fargs:
          params[v] = dyn_vars_and_pars[i]
      for j, p in enumerate(target_pars):
        if p in original_fargs:
          params[p] = dyn_vars_and_pars[num_vars + j]
      for k, v in fixed_vars_and_pars.items():
        if k in original_fargs:
          params[k] = v
      return f(**params)

    return call

  return inner

