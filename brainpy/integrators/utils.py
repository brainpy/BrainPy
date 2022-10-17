# -*- coding: utf-8 -*-


import inspect
from pprint import pprint

import jax.numpy as jnp
import brainpy.math as bm
from brainpy.errors import UnsupportedError

from brainpy import errors

__all__ = [
  'get_args',
  'check_kws',
  'compile_code',
  'check_inits',
  'format_args',
]


def check_kws(parameters, keywords):
  for key, meaning in keywords.items():
    if key in parameters:
      raise errors.CodeError(f'"{key}" is a keyword for '
                             f'numerical solvers in BrainPy, denoting '
                             f'"{meaning}". Please change another name.')


def get_args(f):
  """Get the function arguments.

  >>> def f1(a, b, t, *args, c=1): pass
  >>> get_args(f1)
  (['a', 'b'], ['t', '*args', 'c'], ['a', 'b', 't', '*args', 'c=1'])

  >>> def f2(a, b, *args, c=1, **kwargs): pass
  >>> get_args(f2)
  ValueError: Do not support dict of keyword arguments: **kwargs

  >>> def f3(a, b, t, c=1, d=2): pass
  >>> get_args(f4)
  (['a', 'b'], ['t', 'c', 'd'], ['a', 'b', 't', 'c=1', 'd=2'])

  >>> def f4(a, b, t, *args): pass
  >>> get_args(f4)
  (['a', 'b'], ['t', '*args'], ['a', 'b', 't', '*args'])

  >>> scope = {}
  >>> exec(compile('def f5(a, b, t, *args): pass', '', 'exec'), scope)
  >>> get_args(scope['f5'])
  (['a', 'b'], ['t', '*args'], ['a', 'b', 't', '*args'])

  Parameters
  ----------
  f : callable
      The function.

  Returns
  -------
  args : tuple
      The variable names, the other arguments, and the original args.
  """

  # get the function arguments
  reduced_args = []
  args = []

  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      reduced_args.append(par.name)

    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      reduced_args.append(f'*{par.name}')

    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      raise errors.DiffEqError(f'In BrainPy, numerical integrators do not support KEYWORD_ONLY '
                               f'parameters, e.g., * (error in {f}).')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.DiffEqError(f'In BrainPy, numerical integrators do not support POSITIONAL_ONLY '
                               f'parameters, e.g., / (error in {f}).')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:  # TODO
      raise errors.DiffEqError(f'In BrainPy, numerical integrators do not support VAR_KEYWORD '
                               f'arguments: {str(par)} (error in {f}).')
    else:
      raise errors.DiffEqError(f'Unknown argument type: {par.kind} (error in {f}).')

    args.append(str(par))

  #  variable names
  vars = []
  for a in reduced_args:
    if a == 't':
      break
    vars.append(a)
  else:
    raise ValueError('Do not find time variable "t".')
  pars = reduced_args[len(vars):]
  return vars, pars, args


def compile_code(code_lines, code_scope, func_name, show_code=False):
  code = '\n'.join(code_lines)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  new_f = code_scope[func_name]
  return new_f


def check_inits(inits, variables):
  if isinstance(inits, (tuple, list, bm.JaxArray, jnp.ndarray)):
    assert len(inits) == len(variables), (f'Then number of variables is {len(variables)}, '
                                          f'however we only got {len(inits)} initial values.')
    inits = {v: inits[i] for i, v in enumerate(variables)}
  elif isinstance(inits, dict):
    assert len(inits) == len(variables), (f'Then number of variables is {len(variables)}, '
                                          f'however we only got {len(inits)} initial values.')
  else:
    raise UnsupportedError('Only supports dict/sequence of data for initial values. '
                           f'But we got {type(inits)}: {inits}')
  for key in list(inits.keys()):
    if key not in variables:
      raise ValueError(f'"{key}" is not defined in variables: {variables}')
    val = inits[key]
    if isinstance(val, (float, int)):
      inits[key] = bm.asarray([val], dtype=bm.dftype())
  return inits


def format_args(args, kwargs, arguments):
  all_args = dict()
  for i, arg in enumerate(args):
    all_args[arguments[i]] = arg
  for key, arg in kwargs.items():
    if key in all_args:
      raise ValueError(f'{key} has been provided in *args, '
                       f'but we detect it again in **kwargs.')
    all_args[key] = arg
  return all_args
