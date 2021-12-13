# -*- coding: utf-8 -*-


import numpy as np
import inspect

from brainpy import errors
from brainpy.math import profile

jarray = JaxArray= None


__all__ = [
  'numpy_array',
]


def numpy_array(array):
  global jarray
  if jarray is None:
    try:
      from jax.numpy import ndarray as jarray
    except (ModuleNotFoundError, ImportError):
      pass
  global JaxArray
  if JaxArray is None:
    try:
      from brainpy.math.jax.jaxarray import JaxArray
    except (ModuleNotFoundError, ImportError):
      pass

  if isinstance(array, np.ndarray):
    array = array
  elif (JaxArray is not None) and isinstance(array, JaxArray):
    array = array.numpy()
  elif (jarray is not None) and isinstance(array, jarray):
    array = np.asarray(array)
  else:
    raise ValueError
  return array


def attr_replace(attr):
  return attr.replace('.', '_')


def get_num_indent(code_string, spaces_per_tab=4):
  """Get the indent of a patch of source code.

  Parameters
  ----------
  code_string : str
      The code string.
  spaces_per_tab : int
      The spaces per tab.

  Returns
  -------
  num_indent : int
      The number of the indent.
  """
  lines = code_string.split('\n')
  min_indent = 1000
  for line in lines:
    if line.strip() == '':
      continue
    line = line.replace('\t', ' ' * spaces_per_tab)
    num_indent = len(line) - len(line.lstrip())
    if num_indent < min_indent:
      min_indent = num_indent
  return min_indent


def get_func_body_code(code_string, lambda_func=False):
  """Get the main body code of a function.

  Parameters
  ----------
  code_string : str
      The code string of the function.
  lambda_func : bool
      Whether the code comes from a lambda function.

  Returns
  -------
  code_body : str
      The code body.
  """
  if lambda_func:
    splits = code_string.split(':')
    if len(splits) != 2:
      raise ValueError(f'Can not parse function: \n{code_string}')
    main_code = f'return {":".join(splits[1:])}'
  else:
    func_codes = code_string.split('\n')
    idx = 0
    for i, line in enumerate(func_codes):
      idx += 1
      line = line.replace(' ', '')
      if '):' in line:
        break
    else:
      raise ValueError(f'Can not parse function: \n{code_string}')
    main_code = '\n'.join(func_codes[idx:])
  return main_code


def get_args(f):
  """Get the function arguments.

  Parameters
  ----------
  f : callable
      The function.

  Returns
  -------
  args : tuple
      The variable names, the other arguments, and the original args.
  """

  # 1. get the function arguments
  parameters = inspect.signature(f).parameters

  arguments = []
  for name, par in parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      arguments.append(par.name)

    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      arguments.append(par.name)

    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      raise errors.BrainPyError('Step function do not support positional parameters, e.g., *args')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.BrainPyError('Step function do not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.BrainPyError(f'Step function do not support dict of keyword arguments: {str(par)}')
    else:
      raise errors.BrainPyError(f'Unknown argument type: {par.kind}')

  # 2. check the function arguments
  class_kw = None
  if len(arguments) > 0 and arguments[0] in profile.CLASS_KEYWORDS:
    class_kw = arguments[0]
    arguments = arguments[1:]
  for a in arguments:
    if a in profile.CLASS_KEYWORDS:
      raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                               f'as the first argument.')
  return class_kw, arguments



