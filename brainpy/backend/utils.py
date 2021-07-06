# -*- coding: utf-8 -*-


import math
import inspect

from brainpy import backend
from brainpy import errors

__all__ = [
  'every_to_step_num',
  'attr_replace',
  'get_num_indent',
  'get_func_body_code',
  'get_args',
  'code_lines_to_func',
]


def every_to_step_num(interval):
  num_interval = round(interval / backend.get_dt())
  if math.fmod(interval * 1000, backend.get_dt() * 1000) != 0.:
    print(f'"{interval}" is not an integer multiple of the step '
          f'resolution ("{backend.get_dt()}"). BrainPy adjust it '
          f'to "{num_interval * backend.get_dt()}".')
  return num_interval


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
      raise errors.ModelDefError('Step function do not support positional parameters, e.g., *args')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.ModelDefError('Step function do not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.ModelDefError(f'Step function do not support dict of keyword arguments: {str(par)}')
    else:
      raise errors.ModelDefError(f'Unknown argument type: {par.kind}')

  # 2. check the function arguments
  class_kw = None
  if len(arguments) > 0 and arguments[0] in backend.CLASS_KEYWORDS:
    class_kw = arguments[0]
    arguments = arguments[1:]
  for a in arguments:
    if a in backend.CLASS_KEYWORDS:
      raise errors.DiffEqError(f'Class keywords "{a}" must be defined '
                               f'as the first argument.')
  return class_kw, arguments


def code_lines_to_func(lines, func_name, func_args, scope, remind=''):
  lines_for_compile = [f'    {line}' for line in lines]
  code_for_compile = '\n'.join(lines_for_compile)
  # code = f'def {func_name}({", ".join(func_args)}):\n' + \
  #        f'  try:\n' + \
  #        f'{code_for_compile}\n' + \
  #        f'  except Exception as e:\n'
  code = f'def {func_name}({", ".join(func_args)}):\n' + \
         f'  try:\n' + \
         f'{code_for_compile}\n' + \
         f'  except Exception as e:\n'
  lines_for_debug = [f'[{i+1:3d}] {line}' for i, line in enumerate(code.split('\n'))]
  code_for_debug = '\n'.join(lines_for_debug)
  code += f'    exc_type, exc_obj, exc_tb = sys.exc_info()\n' \
         f'    line_no = exc_tb.tb_lineno\n' \
         f'    raise ValueError("""Error occurred in line %d: \n\n{code_for_debug}\n\n' \
          f'    %s\n{remind}\n""" % (line_no, str(e)))'
  try:
    exec(compile(code, '', 'exec'), scope)
  except Exception as e:
    raise ValueError(f'Compilation function error: \n\n{code}') from e
  func = scope[func_name]
  return code, func

