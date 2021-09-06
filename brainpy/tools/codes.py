# -*- coding: utf-8 -*-

import inspect
import re
from types import LambdaType

__all__ = [
  'copy_doc',
  'code_lines_to_func',

  # tools for code string
  'get_identifiers',
  'indent',
  'deindent',
  'word_replace',

  # other tools
  'is_lambda_function',
  'get_main_code',
  'get_func_source',
  'change_func_name',
]


def copy_doc(source_f):
  def copy(target_f):
    target_f.__doc__ = source_f.__doc__
    return target_f

  return copy


def code_lines_to_func(lines, func_name, func_args, scope, remind=''):
  lines_for_compile = [f'    {line}' for line in lines]
  code_for_compile = '\n'.join(lines_for_compile)
  code = f'def {func_name}({", ".join(func_args)}):\n' + \
         f'  try:\n' + \
         f'{code_for_compile}\n' + \
         f'  except Exception as e:\n'
  code += '    exc_type, exc_obj, exc_tb = sys.exc_info()\n'
  code += '    line_no = exc_tb.tb_lineno\n'
  code += '    raise ValueError(f"Error occurred in line {line_no}: {code_for_debug} {str(e)} {remind}")'
  lines_for_debug = [f'[{i + 1:3d}] {line}' for i, line in enumerate(code.split('\n'))]
  code_for_debug = '\n'.join(lines_for_debug)
  scope['code_for_debug'] = '\n\n' + code_for_debug + '\n\n'
  scope['remind'] = '\n' + remind + '\n'
  try:
    exec(compile(code, '', 'exec'), scope)
  except Exception as e:
    raise ValueError(f'Compilation function error: \n\n{code}') from e
  func = scope[func_name]
  return code, func


######################################
# String tools
######################################


def get_identifiers(expr, include_numbers=False):
  """
  Return all the identifiers in a given string ``expr``, that is everything
  that matches a programming language variable like expression, which is
  here implemented as the regexp ``\\b[A-Za-z_][A-Za-z0-9_]*\\b``.

  Parameters
  ----------
  expr : str
      The string to analyze
  include_numbers : bool, optional
      Whether to include number literals in the output. Defaults to ``False``.

  Returns
  -------
  identifiers : set
      A set of all the identifiers (and, optionally, numbers) in `expr`.

  Examples
  --------
  >>> expr = '3-a*_b+c5+8+f(A - .3e-10, tau_2)*17'
  >>> ids = get_identifiers(expr)
  >>> print(sorted(list(ids)))
  ['A', '_b', 'a', 'c5', 'f', 'tau_2']
  >>> ids = get_identifiers(expr, include_numbers=True)
  >>> print(sorted(list(ids)))
  ['.3e-10', '17', '3', '8', 'A', '_b', 'a', 'c5', 'f', 'tau_2']
  """

  _ID_KEYWORDS = {'and', 'or', 'not', 'True', 'False'}
  identifiers = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_.]*\b', expr))
  # identifiers = set(re.findall(r'\b[A-Za-z_][.?[A-Za-z0-9_]*]*\b', expr))
  if include_numbers:
    # only the number, not a + or -
    pattern = r'(?<=[^A-Za-z_])[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|^[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    numbers = set(re.findall(pattern, expr))
  else:
    numbers = set()
  return (identifiers - _ID_KEYWORDS) | numbers


def indent(text, num_tabs=1, spaces_per_tab=4, tab=None):
  if tab is None:
    tab = ' ' * spaces_per_tab
  indent_ = tab * num_tabs
  indented_string = indent_ + text.replace('\n', '\n' + indent_)
  return indented_string


def deindent(text, num_tabs=None, spaces_per_tab=4, docstring=False):
  text = text.replace('\t', ' ' * spaces_per_tab)
  lines = text.split('\n')
  # if it's a docstring, we search for the common tabulation starting from
  # line 1, otherwise we use all lines
  if docstring:
    start = 1
  else:
    start = 0
  if docstring and len(lines) < 2:  # nothing to do
    return text
  # Find the minimum indentation level
  if num_tabs is not None:
    indent_level = num_tabs * spaces_per_tab
  else:
    line_seq = [len(line) - len(line.lstrip()) for line in lines[start:] if len(line.strip())]
    if len(line_seq) == 0:
      indent_level = 0
    else:
      indent_level = min(line_seq)
  # remove the common indentation
  lines[start:] = [line[indent_level:] for line in lines[start:]]
  return '\n'.join(lines)


def word_replace(expr, substitutions, exclude_dot=True):
  """Applies a dict of word substitutions.

  The dict ``substitutions`` consists of pairs ``(word, rep)`` where each
  word ``word`` appearing in ``expr`` is replaced by ``rep``. Here a 'word'
  means anything matching the regexp ``\\bword\\b``.

  Examples
  --------

  >>> expr = 'a*_b+c5+8+f(A)'
  >>> print(word_replace(expr, {'a':'banana', 'f':'func'}))
  banana*_b+c5+8+func(A)
  """
  for var, replace_var in substitutions.items():
    if exclude_dot:
      expr = re.sub(r'\b(?<!\.)' + var + r'\b(?!\.)', str(replace_var), expr)
    else:
      expr = re.sub(r'\b' + var + r'\b', str(replace_var), expr)
  return expr


######################################
# Other tools
######################################


def change_func_name(f, name):
  f.__name__ = name
  return f


def is_lambda_function(func):
  """Check whether the function is a ``lambda`` function. Comes from
  https://stackoverflow.com/questions/23852423/how-to-check-that-variable-is-a-lambda-function

  Parameters
  ----------
  func : callable function
      The function.

  Returns
  -------
  bool
      True of False.
  """
  return isinstance(func, LambdaType) and func.__name__ == "<lambda>"


def get_func_source(func):
  code = inspect.getsource(func)
  # remove @
  try:
    start = code.index('def ')
    code = code[start:]
  except ValueError:
    pass
  return code


def get_main_code(func, codes=None):
  """Get the main function _code string.

  For lambda function, return the

  Parameters
  ----------
  func : callable, Optional, int, float

  Returns
  -------

  """
  if func is None:
    return ''
  elif callable(func):
    if is_lambda_function(func):
      codes = (codes or get_func_source(func))
      splits = codes.split(':')
      if len(splits) != 2:
        raise ValueError(f'Can not parse function: \n{codes}')
      return f'return {splits[1]}'

    else:
      codes = (codes.split('\n') or inspect.getsourcelines(func)[0])
      idx = 0
      for line in codes:
        idx += 1
        line = line.replace(' ', '')
        if '):' in line:
          break
      else:
        code = "\n".join(codes)
        raise ValueError(f'Can not parse function: \n{code}')
      return ''.join(codes[idx:])
  else:
    raise ValueError(f'Unknown function type: {type(func)}.')
