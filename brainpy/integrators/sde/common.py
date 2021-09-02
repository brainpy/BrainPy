# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy.integrators import constants, utils

_SDE_UNKNOWN_NO = 0


def basic_info(f, g):
  vdt = 'dt'
  if f.__name__.isidentifier():
    func_name = f.__name__
  elif g.__name__.isidentifier():
    func_name = g.__name__
  else:
    global _SDE_UNKNOWN_NO
    func_name = f'unknown_sde{_SDE_UNKNOWN_NO}'
  func_new_name = constants.SDE_INT + func_name
  class_kw, variables, parameters, arguments = utils.get_args(f)
  return vdt, variables, parameters, arguments, func_new_name


def compile_and_assign_attrs(code_lines, code_scope, show_code,
                             variables, parameters, func_name,
                             intg_type, var_type, wiener_type, dt):
  code_scope_old = {key: val for key, val in code_scope.items()}

  # compile functions
  code = '\n'.join(code_lines)
  if show_code:
    print(code)
    print()
    pprint(code_scope)
    print()
  exec(compile(code, '', 'exec'), code_scope)
  new_f = code_scope[func_name]

  # assign values
  new_f.brainpy_data = dict(raw_func=None,
                            code_lines=code_lines,
                            code_scope=code_scope_old,
                            variables=variables,
                            parameters=parameters,
                            dt=dt,
                            func_name=func_name,
                            var_type=var_type)
  return new_f
