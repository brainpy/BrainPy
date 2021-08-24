# -*- coding: utf-8 -*-

from brainpy.integrators import constants, utils
from brainpy.integrators.driver import get_driver

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
  driver = get_driver()(code_scope=code_scope,
                        code_lines=code_lines,
                        func_name=func_name,
                        show_code=show_code)
  call = driver.build()
  return call
