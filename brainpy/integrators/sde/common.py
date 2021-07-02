# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy.integrators import constants
from brainpy.integrators import utils
from brainpy.integrators.integrators import SDEIntegrator

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
  func_new_name = constants.SDE_PREFIX + func_name
  class_kw, variables, parameters, arguments = utils.get_args(f)
  return vdt, variables, parameters, arguments, func_new_name


def compile_and_assign_attrs(code_lines, code_scope, show_code,
                             variables, parameters, func_name,
                             intg_type, var_type, wiener_type, dt):
  driver = backend.get_diffint_driver()(code_scope=code_scope,
                                        code_lines=code_lines,
                                        func_name=func_name,
                                        show_code=show_code)
  call = driver.build()

  integrator = SDEIntegrator(f=code_scope['f'],
                             g=code_scope['g'],
                             intg_type=intg_type,
                             wiener_type=wiener_type,
                             call=call,
                             variables=variables,
                             parameters=parameters,
                             var_type=var_type,
                             dt=dt,
                             code=driver.code)
  return integrator
