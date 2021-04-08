# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy.integrators import constants
from brainpy.integrators import utils

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
    func_new_name = constants.NAME_PREFIX.format('sde') + func_name
    class_kw, variables, parameters, arguments = utils.get_args(f)
    return vdt, variables, parameters, arguments, func_new_name


def compile_and_assign_attrs(code_lines, code_scope, show_code,
                             variables, parameters, func_name,
                             sde_type, var_type, wiener_type, dt):
    driver_cls = backend.get_diffint_driver()
    driver = driver_cls(code_scope=code_scope, code_lines=code_lines,
                        func_name=func_name, show_code=show_code,
                        uploads=dict(variables=variables,
                                     parameters=parameters,
                                     origin_f=code_scope['f'],
                                     origin_g=code_scope['g'],
                                     sde_type=sde_type,
                                     var_type=var_type,
                                     wiener_type=wiener_type,
                                     dt=dt))
    return driver.build()
