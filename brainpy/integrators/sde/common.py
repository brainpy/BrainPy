# -*- coding: utf-8 -*-

from pprint import pprint

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
    func_new_name = constants.NAME_PREFIX + func_name
    class_kw, variables, parameters, arguments = utils.get_args(f)
    return vdt, variables, parameters, arguments, func_new_name


def return_compile_and_assign_attrs(code_lines, code_scope, show_code,
                                    variables, parameters, func_name,
                                    sde_type, var_type, wiener_type, dt):
    # returns
    new_vars = [f'{var}_new' for var in variables]
    code_lines.append(f'  return {", ".join(new_vars)}')

    # compile
    code = '\n'.join(code_lines)
    if show_code:
        print(code)
        print()
        pprint(code_scope)
        print()
    utils.numba_func(code_scope, ['f', 'g'])
    exec(compile(code, '', 'exec'), code_scope)

    # attribute assignment
    new_f = code_scope[func_name]
    new_f.variables = variables
    new_f.parameters = parameters
    new_f.origin_f = code_scope['f']
    new_f.origin_g = code_scope['g']
    new_f.sde_type = sde_type
    new_f.var_type = var_type
    new_f.wiener_type = wiener_type
    new_f.dt = dt
    utils.numba_func(code_scope, func_name)
    return code_scope[func_name]
