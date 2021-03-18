# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy import profile
from brainpy.integrators import constants
from . import common

__all__ = [
    'srk1_strong',
]


def _vector_wiener_terms(code_lines, sde_type, vdt, shape_D, shape_m):
    if sde_type == constants.ITO_SDE:
        I2 = f'0.5*(_term3 - {vdt} * backend.eye({shape_m})) + _a*0.5*{vdt}/math.pi'
    elif sde_type == constants.STRA_SDE:
        I2 = f'0.5*_term3 + _a*0.5*dt/math.pi'
    else:
        raise ValueError(f'Unknown SDE type: {sde_type}. We only supports {constants.SUPPORTED_SDE_TYPE}.')

    if shape_D:
        shape_D = shape_D + '+'

    noise_string = f'''
  # Noise Terms #
  # ----------- #
    
  # single Ito integrals
  _I1 = backend.normal(0., {vdt}_sqrt, {shape_D}({shape_m},))
  # double Ito integrals
  _h = (2.0 / {vdt}) ** 0.5)
  _a = backend.zeros(shape={shape_D}({shape_m}, {shape_m}))
  for _k in range(1, num_iter + 1):
    _x = backend.normal(loc=0., scale=1., size={shape_D}({shape_m}, 1))
    _y = backend.normal(loc=0., scale=1., size={shape_D}(1, {shape_m})) + _h * _I1
    _term1 = backend.matmul(_x, _y)
    _term2 = backend.matmul(backend.reshape(_y, {shape_D}({shape_m}, 1)), 
                            backend.reshape(_x, {shape_D}(1, {shape_m})))
    _a += (_term1 - _term2) / _k
  _I1_rs = backend.reshape(_I1, {shape_D}({shape_m}, 1))
  _term3 = backend.matmul(_I1_rs, backend.reshape(_I1, {shape_D}(1, {shape_m})))
  _I2 = {I2}
  '''
    noise_lines = noise_string.split('\n')
    code_lines.extend(noise_lines)


# ----------
# Wrapper
# ----------


def _srk2_pop_var_vector_wiener(sde_type, code_lines, variables, parameters, vdt):
    # shape information
    # -----
    all_f = [f'f_{var}' for var in variables]
    all_g = [f'g_{var}' for var in variables]
    noise_string = f'''
  {", ".join(all_f)} = f({", ".join(variables + parameters)})  # shape = (..)
  {", ".join(all_g)} = g({", ".join(variables + parameters)})  # shape = (.., m)
  noise_shape = backend.shape(g_x1)
  _D = noise_shape[:-1]
  _m = noise_shape[-1]
  '''
    code_lines.extend(noise_string.split("\n"))

    # noise terms
    _vector_wiener_terms(code_lines, sde_type, vdt, shape_D='_D', shape_m='_m')

    # numerical integration
    # step 1
    # ---
    # g_x1_rs = backend.reshape(g_x1, _D + (1, _m))
    # g_x2_rs = backend.reshape(g_x2, _D + (1, _m))
    for var in variables:
        code_lines.append(f"  g_{var}_rs = backend.reshape(g_{var}, _D+(1, _m))")
    # step 2
    # ---
    # g_H1_x1 = backend.reshape(backend.matmul(g_x1_rs, _I2) / dt_sqrt, _D + (_m,))
    # g_H1_x2 = backend.reshape(backend.matmul(g_x2_rs, _I2) / dt_sqrt, _D + (_m,))
    for var in variables:
        code_lines.append(f'  g_H1_{var} = backend.reshape(backend.matmul(g_{var}_rs, _I2) / {vdt}_sqrt, _D + (_m,))')
    # step 3
    # ---
    # x1_rs = backend.reshape(x1, _D + (1,))
    # x2_rs = backend.reshape(x2, _D + (1,))
    for var in variables:
        code_lines.append(f'  {var}_rs = backend.reshape({var}, _D + (1,))')
    # step 4
    # ---
    # H2_x1 = x1_rs + g_H1_x1
    # H3_x1 = x1_rs - g_H1_x1
    for var in variables:
        code_lines.append(f'  H2_{var} = {var}_rs + g_H1_{var}')
        code_lines.append(f'  H3_{var} = {var}_rs - g_H1_{var}')
    code_lines.append('  ')
    # step 5
    # ---
    # _g_x1 = backend.matmul(g_x1_rs, _I1_rs)
    for var in variables:
        code_lines.append(f'  _g_{var} = backend.matmul(g_{var}_rs, _I1_rs)')
    # step 6
    # ----
    # x1_new = x1 + f_x1 + _g_x1[..., 0, 0]
    for var in variables:
        code_lines.append(f'  {var}_new = {var} + f_{var} + _g_{var}[..., 0, 0]')
    # for _k in range(_m):
    code_lines.append('for _k in range(_m):')
    #   g_x1_H2, g_x2_H2 = g(H2_x1[..., _k], H2_x2[..., _k], t, *args)
    all_H2 = [f'H2_{var}[..., _k]' for var in variables]
    all_g_H2 = [f'g_{var}_H2' for var in variables]
    code_lines.append(f'    {", ".join(all_g_H2)} = g({", ".join(all_H2 + parameters)})')
    #   g_x1_H3, g_x2_H3 = g(H3_x1[..., _k], H3_x2[..., _k], t, *args)
    all_H3 = [f'H3_{var}[..., _k]' for var in variables]
    all_g_H3 = [f'g_{var}_H3' for var in variables]
    code_lines.append(f'    {", ".join(all_g_H3)} = g({", ".join(all_H3 + parameters)})')
    #   x1_new += 0.5 * dt_sqrt * (g_x1_H2[..., _k] - g_x1_H3[..., _k])
    #   x2_new += 0.5 * dt_sqrt * (g_x2_H2[..., _k] - g_x2_H3[..., _k])
    for var in variables:
        code_lines.append(f'    {var}_new += 0.5 * {vdt}_sqrt * (g_{var}_H2[..., _k] - g_{var}_H3[..., _k])')


def _srk2_pop_or_scalar_var_scalar_wiener(sde_type, code_lines, variables, parameters, vdt):
    if sde_type == constants.ITO_SDE:
        I2 = f'0.5 * (_I1 * _I1 - {vdt})'
    elif sde_type == constants.STRA_SDE:
        I2 = f'0.5 * _I1 * _I1'
    else:
        raise ValueError(f'Unknown SDE type: {sde_type}. We only supports {constants.SUPPORTED_SDE_TYPE}.')

    # shape info
    # -----
    all_f = [f'f_{var}' for var in variables]
    all_g = [f'g_{var}' for var in variables]

    code_string = f'''
  {", ".join(all_f)} = f({", ".join(variables + parameters)})  # shape = (..)
  {", ".join(all_g)} = g({", ".join(variables + parameters)})  # shape = (..)

  # single Ito integrals
  _I1 = backend.normal(0., {vdt}_sqrt, backend.shape({variables[0]}))  # shape = (..)
  # double Ito integrals
  _I2 = {I2}  # shape = (..)
  '''
    code_splits = code_string.split('\n')
    code_lines.extend(code_splits)

    # numerical integration
    # -----
    # H1
    for var in variables:
        code_lines.append(f'  g_H1_{var} = g_{var} * _I2 / {vdt}_sqrt  # shape (.., )')
    # H2
    all_H2 = [f'H2_{var}' for var in variables]
    for var in variables:
        code_lines.append(f'  H2_{var} = {var} + g_H1_{var}  # shape (.., )')
    all_g_H2 = [f'g_{var}_H2' for var in variables]
    code_lines.append(f'  {", ".join(all_g_H2)} = g({", ".join(all_H2 + parameters)})')
    code_lines.append(f'  ')
    # H3
    all_H3 = [f'H3_{var}' for var in variables]
    for var in variables:
        code_lines.append(f'  H3_{var} = {var} - g_H1_{var}  # shape (.., )')
    all_g_H3 = [f'g_{var}_H3' for var in variables]
    code_lines.append(f'  {", ".join(all_g_H3)} = g({", ".join(all_H3 + parameters)})')
    code_lines.append(f'  ')
    # final results
    for var in variables:
        code_lines.append(f'  {var}_new = {var} + f_{var} + g_{var} * _I1 '
                          f'+ 0.5 * {vdt}_sqrt * (g_{var}_H2 - g_{var}_H3)')


def _srk1_scalar_var_with_vector_wiener(sde_type, code_lines, variables, parameters, vdt):
    # shape information
    all_f = [f'f_{var}' for var in variables]
    all_g = [f'g_{var}' for var in variables]
    code1 = f'''
  # shape info #
  # ---------- #

  {", ".join(all_f)} = f({", ".join(variables + parameters)})  # shape = ()
  {", ".join(all_g)} = g({", ".join(variables + parameters)})  # shape = (m)
  noise_shape = backend.shape(g_x1)
  _m = noise_shape[0]
  '''
    code_lines.extend(code1.split('\n'))

    # noise term
    _vector_wiener_terms(code_lines, sde_type, vdt, shape_D='', shape_m='_m')

    # numerical integration

    # p1
    # ---
    # g_x1_rs = backend.reshape(g_x1, (1, _m))
    # g_x2_rs = backend.reshape(g_x2, (1, _m))
    for var in variables:
        code_lines.append(f'  g_{var}_rs = backend.reshape(g_{var}, (1, _m))')

    # p2
    # ---
    # g_H1_x1 = backend.matmul(g_x1_rs, _I2) / dt_sqrt  # shape (1, m)
    # g_H1_x2 = backend.matmul(g_x2_rs, _I2) / dt_sqrt  # shape (1, m)
    for var in variables:
        code_lines.append(f'  g_H1_{var} = backend.matmul(g_{var}_rs, _I2) / {vdt}_sqrt  # shape (1, m)')

    # p3
    # ---
    # H2_x1 = x1 + g_H1_x1[0]  # shape (m)
    # H3_x1 = x1 - g_H1_x1[0]  # shape (m)
    for var in variables:
        code_lines.append(f'  H2_{var} = {var} + g_H1_{var}[0]  # shape (m)')
    code_lines.append('  ')

    # p4
    # ---
    # g1_x1 = backend.matmul(g_x1_rs, _I1_rs)  # shape (1, 1)
    # x1_new = x1 + f_x1 + g1_x1[0, 0]  # shape ()
    for var in variables:
        code_lines.append(f'  g1_{var} = backend.matmul(g_{var}_rs, _I1_rs)  # shape (1, 1)')
        code_lines.append(f'  {var}_new = {var} + f_{var} + g1_{var}[0, 0]  # shape ()')

    # p5
    # ---
    # for _k in range(_m):
    #    g_x1_H2, g_x2_H2 = g(H2_x1[_k], H2_x2[_k], t, *args)
    #    g_x1_H3, g_x2_H3 = g(H3_x1[_k], H3_x2[_k], t, *args)
    #    x1_new += 0.5 * dt_sqrt * (g_x1_H2[_k] - g_x1_H3[_k])
    #    x2_new += 0.5 * dt_sqrt * (g_x2_H2[_k] - g_x2_H3[_k])
    code_lines.append('  for _k in range(_m):')
    all_h2_k = [f'H2_{var}[_k]' for var in variables]
    all_g_h2 = [f'g_{var}_H2' for var in variables]
    code_lines.append(f'    {", ".join(all_g_h2)} = g({", ".join(all_h2_k + parameters)})')
    all_h3_k = [f'H3_{var}[_k]' for var in variables]
    all_g_h3 = [f'g_{var}_H3' for var in variables]
    code_lines.append(f'    {", ".join(all_g_h3)} = g({", ".join(all_h3_k + parameters)})')
    for var in variables:
        code_lines.append(f'    {var}_new += 0.5 * {vdt}_sqrt * (g_{var}_H2[_k] - g_{var}_H3[_k])')


def _srk1_system_var_with_vector_wiener(sde_type, code_lines, variables, parameters, vdt):
    # shape information
    code1 = f'''
  # shape infor #
  # ----------- #
    
  f_x = f({", ".join(variables + parameters)})  # shape = (d, ..)
  g_x = g({", ".join(variables + parameters)})  # shape = (d, .., m)
  _shape = backend.shape(g_x)
  _d = _shape[0]
  _m = _shape[-1]
  _D = _shape[1:-1]
  '''
    code_lines.extend(code1.split('\n'))

    # noise term
    _vector_wiener_terms(code_lines, sde_type, vdt, shape_D='_D', shape_m='_m')

    # numerical integration
    code2 = f'''
  # numerical integration #
  # --------------------- #
  
  g_x2 = backend.moveaxis(g_x, 0, -2)  # shape = (.., d, m)
  g_H1_k = backend.matmul(g_x2, _I2) / dt_sqrt  # shape (.., d, m)
  g_H1_k = backend.moveaxis(g_H1_k, -2, 0)  # shape (d, .., m)
  x_rs = backend.reshape(x, (_d,) + _D + (1,))
  H2 = x_rs + g_H1_k  # shape (d, .., m)
  H3 = x_rs - g_H1_k  # shape (d, .., m)
  
  g1 = backend.matmul(g_x2, _I1_rs)  # shape (.., d, 1)
  g1 = backend.moveaxis(g1, -2, 0)  # shape (d, .., 1)
  y = x + f_x + g1[..., 0]  # shape (d, ..)
  for _k in range(_m):
    y += 0.5 * dt_sqrt * g(H2[..., _k], t, *args)[..., _k]
    y -= 0.5 * dt_sqrt * g(H3[..., _k], t, *args)[..., _k]
  '''
    code_lines.extend(code2.split('\n'))


def _srk1_system_var_with_scalar_wiener(sde_type, code_lines, variables, parameters, vdt):
    if sde_type == constants.ITO_SDE:
        I2 = f'0.5 * (_I1 * _I1 - {vdt})'
    elif sde_type == constants.STRA_SDE:
        I2 = f'0.5 * _I1 * _I1'
    else:
        raise ValueError(f'Unknown SDE type: {sde_type}. We only supports {constants.SUPPORTED_SDE_TYPE}.')

    code_string = f'''
  f_x = f({", ".join(variables + parameters)})  # shape = (d, ..)
  g_x = g({", ".join(variables + parameters)})  # shape = (d, ..)
  _shape = backend.shape(g_x)
  _d = _shape[0]
  _D = _shape[1:]

  # single Ito integrals
  _I1 = backend.normal(0., {vdt}_sqrt, _D)  # shape = (..)
  # double Ito integrals
  _I2 = {I2}  # shape = (..)

  # numerical integration #
  # --------------------- #
  g_H1_k = g_x * _I2 / {vdt}_sqrt  # shape (d, ..)
  H2 = x + g_H1_k  # shape (d, ..)
  H3 = x - g_H1_k  # shape (d, ..)

  g1 = g_x * _I1  # shape (d, ..)
  x_new = x + f_x + g1  # shape (d, ..)
  x_new += 0.5 * {vdt}_sqrt * g(H2, {", ".join(parameters)})
  x_new -= 0.5 * {vdt}_sqrt * g(H3, {", ".join(parameters)})
  '''
    code_splits = code_string.split('\n')
    code_lines.extend(code_splits)


def _srk1_wrapper(f, g, dt, sde_type, var_type, wiener_type, show_code, num_iter):
    vdt, variables, parameters, arguments, func_name = common.basic_info(f=f, g=g)

    # 1. code scope
    code_scope = {'f': f, 'g': g, vdt: dt, f'{vdt}_sqrt': dt ** 0.5,
                  'backend': backend, 'num_iter': num_iter}

    # 2. code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    if var_type == constants.SYSTEM_VAR:
        if len(variables) > 1:
            raise ValueError(f'SDE with {constants.SYSTEM_VAR} variable type only '
                             f'supports one system variable. But we got {variables}.')

        if wiener_type == constants.SCALAR_WIENER:
            _srk1_system_var_with_scalar_wiener(sde_type, code_lines, variables, parameters, vdt)
        elif wiener_type == constants.VECTOR_WIENER:
            _srk1_system_var_with_vector_wiener(sde_type, code_lines, variables, parameters, vdt)
        else:
            raise ValueError(f'Unknown Wiener type: {wiener_type}, we only '
                             f'supports {constants.SUPPORTED_WIENER_TYPE}')

    elif var_type == constants.SCALAR_VAR:
        if wiener_type == constants.SCALAR_WIENER:
            _srk2_pop_or_scalar_var_scalar_wiener(sde_type, code_lines, variables, parameters, vdt)
        elif wiener_type == constants.VECTOR_WIENER:
            _srk1_scalar_var_with_vector_wiener(sde_type, code_lines, variables, parameters, vdt)
        else:
            raise ValueError(f'Unknown Wiener type: {wiener_type}, we only '
                             f'supports {constants.SUPPORTED_WIENER_TYPE}')

    elif var_type == constants.POPU_VAR:
        if wiener_type == constants.SCALAR_WIENER:
            _srk2_pop_or_scalar_var_scalar_wiener(sde_type, code_lines, variables, parameters, vdt)
        elif wiener_type == constants.VECTOR_WIENER:
            _srk2_pop_var_vector_wiener(sde_type, code_lines, variables, parameters, vdt)
        else:
            raise ValueError(f'Unknown Wiener type: {wiener_type}, we only '
                             f'supports {constants.SUPPORTED_WIENER_TYPE}')

    else:
        raise ValueError(f'Unknown var type: {var_type}, we only '
                         f'supports {constants.SUPPORTED_VAR_TYPE}')

    # return and compile
    common.return_and_compile(code_lines, code_scope, show_code, variables)
    return code_scope[func_name]


def _srk2_wrapper():
    pass


def _wrap(wrapper, f, g, dt, sde_type, var_type, wiener_type, show_code, num_iter):
    """The base function to format a SRK method.

    Parameters
    ----------
    f : callable
        The drift function of the SDE.
    g : callable
        The diffusion function of the SDE.
    dt : float
        The numerical precision.
    sde_type : str
        "utils.ITO_SDE" : Ito's Stochastic Calculus.
        "utils.STRA_SDE" : Stratonovich's Stochastic Calculus.
    wiener_type : str
    var_type : str
        "scalar" : with the shape of ().
        "population" : with the shape of (N,) or (N1, N2) or (N1, N2, ...).
        "system": with the shape of (d, ), (d, N), or (d, N1, N2).
    show_code : bool
        Whether show the formatted code.

    Returns
    -------
    numerical_func : callable
        The numerical function.
    """

    sde_type = constants.ITO_SDE if sde_type is None else sde_type
    assert sde_type in constants.SUPPORTED_SDE_TYPE, f'Currently, BrainPy only support SDE types: ' \
                                                     f'{constants.SUPPORTED_SDE_TYPE}. But we got {sde_type}.'

    var_type = constants.POPU_VAR if var_type is None else var_type
    assert var_type in constants.SUPPORTED_VAR_TYPE, f'Currently, BrainPy only supports variable types: ' \
                                                     f'{constants.SUPPORTED_VAR_TYPE}. But we got {var_type}.'

    wiener_type = constants.SCALAR_WIENER if wiener_type is None else wiener_type
    assert wiener_type in constants.SUPPORTED_WIENER_TYPE, f'Currently, BrainPy only supports Wiener ' \
                                                           f'Process types: {constants.SUPPORTED_WIENER_TYPE}. ' \
                                                           f'But we got {wiener_type}.'

    show_code = False if show_code is None else show_code
    dt = profile.get_dt() if dt is None else dt
    num_iter = 10 if num_iter is None else num_iter

    if f is not None and g is not None:
        return wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=sde_type,
                       var_type=var_type, wiener_type=wiener_type, num_iter=num_iter)

    elif f is not None:
        return lambda g: wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=sde_type,
                                 var_type=var_type, wiener_type=wiener_type, num_iter=num_iter)

    elif g is not None:
        return lambda f: wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=sde_type,
                                 var_type=var_type, wiener_type=wiener_type, num_iter=num_iter)

    else:
        raise ValueError('Must provide "f" or "g".')


# ------------------
# Numerical methods
# ------------------


def srk1_strong(f=None, g=None, dt=None, sde_type=None, var_type=None, wiener_type=None, num_iter=None, show_code=None):
    return _wrap(_srk1_wrapper, f=f, g=g, dt=dt, sde_type=sde_type, var_type=var_type,
                 wiener_type=wiener_type, show_code=show_code, num_iter=num_iter)


def srk2_strong(f=None, g=None, dt=None, sde_type=None, var_type=None, wiener_type=None, num_iter=None, show_code=None):
    return _wrap(_srk2_wrapper, f=f, g=g, dt=dt, sde_type=sde_type, var_type=var_type,
                 wiener_type=wiener_type, show_code=show_code, num_iter=num_iter)
