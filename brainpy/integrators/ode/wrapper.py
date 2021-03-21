# -*- coding: utf-8 -*-

from pprint import pprint

from brainpy.integrators import constants
from brainpy.integrators import utils

__all__ = [
    'rk_wrapper',
    'adaptive_rk_wrapper',
    'wrapper_of_rk2',
]

_ODE_UNKNOWN_NO = 0


def _f_names(f):
    if f.__name__.isidentifier():
        f_name = f.__name__
    else:
        global _ODE_UNKNOWN_NO
        f_name = f'ode_unknown_{_ODE_UNKNOWN_NO}'
        _ODE_UNKNOWN_NO += 1
    f_new_name = constants.NAME_PREFIX + f_name
    return f_new_name


def _step(vars, dt_var, A, C, code_lines, other_args):
    # steps
    for si, sval in enumerate(A):
        # k-step arguments
        k_args = []
        for v in vars:
            k_arg = f'{v}'
            for j, sv in enumerate(sval):
                if sv not in [0., '0.', '0']:
                    if sv in ['1.', '1', 1.]:
                        k_arg += f' + {dt_var} * d{v}_k{j + 1}'
                    else:
                        k_arg += f' + {dt_var} * d{v}_k{j + 1} * {sv}'
            if k_arg != v:
                name = f'k{si + 1}_{v}_arg'
                code_lines.append(f'  {name} = {k_arg}')
                k_args.append(name)
            else:
                k_args.append(v)

        t_arg = 't'
        if C[si] not in [0., '0.', '0']:
            if C[si] in ['1.', '1', 1.]:
                t_arg += f' + {dt_var}'
            else:
                t_arg += f' + {dt_var} * {C[si]}'
            name = f'k{si + 1}_t_arg'
            code_lines.append(f'  {name} = {t_arg}')
            k_args.append(name)
        else:
            k_args.append(f'{dt_var}')

        # k-step derivative names
        k_derivatives = [f'd{v}_k{si + 1}' for v in vars]

        # k-step code line
        code_lines.append(f'  {", ".join(k_derivatives)} = f('
                          f'{", ".join(k_args + other_args[1:])})')


def _update(vars, dt_var, B, code_lines):
    return_args = []
    for v in vars:
        result = v
        for i, b1 in enumerate(B):
            if b1 not in [0., '0.', '0']:
                result += f' + d{v}_k{i + 1} * {dt_var} * {b1}'
        code_lines.append(f'  {v}_new = {result}')
        return_args.append(f'{v}_new')
    return return_args


def _compile_and_assign_attrs(code_lines, code_scope, show_code,
                              func_name, variables, parameters, dt):
    # compile
    code = '\n'.join(code_lines)
    if show_code:
        print(code)
        print()
        pprint(code_scope)
        print()
    utils.numba_func(code_scope, ['f'])
    exec(compile(code, '', 'exec'), code_scope)

    # attribute assignment
    new_f = code_scope[func_name]
    new_f.variables = variables
    new_f.parameters = parameters
    new_f.origin_f = code_scope['f']
    new_f.dt = dt
    utils.numba_func(code_scope, func_name)
    return code_scope[func_name]


def rk_wrapper(f, show_code, dt, A, B, C):
    """Runge–Kutta methods for ordinary differential equation.

    For the system,

    .. math::

        \frac{d y}{d t}=f(t, y)


    Explicit Runge-Kutta methods take the form

    .. math::

        k_{i}=f\\left(t_{n}+c_{i}h,y_{n}+h\\sum _{j=1}^{s}a_{ij}k_{j}\\right) \\\\
        y_{n+1}=y_{n}+h \\sum_{i=1}^{s} b_{i} k_{i}

    Each method listed on this page is defined by its Butcher tableau,
    which puts the coefficients of the method in a table as follows:

    .. math::

        \\begin{array}{c|cccc}
            c_{1} & a_{11} & a_{12} & \\ldots & a_{1 s} \\\\
            c_{2} & a_{21} & a_{22} & \\ldots & a_{2 s} \\\\
            \\vdots & \vdots & \vdots & \\ddots & \vdots \\\\
            c_{s} & a_{s 1} & a_{s 2} & \\ldots & a_{s s} \\\\
            \\hline & b_{1} & b_{2} & \\ldots & b_{s}
        \\end{array}

    Parameters
    ----------
    f : callable
        The derivative function.
    show_code : bool
        Whether show the formatted code.
    dt : float
        The numerical precision.
    A : tuple, list
        The A matrix in the Butcher tableau.
    B : tuple, list
        The B vector in the Butcher tableau.
    C : tuple, list
        The C vector in the Butcher tableau.

    Returns
    -------
    integral_func : callable
        The one-step numerical integration function.
    """
    class_kw, variables, parameters, arguments = utils.get_args(f)
    dt_var = 'dt'
    func_name = _f_names(f)

    # code scope
    code_scope = {'f': f, 'dt': dt}

    # code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # step stage
    _step(variables, dt_var, A, C, code_lines, parameters)

    # variable update
    return_args = _update(variables, dt_var, B, code_lines)

    # returns
    code_lines.append(f'  return {", ".join(return_args)}')

    # compilation
    return _compile_and_assign_attrs(
        code_lines=code_lines, code_scope=code_scope, show_code=show_code,
        func_name=func_name, variables=variables, parameters=parameters, dt=dt)


def adaptive_rk_wrapper(f, dt, A, B1, B2, C, tol, adaptive, show_code, var_type):
    """Adaptive Runge-Kutta numerical method for ordinary differential equations.

    The embedded methods are designed to produce an estimate of the local
    truncation error of a single Runge-Kutta step, and as result, allow to
    control the error with adaptive stepsize. This is done by having two
    methods in the tableau, one with order p and one with order :math:`p-1`.

    The lower-order step is given by

    .. math::

        y^*_{n+1} = y_n + h\\sum_{i=1}^s b^*_i k_i,

    where the :math:`k_{i}` are the same as for the higher order method. Then the error is

    .. math::

        e_{n+1} = y_{n+1} - y^*_{n+1} = h\\sum_{i=1}^s (b_i - b^*_i) k_i,


    which is :math:`O(h^{p})`. The Butcher Tableau for this kind of method is extended to
    give the values of :math:`b_{i}^{*}`

    .. math::

        \\begin{array}{c|cccc}
            c_1    & a_{11} & a_{12}& \\dots & a_{1s}\\\\
            c_2    & a_{21} & a_{22}& \\dots & a_{2s}\\\\
            \\vdots & \\vdots & \\vdots& \\ddots& \\vdots\\\\
            c_s    & a_{s1} & a_{s2}& \\dots & a_{ss} \\\\
        \\hline & b_1    & b_2   & \\dots & b_s\\\\
               & b_1^*    & b_2^*   & \\dots & b_s^*\\\\
        \\end{array}


    Parameters
    ----------
    f : callable
        The derivative function.
    show_code : bool
        Whether show the formatted code.
    dt : float
        The numerical precision.
    A : tuple, list
        The A matrix in the Butcher tableau.
    B1 : tuple, list
        The B1 vector in the Butcher tableau.
    B2 : tuple, list
        The B2 vector in the Butcher tableau.
    C : tuple, list
        The C vector in the Butcher tableau.
    adaptive : bool
    tol : float
    var_type : str

    Returns
    -------
    integral_func : callable
        The one-step numerical integration function.
    """
    assert var_type in constants.SUPPORTED_VAR_TYPE, \
        f'"var_type" only supports {constants.SUPPORTED_VAR_TYPE}, ' \
        f'not {var_type}.'

    class_kw, variables, parameters, arguments = utils.get_args(f)
    dt_var = 'dt'
    func_name = _f_names(f)

    if adaptive:
        # code scope
        code_scope = {'f': f, 'tol': tol}
        arguments = list(arguments) + ['dt']
    else:
        # code scope
        code_scope = {'f': f, 'dt': dt}

    # code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']
    # stage steps
    _step(variables, dt_var, A, C, code_lines, parameters)
    # variable update
    return_args = _update(variables, dt_var, B1, code_lines)

    # error adaptive item
    if adaptive:
        errors = []
        for v in variables:
            result = []
            for i, (b1, b2) in enumerate(zip(B1, B2)):
                if isinstance(b1, str):
                    b1 = eval(b1)
                if isinstance(b2, str):
                    b2 = eval(b2)
                diff = b1 - b2
                if diff != 0.:
                    result.append(f'd{v}_k{i + 1} * {dt_var} * {diff}')
            if len(result) > 0:
                if var_type == constants.SCALAR_VAR:
                    code_lines.append(f'  {v}_te = abs({" + ".join(result)})')
                else:
                    code_lines.append(f'  {v}_te = sum(abs({" + ".join(result)}))')
                errors.append(f'{v}_te')
        if len(errors) > 0:
            code_lines.append(f'  error = {" + ".join(errors)}')
            code_lines.append(f'  if error > tol:')
            code_lines.append(f'    {dt_var}_new = 0.9 * {dt_var} * (tol / error) ** 0.2')
            code_lines.append(f'  else:')
            code_lines.append(f'    {dt_var}_new = {dt_var}')
            return_args.append(f'{dt_var}_new')

    # returns
    code_lines.append(f'  return {", ".join(return_args)}')

    # compilation
    return _compile_and_assign_attrs(
        code_lines=code_lines, code_scope=code_scope, show_code=show_code,
        func_name=func_name, variables=variables, parameters=parameters, dt=dt)


def wrapper_of_rk2(f, show_code, dt, beta):
    class_kw, variables, parameters, arguments = utils.get_args(f)
    func_name = _f_names(f)

    code_scope = {'f': f, 'dt': dt, 'beta': beta,
                  'k1': 1 - 1 / (2 * beta), 'k2': 1 / (2 * beta)}
    code_lines = [f'def {func_name}({", ".join(arguments)}):']
    # k1
    k1_args = variables + parameters
    k1_vars_d = [f'd{v}_k1' for v in variables]
    code_lines.append(f'  {", ".join(k1_vars_d)} = f({", ".join(k1_args)})')
    # k2
    k2_args = [f'{v} + d{v}_k1 * dt * beta' for v in variables]
    k2_args.append('t + dt * beta')
    k2_args.extend(parameters[1:])
    k2_vars_d = [f'd{v}_k2' for v in variables]
    code_lines.append(f'  {", ".join(k2_vars_d)} = f({", ".join(k2_args)})')
    # returns
    for v, k1, k2 in zip(variables, k1_vars_d, k2_vars_d):
        code_lines.append(f'  {v}_new = {v} + ({k1} * k1 + {k2} * k2) * dt')
    return_vars = [f'{v}_new' for v in variables]
    code_lines.append(f'  return {", ".join(return_vars)}')

    return _compile_and_assign_attrs(
        code_lines=code_lines, code_scope=code_scope, show_code=show_code,
        func_name=func_name, variables=variables, parameters=parameters, dt=dt)

