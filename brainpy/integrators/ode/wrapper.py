# -*- coding: utf-8 -*-

import inspect

from brainpy import backend
from brainpy import errors
from brainpy.backend import ops
from brainpy.integrators import constants
from brainpy.integrators import utils
from brainpy.integrators.ast_analysis import separate_variables

__all__ = [
    'general_rk_wrapper',
    'adaptive_rk_wrapper',
    'rk2_wrapper',
    'exp_euler_wrapper',
]

_ODE_UNKNOWN_NO = 0


class Tools(object):

    @staticmethod
    def f_names(f):
        if f.__name__.isidentifier():
            f_name = f.__name__
        else:
            global _ODE_UNKNOWN_NO
            f_name = f'ode_unknown_{_ODE_UNKNOWN_NO}'
            _ODE_UNKNOWN_NO += 1
        f_new_name = constants.ODE_PREFIX + f_name
        return f_new_name

    @staticmethod
    def step(class_kw, vars, dt_var, A, C, code_lines, other_args):
        # steps
        for si, sval in enumerate(A):
            # k-step arguments
            k_args = []
            for v in vars:
                k_arg = f'{v}'
                for j, sv in enumerate(sval):
                    if sv not in [0., '0.0', '0.', '0']:
                        if sv in ['1.0', '1.', '1', 1.]:
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
                k_args.append(t_arg)

            # k-step derivative names
            k_derivatives = [f'd{v}_k{si + 1}' for v in vars]

            # k-step code line
            code_lines.append(f'  {", ".join(k_derivatives)} = f('
                              f'{", ".join(class_kw + k_args + other_args[1:])})')

    @staticmethod
    def update(vars, dt_var, B, code_lines):
        return_args = []
        for v in vars:
            result = v
            for i, b1 in enumerate(B):
                if b1 not in [0., '0.', '0']:
                    result += f' + d{v}_k{i + 1} * {dt_var} * {b1}'
            code_lines.append(f'  {v}_new = {result}')
            return_args.append(f'{v}_new')
        return return_args

    @staticmethod
    def compile_and_assign_attrs(code_lines, code_scope, show_code,
                                 func_name, variables, parameters,
                                 dt, var_type):
        driver_cls = backend.get_diffint_driver()
        driver = driver_cls(code_scope=code_scope,
                            code_lines=code_lines,
                            func_name=func_name,
                            show_code=show_code,
                            uploads=dict(variables=variables,
                                         parameters=parameters,
                                         origin_f=code_scope['f'],
                                         var_type=var_type,
                                         dt=dt))
        return driver.build()


def general_rk_wrapper(f, show_code, dt, A, B, C, var_type, im_return):
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
    func_name = Tools.f_names(f)

    # code scope
    code_scope = {'f': f, 'dt': dt}

    # code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # step stage
    Tools.step(class_kw, variables, dt_var, A, C, code_lines, parameters)

    # variable update
    return_args = Tools.update(variables, dt_var, B, code_lines)

    # returns
    code_lines.append(f'  return {", ".join(return_args)}')

    # compilation
    return Tools.compile_and_assign_attrs(code_lines=code_lines,
                                          code_scope=code_scope,
                                          show_code=show_code,
                                          func_name=func_name,
                                          variables=variables,
                                          parameters=parameters,
                                          dt=dt,
                                          var_type=var_type)


def adaptive_rk_wrapper(f, dt, A, B1, B2, C, tol, adaptive, show_code, var_type, im_return):
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
    if var_type not in constants.SUPPORTED_VAR_TYPE:
        raise errors.IntegratorError(f'"var_type" only supports {constants.SUPPORTED_VAR_TYPE}, not {var_type}.')

    class_kw, variables, parameters, arguments = utils.get_args(f)
    dt_var = 'dt'
    func_name = Tools.f_names(f)

    if adaptive:
        # code scope
        code_scope = {'f': f, 'tol': tol}
        arguments = list(arguments) + [f'dt={dt}']
    else:
        # code scope
        code_scope = {'f': f, 'dt': dt}

    # code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']
    # stage steps
    Tools.step(class_kw, variables, dt_var, A, C, code_lines, parameters)
    # variable update
    return_args = Tools.update(variables, dt_var, B1, code_lines)

    # error adaptive item
    if adaptive:
        errors_ = []
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
                errors_.append(f'{v}_te')
        if len(errors_) > 0:
            code_lines.append(f'  error = {" + ".join(errors_)}')
            code_lines.append(f'  if error > tol:')
            code_lines.append(f'    {dt_var}_new = 0.9 * {dt_var} * (tol / error) ** 0.2')
            code_lines.append(f'  else:')
            code_lines.append(f'    {dt_var}_new = {dt_var}')
            return_args.append(f'{dt_var}_new')

    # returns
    code_lines.append(f'  return {", ".join(return_args)}')

    # compilation
    return Tools.compile_and_assign_attrs(code_lines=code_lines,
                                          code_scope=code_scope,
                                          show_code=show_code,
                                          func_name=func_name,
                                          variables=variables,
                                          parameters=parameters,
                                          dt=dt,
                                          var_type=var_type)


def rk2_wrapper(f, show_code, dt, beta, var_type, im_return):
    class_kw, variables, parameters, arguments = utils.get_args(f)
    func_name = Tools.f_names(f)

    code_scope = {'f': f, 'dt': dt, 'beta': beta,
                  '_k1': 1 - 1 / (2 * beta), '_k2': 1 / (2 * beta)}
    code_lines = [f'def {func_name}({", ".join(arguments)}):']
    # k1
    k1_args = variables + parameters
    k1_vars_d = [f'd{v}_k1' for v in variables]
    code_lines.append(f'  {", ".join(k1_vars_d)} = f({", ".join(class_kw + k1_args)})')
    # k2
    k2_args = [f'{v} + d{v}_k1 * dt * beta' for v in variables]
    k2_args.append('t + dt * beta')
    k2_args.extend(parameters[1:])
    k2_vars_d = [f'd{v}_k2' for v in variables]
    code_lines.append(f'  {", ".join(k2_vars_d)} = f({", ".join(class_kw + k2_args)})')
    # returns
    for v, k1, k2 in zip(variables, k1_vars_d, k2_vars_d):
        code_lines.append(f'  {v}_new = {v} + ({k1} * _k1 + {k2} * _k2) * dt')
    return_vars = [f'{v}_new' for v in variables]
    code_lines.append(f'  return {", ".join(return_vars)}')

    return Tools.compile_and_assign_attrs(code_lines=code_lines,
                                          code_scope=code_scope,
                                          show_code=show_code,
                                          func_name=func_name,
                                          variables=variables,
                                          parameters=parameters,
                                          dt=dt,
                                          var_type=var_type)


def exp_euler_wrapper(f, show_code, dt, var_type, im_return):
    try:
        import sympy
        from brainpy.integrators import sympy_analysis
    except ModuleNotFoundError:
        raise errors.PackageMissingError('SymPy must be installed when using exponential euler methods.')

    if var_type == constants.SYSTEM_VAR:
        raise errors.IntegratorError(f'Exponential Euler method do not support {var_type} variable type.')

    dt_var = 'dt'
    class_kw, variables, parameters, arguments = utils.get_args(f)
    func_name = Tools.f_names(f)

    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # code scope
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(dict(closure_vars.globals))
    code_scope[dt_var] = dt
    code_scope['f'] = f
    code_scope['exp'] = ops.exp

    analysis = separate_variables(f)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, vars) in enumerate(variables_for_returns.items()):
        # separate variables
        sd_variables = []
        for v in vars:
            if len(v) > 1:
                raise ValueError('Cannot analyze multi-assignment code line.')
            sd_variables.append(v[0])
        expressions = expressions_for_returns[key]
        var_name = variables[vi]
        diff_eq = sympy_analysis.SingleDiffEq(var_name=var_name,
                                              variables=sd_variables,
                                              expressions=expressions,
                                              derivative_expr=key,
                                              scope=code_scope,
                                              func_name=func_name)

        f_expressions = diff_eq.get_f_expressions(substitute_vars=diff_eq.var_name)

        # code lines
        code_lines.extend([f"  {str(expr)}" for expr in f_expressions[:-1]])

        # get the linear system using sympy
        f_res = f_expressions[-1]
        df_expr = sympy_analysis.str2sympy(f_res.code).expr.expand()
        s_df = sympy.Symbol(f"{f_res.var_name}")
        code_lines.append(f'  {s_df.name} = {sympy_analysis.sympy2str(df_expr)}')
        var = sympy.Symbol(diff_eq.var_name, real=True)

        # get df part
        s_linear = sympy.Symbol(f'_{diff_eq.var_name}_linear')
        s_linear_exp = sympy.Symbol(f'_{diff_eq.var_name}_linear_exp')
        s_df_part = sympy.Symbol(f'_{diff_eq.var_name}_df_part')
        if df_expr.has(var):
            # linear
            linear = sympy.collect(df_expr, var, evaluate=False)[var]
            code_lines.append(f'  {s_linear.name} = {sympy_analysis.sympy2str(linear)}')
            # linear exponential
            linear_exp = sympy.exp(linear * dt)
            code_lines.append(f'  {s_linear_exp.name} = {sympy_analysis.sympy2str(linear_exp)}')
            # df part
            df_part = (s_linear_exp - 1) / s_linear * s_df
            code_lines.append(f'  {s_df_part.name} = {sympy_analysis.sympy2str(df_part)}')

        else:
            # linear exponential
            code_lines.append(f'  {s_linear_exp.name} = sqrt({dt})')
            # df part
            code_lines.append(f'  {s_df_part.name} = {sympy_analysis.sympy2str(dt * s_df)}')

        # update expression
        update = var + s_df_part

        # The actual update step
        code_lines.append(f'  {diff_eq.var_name}_new = {sympy_analysis.sympy2str(update)}')
        code_lines.append('')

    code_lines.append(f'  return {", ".join([f"{v}_new" for v in variables])}')
    return Tools.compile_and_assign_attrs(code_lines=code_lines,
                                          code_scope=code_scope,
                                          show_code=show_code,
                                          func_name=func_name,
                                          variables=variables,
                                          parameters=parameters,
                                          dt=dt,
                                          var_type=var_type)
