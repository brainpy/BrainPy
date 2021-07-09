# -*- coding: utf-8 -*-

import inspect

from brainpy import errors, math
from brainpy.integrators import constants, utils
from brainpy.integrators.analysis_by_ast import separate_variables
from brainpy.integrators.ode import common

__all__ = [
  'general_rk_wrapper',
  'adaptive_rk_wrapper',
  'rk2_wrapper',
  'exp_euler_wrapper',
]

_f_kw = 'f'
_dt_kw = 'dt'


def general_rk_wrapper(f, show_code, dt, A, B, C, var_type):
  """Runge–Kutta methods for ordinary differential equation.

  For the system,

  .. backend::

      \frac{d y}{d t}=f(t, y)


  Explicit Runge-Kutta methods take the form

  .. backend::

      k_{i}=f\\left(t_{n}+c_{i}h,y_{n}+h\\sum _{j=1}^{s}a_{ij}k_{j}\\right) \\\\
      y_{n+1}=y_{n}+h \\sum_{i=1}^{s} b_{i} k_{i}

  Each method listed on this page is defined by its Butcher tableau,
  which puts the coefficients of the method in a table as follows:

  .. backend::

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
  func_name = common.f_names(f)

  keywords = {
    _f_kw: 'the derivative function',
    _dt_kw: 'the precision of numerical integration'
  }
  for v in variables:
    keywords[f'{v}_new'] = 'the intermediate value'
    for i in range(1, len(A) + 1):
      keywords[f'd{v}_k{i}'] = 'the intermediate value'
    for i in range(2, len(A) + 1):
      keywords[f'k{i}_{v}_arg'] = 'the intermediate value'
      keywords[f'k{i}_t_arg'] = 'the intermediate value'
  utils.check_kws(arguments, keywords)

  # code scope
  code_scope = {'f': f, _dt_kw: dt}

  # code lines
  code_lines = [f'def {func_name}({", ".join(arguments)}):']

  # step stage
  common.step(class_kw, variables, _dt_kw, A, C, code_lines, parameters)

  # variable update
  return_args = common.update(variables, _dt_kw, B, code_lines)

  # returns
  code_lines.append(f'  return {", ".join(return_args)}')

  # compilation
  return common.compile_and_assign_attrs(code_lines=code_lines,
                                         code_scope=code_scope,
                                         show_code=show_code,
                                         func_name=func_name,
                                         variables=variables,
                                         parameters=parameters,
                                         dt=dt,
                                         var_type=var_type)


def adaptive_rk_wrapper(f, dt, A, B1, B2, C, tol, adaptive, show_code, var_type):
  """Adaptive Runge-Kutta numerical method for ordinary differential equations.

  The embedded methods are designed to produce an estimate of the local
  truncation error of a single Runge-Kutta step, and as result, allow to
  control the error with adaptive stepsize. This is done by having two
  methods in the tableau, one with order p and one with order :backend:`p-1`.

  The lower-order step is given by

  .. backend::

      y^*_{n+1} = y_n + h\\sum_{i=1}^s b^*_i k_i,

  where the :backend:`k_{i}` are the same as for the higher order method. Then the error is

  .. backend::

      e_{n+1} = y_{n+1} - y^*_{n+1} = h\\sum_{i=1}^s (b_i - b^*_i) k_i,


  which is :backend:`O(h^{p})`. The Butcher Tableau for this kind of method is extended to
  give the values of :backend:`b_{i}^{*}`

  .. backend::

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
  func_name = common.f_names(f)

  keywords = {
    _f_kw: 'the derivative function',
    _dt_kw: 'the precision of numerical integration',
  }
  for v in variables:
    keywords[f'{v}_new'] = 'the intermediate value'
    for i in range(1, len(A) + 1):
      keywords[f'd{v}_k{i}'] = 'the intermediate value'
    for i in range(2, len(A) + 1):
      keywords[f'k{i}_{v}_arg'] = 'the intermediate value'
      keywords[f'k{i}_t_arg'] = 'the intermediate value'

  if adaptive:
    keywords['dt_new'] = 'the new numerical precision "dt"'
    keywords['tol'] = 'the tolerance for the local truncation error'
    keywords['error'] = 'the local truncation error'
    for v in variables:
      keywords[f'{v}_te'] = 'the local truncation error'
    # code scope
    code_scope = {_f_kw: f, 'tol': tol}
    arguments = list(arguments) + [f'{_dt_kw}={dt}']
  else:
    # code scope
    code_scope = {_f_kw: f, _dt_kw: dt}
  utils.check_kws(arguments, keywords)

  # code lines
  code_lines = [f'def {func_name}({", ".join(arguments)}):']
  # stage steps
  common.step(class_kw, variables, _dt_kw, A, C, code_lines, parameters)
  # variable update
  return_args = common.update(variables, _dt_kw, B1, code_lines)

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
          result.append(f'd{v}_k{i + 1} * {_dt_kw} * {diff}')
      if len(result) > 0:
        if var_type == constants.SCALAR_VAR:
          code_lines.append(f'  {v}_te = abs({" + ".join(result)})')
        else:
          code_lines.append(f'  {v}_te = sum(abs({" + ".join(result)}))')
        errors_.append(f'{v}_te')
    if len(errors_) > 0:
      code_lines.append(f'  error = {" + ".join(errors_)}')
      code_lines.append(f'  if error > tol:')
      code_lines.append(f'    {_dt_kw}_new = 0.9 * {_dt_kw} * (tol / error) ** 0.2')
      code_lines.append(f'  else:')
      code_lines.append(f'    {_dt_kw}_new = {_dt_kw}')
      return_args.append(f'{_dt_kw}_new')

  # returns
  code_lines.append(f'  return {", ".join(return_args)}')

  # compilation
  return common.compile_and_assign_attrs(code_lines=code_lines,
                                         code_scope=code_scope,
                                         show_code=show_code,
                                         func_name=func_name,
                                         variables=variables,
                                         parameters=parameters,
                                         dt=dt,
                                         var_type=var_type)


def rk2_wrapper(f, show_code, dt, beta, var_type):
  class_kw, variables, parameters, arguments = utils.get_args(f)
  func_name = common.f_names(f)

  keywords = {
    _f_kw: 'the derivative function',
    _dt_kw: 'the precision of numerical integration',
    'beta': 'the parameters in RK2 method',
    '_k1': 'the parameters in RK2 method',
    '_k2': 'the parameters in RK2 method',
  }
  for v in variables:
    keywords[f'{v}_new'] = 'the intermediate value'
    for i in range(1, 3):
      keywords[f'd{v}_k{i}'] = 'the intermediate value'
    for i in range(2, 3):
      keywords[f'k{i}_{v}_arg'] = 'the intermediate value'
      keywords[f'k{i}_t_arg'] = 'the intermediate value'
  utils.check_kws(arguments, keywords)

  code_scope = {_f_kw: f,
                _dt_kw: dt,
                'beta': beta,
                '_k1': 1 - 1 / (2 * beta),
                '_k2': 1 / (2 * beta)}
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

  return common.compile_and_assign_attrs(code_lines=code_lines,
                                         code_scope=code_scope,
                                         show_code=show_code,
                                         func_name=func_name,
                                         variables=variables,
                                         parameters=parameters,
                                         dt=dt,
                                         var_type=var_type)


def exp_euler_wrapper(f, show_code, dt, var_type):
  try:
    import sympy
    from brainpy.integrators import analysis_by_sympy
  except ModuleNotFoundError:
    raise errors.PackageMissingError('SymPy must be installed when '
                                     'using exponential euler methods.')

  if var_type == constants.SYSTEM_VAR:
    raise errors.IntegratorError(f'Exponential Euler method do not '
                                 f'support {var_type} variable type.')

  class_kw, variables, parameters, arguments = utils.get_args(f)
  func_name = common.f_names(f)
  keywords = {
    _f_kw: 'the derivative function',
    _dt_kw: 'the precision of numerical integration',
    'exp': 'the exponential function',
  }
  for v in variables:
    keywords[f'{v}_new'] = 'the intermediate value'
  utils.check_kws(arguments, keywords)

  code_lines = [f'def {func_name}({", ".join(arguments)}):']

  # code scope
  closure_vars = inspect.getclosurevars(f)
  code_scope = dict(closure_vars.nonlocals)
  code_scope.update(dict(closure_vars.globals))
  code_scope[_dt_kw] = dt
  code_scope[_f_kw] = f
  code_scope['exp'] = math.exp

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
    diff_eq = analysis_by_sympy.SingleDiffEq(var_name=var_name,
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
    df_expr = analysis_by_sympy.str2sympy(f_res.code).expr.expand()
    s_df = sympy.Symbol(f"{f_res.var_name}")
    code_lines.append(f'  {s_df.name} = {analysis_by_sympy.sympy2str(df_expr)}')
    var = sympy.Symbol(diff_eq.var_name, real=True)

    # get df part
    s_linear = sympy.Symbol(f'_{diff_eq.var_name}_linear')
    s_linear_exp = sympy.Symbol(f'_{diff_eq.var_name}_linear_exp')
    s_df_part = sympy.Symbol(f'_{diff_eq.var_name}_df_part')
    if df_expr.has(var):
      # linear
      linear = sympy.collect(df_expr, var, evaluate=False)[var]
      code_lines.append(f'  {s_linear.name} = {analysis_by_sympy.sympy2str(linear)}')
      # linear exponential
      linear_exp = sympy.exp(linear * dt)
      code_lines.append(f'  {s_linear_exp.name} = {analysis_by_sympy.sympy2str(linear_exp)}')
      # df part
      df_part = (s_linear_exp - 1) / s_linear * s_df
      code_lines.append(f'  {s_df_part.name} = {analysis_by_sympy.sympy2str(df_part)}')

    else:
      # linear exponential
      code_lines.append(f'  {s_linear_exp.name} = {dt} ** 0.5')
      # df part
      code_lines.append(f'  {s_df_part.name} = {analysis_by_sympy.sympy2str(dt * s_df)}')

    # update expression
    update = var + s_df_part

    # The actual update step
    code_lines.append(f'  {diff_eq.var_name}_new = {analysis_by_sympy.sympy2str(update)}')
    code_lines.append('')

  code_lines.append(f'  return {", ".join([f"{v}_new" for v in variables])}')
  return common.compile_and_assign_attrs(code_lines=code_lines,
                                         code_scope=code_scope,
                                         show_code=show_code,
                                         func_name=func_name,
                                         variables=variables,
                                         parameters=parameters,
                                         dt=dt,
                                         var_type=var_type)
