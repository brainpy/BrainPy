# -*- coding: utf-8 -*-

from brainpy import errors, math
from brainpy.integrators import constants
from . import common

__all__ = [
  'srk1w1_scalar',
  'srk2w1_scalar',
  'KlPl_scalar',
]


class Tools(object):
  @staticmethod
  def _noise_terms(code_lines, variables, vdt, triple_integral=True):
    # num_vars = len(variables)
    # if num_vars > 1:
    #     code_lines.append(f'  all_I1 = math.normal(0.0, dt_sqrt, ({num_vars},)+math.shape({variables[0]}))')
    #     code_lines.append(f'  all_I0 = math.normal(0.0, dt_sqrt, ({num_vars},)+math.shape({variables[0]}))')
    #     code_lines.append(f'  all_I10 = 0.5 * {vdt} * (all_I1 + all_I0 / 3.0 ** 0.5)')
    #     code_lines.append(f'  all_I11 = 0.5 * (all_I1 ** 2 - {vdt})')
    #     if triple_integral:
    #         code_lines.append(f'  all_I111 = (all_I1 ** 3 - 3 * {vdt} * all_I1) / 6')
    #     code_lines.append(f'  ')
    #     for i, var in enumerate(variables):
    #         code_lines.append(f'  {var}_I1 = all_I1[{i}]')
    #         code_lines.append(f'  {var}_I0 = all_I0[{i}]')
    #         code_lines.append(f'  {var}_I10 = all_I10[{i}]')
    #         code_lines.append(f'  {var}_I11 = all_I11[{i}]')
    #         if triple_integral:
    #             code_lines.append(f'  {var}_I111 = all_I111[{i}]')
    #         code_lines.append(f'  ')
    # else:
    #     var = variables[0]
    #     code_lines.append(f'  {var}_I1 = math.normal(0.0, dt_sqrt, math.shape({var}))')
    #     code_lines.append(f'  {var}_I0 = math.normal(0.0, dt_sqrt, math.shape({var}))')
    #     code_lines.append(f'  {var}_I10 = 0.5 * {vdt} * ({var}_I1 + {var}_I0 / 3.0 ** 0.5)')
    #     code_lines.append(f'  {var}_I11 = 0.5 * ({var}_I1 ** 2 - {vdt})')
    #     if triple_integral:
    #         code_lines.append(f'  {var}_I111 = ({var}_I1 ** 3 - 3 * {vdt} * {var}_I1) / 6')
    #     code_lines.append('  ')

    for var in variables:
      code_lines.append(f'  {var}_I1 = math.random.normal(0.000, dt_sqrt, math.shape({var}))')
      code_lines.append(f'  {var}_I0 = math.random.normal(0.000, dt_sqrt, math.shape({var}))')
      code_lines.append(f'  {var}_I10 = 0.5 * {vdt} * ({var}_I1 + {var}_I0 / 3.0 ** 0.5)')
      code_lines.append(f'  {var}_I11 = 0.5 * ({var}_I1 ** 2 - {vdt})')
      if triple_integral:
        code_lines.append(f'  {var}_I111 = ({var}_I1 ** 3 - 3 * {vdt} * {var}_I1) / 6')
      code_lines.append('  ')

  @staticmethod
  def _state1(code_lines, variables, parameters):
    f_names = [f'{var}_f_H0s1' for var in variables]
    g_names = [f'{var}_g_H1s1' for var in variables]
    code_lines.append(f'  {", ".join(f_names)} = f({", ".join(variables + parameters)})')
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(variables + parameters)})')
    code_lines.append('  ')


class Wrappers(object):
  @staticmethod
  def srk1w1(f, g, dt, show_code, intg_type, var_type, wiener_type):
    vdt, variables, parameters, arguments, func_name = common.basic_info(f=f, g=g)

    # 1. code scope
    code_scope = {'f': f, 'g': g, vdt: dt, f'{vdt}_sqrt': dt ** 0.5, 'math': math}

    # 2. code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # 2.1 noise
    Tools._noise_terms(code_lines, variables, vdt, triple_integral=True)

    # 2.2 stage 1
    Tools._state1(code_lines, variables, parameters)

    # 2.3 stage 2
    all_H0s2, all_H1s2 = [], []
    for var in variables:
      code_lines.append(f'  {var}_H0s2 = {var} + {vdt} * 0.75 * {var}_f_H0s1 + '
                        f'1.5 * {var}_g_H1s1 * {var}_I10 / {vdt}')
      all_H0s2.append(f'{var}_H0s2')
      code_lines.append(f'  {var}_H1s2 = {var} + {vdt} * 0.25 * {var}_f_H0s1 + '
                        f'dt_sqrt * 0.5 * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    all_H0s2.append(f't + 0.75 * {vdt}')  # t
    all_H1s2.append(f't + 0.25 * {vdt}')  # t
    f_names = [f'{var}_f_H0s2' for var in variables]
    code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s2 + parameters[1:])})')
    g_names = [f'{var}_g_H1s2' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + parameters[1:])})')
    code_lines.append('  ')

    # 2.4 state 3
    all_H1s3 = []
    for var in variables:
      code_lines.append(f'  {var}_H1s3 = {var} + {vdt} * {var}_f_H0s1 - dt_sqrt * {var}_g_H1s1')
      all_H1s3.append(f'{var}_H1s3')
    all_H1s3.append(f't + {vdt}')  # t
    g_names = [f'{var}_g_H1s3' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s3 + parameters[1:])})')
    code_lines.append('  ')

    # 2.5 state 4
    all_H1s4 = []
    for var in variables:
      code_lines.append(f'  {var}_H1s4 = {var} + 0.25 * {vdt} * {var}_f_H0s1 + dt_sqrt * '
                        f'(-5 * {var}_g_H1s1 + 3 * {var}_g_H1s2 + 0.5 * {var}_g_H1s3)')
      all_H1s4.append(f'{var}_H1s4')
    all_H1s4.append(f't + 0.25 * {vdt}')  # t
    g_names = [f'{var}_g_H1s4' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s4 + parameters[1:])})')
    code_lines.append('  ')

    # 2.6 final stage
    for var in variables:
      code_lines.append(f'  {var}_f1 = {var}_f_H0s1/3 + {var}_f_H0s2 * 2/3')
      code_lines.append(
        f'  {var}_g1 = -{var}_I1 - {var}_I11/dt_sqrt + 2 * {var}_I10/{vdt} - 2 * {var}_I111/{vdt}')
      code_lines.append(f'  {var}_g2 = {var}_I1 * 4/3 + {var}_I11 / dt_sqrt * 4/3 - '
                        f'{var}_I10 / {vdt} * 4/3 + {var}_I111 / {vdt} * 5/3')
      code_lines.append(f'  {var}_g3 = {var}_I1 * 2/3 - {var}_I11/dt_sqrt/3 - '
                        f'{var}_I10 / {vdt} * 2/3 - {var}_I111 / {vdt} * 2/3')
      code_lines.append(f'  {var}_g4 = {var}_I111 / {vdt}')
      code_lines.append(f'  {var}_new = {var} + {vdt} * {var}_f1 + {var}_g1 * {var}_g_H1s1 + '
                        f'{var}_g2 * {var}_g_H1s2 + {var}_g3 * {var}_g_H1s3 + {var}_g4 * {var}_g_H1s4')
      code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in variables]
    code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    return common.compile_and_assign_attrs(
      code_lines=code_lines, code_scope=code_scope, show_code=show_code,
      variables=variables, parameters=parameters, func_name=func_name,
      intg_type=intg_type, var_type=var_type, wiener_type=wiener_type, dt=dt)

  @staticmethod
  def srk2w1(f, g, dt, show_code, intg_type, var_type, wiener_type):
    vdt, variables, parameters, arguments, func_name = common.basic_info(f=f, g=g)

    # 1. code scope
    code_scope = {'f': f, 'g': g, vdt: dt, f'{vdt}_sqrt': dt ** 0.5, 'math': math}

    # 2. code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # 2.1 noise
    Tools._noise_terms(code_lines, variables, vdt, triple_integral=True)

    # 2.2 stage 1
    Tools._state1(code_lines, variables, parameters)

    # 2.3 stage 2
    # ----
    # H0s2 = x + dt * f_H0s1
    # H1s2 = x + dt * 0.25 * f_H0s1 - dt_sqrt * 0.5 * g_H1s1
    # f_H0s2 = f(H0s2, t + dt, *args)
    # g_H1s2 = g(H1s2, t + 0.25 * dt, *args)
    all_H0s2, all_H1s2 = [], []
    for var in variables:
      code_lines.append(f'  {var}_H0s2 = {var} + {vdt} * {var}_f_H0s1')
      all_H0s2.append(f'{var}_H0s2')
      code_lines.append(f'  {var}_H1s2 = {var} + {vdt} * 0.25 * {var}_f_H0s1 - '
                        f'dt_sqrt * 0.5 * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    all_H0s2.append(f't + {vdt}')  # t
    all_H1s2.append(f't + 0.25 * {vdt}')  # t
    f_names = [f'{var}_f_H0s2' for var in variables]
    code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s2 + parameters[1:])})')
    g_names = [f'{var}_g_H1s2' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + parameters[1:])})')
    code_lines.append('  ')

    # 2.4 state 3
    # ---
    # H0s3 = x + dt * (0.25 * f_H0s1 + 0.25 * f_H0s2) + (g_H1s1 + 0.5 * g_H1s2) * I10 / dt
    # H1s3 = x + dt * f_H0s1 + dt_sqrt * g_H1s1
    # f_H0s3 = g(H0s3, t + 0.5 * dt, *args)
    # g_H1s3 = g(H1s3, t + dt, *args)
    all_H0s3, all_H1s3 = [], []
    for var in variables:
      code_lines.append(f'  {var}_H0s3 = {var} + {vdt} * (0.25 * {var}_f_H0s1 + 0.25 * {var}_f_H0s2) + '
                        f'({var}_g_H1s1 + 0.5 * {var}_g_H1s2) * {var}_I10 / {vdt}')
      all_H0s3.append(f'{var}_H0s3')
      code_lines.append(f'  {var}_H1s3 = {var} + {vdt} * {var}_f_H0s1 + dt_sqrt * {var}_g_H1s1')
      all_H1s3.append(f'{var}_H1s3')
    all_H0s3.append(f't + 0.5 * {vdt}')  # t
    all_H1s3.append(f't + {vdt}')  # t
    f_names = [f'{var}_f_H0s3' for var in variables]
    g_names = [f'{var}_g_H1s3' for var in variables]
    code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s3 + parameters[1:])})')
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s3 + parameters[1:])})')
    code_lines.append('  ')

    # 2.5 state 4
    # ----
    # H1s4 = x + dt * 0.25 * f_H0s3 + dt_sqrt * (2 * g_H1s1 - g_H1s2 + 0.5 * g_H1s3)
    # g_H1s4 = g(H1s4, t + 0.25 * dt, *args)
    all_H1s4 = []
    for var in variables:
      code_lines.append(f'  {var}_H1s4 = {var} + 0.25 * {vdt} * {var}_f_H0s1 + dt_sqrt * '
                        f'(2 * {var}_g_H1s1 - {var}_g_H1s2 + 0.5 * {var}_g_H1s3)')
      all_H1s4.append(f'{var}_H1s4')
    all_H1s4.append(f't + 0.25 * {vdt}')  # t
    g_names = [f'{var}_g_H1s4' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s4 + parameters[1:])})')
    code_lines.append('  ')

    # 2.6 final stage
    # ----
    # f1 = f_H0s1 / 6 + f_H0s2 / 6 + f_H0s3 * 2 / 3
    # g1 = - I1 + I11 / dt_sqrt + 2 * I10 / dt - 2 * I111 / dt
    # g2 = I1 * 4 / 3 - I11 / dt_sqrt * 4 / 3 - I10 / dt * 4 / 3 + I111 / dt * 5 / 3
    # g3 = I1 * 2 / 3 + I11 / dt_sqrt / 3 - I10 / dt * 2 / 3 - I111 / dt * 2 / 3
    # g4 = I111 / dt
    # y1 = x + dt * f1 + g1 * g_H1s1 + g2 * g_H1s2 + g3 * g_H1s3 + g4 * g_H1s4
    for var in variables:
      code_lines.append(f'  {var}_f1 = {var}_f_H0s1/6 + {var}_f_H0s2/6 + {var}_f_H0s3*2/3')
      code_lines.append(
        f'  {var}_g1 = -{var}_I1 + {var}_I11/dt_sqrt + 2 * {var}_I10/{vdt} - 2 * {var}_I111/{vdt}')
      code_lines.append(f'  {var}_g2 = {var}_I1 * 4/3 - {var}_I11 / dt_sqrt * 4/3 - '
                        f'{var}_I10 / {vdt} * 4/3 + {var}_I111 / {vdt} * 5/3')
      code_lines.append(f'  {var}_g3 = {var}_I1 * 2/3 + {var}_I11/dt_sqrt/3 - '
                        f'{var}_I10 / {vdt} * 2/3 - {var}_I111 / {vdt} * 2/3')
      code_lines.append(f'  {var}_g4 = {var}_I111 / {vdt}')
      code_lines.append(f'  {var}_new = {var} + {vdt} * {var}_f1 + {var}_g1 * {var}_g_H1s1 + '
                        f'{var}_g2 * {var}_g_H1s2 + {var}_g3 * {var}_g_H1s3 + {var}_g4 * {var}_g_H1s4')
      code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in variables]
    code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    return common.compile_and_assign_attrs(
      code_lines=code_lines, code_scope=code_scope, show_code=show_code,
      variables=variables, parameters=parameters, func_name=func_name,
      intg_type=intg_type, var_type=var_type, wiener_type=wiener_type, dt=dt)

  @staticmethod
  def KlPl(f, g, dt, show_code, intg_type, var_type, wiener_type):
    vdt, variables, parameters, arguments, func_name = common.basic_info(f=f, g=g)

    # 1. code scope
    code_scope = {'f': f, 'g': g, vdt: dt, f'{vdt}_sqrt': dt ** 0.5, 'math': math}

    # 2. code lines
    code_lines = [f'def {func_name}({", ".join(arguments)}):']

    # 2.1 noise
    Tools._noise_terms(code_lines, variables, vdt, triple_integral=False)

    # 2.2 stage 1
    Tools._state1(code_lines, variables, parameters)

    # 2.3 stage 2
    # ----
    # H1s2 = x + dt * f_H0s1 + dt_sqrt * g_H1s1
    # g_H1s2 = g(H1s2, t0, *args)
    all_H1s2 = []
    for var in variables:
      code_lines.append(f'  {var}_H1s2 = {var} + {vdt} * {var}_f_H0s1 + dt_sqrt * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    g_names = [f'{var}_g_H1s2' for var in variables]
    code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + parameters)})')
    code_lines.append('  ')

    # 2.4 final stage
    # ----
    # g1 = (I1 - I11 / dt_sqrt + I10 / dt)
    # g2 = I11 / dt_sqrt
    # y1 = x + dt * f_H0s1 + g1 * g_H1s1 + g2 * g_H1s2
    for var in variables:
      code_lines.append(f'  {var}_g1 = -{var}_I1 + {var}_I11/dt_sqrt + {var}_I10/{vdt}')
      code_lines.append(f'  {var}_g2 = {var}_I11 / dt_sqrt')
      code_lines.append(f'  {var}_new = {var} + {vdt} * {var}_f_H0s1 + '
                        f'{var}_g1 * {var}_g_H1s1 + {var}_g2 * {var}_g_H1s2')
      code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in variables]
    code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    return common.compile_and_assign_attrs(
      code_lines=code_lines, code_scope=code_scope, show_code=show_code,
      variables=variables, parameters=parameters, func_name=func_name,
      intg_type=intg_type, var_type=var_type, wiener_type=wiener_type, dt=dt)

  @staticmethod
  def wrap(wrapper, f, g, dt, intg_type, var_type, wiener_type, show_code):
    """The base function to format a SRK method.

    Parameters
    ----------
    f : callable
        The drift function of the SDE_INT.
    g : callable
        The diffusion function of the SDE_INT.
    dt : float
        The numerical precision.
    intg_type : str
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

    var_type = constants.POP_VAR if var_type is None else var_type
    intg_type = constants.ITO_SDE if intg_type is None else intg_type
    wiener_type = constants.SCALAR_WIENER if wiener_type is None else wiener_type
    if var_type not in constants.SUPPORTED_VAR_TYPE:
      raise errors.IntegratorError(f'Currently, BrainPy only supports variable types: '
                                   f'{constants.SUPPORTED_VAR_TYPE}. But we got {var_type}.')
    if intg_type != constants.ITO_SDE:
      raise errors.IntegratorError(f'SRK method for SDEs with scalar noise only supports Ito SDE_INT type, '
                                   f'but we got {intg_type} integral.')
    if wiener_type != constants.SCALAR_WIENER:
      raise errors.IntegratorError(f'SRK method for SDEs with scalar noise only supports scalar '
                                   f'Wiener Process, but we got "{wiener_type}" noise.')

    show_code = False if show_code is None else show_code
    dt = math.get_dt() if dt is None else dt

    if f is not None and g is not None:
      return wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=intg_type,
                     var_type=var_type, wiener_type=wiener_type)

    elif f is not None:
      return lambda g: wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=intg_type,
                               var_type=var_type, wiener_type=wiener_type)

    elif g is not None:
      return lambda f: wrapper(f=f, g=g, dt=dt, show_code=show_code, sde_type=intg_type,
                               var_type=var_type, wiener_type=wiener_type)

    else:
      raise ValueError('Must provide "f" or "g".')


def srk1w1_scalar(f=None, g=None, dt=None, intg_type=None, var_type=None, wiener_type=None, show_code=None):
  """Order 2.0 weak SRK methods for SDEs with scalar Wiener process.

  This method has have strong orders :backend:`(p_d, p_s) = (2.0,1.5)`.

  The Butcher table is:

  .. math::

      \\begin{array}{l|llll|llll|llll}
          0   &&&&&  &&&&  &&&& \\\\
          3/4 &3/4&&&& 3/2&&& &&&& \\\\
          0   &0&0&0&& 0&0&0&& &&&&\\\\
          \\hline
          0 \\\\
          1/4 & 1/4&&& & 1/2&&&\\\\
          1 & 1&0&&& -1&0&\\\\
          1/4& 0&0&1/4&&  -5&3&1/2\\\\
          \\hline
          & 1/3& 2/3& 0 & 0 & -1 & 4/3 & 2/3&0 & -1 &4/3 &-1/3 &0 \\\\
          \\hline
          & &&&& 2 &-4/3 & -2/3 & 0 & -2 & 5/3 & -2/3 & 1
      \\end{array}


  References
  ----------

  .. [1] Rößler, Andreas. "Strong and weak approximation methods for stochastic differential
          equations—some recent developments." Recent developments in applied probability and
          statistics. Physica-Verlag HD, 2010. 127-153.
  .. [2] Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of
          stochastic differential equations." SIAM Journal on Numerical Analysis 48.3
          (2010): 922-952.

  """
  return Wrappers.wrap(Wrappers.srk1w1, f=f, g=g, dt=dt, intg_type=intg_type, var_type=var_type,
                       wiener_type=wiener_type, show_code=show_code)


def srk2w1_scalar(f=None, g=None, dt=None, intg_type=None, var_type=None, wiener_type=None, show_code=None):
  """Order 1.5 Strong SRK Methods for SDEs witdt Scalar Noise.

  This method has have strong orders :backend:`(p_d, p_s) = (3.0,1.5)`.

  The Butcher table is:

  .. math::

      \\begin{array}{c|cccc|cccc|ccc|}
          0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\\\
          1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\\\
          1 / 2 & 1 / 4 & 1 / 4 & 0 & 0 & 1 & 1 / 2 & 0 & 0 & & & & \\\\
          0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\\\
          \\hline 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\\\
          1 / 4 & 1 / 4 & 0 & 0 & 0 & -1 / 2 & 0 & 0 & 0 & & & & \\\\
          1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & & & & \\\\
          1 / 4 & 0 & 0 & 1 / 4 & 0 & 2 & -1 & 1 / 2 & 0 & & & & \\\\
          \\hline & 1 / 6 & 1 / 6 & 2 / 3 & 0 & -1 & 4 / 3 & 2 / 3 & 0 & -1 & -4 / 3 & 1 / 3 & 0 \\\\
          \\hline & & & & &2 & -4 / 3 & -2 / 3 & 0 & -2 & 5 / 3 & -2 / 3 & 1
      \\end{array}


  References
  ----------

  [1] Rößler, Andreas. "Strong and weak approximation methods for stochastic differential
      equations—some recent developments." Recent developments in applied probability and
      statistics. Physica-Verlag HD, 2010. 127-153.
  [2] Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of
      stochastic differential equations." SIAM Journal on Numerical Analysis 48.3
      (2010): 922-952.
  """
  return Wrappers.wrap(Wrappers.srk2w1, f=f, g=g, dt=dt, intg_type=intg_type, var_type=var_type,
                       wiener_type=wiener_type, show_code=show_code)


def KlPl_scalar(f=None, g=None, dt=None, intg_type=None, var_type=None, wiener_type=None, show_code=None):
  """Order 1.0 Strong SRK Methods for SDEs with Scalar Noise.

  This method has have orders :backend:`p_s = 1.0`.

  The Butcher table is:

  .. math::

      \\begin{array}{c|cc|cc|cc|c}
          0 & 0 & 0 & 0 & 0 & & \\\\
          0 & 0 & 0 & 0 & 0 & & \\\\
          \\hline 0 & 0 & 0 & 0 & 0 & & \\\\
          0 & 1 & 0 & 1 & 0 & & \\\\
          \\hline 0 & 1 & 0 & 1 & 0 & -1 & 1 \\\\
          \\hline & & & 1 & 0 & 0 & 0
      \\end{array}

  References
  ----------

  [1] P. E. Kloeden, E. Platen, Numerical Solution of Stochastic Differential
      Equations, 2nd Edition, Springer, Berlin Heidelberg New York, 1995.
  """
  return Wrappers.wrap(Wrappers.KlPl, f=f, g=g, dt=dt, intg_type=intg_type,
                       var_type=var_type, wiener_type=wiener_type, show_code=show_code)
