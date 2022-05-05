# -*- coding: utf-8 -*-

from brainpy.integrators import constants, utils
from brainpy.integrators.sde.base import SDEIntegrator
from .generic import register_sde_integrator

__all__ = [
  'SRK1W1',
  'SRK2W1',
  'KlPl',
]


def _noise_terms(code_lines, variables, triple_integral=True):
  # num_vars = len(variables)
  # if num_vars > 1:
  #     code_lines.append(f'  all_I1 = math.normal(0.0, dt_sqrt, ({num_vars},)+math.shape({variables[0]}))')
  #     code_lines.append(f'  all_I0 = math.normal(0.0, dt_sqrt, ({num_vars},)+math.shape({variables[0]}))')
  #     code_lines.append(f'  all_I10 = 0.5 * {constants.DT} * (all_I1 + all_I0 / 3.0 ** 0.5)')
  #     code_lines.append(f'  all_I11 = 0.5 * (all_I1 ** 2 - {constants.DT})')
  #     if triple_integral:
  #         code_lines.append(f'  all_I111 = (all_I1 ** 3 - 3 * {constants.DT} * all_I1) / 6')
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
  #     code_lines.append(f'  {var}_I10 = 0.5 * {constants.DT} * ({var}_I1 + {var}_I0 / 3.0 ** 0.5)')
  #     code_lines.append(f'  {var}_I11 = 0.5 * ({var}_I1 ** 2 - {constants.DT})')
  #     if triple_integral:
  #         code_lines.append(f'  {var}_I111 = ({var}_I1 ** 3 - 3 * {constants.DT} * {var}_I1) / 6')
  #     code_lines.append('  ')

  for var in variables:
    code_lines.append(f'  {var}_I1 = dt_sqrt * random.randn(*math.shape({var}))')
    code_lines.append(f'  {var}_I0 = dt_sqrt * random.randn(*math.shape({var}))')
    code_lines.append(f'  {var}_I10 = 0.5 * {constants.DT} * ({var}_I1 + {var}_I0 / 3.0 ** 0.5)')
    code_lines.append(f'  {var}_I11 = 0.5 * ({var}_I1 ** 2 - {constants.DT})')
    if triple_integral:
      code_lines.append(f'  {var}_I111 = ({var}_I1 ** 3 - 3 * {constants.DT} * {var}_I1) / 6')
    code_lines.append('  ')


def _state1(code_lines, variables, parameters):
  f_names = [f'{var}_f_H0s1' for var in variables]
  g_names = [f'{var}_g_H1s1' for var in variables]
  code_lines.append(f'  {", ".join(f_names)} = f({", ".join(variables + parameters)})')
  code_lines.append(f'  {", ".join(g_names)} = g({", ".join(variables + parameters)})')
  code_lines.append('  ')


class SRK1W1(SDEIntegrator):
  r"""Order 2.0 weak SRK methods for SDEs with scalar Wiener process.

  This method has have strong orders :math:`(p_d, p_s) = (2.0,1.5)`.

  The Butcher table is:

  .. math::

      \begin{array}{l|llll|llll|llll}
          0   &&&&&  &&&&  &&&& \\
          3/4 &3/4&&&& 3/2&&& &&&& \\
          0   &0&0&0&& 0&0&0&& &&&&\\
          \hline
          0 \\
          1/4 & 1/4&&& & 1/2&&&\\
          1 & 1&0&&& -1&0&\\
          1/4& 0&0&1/4&&  -5&3&1/2\\
          \hline
          & 1/3& 2/3& 0 & 0 & -1 & 4/3 & 2/3&0 & -1 &4/3 &-1/3 &0 \\
          \hline
          & &&&& 2 &-4/3 & -2/3 & 0 & -2 & 5/3 & -2/3 & 1
      \end{array}


  References
  ----------

  .. [1] Rößler, Andreas. "Strong and weak approximation methods for stochastic differential
          equations—some recent developments." Recent developments in applied probability and
          statistics. Physica-Verlag HD, 2010. 127-153.
  .. [2] Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of
          stochastic differential equations." SIAM Journal on Numerical Analysis 48.3
          (2010): 922-952.

  """

  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None, state_delays=None):
    super(SRK1W1, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                                 var_type=var_type, intg_type=intg_type,
                                 wiener_type=wiener_type, state_delays=state_delays)
    assert self.wiener_type == constants.SCALAR_WIENER
    self.build()

  def build(self):
    # 2. code lines
    self.code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 noise
    _noise_terms(self.code_lines, self.variables, triple_integral=True)

    # 2.2 stage 1
    _state1(self.code_lines, self.variables, self.parameters)

    # 2.3 stage 2
    all_H0s2, all_H1s2 = [], []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H0s2 = {var} + {constants.DT} * 0.75 * {var}_f_H0s1 + '
                             f'1.5 * {var}_g_H1s1 * {var}_I10 / {constants.DT}')
      all_H0s2.append(f'{var}_H0s2')
      self.code_lines.append(f'  {var}_H1s2 = {var} + {constants.DT} * 0.25 * {var}_f_H0s1 + '
                             f'dt_sqrt * 0.5 * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    all_H0s2.append(f't + 0.75 * {constants.DT}')  # t
    all_H1s2.append(f't + 0.25 * {constants.DT}')  # t
    f_names = [f'{var}_f_H0s2' for var in self.variables]
    self.code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s2 + self.parameters[1:])})')
    g_names = [f'{var}_g_H1s2' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.4 state 3
    all_H1s3 = []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H1s3 = {var} + {constants.DT} * {var}_f_H0s1 - dt_sqrt * {var}_g_H1s1')
      all_H1s3.append(f'{var}_H1s3')
    all_H1s3.append(f't + {constants.DT}')  # t
    g_names = [f'{var}_g_H1s3' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s3 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.5 state 4
    all_H1s4 = []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H1s4 = {var} + 0.25 * {constants.DT} * {var}_f_H0s1 + dt_sqrt * '
                             f'(-5 * {var}_g_H1s1 + 3 * {var}_g_H1s2 + 0.5 * {var}_g_H1s3)')
      all_H1s4.append(f'{var}_H1s4')
    all_H1s4.append(f't + 0.25 * {constants.DT}')  # t
    g_names = [f'{var}_g_H1s4' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s4 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.6 final stage
    for var in self.variables:
      self.code_lines.append(f'  {var}_f1 = {var}_f_H0s1/3 + {var}_f_H0s2 * 2/3')
      self.code_lines.append(
        f'  {var}_g1 = -{var}_I1 - {var}_I11/dt_sqrt + 2 * {var}_I10/{constants.DT} - 2 * {var}_I111/{constants.DT}')
      self.code_lines.append(f'  {var}_g2 = {var}_I1 * 4/3 + {var}_I11 / dt_sqrt * 4/3 - '
                             f'{var}_I10 / {constants.DT} * 4/3 + {var}_I111 / {constants.DT} * 5/3')
      self.code_lines.append(f'  {var}_g3 = {var}_I1 * 2/3 - {var}_I11/dt_sqrt/3 - '
                             f'{var}_I10 / {constants.DT} * 2/3 - {var}_I111 / {constants.DT} * 2/3')
      self.code_lines.append(f'  {var}_g4 = {var}_I111 / {constants.DT}')
      self.code_lines.append(f'  {var}_new = {var} + {constants.DT} * {var}_f1 + {var}_g1 * {var}_g_H1s1 + '
                             f'{var}_g2 * {var}_g_H1s2 + {var}_g3 * {var}_g_H1s3 + {var}_g4 * {var}_g_H1s4')
      self.code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in self.variables]
    self.code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


register_sde_integrator('srk1w1', SRK1W1)


class SRK2W1(SDEIntegrator):
  r"""Order 1.5 Strong SRK Methods for SDEs with Scalar Noise.

  This method has have strong orders :math:`(p_d, p_s) = (3.0,1.5)`.

  The Butcher table is:

  .. math::

      \begin{array}{c|cccc|cccc|ccc|}
          0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\
          1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\
          1 / 2 & 1 / 4 & 1 / 4 & 0 & 0 & 1 & 1 / 2 & 0 & 0 & & & & \\
          0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\
          \hline 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & & & & \\
          1 / 4 & 1 / 4 & 0 & 0 & 0 & -1 / 2 & 0 & 0 & 0 & & & & \\
          1 & 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & & & & \\
          1 / 4 & 0 & 0 & 1 / 4 & 0 & 2 & -1 & 1 / 2 & 0 & & & & \\
          \hline & 1 / 6 & 1 / 6 & 2 / 3 & 0 & -1 & 4 / 3 & 2 / 3 & 0 & -1 & -4 / 3 & 1 / 3 & 0 \\
          \hline & & & & &2 & -4 / 3 & -2 / 3 & 0 & -2 & 5 / 3 & -2 / 3 & 1
      \end{array}


  References
  ----------

  .. [1] Rößler, Andreas. "Strong and weak approximation methods for stochastic differential
         equations—some recent developments." Recent developments in applied probability and
         statistics. Physica-Verlag HD, 2010. 127-153.
  .. [2] Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of
         stochastic differential equations." SIAM Journal on Numerical Analysis 48.3
         (2010): 922-952.
  """

  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None, state_delays=None):
    super(SRK2W1, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                                 var_type=var_type, intg_type=intg_type,
                                 wiener_type=wiener_type, state_delays=state_delays)
    assert self.wiener_type == constants.SCALAR_WIENER
    self.build()

  def build(self):
    self.code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 noise
    _noise_terms(self.code_lines, self.variables, triple_integral=True)

    # 2.2 stage 1
    _state1(self.code_lines, self.variables, self.parameters)

    # 2.3 stage 2
    # ----
    # H0s2 = x + dt * f_H0s1
    # H1s2 = x + dt * 0.25 * f_H0s1 - dt_sqrt * 0.5 * g_H1s1
    # f_H0s2 = f(H0s2, t + dt, *args)
    # g_H1s2 = g(H1s2, t + 0.25 * dt, *args)
    all_H0s2, all_H1s2 = [], []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H0s2 = {var} + {constants.DT} * {var}_f_H0s1')
      all_H0s2.append(f'{var}_H0s2')
      self.code_lines.append(f'  {var}_H1s2 = {var} + {constants.DT} * 0.25 * {var}_f_H0s1 - '
                             f'dt_sqrt * 0.5 * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    all_H0s2.append(f't + {constants.DT}')  # t
    all_H1s2.append(f't + 0.25 * {constants.DT}')  # t
    f_names = [f'{var}_f_H0s2' for var in self.variables]
    self.code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s2 + self.parameters[1:])})')
    g_names = [f'{var}_g_H1s2' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.4 state 3
    # ---
    # H0s3 = x + dt * (0.25 * f_H0s1 + 0.25 * f_H0s2) + (g_H1s1 + 0.5 * g_H1s2) * I10 / dt
    # H1s3 = x + dt * f_H0s1 + dt_sqrt * g_H1s1
    # f_H0s3 = g(H0s3, t + 0.5 * dt, *args)
    # g_H1s3 = g(H1s3, t + dt, *args)
    all_H0s3, all_H1s3 = [], []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H0s3 = {var} + {constants.DT} * (0.25 * {var}_f_H0s1 + 0.25 * {var}_f_H0s2) + '
                             f'({var}_g_H1s1 + 0.5 * {var}_g_H1s2) * {var}_I10 / {constants.DT}')
      all_H0s3.append(f'{var}_H0s3')
      self.code_lines.append(f'  {var}_H1s3 = {var} + {constants.DT} * {var}_f_H0s1 + dt_sqrt * {var}_g_H1s1')
      all_H1s3.append(f'{var}_H1s3')
    all_H0s3.append(f't + 0.5 * {constants.DT}')  # t
    all_H1s3.append(f't + {constants.DT}')  # t
    f_names = [f'{var}_f_H0s3' for var in self.variables]
    g_names = [f'{var}_g_H1s3' for var in self.variables]
    self.code_lines.append(f'  {", ".join(f_names)} = f({", ".join(all_H0s3 + self.parameters[1:])})')
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s3 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.5 state 4
    # ----
    # H1s4 = x + dt * 0.25 * f_H0s3 + dt_sqrt * (2 * g_H1s1 - g_H1s2 + 0.5 * g_H1s3)
    # g_H1s4 = g(H1s4, t + 0.25 * dt, *args)
    all_H1s4 = []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H1s4 = {var} + 0.25 * {constants.DT} * {var}_f_H0s1 + dt_sqrt * '
                             f'(2 * {var}_g_H1s1 - {var}_g_H1s2 + 0.5 * {var}_g_H1s3)')
      all_H1s4.append(f'{var}_H1s4')
    all_H1s4.append(f't + 0.25 * {constants.DT}')  # t
    g_names = [f'{var}_g_H1s4' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s4 + self.parameters[1:])})')
    self.code_lines.append('  ')

    # 2.6 final stage
    # ----
    # f1 = f_H0s1 / 6 + f_H0s2 / 6 + f_H0s3 * 2 / 3
    # g1 = - I1 + I11 / dt_sqrt + 2 * I10 / dt - 2 * I111 / dt
    # g2 = I1 * 4 / 3 - I11 / dt_sqrt * 4 / 3 - I10 / dt * 4 / 3 + I111 / dt * 5 / 3
    # g3 = I1 * 2 / 3 + I11 / dt_sqrt / 3 - I10 / dt * 2 / 3 - I111 / dt * 2 / 3
    # g4 = I111 / dt
    # y1 = x + dt * f1 + g1 * g_H1s1 + g2 * g_H1s2 + g3 * g_H1s3 + g4 * g_H1s4
    for var in self.variables:
      self.code_lines.append(f'  {var}_f1 = {var}_f_H0s1/6 + {var}_f_H0s2/6 + {var}_f_H0s3*2/3')
      self.code_lines.append(
        f'  {var}_g1 = -{var}_I1 + {var}_I11/dt_sqrt + 2 * {var}_I10/{constants.DT} - 2 * {var}_I111/{constants.DT}')
      self.code_lines.append(f'  {var}_g2 = {var}_I1 * 4/3 - {var}_I11 / dt_sqrt * 4/3 - '
                             f'{var}_I10 / {constants.DT} * 4/3 + {var}_I111 / {constants.DT} * 5/3')
      self.code_lines.append(f'  {var}_g3 = {var}_I1 * 2/3 + {var}_I11/dt_sqrt/3 - '
                             f'{var}_I10 / {constants.DT} * 2/3 - {var}_I111 / {constants.DT} * 2/3')
      self.code_lines.append(f'  {var}_g4 = {var}_I111 / {constants.DT}')
      self.code_lines.append(f'  {var}_new = {var} + {constants.DT} * {var}_f1 + {var}_g1 * {var}_g_H1s1 + '
                             f'{var}_g2 * {var}_g_H1s2 + {var}_g3 * {var}_g_H1s3 + {var}_g4 * {var}_g_H1s4')
      self.code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in self.variables]
    self.code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


register_sde_integrator('srk2w1', SRK2W1)


class KlPl(SDEIntegrator):
  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None, state_delays=None):
    super(KlPl, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                               var_type=var_type, intg_type=intg_type,
                               wiener_type=wiener_type, state_delays=state_delays)
    assert self.wiener_type == constants.SCALAR_WIENER
    self.build()

  def build(self):
    self.code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 noise
    _noise_terms(self.code_lines, self.variables, triple_integral=False)

    # 2.2 stage 1
    _state1(self.code_lines, self.variables, self.parameters)

    # 2.3 stage 2
    # ----
    # H1s2 = x + dt * f_H0s1 + dt_sqrt * g_H1s1
    # g_H1s2 = g(H1s2, t0, *args)
    all_H1s2 = []
    for var in self.variables:
      self.code_lines.append(f'  {var}_H1s2 = {var} + {constants.DT} * {var}_f_H0s1 + dt_sqrt * {var}_g_H1s1')
      all_H1s2.append(f'{var}_H1s2')
    g_names = [f'{var}_g_H1s2' for var in self.variables]
    self.code_lines.append(f'  {", ".join(g_names)} = g({", ".join(all_H1s2 + self.parameters)})')
    self.code_lines.append('  ')

    # 2.4 final stage
    # ----
    # g1 = (I1 - I11 / dt_sqrt + I10 / dt)
    # g2 = I11 / dt_sqrt
    # y1 = x + dt * f_H0s1 + g1 * g_H1s1 + g2 * g_H1s2
    for var in self.variables:
      self.code_lines.append(f'  {var}_g1 = -{var}_I1 + {var}_I11/dt_sqrt + {var}_I10/{constants.DT}')
      self.code_lines.append(f'  {var}_g2 = {var}_I11 / dt_sqrt')
      self.code_lines.append(f'  {var}_new = {var} + {constants.DT} * {var}_f_H0s1 + '
                             f'{var}_g1 * {var}_g_H1s1 + {var}_g2 * {var}_g_H1s2')
      self.code_lines.append('  ')

    # returns
    new_vars = [f'{var}_new' for var in self.variables]
    self.code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


register_sde_integrator('klpl', KlPl)
