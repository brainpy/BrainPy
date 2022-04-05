# -*- coding: utf-8 -*-

import inspect

from brainpy import errors, math
from brainpy.integrators import constants, utils
from brainpy.integrators.analysis_by_ast import separate_variables
from brainpy.integrators.sde.base import SDEIntegrator
from .generic import register_sde_integrator

try:
  import sympy
  from brainpy.integrators import analysis_by_sympy
except (ModuleNotFoundError, ImportError):
  sympy = analysis_by_sympy = None

__all__ = [
  'Euler',
  'Heun',
  'Milstein',
  'ExponentialEuler',
]


def df_and_dg(code_lines, variables, parameters):
  # 1. df
  # df = f(x, t, *args)
  all_df = [f'{var}_df' for var in variables]
  code_lines.append(f'  {", ".join(all_df)} = f({", ".join(variables + parameters)})')

  # 2. dg
  # dg = g(x, t, *args)
  all_dg = [f'{var}_dg' for var in variables]
  code_lines.append(f'  {", ".join(all_dg)} = g({", ".join(variables + parameters)})')
  code_lines.append('  ')


def dfdt(code_lines, variables):
  for var in variables:
    code_lines.append(f'  {var}_dfdt = {var}_df * {constants.DT}')
  code_lines.append('  ')


def noise_terms(code_lines, variables):
  # num_vars = len(variables)
  # if num_vars > 1:
  #     code_lines.append(f'  all_dW = math.normal(0.0, dt_sqrt, ({num_vars},)+math.shape({variables[0]}_dg))')
  #     for i, var in enumerate(variables):
  #         code_lines.append(f'  {var}_dW = all_dW[{i}]')
  # else:
  #     var = variables[0]
  #     code_lines.append(f'  {var}_dW = math.normal(0.0, dt_sqrt, math.shape({var}))')
  # code_lines.append('  ')

  for var in variables:
    code_lines.append(f'  {var}_dW = random.normal(0.000, dt_sqrt, math.shape({var})).value')
  code_lines.append('  ')


class Euler(SDEIntegrator):
  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None,
               state_delays=None):
    super(Euler, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                                var_type=var_type, intg_type=intg_type,
                                wiener_type=wiener_type, state_delays=state_delays)
    self.build()

  def build(self):
    self.code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 df, dg
    df_and_dg(self.code_lines, self.variables, self.parameters)

    # 2.2 dfdt
    dfdt(self.code_lines, self.variables)

    # 2.3 dW
    noise_terms(self.code_lines, self.variables)

    # 2.3 dgdW
    # ----
    # SCALAR_WIENER : dg * dW
    # VECTOR_WIENER : math.sum(dg * dW, axis=-1)

    if self.wiener_type == constants.SCALAR_WIENER:
      for var in self.variables:
        self.code_lines.append(f'  {var}_dgdW = {var}_dg * {var}_dW')
    else:
      for var in self.variables:
        self.code_lines.append(f'  {var}_dgdW = math.sum({var}_dg * {var}_dW, axis=-1)')
    self.code_lines.append('  ')

    if self.intg_type == constants.ITO_SDE:
      # 2.4 new var
      # ----
      # y = x + dfdt + dgdW
      for var in self.variables:
        self.code_lines.append(f'  {var}_new = {var} + {var}_dfdt + {var}_dgdW')
      self.code_lines.append('  ')

    elif self.intg_type == constants.STRA_SDE:
      # 2.4  y_bar = x + math.sum(dgdW, axis=-1)
      all_bar = [f'{var}_bar' for var in self.variables]
      for var in self.variables:
        self.code_lines.append(f'  {var}_bar = {var} + {var}_dgdW')
      self.code_lines.append('  ')

      # 2.5  dg_bar = g(y_bar, t, *args)
      all_dg_bar = [f'{var}_dg_bar' for var in self.variables]
      self.code_lines.append(f'  {", ".join(all_dg_bar)} = g({", ".join(all_bar + self.parameters)})')

      # 2.6 dgdW2
      # ----
      # SCALAR_WIENER : dgdW2 = dg_bar * dW
      # VECTOR_WIENER : dgdW2 = math.sum(dg_bar * dW, axis=-1)
      if self.wiener_type == constants.SCALAR_WIENER:
        for var in self.variables:
          self.code_lines.append(f'  {var}_dgdW2 = {var}_dg_bar * {var}_dW')
      else:
        for var in self.variables:
          self.code_lines.append(f'  {var}_dgdW2 = math.sum({var}_dg_bar * {var}_dW, axis=-1)')
      self.code_lines.append('  ')

      # 2.7 new var
      # ----
      # y = x + dfdt + 0.5 * (dgdW + dgdW2)
      for var in self.variables:
        self.code_lines.append(f'  {var}_new = {var} + {var}_dfdt + 0.5 * ({var}_dgdW + {var}_dgdW2)')
      self.code_lines.append('  ')
    else:
      raise ValueError(f'Unknown SDE_INT type: {self.intg_type}. We only '
                       f'supports {constants.SUPPORTED_INTG_TYPE}.')

    # returns
    new_vars = [f'{var}_new' for var in self.variables]
    self.code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)


register_sde_integrator('euler', Euler)


class Heun(Euler):
  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None,
               state_delays=None):
    if intg_type != constants.STRA_SDE:
      raise errors.IntegratorError(f'Heun method only supports Stranovich integral of SDEs, '
                                   f'but we got {intg_type} integral.')
    super(Heun, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                               var_type=var_type, intg_type=intg_type,
                               wiener_type=wiener_type, state_delays=state_delays)
    self.build()


register_sde_integrator('heun', Heun)


class Milstein(SDEIntegrator):
  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None,
               state_delays=None):
    super(Milstein, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                                   var_type=var_type, intg_type=intg_type,
                                   wiener_type=wiener_type, state_delays=state_delays)
    self.build()

  def build(self):
    # 2. code lines
    self.code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 df, dg
    df_and_dg(self.code_lines, self.variables, self.parameters)

    # 2.2 dfdt
    dfdt(self.code_lines, self.variables)

    # 2.3 dW
    noise_terms(self.code_lines, self.variables)

    # 2.3 dgdW
    # ----
    # dg * dW
    for var in self.variables:
      self.code_lines.append(f'  {var}_dgdW = {var}_dg * {var}_dW')
    self.code_lines.append('  ')

    # 2.4  df_bar = x + dfdt + math.sum(dg * dt_sqrt, axis=-1)
    all_df_bar = [f'{var}_df_bar' for var in self.variables]
    if self.wiener_type == constants.SCALAR_WIENER:
      for var in self.variables:
        self.code_lines.append(f'  {var}_df_bar = {var} + {var}_dfdt + {var}_dg * {constants.DT}_sqrt')
    else:
      for var in self.variables:
        self.code_lines.append(f'  {var}_df_bar = {var} + {var}_dfdt + math.sum('
                               f'{var}_dg * {constants.DT}_sqrt, axis=-1)')

    # 2.5  dg_bar = g(y_bar, t, *args)
    all_dg_bar = [f'{var}_dg_bar' for var in self.variables]
    self.code_lines.append(f'  {", ".join(all_dg_bar)} = g({", ".join(all_df_bar + self.parameters)})')
    self.code_lines.append('  ')

    # 2.6 dgdW2
    # ----
    # dgdW2 = 0.5 * (dg_bar - dg) * (dW * dW / dt_sqrt - dt_sqrt)
    if self.intg_type == constants.ITO_SDE:
      for var in self.variables:
        self.code_lines.append(f'  {var}_dgdW2 = 0.5 * ({var}_dg_bar - {var}_dg) * '
                               f'({var}_dW * {var}_dW / {constants.DT}_sqrt - {constants.DT}_sqrt)')
    elif self.intg_type == constants.STRA_SDE:
      for var in self.variables:
        self.code_lines.append(f'  {var}_dgdW2 = 0.5 * ({var}_dg_bar - {var}_dg) * '
                               f'{var}_dW * {var}_dW / {constants.DT}_sqrt')
    else:
      raise ValueError(f'Unknown SDE_INT type: {self.intg_type}')
    self.code_lines.append('  ')

    # 2.7 new var
    # ----
    # SCALAR_WIENER : y = x + dfdt + dgdW + dgdW2
    # VECTOR_WIENER : y = x + dfdt + math.sum(dgdW + dgdW2, axis=-1)
    if self.wiener_type == constants.SCALAR_WIENER:
      for var in self.variables:
        self.code_lines.append(f'  {var}_new = {var} + {var}_dfdt + {var}_dgdW + {var}_dgdW2')
    elif self.wiener_type == constants.VECTOR_WIENER:
      for var in self.variables:
        self.code_lines.append(f'  {var}_new = {var} + {var}_dfdt + math.sum({var}_dgdW + {var}_dgdW2, axis=-1)')
    else:
      raise ValueError(f'Unknown Wiener Process : {self.wiener_type}')
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


register_sde_integrator('milstein', Milstein)


class ExponentialEuler(SDEIntegrator):
  r"""First order, explicit exponential Euler method.

  For a SDE equation of the form

  .. math::

      d y=(Ay+ F(y))dt + g(y)dW(t) = f(y)dt + g(y)dW(t), \quad y(0)=y_{0}

  its schema is given by [1]_

  .. math::

      y_{n+1} & =e^{\Delta t A}(y_{n}+ g(y_n)\Delta W_{n})+\varphi(\Delta t A) F(y_{n}) \Delta t \\
       &= y_n + \Delta t \varphi(\Delta t A) f(y) + e^{\Delta t A}g(y_n)\Delta W_{n}

  where :math:`\varphi(z)=\frac{e^{z}-1}{z}`.

  References
  ----------
  .. [1] Erdoğan, Utku, and Gabriel J. Lord. "A new class of exponential integrators for stochastic
         differential equations with multiplicative noise." arXiv preprint arXiv:1608.07096 (2016).
  """

  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None,
               state_delays=None):
    super(ExponentialEuler, self).__init__(f=f, g=g, dt=dt, show_code=show_code, name=name,
                                           var_type=var_type, intg_type=intg_type,
                                           wiener_type=wiener_type, state_delays=state_delays)
    self.build()

  def build(self):
    # if math.get_backend_name() == 'jax':
    #   raise NotImplementedError
    # else:
    self.symbolic_build()

  def autograd_build(self):
    pass

  def symbolic_build(self):
    if self.var_type == constants.SYSTEM_VAR:
      raise errors.IntegratorError(f'Exponential Euler method do not support {self.var_type} variable type.')
    if self.intg_type != constants.ITO_SDE:
      raise errors.IntegratorError(f'Exponential Euler method only supports Ito integral, but we got {self.intg_type}.')

    if sympy is None or analysis_by_sympy is None:
      raise errors.PackageMissingError('SymPy must be installed when '
                                       'using exponential integrators.')

    # check bound method
    if hasattr(self.derivative[constants.F], '__self__'):
      self.code_lines = [f'def {self.func_name}({", ".join(["self"] + list(self.arguments))}):']

    # 1. code scope
    closure_vars = inspect.getclosurevars(self.derivative[constants.F])
    self.code_scope.update(closure_vars.nonlocals)
    self.code_scope.update(dict(closure_vars.globals))
    self.code_scope['math'] = math

    # 2. code lines
    code_lines = self.code_lines
    # code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']
    code_lines.append(f'  {constants.DT}_sqrt = {constants.DT} ** 0.5')

    # 2.1 dg
    # dg = g(x, t, *args)
    all_dg = [f'{var}_dg' for var in self.variables]
    code_lines.append(f'  {", ".join(all_dg)} = g({", ".join(self.variables + self.parameters)})')
    code_lines.append('  ')

    # 2.2 dW
    noise_terms(code_lines, self.variables)

    # 2.3 dgdW
    # ----
    # SCALAR_WIENER : dg * dW
    # VECTOR_WIENER : math.sum(dg * dW, axis=-1)

    if self.wiener_type == constants.SCALAR_WIENER:
      for var in self.variables:
        code_lines.append(f'  {var}_dgdW = {var}_dg * {var}_dW')
    else:
      for var in self.variables:
        code_lines.append(f'  {var}_dgdW = math.sum({var}_dg * {var}_dW, axis=-1)')
    code_lines.append('  ')

    # 2.4 new var
    # ----
    analysis = separate_variables(self.derivative[constants.F])
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
      var_name = self.variables[vi]
      diff_eq = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                               variables=sd_variables,
                                               expressions=expressions,
                                               derivative_expr=key,
                                               scope=self.code_scope,
                                               func_name=self.func_name)

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
        code_lines.append(f'  {s_linear_exp.name} = math.exp({analysis_by_sympy.sympy2str(linear)} * {constants.DT})')
        # df part
        df_part = (s_linear_exp - 1) / s_linear * s_df
        code_lines.append(f'  {s_df_part.name} = {analysis_by_sympy.sympy2str(df_part)}')

      else:
        # linear exponential
        code_lines.append(f'  {s_linear_exp.name} = {constants.DT}_sqrt')
        # df part
        code_lines.append(f'  {s_df_part.name} = {s_df.name} * {constants.DT}')

      # update expression
      update = var + s_df_part

      # The actual update step
      code_lines.append(f'  {diff_eq.var_name}_new = {analysis_by_sympy.sympy2str(update)} + {var_name}_dgdW')
      code_lines.append('')

    # returns
    new_vars = [f'{var}_new' for var in self.variables]
    code_lines.append(f'  return {", ".join(new_vars)}')

    # return and compile
    self.integral = utils.compile_code(
      code_scope={k: v for k, v in self.code_scope.items()},
      code_lines=self.code_lines,
      show_code=self.show_code,
      func_name=self.func_name)

    if hasattr(self.derivative[constants.F], '__self__'):
      host = self.derivative[constants.F].__self__
      self.integral = self.integral.__get__(host, host.__class__)


register_sde_integrator('exponential_euler', ExponentialEuler)
register_sde_integrator('exp_euler', ExponentialEuler)
