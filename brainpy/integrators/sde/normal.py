# -*- coding: utf-8 -*-

from typing import Union, Callable, Dict, Sequence

from brainpy import errors, math as bm
from brainpy.base import Collector
from brainpy.integrators import constants, utils, joint_eq
from brainpy.integrators.sde.base import SDEIntegrator
from .generic import register_sde_integrator

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

  def __init__(
      self,
      f: Callable,
      g: Callable,
      dt: float = None,
      name: str = None,
      show_code: bool = False,
      var_type: str = None,
      intg_type: str = None,
      wiener_type: str = None,
      dyn_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      state_delays: Dict[str, bm.AbstractDelay] = None
  ):
    super(ExponentialEuler, self).__init__(f=f,
                                           g=g,
                                           dt=dt,
                                           show_code=show_code,
                                           name=name,
                                           var_type=var_type,
                                           intg_type=intg_type,
                                           wiener_type=wiener_type,
                                           state_delays=state_delays)

    if self.intg_type == constants.STRA_SDE:
      raise NotImplementedError(f'{self.__class__.__name__} does not support integral type of {constants.STRA_SDE}. '
                                f'It only supports {constants.ITO_SDE} now. ')
    self.dyn_vars = dyn_vars

    # build the integrator
    self.code_lines = []
    self.code_scope = {}
    self.integral = self.build()

  def build(self):
    all_vars, all_pars = [], []
    integrals, arg_names = [], []
    a = self._build_integrator(self.f)
    for integral, vars, _ in a:
      integrals.append(integral)
      for var in vars:
        if var not in all_vars:
          all_vars.append(var)
    for _, vars, pars in a:
      for par in pars:
        if (par not in all_vars) and (par not in all_pars):
          all_pars.append(par)
      arg_names.append(vars + pars + ['dt'])
    all_pars.append('dt')
    all_vps = all_vars + all_pars

    def integral_func(*args, **kwargs):
      # format arguments
      params_in = Collector()
      for i, arg in enumerate(args):
        params_in[all_vps[i]] = arg
      params_in.update(kwargs)
      dt = params_in.pop('dt', self.dt)

      # diffusion part
      noises = self.g(**params_in)

      # call integrals
      results = []
      params_in['dt'] = dt
      for i, int_fun in enumerate(integrals):
        _key = arg_names[i][0]
        r = int_fun(params_in[_key], **{arg: params_in[arg] for arg in arg_names[i][1:] if arg in params_in})
        if self.wiener_type == constants.SCALAR_WIENER:
          n = noises[i]
        else:
          if bm.ndim(noises[i]) != bm.ndim(r) + 1:
            raise ValueError(f'The dimension of the noise does not match when setting {constants.VECTOR_WIENER}. '
                             f'We got the dimension of noise {bm.ndim(noises[i])}, but we expect {bm.ndim(r) + 1}.')
          n = bm.sum(noises[i], axis=0)
        n = n * self.rng.randn(*bm.shape(r)) * bm.sqrt(params_in['dt'])
        results.append(r + n)
      return results if isinstance(self.f, joint_eq.JointEq) else results[0]

    return integral_func

  def _build_integrator(self, f):
    if isinstance(f, joint_eq.JointEq):
      results = []
      for sub_eq in f.eqs:
        results.extend(self._build_integrator(sub_eq))
      return results

    else:
      vars, pars, _ = utils.get_args(f)

      # checking
      if len(vars) != 1:
        raise errors.DiffEqError(constants.exp_error_msg.format(cls=self.__class__.__name__,
                                                                vars=str(vars),
                                                                eq=str(f)))

      # gradient function
      value_and_grad = bm.vector_grad(f, argnums=0, dyn_vars=self.dyn_vars, return_value=True)

      # integration function
      def integral(*args, **kwargs):
        assert len(args) > 0
        dt = kwargs.pop('dt', self.dt)
        linear, derivative = value_and_grad(*args, **kwargs)
        phi = bm.where(linear == 0., bm.ones_like(linear), (bm.exp(dt * linear) - 1) / (dt * linear))
        return args[0] + dt * phi * derivative

      return [(integral, vars, pars), ]


register_sde_integrator('exponential_euler', ExponentialEuler)
register_sde_integrator('exp_euler', ExponentialEuler)
register_sde_integrator('exp_euler_auto', ExponentialEuler)
register_sde_integrator('exp_auto', ExponentialEuler)
