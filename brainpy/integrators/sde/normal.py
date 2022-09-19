# -*- coding: utf-8 -*-

from typing import Union, Callable, Dict, Sequence

import jax.numpy as jnp

from brainpy import errors, math as bm
from brainpy.base import Collector
from brainpy.integrators import constants, utils, joint_eq
from brainpy.integrators.sde.base import SDEIntegrator
from .generic import register_sde_integrator
from brainpy.integrators.utils import format_args
from brainpy.integrators.constants import DT

__all__ = [
  'Euler',
  'Heun',
  'Milstein',
  'MilsteinGradFree',
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
  for var in variables:
    code_lines.append(f'  if {var}_dg is not None:')
    code_lines.append(f'    {var}_dW = random.normal(0.000, dt_sqrt, math.shape({var})).value')
  code_lines.append('  ')


class Euler(SDEIntegrator):
  r"""Euler method for the Ito and Stratonovich integrals.

  For Ito schema, the Euler method (also called as Euler-Maruyama method) is given by:
  
  .. math::
      
     \begin{aligned}
      Y_{n+1} &=Y_{n}+f\left(Y_{n}\right) h_{n}+g\left(Y_{n}\right) \Delta W_{n} \\
      \Delta W_{n} &=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
      \end{aligned}

  As the order of convergence for the Euler-Maruyama method is low (strong
  order of convergence 0.5, weak order of convergence 1), the numerical results
  are inaccurate unless a small step size is used. In fact, Euler-Maruyama
  represents the order 0.5 strong Taylor scheme.

  For Stratonovich scheme, the Euler-Heun method has to be used instead of the Euler-Maruyama method

  .. math::

     \begin{aligned}
      Y_{n+1} &=Y_{n}+f_{n} h+\frac{1}{2}\left[g_{n}+g\left(\bar{Y}_{n}\right)\right] \Delta W_{n} \\
      \bar{Y}_{n} &=Y_{n}+g_{n} \Delta W_{n} \\
      \Delta W_{n} &=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
      \end{aligned}


  See Also
  --------
  Heun

  """

  def __init__(
      self, f, g, dt=None, name=None, show_code=False,
      var_type=None, intg_type=None, wiener_type=None,
      state_delays=None, dyn_vars=None
  ):
    super(Euler, self).__init__(f=f, g=g, dt=dt, name=name,
                                var_type=var_type, intg_type=intg_type,
                                wiener_type=wiener_type,
                                state_delays=state_delays,
                                dyn_vars=dyn_vars)

    self.set_integral(self.step)

  def step(self, *args, **kwargs):
    all_args = format_args(args, kwargs, self.arg_names)
    dt = all_args.pop(DT, self.dt)

    # drift values
    drifts = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(drifts, (bm.ndarray, jnp.ndarray)):
        raise ValueError('Drift values must be a tensor when there '
                         'is only one variable in the equation.')
      drifts = {self.variables[0]: drifts}
    else:
      if not isinstance(drifts, (tuple, list)):
        raise ValueError('Drift values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      drifts = {var: drifts[i] for i, var in enumerate(self.variables)}

    # diffusion values
    diffusions = self.g(**all_args)
    if len(self.variables) == 1:
      # if not isinstance(diffusions, (bm.ndarray, jnp.ndarray)):
      #   raise ValueError('Diffusion values must be a tensor when there '
      #                    'is only one variable in the equation.')
      diffusions = {self.variables[0]: diffusions}
    else:
      if not isinstance(diffusions, (tuple, list)):
        raise ValueError('Diffusion values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      diffusions = {var: diffusions[i] for i, var in enumerate(self.variables)}
    if self.wiener_type == constants.VECTOR_WIENER:
      for key, val in diffusions.items():
        if val is not None and bm.ndim(val) == 0:
          raise ValueError(f"{constants.VECTOR_WIENER} wiener process needs multiple "
                           f"dimensional diffusion value. But we got a scale value for "
                           f"variable {key}.")

    # integral results
    integrals = []
    if self.intg_type == constants.ITO_SDE:
      for key in self.variables:
        integral = all_args[key] + drifts[key] * dt
        if diffusions[key] is not None:
          shape = bm.shape(all_args[key])
          if self.wiener_type == constants.SCALAR_WIENER:
            integral += diffusions[key] * self.rng.randn(*shape) * bm.sqrt(dt)
          else:
            shape += bm.shape(diffusions[key])[-1:]
            integral += bm.sum(diffusions[key] * self.rng.randn(*shape), axis=-1) * bm.sqrt(dt)
        integrals.append(integral)

    else:
      # \bar{Y}_{n}=Y_{n}+g_{n} \Delta W_{n}
      all_args_bar = {key: val for key, val in all_args.items()}
      all_noises = {}
      for key in self.variables:
        if diffusions[key] is None:
          all_args_bar[key] = all_args[key]
        else:
          shape = bm.shape(all_args[key])
          if self.wiener_type == constants.VECTOR_WIENER:
            noise_shape = bm.shape(diffusions[key])
            self._check_vector_wiener_dim(noise_shape, shape)
            shape += noise_shape[-1:]
          noise = self.rng.randn(*shape)
          all_noises[key] = noise * bm.sqrt(dt)
          if self.wiener_type == constants.VECTOR_WIENER:
            y_bar = all_args[key] + bm.sum(diffusions[key] * noise, axis=-1)
          else:
            y_bar = all_args[key] + diffusions[key] * noise
          all_args_bar[key] = y_bar
      # g(\bar{Y}_{n})
      diffusion_bars = self.g(**all_args_bar)
      if len(self.variables) == 1:
        diffusion_bars = {self.variables[0]: diffusion_bars}
      else:
        diffusion_bars = {var: diffusion_bars[i] for i, var in enumerate(self.variables)}
      # Y_{n+1}=Y_{n}+f_{n} h+\frac{1}{2}\left[g_{n}+g\left(\bar{Y}_{n}\right)\right] \Delta W_{n}
      for key in self.variables:
        integral = all_args[key] + drifts[key] * dt
        if diffusion_bars[key] is not None:
          integral += (diffusions[key] + diffusion_bars[key]) / 2 * all_noises[key]
        integrals.append(integral)

    # return integrals
    if len(self.variables) == 1:
      return integrals[0]
    else:
      return integrals


register_sde_integrator('euler', Euler)


class Heun(Euler):
  r"""The Euler-Heun method for Stratonovich integral scheme.

  Its mathematical expression is given by

  .. math::

   \begin{aligned}
    Y_{n+1} &=Y_{n}+f_{n} h+\frac{1}{2}\left[g_{n}+g\left(\bar{Y}_{n}\right)\right] \Delta W_{n} \\
    \bar{Y}_{n} &=Y_{n}+g_{n} \Delta W_{n} \\
    \Delta W_{n} &=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
    \end{aligned}


  See Also
  --------
  Euler

  """

  def __init__(self, f, g, dt=None, name=None, show_code=False,
               var_type=None, intg_type=None, wiener_type=None,
               state_delays=None, dyn_vars=None):
    if intg_type != constants.STRA_SDE:
      raise errors.IntegratorError(f'Heun method only supports Stranovich '
                                   f'integral of SDEs, but we got {intg_type} integral.')
    super(Heun, self).__init__(f=f, g=g, dt=dt, name=name,
                               var_type=var_type, intg_type=intg_type,
                               wiener_type=wiener_type, state_delays=state_delays,
                               dyn_vars=dyn_vars)


register_sde_integrator('heun', Heun)


class Milstein(SDEIntegrator):
  r"""Milstein method for Ito or Stratonovich integrals.

  The Milstein scheme represents the order 1.0 strong Taylor scheme. For the Ito integral,

  .. math::

     \begin{aligned}
      &Y_{n+1}=Y_{n}+f_{n} h+g_{n} \Delta W_{n}+\frac{1}{2} g_{n} g_{n}^{\prime}\left[\left(\Delta W_{n}\right)^{2}-h\right] \\
      &\Delta W_{n}=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
      \end{aligned}

  where :math:`g_{n}^{\prime}=\frac{d g\left(Y_{n}\right)}{d Y_{n}}` is the first derivative of :math:`g_n`.


  For the Stratonovich integral, the Milstein method is given by

  .. math::

     \begin{aligned}
     &Y_{n+1}=Y_{n}+f_{n} h+g_{n} \Delta W_{n}+\frac{1}{2} g_{n} g_{n}^{\prime}\left(\Delta W_{n}\right)^{2} \\
     &\Delta W_{n}=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
     \end{aligned}

  """

  def __init__(
      self,
      f: Callable,
      g: Callable,
      dt: float = None,
      name: str = None,
      show_code=False,
      var_type: str = None,
      intg_type: str = None,
      wiener_type: str = None,
      state_delays: Dict[str, bm.AbstractDelay] = None,
      dyn_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
  ):
    super(Milstein, self).__init__(f=f,
                                   g=g,
                                   dt=dt,
                                   name=name,
                                   var_type=var_type,
                                   intg_type=intg_type,
                                   wiener_type=wiener_type,
                                   state_delays=state_delays,
                                   dyn_vars=dyn_vars)
    self.set_integral(self.step)

  def _get_g_grad(self, f, allow_raise=False, need_grad=True):
    if isinstance(f, joint_eq.JointEq):
      results = []
      state = True
      for sub_eq in f.eqs:
        r, r_state = self._get_g_grad(sub_eq, allow_raise, need_grad)
        results.extend(r)
        state &= r_state
      return results, state
    else:
      res = [None, None, None]
      state = True
      try:
        vars, pars, _ = utils.get_args(f)
        if len(vars) != 1:
          raise errors.DiffEqError(constants.multi_vars_msg.format(cls=self.__class__.__name__,
                                                                   vars=str(vars), eq=str(f)))
        res[1] = vars
        res[2] = pars
      except errors.DiffEqError as e:
        state = False
        if not allow_raise:
          raise e
      if need_grad:
        res[0] = bm.vector_grad(f, argnums=0, dyn_vars=self.dyn_vars)
      return [tuple(res)], state

  def step(self, *args, **kwargs):
    # parse grad function and individual arguments
    parses, state = self._get_g_grad(self.g, allow_raise=False, need_grad=True)
    if not state:
      parses2 = self._get_g_grad(self.f, allow_raise=True, need_grad=False)
      if len(parses2) != len(parses):
        raise ValueError(f'"f" and "g" should defined with JointEq both, and should '
                         f'keep the same structure.')
      parses = [a[:1] + b[1:] for a, b in zip(parses, parses2)]

    # input arguments
    all_args = format_args(args, kwargs, self.arg_names)
    dt = all_args.pop(DT, self.dt)

    # drift values
    drifts = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(drifts, (bm.ndarray, jnp.ndarray)):
        raise ValueError('Drift values must be a tensor when there '
                         'is only one variable in the equation.')
      drifts = {self.variables[0]: drifts}
    else:
      if not isinstance(drifts, (tuple, list)):
        raise ValueError('Drift values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      drifts = {var: drifts[i] for i, var in enumerate(self.variables)}

    # diffusion values
    diffusions = self.g(**all_args)
    if len(self.variables) == 1:
      if not isinstance(diffusions, (bm.ndarray, jnp.ndarray)):
        raise ValueError('Diffusion values must be a tensor when there '
                         'is only one variable in the equation.')
      diffusions = {self.variables[0]: diffusions}
    else:
      if not isinstance(diffusions, (tuple, list)):
        raise ValueError('Diffusion values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      diffusions = {var: diffusions[i] for i, var in enumerate(self.variables)}
    if self.wiener_type == constants.VECTOR_WIENER:
      for key, val in diffusions.items():
        if val is not None and bm.ndim(val) == 0:
          raise ValueError(f"{constants.VECTOR_WIENER} wiener process needs multiple "
                           f"dimensional diffusion value. But we got a scale value for "
                           f"variable {key}.")

    # derivative of diffusion parts
    all_dg = {}
    for i, key in enumerate(self.variables):
      f_dg, vars_, pars_ = parses[i]
      vps = vars_ + pars_
      all_dg[key] = f_dg(all_args[vps[0]], **{arg: all_args[arg] for arg in vps[1:] if arg in all_args})

    # integral results
    integrals = []
    for i, key in enumerate(self.variables):
      integral = all_args[key] + drifts[key] * dt
      if diffusions[key] is not None:
        shape = bm.shape(all_args[key])
        if self.wiener_type == constants.VECTOR_WIENER:
          noise_shape = bm.shape(diffusions[key])
          self._check_vector_wiener_dim(noise_shape, shape)
          shape += noise_shape[-1:]
        noise = self.rng.randn(*shape) * bm.sqrt(dt)
        if self.wiener_type == constants.VECTOR_WIENER:
          integral += bm.sum(diffusions[key] * noise, axis=-1)
        else:
          integral += diffusions[key] * noise
        noise_p2 = (noise ** 2 - dt) if self.intg_type == constants.ITO_SDE else noise ** 2
        diffusion = diffusions[key] * all_dg[key] / 2 * noise_p2
        diffusion = bm.sum(diffusion, axis=-1) if self.wiener_type == constants.VECTOR_WIENER else diffusion
        integral += diffusion
      integrals.append(integral)
    return integrals if len(self.variables) > 1 else integrals[0]


register_sde_integrator('milstein', Milstein)


class MilsteinGradFree(SDEIntegrator):
  r"""Derivative-free Milstein method for Ito or Stratonovich integrals.

  The following implementation approximates the frist derivative of :math:`g` thanks to a Runge-Kutta approach.
  For the Ito integral, the derivative-free Milstein method is given by

  .. math::

     \begin{aligned}
    Y_{n+1} &=Y_{n}+f_{n} h+g_{n} \Delta W_{n}+\frac{1}{2 \sqrt{h}}\left[g\left(\bar{Y}_{n}\right)-g_{n}\right]\left[\left(\Delta W_{n}\right)^{2}-h\right] \\
    \bar{Y}_{n} &=Y_{n}+f_{n} h+g_{n} \sqrt{h} \\
    \Delta W_{n} &=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
    \end{aligned}


  For the Stratonovich integral, the derivative-free Milstein method is given by

  .. math::

     \begin{aligned}
    Y_{n+1} &=Y_{n}+f_{n} h+g_{n} \Delta W_{n}+\frac{1}{2 \sqrt{h}}\left[g\left(\bar{Y}_{n}\right)-g_{n}\right]\left(\Delta W_{n}\right)^{2} \\
    \bar{Y}_{n} &=Y_{n}+f_{n} h+g_{n} \sqrt{h} \\
    \Delta W_{n} &=\left[W_{t+h}-W_{t}\right] \sim \sqrt{h} \mathcal{N}(0,1)
    \end{aligned}

  """

  def __init__(
      self,
      f: Callable,
      g: Callable,
      dt: float = None,
      name: str = None,
      show_code=False,
      var_type: str = None,
      intg_type: str = None,
      wiener_type: str = None,
      state_delays: Dict[str, bm.AbstractDelay] = None,
      dyn_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
  ):
    super(MilsteinGradFree, self).__init__(f=f,
                                           g=g,
                                           dt=dt,
                                           name=name,
                                           var_type=var_type,
                                           intg_type=intg_type,
                                           wiener_type=wiener_type,
                                           state_delays=state_delays,
                                           dyn_vars=dyn_vars)
    self.set_integral(self.step)

  def step(self, *args, **kwargs):
    # input arguments
    all_args = format_args(args, kwargs, self.arg_names)
    dt = all_args.pop(DT, self.dt)

    # drift values
    drifts = self.f(**all_args)
    if len(self.variables) == 1:
      if not isinstance(drifts, (bm.ndarray, jnp.ndarray)):
        raise ValueError('Drift values must be a tensor when there '
                         'is only one variable in the equation.')
      drifts = {self.variables[0]: drifts}
    else:
      if not isinstance(drifts, (tuple, list)):
        raise ValueError('Drift values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      drifts = {var: drifts[i] for i, var in enumerate(self.variables)}

    # diffusion values
    diffusions = self.g(**all_args)
    if len(self.variables) == 1:
      if not isinstance(diffusions, (bm.ndarray, jnp.ndarray)):
        raise ValueError('Diffusion values must be a tensor when there '
                         'is only one variable in the equation.')
      diffusions = {self.variables[0]: diffusions}
    else:
      if not isinstance(diffusions, (tuple, list)):
        raise ValueError('Diffusion values must be a list/tuple of tensors '
                         'when there are multiple variables in the equation.')
      diffusions = {var: diffusions[i] for i, var in enumerate(self.variables)}
    if self.wiener_type == constants.VECTOR_WIENER:
      for key, val in diffusions.items():
        if val is not None and bm.ndim(val) == 0:
          raise ValueError(f"{constants.VECTOR_WIENER} wiener process needs multiple "
                           f"dimensional diffusion value. But we got a scale value for "
                           f"variable {key}.")

    # intermediate results
    y_bars = {k: v for k, v in all_args.items()}
    for key in self.variables:
      bar = all_args[key] + drifts[key] * dt
      if diffusions[key] is not None:
        bar += diffusions[key] * bm.sqrt(dt)
      y_bars[key] = bar
    diffusion_bars = self.g(**y_bars)
    if len(self.variables) == 1:
      diffusion_bars = {self.variables[0]: diffusion_bars}
    else:
      diffusion_bars = {var: diffusion_bars[i] for i, var in enumerate(self.variables)}

    # integral results
    integrals = []
    for i, key in enumerate(self.variables):
      integral = all_args[key] + drifts[key] * dt
      if diffusions[key] is not None:
        shape = bm.shape(all_args[key])
        if self.wiener_type == constants.VECTOR_WIENER:
          noise_shape = bm.shape(diffusions[key])
          self._check_vector_wiener_dim(noise_shape, shape)
          shape += noise_shape[-1:]
        noise = self.rng.randn(*shape) * bm.sqrt(dt)
        if self.wiener_type == constants.VECTOR_WIENER:
          integral += bm.sum(diffusions[key] * noise, axis=-1)
        else:
          integral += diffusions[key] * noise
        noise_p2 = (noise ** 2 - dt) if self.intg_type == constants.ITO_SDE else noise ** 2
        minus = (diffusion_bars[key] - diffusions[key]) / 2 / bm.sqrt(dt)
        if self.wiener_type == constants.VECTOR_WIENER:
          integral += minus * bm.sum(noise_p2, axis=-1)
        else:
          integral += minus * noise_p2
      integrals.append(integral)
    return integrals if len(self.variables) > 1 else integrals[0]


register_sde_integrator('milstein2', Milstein)
register_sde_integrator('milstein_grad_free', Milstein)


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


  See Also
  --------
  Euler, Heun, Milstein
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
                                           dyn_vars=dyn_vars,
                                           state_delays=state_delays)

    if self.intg_type == constants.STRA_SDE:
      raise NotImplementedError(f'{self.__class__.__name__} does not support integral type of {constants.STRA_SDE}. '
                                f'It only supports {constants.ITO_SDE} now. ')

    # build the integrator
    self.integral = self.build()

  def build(self):
    parses = self._build_integrator(self.f)
    all_vps = self.variables + self.parameters

    def integral_func(*args, **kwargs):
      # format arguments
      params_in = Collector()
      for i, arg in enumerate(args):
        params_in[all_vps[i]] = arg
      params_in.update(kwargs)
      dt = params_in.pop(constants.DT, self.dt)

      # diffusion part
      diffusions = self.g(**params_in)

      # call integrals
      results = []
      params_in[constants.DT] = dt
      for i, parse in enumerate(parses):
        f_integral, vars_, pars_ = parse
        vps = vars_ + pars_ + [constants.DT]
        # integral of the drift part
        r = f_integral(params_in[vps[0]], **{arg: params_in[arg] for arg in vps[1:] if arg in params_in})
        if isinstance(diffusions, (tuple, list)):
          diffusion = diffusions[i]
        else:
          assert len(parses) == 1
          diffusion = diffusions
        # diffusion part
        shape = bm.shape(params_in[vps[0]])
        if diffusion is not None:
          if self.wiener_type == constants.VECTOR_WIENER:
            noise_shape = bm.shape(diffusion)
            self._check_vector_wiener_dim(noise_shape, shape)
            shape += noise_shape[-1:]
            diffusion = bm.sum(diffusion * self.rng.randn(*shape), axis=-1)
          else:
            diffusion = diffusion * self.rng.randn(*shape)
          r += diffusion * bm.sqrt(params_in[constants.DT])
        # final result
        results.append(r)
      return results if len(self.variables) > 1 else results[0]

    return integral_func

  def _build_integrator(self, f):
    if isinstance(f, joint_eq.JointEq):
      results = []
      for sub_eq in f.eqs:
        results.extend(self._build_integrator(sub_eq))
      return results

    else:
      vars, pars, _ = utils.get_args(f)
      if len(vars) != 1:
        raise errors.DiffEqError(constants.multi_vars_msg.format(cls=self.__class__.__name__,
                                                                 vars=str(vars), eq=str(f)))
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
