# -*- coding: utf-8 -*-

import inspect
import logging
from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.scipy.optimize import minimize

import brainpy.math as bm
from brainpy import errors, tools
from brainpy.analysis.numeric import utils, solver
from brainpy.analysis import constants as C
from brainpy.base.collector import Collector

logger = logging.getLogger('brainpy.analysis.numeric')

__all__ = [
  'LowDimAnalyzer',
  'LowDimAnalyzer1D',
  'LowDimAnalyzer2D',
]


def _update_scope(scope):
  scope['math'] = bm
  scope['bm'] = bm


def _dict_copy(target):
  assert isinstance(target, dict)
  return {k: v for k, v in target.items()}


def _get_args(f):
  reduced_args = []
  for name, par in inspect.signature(f).parameters.items():
    if par.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.KEYWORD_ONLY:
      reduced_args.append(par.name)
    elif par.kind is inspect.Parameter.VAR_POSITIONAL:
      raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.POSITIONAL_ONLY:
      raise errors.DiffEqError('Don not support positional only parameters, e.g., /')
    elif par.kind is inspect.Parameter.VAR_KEYWORD:
      raise errors.DiffEqError(f'Don not support dict of keyword arguments: {str(par)}')
    else:
      raise errors.DiffEqError(f'Unknown argument type: {par.kind}')

  var_names = []
  for a in reduced_args:
    if a == 't': break
    var_names.append(a)
  else:
    raise ValueError('Do not find time variable "t".')
  return var_names, reduced_args


# def _get_substitution(substitute_var: str,
#                       target_var: str,
#                       eq_group: dict,
#                       target_var_names: List[str],
#                       target_par_names: List[str],
#                       escape_sympy_solver: bool,
#                       timeout_len: float,
#                       eq_y_scope: dict):
#   results = tools.DictPlus()
#   if not escape_sympy_solver:
#     y_symbol = sympy.Symbol(target_var, real=True)
#     code = eq_group["sub_exprs"][-1].code
#     y_eq = analysis_by_sympy.str2sympy(code).expr
#
#     eq_y_scope.update(eq_group['diff_eq'].func_scope)
#     _update_scope(eq_y_scope)
#
#     argument = ', '.join(target_var_names + target_par_names)
#
#     try:
#       logger.warning(f'SymPy solve "{eq_group["func_name"]}({argument}) = 0" to '
#                      f'"{target_var} = f({substitute_var}, {",".join(target_var_names[2:] + target_par_names)})", ')
#       # solve the expression
#       f = tools.timeout(timeout_len)(lambda: sympy.solve(y_eq, y_symbol))
#       y_by_x_in_y_eq = f()
#       if len(y_by_x_in_y_eq) > 1:
#         raise NotImplementedError('Do not support multiple values.')
#       y_by_x_in_y_eq = analysis_by_sympy.sympy2str(y_by_x_in_y_eq[0])
#
#       # check
#       all_vars = set(eq_y_scope.keys())
#       all_vars.update(target_var_names + target_par_names)
#       unknown_symbols = utils.unknown_symbol(y_by_x_in_y_eq, all_vars)
#       if len(unknown_symbols):
#         logger.warning(f'\tfailed because contain unknown symbols: {unknown_symbols}.')
#         results['status'] = 'sympy_failed'
#         results['subs'] = []
#         results['f'] = None
#       else:
#         logger.warning('\tsuccess.')
#         # substituted codes
#         subs_codes = [f'{expr.var_name} = {expr.code}' for expr in eq_group["sub_exprs"][:-1]]
#         subs_codes.append(f'{target_var} = {y_by_x_in_y_eq}')
#
#         # compile the function
#         func_code = f'def func({substitute_var}, {",".join(target_var_names[2:] + target_par_names)}):\n'
#         for expr in eq_group["sub_exprs"][:-1]:
#           func_code += f'  {expr.var_name} = {expr.code}\n'
#         func_code += f'  return {y_by_x_in_y_eq}'
#         exec(compile(func_code, '', 'exec'), eq_y_scope)
#
#         # set results
#         results['status'] = 'sympy_success'
#         results['subs'] = subs_codes
#         results['f'] = eq_y_scope['func']
#
#     except NotImplementedError:
#       logger.warning('\tfailed because the equation is too complex.')
#       results['status'] = 'sympy_failed'
#       results['subs'] = []
#       results['f'] = None
#     except KeyboardInterrupt:
#       logger.warning(f'\tfailed because {timeout_len} s timeout.')
#       results['status'] = 'sympy_failed'
#       results['subs'] = []
#       results['f'] = None
#   else:
#     results['status'] = 'escape'
#     results['subs'] = []
#     results['f'] = None
#   return results


def _std_derivative(original_fargs, target_vars, target_pars):
  var = original_fargs[0]
  num_vars = len(target_vars)

  def inner(f):
    def call(*dyn_vars_and_pars, **fixed_vars_and_pars):
      params = dict()
      for i, v in enumerate(target_vars):
        if (v != var) and (v in original_fargs):
          params[v] = dyn_vars_and_pars[i]
      for j, p in enumerate(target_pars):
        if p in original_fargs:
          params[p] = dyn_vars_and_pars[num_vars + j]
      for k, v in fixed_vars_and_pars.items():
        if k in original_fargs:
          params[k] = v
      return f(dyn_vars_and_pars[target_vars.index(var)], 0., **params)

    return call

  return inner


class LowDimAnalyzer(object):
  r"""Dynamics Analyzer for Neuron Models.

  This class is a base class which aims for analyze the analysis in
  neuron models. A neuron model is characterized by a series of dynamical
  variables and parameters:

  .. math::

      {dF \over dt} = F(v_1, v_2, ..., p_1, p_2, ...)

  where :math:`v_1, v_2` are variables, :math:`p_1, p_2` are parameters.

  Parameters
  ----------
  model : Any
      A model of the population, the integrator function,
      or a list/tuple of integrator functions.
  target_vars : dict
      The target/dynamical variables.
  fixed_vars : dict
      The fixed variables.
  target_pars : dict, optional
      The parameters which can be dynamical varied.
  pars_update : dict, optional
      The parameters to update.
  resolutions : float, dict
      The resolution for numerical iterative solvers. Default is 0.1. It can set the
      numerical resolution of dynamical variables or dynamical parameters. For example,

      - set ``numerical_resolution=0.1`` will generalize it to all variables and parameters;
      - set ``numerical_resolution={var1: 0.1, var2: 0.2, par1: 0.1, par2: 0.05}`` will specify
        the particular resolutions to variables and parameters.
      - Moreover, you can also set ``numerical_resolution={var1: np.array([...]), var2: 0.1}``
        to specify the search points need to explore for variable `var1`. This will be useful
        to set sense search points at some inflection points.
  - **perturbation**: float. The small perturbation used to solve the function derivative.
  - **sympy_solver_timeout**: float, with the unit of second. The maximum  time allowed
    to use sympy solver to get the variable relationship.
  - **escape_sympy_solver**: bool. Whether escape to use sympy solver, and directly use
    numerical optimization method to solve the nullcline and fixed points.
  - **lim_scale**: float. The axis limit scale factor. Default is 1.05. The setting means
    the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.
  """

  def __init__(self,
               model,
               target_vars,
               fixed_vars=None,
               target_pars=None,
               pars_update=None,
               resolutions=None,
               jit_device=None,
               escape_sympy_solver=False,
               sympy_solver_timeout=5.,
               lim_scale=1.05,
               options=None, ):
    # model
    # -----
    self.model = utils.integrators_into_model(model, )

    # target variables
    # ----------------
    if not isinstance(target_vars, dict):
      raise errors.AnalyzerError('"target_vars" must be a dict, with the format of '
                                 '{"var1": (var1_min, var1_max)}.')
    self.target_vars = Collector(target_vars)
    self.target_var_names = list(self.target_vars.keys())  # list of target vars
    for key in self.target_vars.keys():
      if key not in self.model.variables:
        raise errors.AnalyzerError(f'{key} is not a dynamical variable in {self.model}.')

    # fixed variables
    # ----------------
    if fixed_vars is None:
      fixed_vars = dict()
    if not isinstance(fixed_vars, dict):
      raise errors.AnalyzerError('"fixed_vars" must be a dict with the format '
                                 'of {"var1": val1, "var2": val2}.')
    for key in fixed_vars.keys():
      if key not in self.model.variables:
        raise ValueError(f'{key} is not a dynamical variable in {self.model}.')
    self.fixed_vars = Collector(fixed_vars)

    # check duplicate
    for key in self.fixed_vars.keys():
      if key in self.target_vars:
        raise errors.AnalyzerError(f'"{key}" is defined as a target variable in "target_vars", '
                                   f'but also defined as a fixed variable in "fixed_vars".')

    # parameters to update
    # ---------------------
    if pars_update is None:
      pars_update = dict()
    if not isinstance(pars_update, dict):
      raise errors.AnalyzerError('"pars_update" must be a dict with the format '
                                 'of {"par1": val1, "par2": val2}.')
    pars_update = Collector(pars_update)
    for key in pars_update.keys():
      if key not in self.model.parameters:
        raise errors.AnalyzerError(f'"{key}" is not a valid parameter in "{self.model}" model.')
    self.pars_update = pars_update

    # dynamical parameters
    # ---------------------
    if target_pars is None:
      target_pars = dict()
    if not isinstance(target_pars, dict):
      raise errors.AnalyzerError('"target_pars" must be a dict with the format of {"par1": (val1, val2)}.')
    for key in target_pars.keys():
      if key not in self.model.parameters:
        raise errors.AnalyzerError(f'"{key}" is not a valid parameter in "{self.model}" model.')
    self.target_pars = Collector(target_pars)
    self.target_par_names = list(self.target_pars.keys())  # list of target_pars

    # check duplicate
    for key in self.pars_update.keys():
      if key in self.target_pars:
        raise errors.AnalyzerError(f'"{key}" is defined as a target parameter in "target_pars", '
                                   f'but also defined as a fixed parameter in "pars_update".')

    # resolutions for numerical methods
    # ---------------------------------
    self.resolutions = dict()
    _target_vp = self.target_vars + self.target_pars
    if resolutions is None:
      for key, lim in self.target_vars.items():
        self.resolutions[key] = bm.linspace(*lim, 20)
      for key, lim in self.target_pars.items():
        self.resolutions[key] = bm.linspace(*lim, 20)
    elif isinstance(resolutions, float):
      for key, lim in self.target_vars.items():
        self.resolutions[key] = bm.arange(*lim, resolutions)
      for key, lim in self.target_pars.items():
        self.resolutions[key] = bm.arange(*lim, resolutions)
    elif isinstance(resolutions, dict):
      for key in self.target_var_names + self.target_par_names:
        if key not in resolutions:
          self.resolutions[key] = bm.linspace(*_target_vp[key], 20)
          continue
        resolution = resolutions[key]
        if isinstance(resolution, float):
          self.resolutions[key] = bm.arange(*_target_vp[key], resolution)
        elif isinstance(resolution, (bm.ndarray, np.ndarray, jnp.ndarray)):
          if not np.ndim(resolution) == 1:
            raise errors.AnalyzerError(f'resolution must be a 1D vector, but get its '
                                       f'shape with {resolution.shape}.')
          self.resolutions[key] = bm.asarray(resolution)
        else:
          raise errors.AnalyzerError(f'Unknown resolution setting: {key}: {resolution}')
    else:
      raise errors.AnalyzerError(f'Unknown resolution type: {type(resolutions)}')

    # other settings
    # --------------
    if options is None:
      options = dict()
    self.options = options
    self.jit_device = jit_device
    self.escape_sympy_solver = escape_sympy_solver
    self.sympy_solver_timeout = sympy_solver_timeout
    self.lim_scale = lim_scale

    # A dict to store the analyzed results
    # -------------------------------------
    # 'dxdt' : The differential function ``f`` of the first variable ``x``.
    #          It can be used as ``dxdt(x, y, ...)``.
    # 'dydt' : The differential function ``g`` of the second variable ``y``.
    #          It can be used as ``dydt(x, y, ...)``.
    # 'dfdx' : The derivative of ``f`` by ``x``. It can be used as ``dfdx(x, y, ...)``.
    # 'dfdy' : The derivative of ``f`` by ``y``. It can be used as ``dfdy(x, y, ...)``.
    # 'dgdx' : The derivative of ``g`` by ``x``. It can be used as ``dgdx(x, y, ...)``.
    # 'dgdy' : The derivative of ``g`` by ``y``. It can be used as ``dgdy(x, y, ...)``.
    # 'jacobian' : The jacobian matrix. It can be used as ``jacobian(x, y, ...)``.
    # 'fixed_point' : The fixed point.
    # 'y_by_x_in_fy' :
    # 'x_by_y_in_fy' :
    # 'y_by_x_in_fx' :
    # 'x_by_y_in_fx' :
    self.analyzed_results = tools.DictPlus()

  @property
  def target_eqs(self):
    if 'target_eqs' not in self.analyzed_results:
      var2eq = {eq.var_name: eq for eq in self.model.analyzers}
      target_eqs = tools.DictPlus()
      for key in self.target_vars.keys():
        if key not in var2eq:
          raise errors.AnalyzerError(f'target "{key}" is not a dynamical variable.')
        diff_eq = var2eq[key]
        sub_exprs = diff_eq.get_f_expressions(substitute_vars=list(self.target_vars.keys()))
        old_exprs = diff_eq.get_f_expressions(substitute_vars=None)
        target_eqs[key] = tools.DictPlus(sub_exprs=sub_exprs,
                                         old_exprs=old_exprs,
                                         diff_eq=diff_eq,
                                         func_name=diff_eq.func_name)
      self.analyzed_results['target_eqs'] = target_eqs
    return self.analyzed_results['target_eqs']


class LowDimAnalyzer1D(LowDimAnalyzer):
  r"""Neuron analysis analyzer for 1D system.

  It supports the analysis of 1D dynamical system.

  .. math::

      {dx \over dt} = f(x, t)

  Actually, the analysis for 1D system is purely analytical. It do not
  rely on SymPy.
  """

  def __init__(self, *args, **kwargs):
    super(LowDimAnalyzer1D, self).__init__(*args, **kwargs)
    self.x_var = self.target_var_names[0]

  @property
  def F_fx(self):
    """Make the standard function call of :math:`f_x (*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_fx not in self.analyzed_results:
      _, arguments = _get_args(self.model.F[self.x_var])
      wrapper = _std_derivative(arguments, self.target_var_names, self.target_par_names)
      f = wrapper(self.model.F[self.x_var])
      f = partial(f, **(self.pars_update + self.fixed_vars))
      self.analyzed_results[C.F_fx] = bm.jit(f, device=self.jit_device)
    return self.analyzed_results[C.F_fx]

  @property
  def F_dfxdx(self):
    """The function to evaluate :math:`\frac{df_x(*\mathrm{vars}, *\mathrm{pars})}{dx}`."""
    if C.F_dfxdx not in self.analyzed_results:
      dfx = bm.vector_grad(self.F_fx, argnums=0)
      self.analyzed_results[C.F_dfxdx] = bm.jit(dfx, device=self.jit_device)
    return self.analyzed_results[C.F_dfxdx]

  @property
  def F_fixed_points(self):
    """The function to evalute fixed points :math:`F(*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_fixed_point not in self.analyzed_results:
      f = lambda candidates: solver.roots_of_1d_by_x(self.F_fx, candidates)
      self.analyzed_results[C.F_fixed_point] = f
    return self.analyzed_results[C.F_fixed_point]


class LowDimAnalyzer2D(LowDimAnalyzer1D):
  r"""Neuron analysis analyzer for 2D system.

  It supports the analysis of 2D dynamical system.

  .. math::

      {dx \over dt} = fx(x, t, y)

      {dy \over dt} = fy(y, t, x)
  """

  def __init__(self, *args, **kwargs):
    super(LowDimAnalyzer2D, self).__init__(*args, **kwargs)

    self.y_var = self.target_var_names[1]

    # # options
    # # ---------
    # options = kwargs.get('options', dict())
    # if options is None: options = dict()
    # assert isinstance(options, dict)
    # for a in [C.y_by_x_in_fy, C.y_by_x_in_fx, C.x_by_y_in_fx, C.x_by_y_in_fy]:
    #   if a in options:
    #     # check "subs"
    #     subs = options[a]
    #     if isinstance(subs, str):
    #       subs = [subs]
    #     elif isinstance(subs, (tuple, list)):
    #       subs = subs
    #       for s in subs:
    #         assert isinstance(s, str)
    #     else:
    #       raise ValueError(f'Unknown setting of "{a}": {subs}')
    #
    #     # check "f"
    #     scope = _dict_copy(self.pars_update)
    #     scope.update(self.fixed_vars)
    #     _update_scope(scope)
    #     if a.startswith('fy::'):
    #       scope.update(self.fy_eqs['diff_eq'].func_scope)
    #     else:
    #       scope.update(self.fx_eqs['diff_eq'].func_scope)
    #
    #     # function code
    #     argument = ",".join(self.target_var_names[2:] + self.target_par_names)
    #     if a.endswith('y=f(x)'):
    #       func_codes = [f'def func({self.x_var}, {argument}):\n']
    #     else:
    #       func_codes = [f'def func({self.y_var}, {argument}):\n']
    #     func_codes.extend(subs)
    #     func_codes.append(f'return {subs[-1].split("=")[0]}')
    #
    #     # function compilation
    #     exec(compile("\n  ".join(func_codes), '', 'exec'), scope)
    #     f = scope['func']
    #
    #     # results
    #     self.analyzed_results[a] = tools.DictPlus(status='sympy_success', subs=subs, f=f)

  @property
  def F_fy(self):
    """The function to evaluate :math:`f_y(*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_fy not in self.analyzed_results:
      variables, arguments = _get_args(self.model.F[self.y_var])
      wrapper = _std_derivative(arguments, self.target_var_names, self.target_par_names)
      f = wrapper(self.model.F[self.y_var])
      f = partial(f, **(self.pars_update + self.fixed_vars))
      self.analyzed_results[C.F_fy] = bm.jit(f, device=self.jit_device)
    return self.analyzed_results[C.F_fy]

  @property
  def F_dfxdy(self):
    """The function to evaluate :math:`\frac{df_x (*\mathrm{vars}, *\mathrm{pars})}{dy}`."""
    if C.F_dfxdy not in self.analyzed_results:
      dfxdy = bm.vector_grad(self.F_fx, argnums=1)
      self.analyzed_results[C.F_dfxdy] = bm.jit(dfxdy, device=self.jit_device)
    return self.analyzed_results[C.F_dfxdy]

  @property
  def F_dfydx(self):
    """The function to evaluate :math:`\frac{df_y (*\mathrm{vars}, *\mathrm{pars})}{dx}`."""
    if C.F_dfydx not in self.analyzed_results:
      dfydx = bm.vector_grad(self.F_fy, argnums=0)
      self.analyzed_results[C.F_dfydx] = bm.jit(dfydx, device=self.jit_device)
    return self.analyzed_results[C.F_dfydx]

  @property
  def F_dfydy(self):
    """The function to evaluate :math:`\frac{df_y (*\mathrm{vars}, *\mathrm{pars})}{dy}`."""
    if C.F_dfydy not in self.analyzed_results:
      dfydy = bm.vector_grad(self.F_fy, argnums=1)
      self.analyzed_results[C.F_dfydy] = bm.jit(dfydy, device=self.jit_device)
    return self.analyzed_results[C.F_dfydy]

  @property
  def F_jacobian(self):
    """The function to evaluate :math:`J(*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_jacobian not in self.analyzed_results:
      F_fx = solver.f_without_jaxarray_return(self.F_fx)
      F_fy = solver.f_without_jaxarray_return(self.F_fy)

      @partial(bm.jacobian, argnums=(0, 1))
      def f_jacobian(*var_and_pars):
        return F_fx(*var_and_pars), F_fy(*var_and_pars)

      def call(*var_and_pars):
        var_and_pars = tuple((vp.value if isinstance(vp, bm.JaxArray) else vp) for vp in var_and_pars)
        return jnp.array(bm.jit(f_jacobian, device=self.jit_device)(*var_and_pars))

      self.analyzed_results[C.F_jacobian] = call
    return self.analyzed_results[C.F_jacobian]

  def F_fx_nullcline_by_opt(self, coords=None):
    """Get the function to solve :math:`fx` nullcline by using numerical optimization method."""
    # check coordinates
    if coords is None:
      coords = self.x_var + '-' + self.y_var
    key = f'{C.F_fx_nullcline_by_opt},{coords}'
    if key not in self.analyzed_results:
      # check coordinates
      _splits = [a.strip() for a in coords.split('-')]
      assert len(_splits) == 2
      if self.x_var not in _splits:
        raise ValueError(f'Variable "{self.x_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      if self.y_var not in _splits:
        raise ValueError(f'Variable "{self.y_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      target_to_opt = _splits[1]

      # optimization function
      if target_to_opt == self.y_var:
        f_to_opt = lambda y, *fixed_vp: self.F_fx(fixed_vp[0], y, *fixed_vp[2:])
      else:
        f_to_opt = lambda x, *fixed_vp: self.F_fx(x, *fixed_vp[1:])
      f_opt = lambda candidates, *args: solver.roots_of_1d_by_x(f_to_opt, candidates, args)
      self.analyzed_results[key] = f_opt
    return self.analyzed_results[key]

  def F_fy_nullcline_by_opt(self, coords=None):
    """Get the function to solve Y nullcline by using numerical optimization method.

    Parameters
    ----------
    coords : str
        The coordination.
    """
    if coords is None:
      coords = self.x_var + '-' + self.y_var
    key = f'{C.F_fy_nullcline_by_opt},{coords}'
    if key not in self.analyzed_results:
      # check coordinates
      _splits = [a.strip() for a in coords.split('-')]
      assert len(_splits) == 2
      if self.x_var not in _splits:
        raise ValueError(f'Variable "{self.x_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      if self.y_var not in _splits:
        raise ValueError(f'Variable "{self.y_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      target_to_opt = _splits[1]

      # optimization function
      if target_to_opt == self.y_var:
        f_to_opt = lambda y, *fixed_vp: self.F_fy(fixed_vp[0], y, *fixed_vp[2:])
      else:
        f_to_opt = lambda x, *fixed_vp: self.F_fy(x, *fixed_vp[1:])
      f_opt = lambda candidates, *args: solver.roots_of_1d_by_x(f_to_opt, candidates, args)
      self.analyzed_results[key] = f_opt
    return self.analyzed_results[key]

  def _get_fx_nullcline_points(self, *args, **kwargs):
    raise NotImplementedError

  def _get_fy_nullcline_points(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def F_fixed_points(self, tol_unique=1e-2):
    """The function to evaluate the fixed point by :math:`F(*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_fixed_point not in self.analyzed_results:
      fx_nullcline_points = self._get_fx_nullcline_points()
      candidates = jnp.asarray(np.stack(fx_nullcline_points).T)
      fx = solver.f_without_jaxarray_return(self.F_fx)
      fy = solver.f_without_jaxarray_return(self.F_fy)

      def aux_fun(xy, *args):
        dx = fx(xy[0], xy[1], *args)
        dy = fy(xy[0], xy[1], *args)
        return (dx ** 2 + dy ** 2).sum()

      def opt_fun(xy_init, *args):
        return minimize(aux_fun, xy_init, args=args, method='BFGS')

      f = bm.jit(bm.vmap(opt_fun, in_axes=(0, None)))

      def call_func(*args):
        points = f(candidates, *args)
        points = np.asarray(points.x)
        fps, _ = utils.keep_unique(points, tol=tol_unique)
        return fps

      self.analyzed_results[C.F_fixed_point] = call_func
      self.analyzed_results[C.F_fixed_point_opt] = opt_fun
      self.analyzed_results[C.F_fixed_point_aux] = aux_fun

    return self.analyzed_results[C.F_fixed_point]

  @property
  def F_fixed_point_opt(self):
    if C.F_fixed_point_opt not in self.analyzed_results:
      f = self.F_fixed_points
    return self.analyzed_results[C.F_fixed_point_opt]

  @property
  def F_fixed_point_aux(self):
    if C.F_fixed_point_aux not in self.analyzed_results:
      f = self.F_fixed_points
    return self.analyzed_results[C.F_fixed_point_aux]

  # @property
  # def fx_eqs(self):
  #   return self.target_eqs[self.x_var]
  #
  # @property
  # def fy_eqs(self):
  #   return self.target_eqs[self.y_var]
  #
  # @property
  # def y_by_x_in_fy(self):
  #   """Get the expression of "y=f(x)" in :math:`f_y` equation.
  #
  #   Specifically, ``self.analyzed_results['y_by_x_in_fy']`` is a Dict,
  #   with the following keywords:
  #
  #   - status : 'sympy_success', 'sympy_failed', 'escape'
  #   - subs : substituted expressions (relationship) of y_by_x
  #   - f : function of y_by_x
  #   """
  #   if C.y_by_x_in_fy not in self.analyzed_results:
  #     eq_y_scope = _dict_copy(self.pars_update)
  #     eq_y_scope.update(self.fixed_vars)
  #     results = _get_substitution(substitute_var=self.x_var,
  #                                 target_var=self.y_var,
  #                                 eq_group=self.fy_eqs,
  #                                 target_var_names=self.target_var_names,
  #                                 target_par_names=self.target_par_names,
  #                                 escape_sympy_solver=self.escape_sympy_solver,
  #                                 timeout_len=self.sympy_solver_timeout,
  #                                 eq_y_scope=eq_y_scope)
  #     self.analyzed_results[C.y_by_x_in_fy] = results
  #   return self.analyzed_results[C.y_by_x_in_fy]
  #
  # @property
  # def y_by_x_in_fx(self):
  #   """Get the expression of "y_by_x_in_fx".
  #
  #   Specifically, ``self.analyzed_results['y_by_x_in_fx']`` is a Dict,
  #   with the following keywords:
  #
  #   - status : 'sympy_success', 'sympy_failed', 'escape'
  #   - subs : substituted expressions (relationship) of y_by_x
  #   - f : function of y_by_x
  #   """
  #   if C.y_by_x_in_fx not in self.analyzed_results:
  #     eq_y_scope = _dict_copy(self.pars_update)
  #     eq_y_scope.update(self.fixed_vars)
  #     results = _get_substitution(substitute_var=self.x_var,
  #                                 target_var=self.y_var,
  #                                 eq_group=self.fx_eqs,
  #                                 target_var_names=self.target_var_names,
  #                                 target_par_names=self.target_par_names,
  #                                 escape_sympy_solver=self.escape_sympy_solver,
  #                                 timeout_len=self.sympy_solver_timeout,
  #                                 eq_y_scope=eq_y_scope)
  #     self.analyzed_results[C.y_by_x_in_fx] = results
  #   return self.analyzed_results[C.y_by_x_in_fx]
  #
  # @property
  # def x_by_y_in_fy(self):
  #   """Get the expression of "x_by_y_in_fy".
  #
  #   Specifically, ``self.analyzed_results['x_by_y_in_fy']`` is a Dict,
  #   with the following keywords:
  #
  #   - status : 'sympy_success', 'sympy_failed', 'escape'
  #   - subs : substituted expressions (relationship) of x_by_y
  #   - f : function of x_by_y
  #   """
  #   if C.x_by_y_in_fy not in self.analyzed_results:
  #     eq_y_scope = _dict_copy(self.pars_update)
  #     eq_y_scope.update(self.fixed_vars)
  #     results = _get_substitution(substitute_var=self.y_var,
  #                                 target_var=self.x_var,
  #                                 eq_group=self.fy_eqs,
  #                                 target_var_names=self.target_var_names,
  #                                 target_par_names=self.target_par_names,
  #                                 escape_sympy_solver=self.escape_sympy_solver,
  #                                 timeout_len=self.sympy_solver_timeout,
  #                                 eq_y_scope=eq_y_scope)
  #     self.analyzed_results[C.x_by_y_in_fy] = results
  #   return self.analyzed_results[C.x_by_y_in_fy]
  #
  # @property
  # def x_by_y_in_fx(self):
  #   """Get the expression of "x_by_y_in_fx".
  #
  #   Specifically, ``self.analyzed_results['x_by_y_in_fx']`` is a Dict,
  #   with the following keywords:
  #
  #   - status : 'sympy_success', 'sympy_failed', 'escape'
  #   - subs : substituted expressions (relationship) of x_by_y
  #   - f : function of x_by_y
  #   """
  #   if C.x_by_y_in_fx not in self.analyzed_results:
  #     eq_y_scope = _dict_copy(self.pars_update)
  #     eq_y_scope.update(self.fixed_vars)
  #     results = _get_substitution(substitute_var=self.y_var,
  #                                 target_var=self.x_var,
  #                                 eq_group=self.fx_eqs,
  #                                 target_var_names=self.target_var_names,
  #                                 target_par_names=self.target_par_names,
  #                                 escape_sympy_solver=self.escape_sympy_solver,
  #                                 timeout_len=self.sympy_solver_timeout,
  #                                 eq_y_scope=eq_y_scope)
  #     self.analyzed_results[C.x_by_y_in_fx] = results
  #   return self.analyzed_results[C.x_by_y_in_fx]
