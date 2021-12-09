# -*- coding: utf-8 -*-

import inspect
import logging
from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.scipy.optimize import minimize

import brainpy.math as bm
from brainpy import errors, tools
from brainpy.analysis import constants as C, utils
from brainpy.base.collector import Collector

logger = logging.getLogger('brainpy.analysis')

__all__ = [
  'LowDimAnalyzer',
  'Num1DAnalyzer',
  'Num2DAnalyzer',
]


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
  r"""Dynamics Analyzer for Low-dimensional Dynamical Systems.

  A dynamical model is characterized by a series of dynamical
  variables and parameters:

  .. math::

      {dF \over dt} = F(v_1, v_2, ..., p_1, p_2, ...)

  where :math:`v_1, v_2` are variables, :math:`p_1, p_2` are parameters.

  .. note::
    ``LowDimAnalyzer`` cannot analyze dynamical system depends on time :math`t`.

  Parameters
  ----------
  model : Any, ODEIntegrator, sequence of ODEIntegrator, DynamicalSystem
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

      - set ``resolutions=0.1`` will generalize it to all variables and parameters;
      - set ``resolutions={var1: 0.1, var2: 0.2, par1: 0.1, par2: 0.05}`` will specify
        the particular resolutions to variables and parameters.
      - Moreover, you can also set ``resolutions={var1: JaxArray([...]), var2: 0.1}``
        to specify the search points need to explore for variable `var1`.
        This will be useful to set sense search points at some inflection points.
  lim_scale: float
    The axis limit scale factor. Default is 1.05. The setting means
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
               lim_scale=1.05,
               options=None, ):
    # model
    # -----
    self.model = utils.model_transform(model)

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


class Num1DAnalyzer(LowDimAnalyzer):
  r"""Neuron analysis analyzer for 1D system.

  It supports the analysis of 1D dynamical system.

  .. math::

      {dx \over dt} = f(x, t)

  Actually, the analysis for 1D system is purely analytical. It do not
  rely on SymPy.
  """

  def __init__(self, *args, **kwargs):
    super(Num1DAnalyzer, self).__init__(*args, **kwargs)
    self.x_var = self.target_var_names[0]
    if len(self.target_vars) < 1:
      raise errors.AnalyzerError(f'{Num1DAnalyzer.__name__} only supports dynamical system '
                                 f'with >= 1 variables. But we got {len(self.target_vars)} '
                                 f'variables in {self.model}.')

  @property
  def F_fx(self):
    """Make the standard function call of :math:`f_x (*\mathrm{vars}, *\mathrm{pars})`.

    This function has been transformed into the standard call.
    For instance, if the user has the ``target_vars=("v1", "v2")`` and
    the ``target_pars=("p1", "p2")``, while the first function is defined as:

    >>> def f1(v1, t, p1):
    >>>   return something

    However, after the stransformation, this function should be called as:

    >>> self.F_fx(v1, v2, p1, p2)
    """
    if C.F_fx not in self.analyzed_results:
      _, arguments = _get_args(self.model.F[self.x_var])
      wrapper = _std_derivative(arguments, self.target_var_names, self.target_par_names)
      f = wrapper(self.model.F[self.x_var])
      f = partial(f, **(self.pars_update + self.fixed_vars))
      f = utils.f_without_jaxarray_return(f)
      self.analyzed_results[C.F_fx] = bm.jit(f, device=self.jit_device)
    return self.analyzed_results[C.F_fx]

  @property
  def F_vmap_fx(self):
    if C.F_vmap_fx not in self.analyzed_results:
      self.analyzed_results[C.F_vmap_fx] = bm.jit(bm.vmap(self.F_fx), device=self.jit_device)
    return self.analyzed_results[C.F_vmap_fx]

  @property
  def F_dfxdx(self):
    """The function to evaluate :math:`\frac{df_x(*\mathrm{vars}, *\mathrm{pars})}{dx}`."""
    if C.F_dfxdx not in self.analyzed_results:
      dfx = bm.vector_grad(self.F_fx, argnums=0)
      self.analyzed_results[C.F_dfxdx] = bm.jit(dfx, device=self.jit_device)
    return self.analyzed_results[C.F_dfxdx]

  @property
  def F_fixed_point_aux(self):
    if C.F_fixed_point_aux not in self.analyzed_results:
      def aux_fun(x, *args):
        return jnp.abs(self.F_fx(x, *args)).sum()

      self.analyzed_results[C.F_fixed_point_aux] = aux_fun
    return self.analyzed_results[C.F_fixed_point_aux]

  @property
  def F_vmap_fp_aux(self):
    if C.F_vmap_fp_aux not in self.analyzed_results:
      # The arguments of this function are:
      # ---
      # "X": a two-dimensional matrix: (num_batch, num_var)
      # "args": a list of one-dimensional vectors, each has the shape of (num_batch,)
      self.analyzed_results[C.F_vmap_fp_aux] = bm.jit(bm.vmap(self.F_fixed_point_aux))
    return self.analyzed_results[C.F_vmap_fp_aux]

  @property
  def F_fixed_point_opt(self):
    if C.F_fixed_point_opt not in self.analyzed_results:
      def f(start_and_end, *args):
        return utils.jax_brentq(self.F_fx)(start_and_end[0], start_and_end[1], args)

      self.analyzed_results[C.F_fixed_point_opt] = f
    return self.analyzed_results[C.F_fixed_point_opt]

  @property
  def F_vmap_fp_opt(self):
    if C.F_vmap_fp_opt not in self.analyzed_results:
      # The arguments of this function are:
      # ---
      # "X": a two-dimensional matrix: (num_batch, num_var)
      # "args": a list of one-dimensional vectors, each has the shape of (num_batch,)
      self.analyzed_results[C.F_vmap_fp_opt] = bm.jit(bm.vmap(self.F_fixed_point_opt))
    return self.analyzed_results[C.F_vmap_fp_opt]

  def _get_fixed_points(self, candidates, *args, num_seg=None, tol_loss=1e-7, loss_screen=None):
    """

    "candidates" and "args" can be obtained through:

    >>> all_candidates = []
    >>> all_par1 = []
    >>> all_par2 = []
    >>> for p1 in par1_list:
    >>>   for p2 in par2_list:
    >>>     xs = self.resolutions[self.x_var]
    >>>     all_candidates.append(xs)
    >>>     all_par1.append(jnp.ones_like(xs) * p1)
    >>>     all_par2.append(jnp.ones_like(xs) * p2)

    Parameters
    ----------
    candidates
    args
    tol_loss
    loss_screen

    Returns
    -------

    """
    # candidates: xs, a vector with the length of self.resolutions[self.x_var]
    # args: parameters, a list/tuple of vectors
    candidates = candidates.value if isinstance(candidates, bm.JaxArray) else candidates
    selected_ids = np.arange(len(candidates))
    args = tuple(a.value if isinstance(candidates, bm.JaxArray) else a for a in args)
    for a in args: assert len(a) == len(candidates)
    if num_seg is None:
      num_seg = len(self.resolutions[self.x_var])
    assert isinstance(num_seg, int)

    # get the signs
    signs = jnp.sign(self.F_vmap_fx(candidates, *args))
    signs = signs.reshape((num_seg, -1))
    par_len = signs.shape[1]
    signs1 = signs.at[-1].set(1)
    signs2 = jnp.vstack((signs[1:], signs[:1])).at[-1].set(1)
    ids = jnp.where((signs1 * signs2).flatten() <= 0)[0]
    if len(ids) <= 0:
      return [], []

    # selected the proper candidates to optimize fixed points
    selected_ids = selected_ids[np.asarray(ids)]
    starts = candidates[ids]
    ends = candidates[ids + par_len]
    X = jnp.stack((starts, ends)).T
    args = tuple(a[ids] for a in args)

    # optimize the fixed points
    res = self.F_vmap_fp_opt(X, *args)
    losses = self.F_vmap_fp_aux(res['root'], *args)
    valid_or_not = jnp.logical_and(res['status'] == utils._ECONVERGED, losses <= tol_loss)
    ids = np.asarray(jnp.where(valid_or_not)[0])
    fps = np.asarray(res['root'])[ids]
    selected_ids = selected_ids[np.asarray(ids)]
    return fps, selected_ids


class Num2DAnalyzer(Num1DAnalyzer):
  r"""Neuron analysis analyzer for 2D system.

  It supports the analysis of 2D dynamical system.

  .. math::

      {dx \over dt} = fx(x, t, y)

      {dy \over dt} = fy(y, t, x)
  """

  def __init__(self, *args, **kwargs):
    super(Num2DAnalyzer, self).__init__(*args, **kwargs)
    if len(self.target_vars) < 2:
      raise errors.AnalyzerError(f'{Num1DAnalyzer.__name__} only supports dynamical system '
                                 f'with >= 2 variables. But we got {len(self.target_vars)} '
                                 f'variables in {self.model}.')
    self.y_var = self.target_var_names[1]

  @property
  def F_fy(self):
    """The function to evaluate :math:`f_y(*\mathrm{vars}, *\mathrm{pars})`.

    This function has been transformed into the standard call.
    For instance, if the user has the ``target_vars=("v1", "v2")`` and
    the ``target_pars=("p1", "p2")``, while the first function is defined as:

    >>> def f1(v1, t, p1):
    >>>   return something

    However, after the stransformation, this function should be called as:

    >>> self.F_fy(v1, v2, p1, p2)
    """
    if C.F_fy not in self.analyzed_results:
      variables, arguments = _get_args(self.model.F[self.y_var])
      wrapper = _std_derivative(arguments, self.target_var_names, self.target_par_names)
      f = wrapper(self.model.F[self.y_var])
      f = partial(f, **(self.pars_update + self.fixed_vars))
      f = utils.f_without_jaxarray_return(f)
      self.analyzed_results[C.F_fy] = bm.jit(f, device=self.jit_device)
    return self.analyzed_results[C.F_fy]

  @property
  def F_vmap_fy(self):
    if C.F_vmap_fy not in self.analyzed_results:
      self.analyzed_results[C.F_vmap_fy] = bm.jit(bm.vmap(self.F_fy), device=self.jit_device)
    return self.analyzed_results[C.F_vmap_fy]

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
      @partial(bm.jacobian, argnums=(0, 1))
      def f_jacobian(*var_and_pars):
        return self.F_fx(*var_and_pars), self.F_fy(*var_and_pars)

      def call(*var_and_pars):
        var_and_pars = tuple((vp.value if isinstance(vp, bm.JaxArray) else vp) for vp in var_and_pars)
        return jnp.array(bm.jit(f_jacobian, device=self.jit_device)(*var_and_pars))

      self.analyzed_results[C.F_jacobian] = call
    return self.analyzed_results[C.F_jacobian]

  @property
  def F_fixed_point_opt(self):
    if C.F_fixed_point_opt not in self.analyzed_results:
      def opt_fun(xy_init, *args):
        # "xy_init" is a vector with length 2,
        # "args: is a tuple of scalar
        return minimize(self.F_fixed_point_aux, xy_init, args=args, method='BFGS')

      self.analyzed_results[C.F_fixed_point_opt] = opt_fun
    return self.analyzed_results[C.F_fixed_point_opt]

  @property
  def F_fixed_point_aux(self):
    if C.F_fixed_point_aux not in self.analyzed_results:
      def aux_fun(xy, *args):
        # "xy" is a vector with length 2,
        # "args": is a tuple of scalar
        dx = self.F_fx(xy[0], xy[1], *args)
        dy = self.F_fy(xy[0], xy[1], *args)
        # return (jnp.abs(dx) + jnp.abs(dy)).sum()
        return (dx ** 2 + dy ** 2).sum()

      self.analyzed_results[C.F_fixed_point_aux] = aux_fun
    return self.analyzed_results[C.F_fixed_point_aux]

  def _fp_filter(self, x_values, y_values, par_values, aux_filter=0.):
    if aux_filter > 0.:
      losses = self.F_vmap_fp_aux(jnp.stack([x_values, y_values]).T, *par_values)
      ids = jnp.where(losses < aux_filter)[0]
      x_values = x_values[ids]
      y_values = y_values[ids]
      par_values = tuple(p[ids] for p in par_values)
    return x_values, y_values, par_values

  def _get_fp_candidates(self, num_segments=1, num_rank=100):
    logger.warning(f"I am filtering out fixed point candidates with auxiliary function ...")
    all_xs = []
    all_ys = []
    all_ps = tuple([] for _ in range(len(self.target_par_names)))

    # points of variables and parameters
    xs = self.resolutions[self.x_var].value
    ys = self.resolutions[self.y_var].value
    P = tuple(self.resolutions[p].value for p in self.target_par_names)
    f_select = bm.jit(bm.vmap(lambda vals, ids: vals[ids], in_axes=(1, 1)))

    # num seguments
    if isinstance(num_segments, int):
      num_segments = tuple([num_segments] * len(self.target_par_names))
    assert isinstance(num_segments, (tuple, list)) and len(num_segments) == len(self.target_par_names)
    arg_lens = tuple(len(p) for p in P)
    arg_pre_len = tuple(int(np.ceil(l / num_segments[i])) for i, l in enumerate(arg_lens))
    arg_id_segments = tuple(np.arange(0, l, arg_pre_len[i]) for i, l in enumerate(arg_lens))
    arg_id_segments = tuple(ids.flatten() for ids in np.meshgrid(*arg_id_segments))
    if len(arg_id_segments) == 0:
      arg_id_segments = ((0,),)
    for _j, ids in enumerate(zip(*arg_id_segments)):
      if len(arg_id_segments[0]) > 1:
        logger.warning(f"{C.prefix}segment {_j} ...")

      ps = tuple(p[ids[i]: ids[i] + arg_pre_len[i]] for i, p in enumerate(P))
      # change the position of meshgrid values
      vps = tuple((v.value if isinstance(v, bm.JaxArray) else v) for v in ((xs, ys) + ps))
      mesh_values = jnp.meshgrid(*vps)
      mesh_values = tuple(jnp.moveaxis(m, 0, 1) for m in mesh_values)
      mesh_values = tuple(m.flatten() for m in mesh_values)
      # function outputs
      losses = self.F_vmap_fp_aux(jnp.stack([mesh_values[0], mesh_values[1]]).T, *mesh_values[2:])
      shape = (len(xs) * len(ys), -1)
      losses = losses.reshape(shape)
      argsorts = jnp.argsort(losses, axis=0)[:num_rank]
      all_xs.append(f_select(mesh_values[0].reshape(shape), argsorts).flatten())
      all_ys.append(f_select(mesh_values[1].reshape(shape), argsorts).flatten())
      for i, p in enumerate(ps):
        all_ps[i].append(f_select(mesh_values[i+2].reshape(shape), argsorts).flatten())
    all_xys = jnp.vstack([jnp.concatenate(all_xs), jnp.concatenate(all_ys)]).T
    all_ps = tuple(jnp.concatenate(p) for p in all_ps)
    return (all_xys, all_ps)

  def _get_fx_nullcline_points(self, coords=None, tol=1e-7, num_segments=1, fp_aux_filter=0.):
    coords = (self.x_var + '-' + self.y_var) if coords is None else coords
    key = C.fx_nullcline_points + ',' + coords
    if key not in self.analyzed_results:
      logger.warning("I am evaluating fx-nullcline by optimization ...")

      all_losses = []
      all_x_values_in_fx = []
      all_y_values_in_fx = []
      all_p_values_in_fx = tuple([] for _ in range(len(self.target_par_names)))

      # auxiliary functions
      f2 = lambda y, x, *pars: self.F_fx(x, y, *pars)
      vmap_f2 = bm.jit(bm.vmap(f2), device=self.jit_device)
      vmap_brentq_f2 = bm.jit(bm.vmap(utils.jax_brentq(f2)), device=self.jit_device)
      vmap_brentq_f1 = bm.jit(bm.vmap(utils.jax_brentq(self.F_fx)), device=self.jit_device)

      # points of variables and parameters
      xs = self.resolutions[self.x_var].value
      ys = self.resolutions[self.y_var].value
      P = tuple(self.resolutions[p].value for p in self.target_par_names)

      # num seguments
      if isinstance(num_segments, int):
        num_segments = tuple([num_segments] * len(self.target_par_names))
      assert isinstance(num_segments, (tuple, list)) and len(num_segments) == len(self.target_par_names)
      arg_lens = tuple(len(p) for p in P)
      arg_pre_len = tuple(int(np.ceil(l / num_segments[i])) for i, l in enumerate(arg_lens))
      arg_id_segments = tuple(np.arange(0, l, arg_pre_len[i]) for i, l in enumerate(arg_lens))
      arg_id_segments = tuple(ids.flatten() for ids in np.meshgrid(*arg_id_segments))
      if len(arg_id_segments) == 0:
        arg_id_segments = ((0,),)
      for _j, ids in enumerate(zip(*arg_id_segments)):
        if len(arg_id_segments[0]) > 1:
          logger.warning(f"{C.prefix}segment {_j} ...")

        Ps = tuple(p[ids[i]: ids[i] + arg_pre_len[i]] for i, p in enumerate(P))
        if coords == self.x_var + '-' + self.y_var:
          x0s, x1s, vps = utils.brentq_candidates(self.F_vmap_fx, *((xs, ys) + Ps))
          x_values_in_fx, out_args = utils.brentq_roots2(vmap_brentq_f1, x0s, x1s, *vps)
          y_values_in_fx = out_args[0]
          p_values_in_fx = out_args[1:]
          x_values_in_fx, y_values_in_fx, p_values_in_fx = \
            self._fp_filter(x_values_in_fx, y_values_in_fx, p_values_in_fx, fp_aux_filter)
        elif coords == self.y_var + '-' + self.x_var:
          x0s, x1s, vps = utils.brentq_candidates(vmap_f2, *((ys, xs) + Ps))
          y_values_in_fx, out_args = utils.brentq_roots2(vmap_brentq_f2, x0s, x1s, *vps)
          x_values_in_fx = out_args[0]
          p_values_in_fx = out_args[1:]
          x_values_in_fx, y_values_in_fx, p_values_in_fx = \
            self._fp_filter(x_values_in_fx, y_values_in_fx, p_values_in_fx, fp_aux_filter)
        else:
          raise ValueError
        losses = self.F_vmap_fx(x_values_in_fx, y_values_in_fx, *p_values_in_fx)
        all_losses.append(losses)
        all_x_values_in_fx.append(x_values_in_fx)
        all_y_values_in_fx.append(y_values_in_fx)
        for i, arg in enumerate(p_values_in_fx):
          all_p_values_in_fx[i].append(arg)

      all_losses = jnp.concatenate(all_losses)
      all_x_values_in_fx = jnp.concatenate(all_x_values_in_fx)
      all_y_values_in_fx = jnp.concatenate(all_y_values_in_fx)
      all_p_values_in_fx = tuple(jnp.concatenate(p) for p in all_p_values_in_fx)
      ids = jnp.where(all_losses < tol)[0]
      all_x_values_in_fx = all_x_values_in_fx[ids]
      all_y_values_in_fx = all_y_values_in_fx[ids]
      all_p_values_in_fx = tuple(a[ids] for a in all_p_values_in_fx)
      all_xy_values = jnp.stack([all_x_values_in_fx, all_y_values_in_fx]).T
      self.analyzed_results[key] = (all_xy_values,) + all_p_values_in_fx
    return self.analyzed_results[key]

  def _get_fy_nullcline_points(self, coords=None, tol=1e-7, num_segments=1, fp_aux_filter=0.):
    coords = (self.x_var + '-' + self.y_var) if coords is None else coords
    key = C.fy_nullcline_points + ',' + coords
    if key not in self.analyzed_results:
      logger.warning("I am evaluating fy-nullcline by optimization ...")

      all_losses = []
      all_x_values_in_fy = []
      all_y_values_in_fy = []
      all_p_values_in_fy = tuple([] for _ in range(len(self.target_par_names)))

      # auxiliary functions
      f2 = lambda y, x, *pars: self.F_fy(x, y, *pars)
      vmap_f2 = bm.jit(bm.vmap(f2), device=self.jit_device)
      vmap_brentq_f2 = bm.jit(bm.vmap(utils.jax_brentq(f2)), device=self.jit_device)
      vmap_brentq_f1 = bm.jit(bm.vmap(utils.jax_brentq(self.F_fy)), device=self.jit_device)

      if isinstance(num_segments, int):
        num_segments = tuple([num_segments] * len(self.target_par_names))
      assert isinstance(num_segments, (tuple, list)) and len(num_segments) == len(self.target_par_names)
      xs = self.resolutions[self.x_var].value
      ys = self.resolutions[self.y_var].value
      P = tuple(self.resolutions[p].value for p in self.target_par_names)
      arg_lens = tuple(len(p) for p in P)
      arg_pre_len = tuple(int(np.ceil(l / num_segments[i])) for i, l in enumerate(arg_lens))
      arg_id_segments = tuple(np.arange(0, l, arg_pre_len[i]) for i, l in enumerate(arg_lens))
      arg_id_segments = tuple(ids.flatten() for ids in np.meshgrid(*arg_id_segments))
      if len(arg_id_segments) == 0:
        arg_id_segments = ((0,),)
      for _j, ids in enumerate(zip(*arg_id_segments)):
        if len(arg_id_segments[0]) > 1:
          logger.warning(f"{C.prefix}segment {_j} ...")

        Ps = tuple(p[ids[i]: ids[i] + arg_pre_len[i]] for i, p in enumerate(P))
        if coords == self.x_var + '-' + self.y_var:
          starts, ends, vps = utils.brentq_candidates(self.F_vmap_fy, *((xs, ys) + Ps))
          x_values_in_fy, out_args = utils.brentq_roots2(vmap_brentq_f1, starts, ends, *vps)
          y_values_in_fy = out_args[0]
          p_values_in_fy = out_args[1:]
          x_values_in_fy, y_values_in_fy, p_values_in_fy = \
            self._fp_filter(x_values_in_fy, y_values_in_fy, p_values_in_fy, fp_aux_filter)
        elif coords == self.y_var + '-' + self.x_var:
          starts, ends, vps = utils.brentq_candidates(vmap_f2, *((ys, xs) + Ps))
          y_values_in_fy, out_args = utils.brentq_roots2(vmap_brentq_f2, starts, ends, *vps)
          x_values_in_fy = out_args[0]
          p_values_in_fy = out_args[1:]
          x_values_in_fy, y_values_in_fy, p_values_in_fy = \
            self._fp_filter(x_values_in_fy, y_values_in_fy, p_values_in_fy, fp_aux_filter)
        else:
          raise ValueError
        losses = self.F_vmap_fy(x_values_in_fy, y_values_in_fy, *p_values_in_fy)
        all_losses.append(losses)
        all_x_values_in_fy.append(x_values_in_fy)
        all_y_values_in_fy.append(y_values_in_fy)
        for i, arg in enumerate(p_values_in_fy):
          all_p_values_in_fy[i].append(arg)
      all_losses = jnp.concatenate(all_losses)
      all_x_values_in_fy = jnp.concatenate(all_x_values_in_fy)
      all_y_values_in_fy = jnp.concatenate(all_y_values_in_fy)
      all_p_values_in_fy = tuple(jnp.concatenate(p) for p in all_p_values_in_fy)
      ids = jnp.where(all_losses < tol)[0]
      all_x_values_in_fy = all_x_values_in_fy[ids]
      all_y_values_in_fy = all_y_values_in_fy[ids]
      all_p_values_in_fy = tuple(a[ids] for a in all_p_values_in_fy)
      all_xy_values = jnp.stack([all_x_values_in_fy, all_y_values_in_fy]).T
      self.analyzed_results[key] = (all_xy_values,) + all_p_values_in_fy
    return self.analyzed_results[key]

  def _get_fixed_points(self, candidates, *args, tol_loss=1e-7,
                        tol_unique=1e-2, loss_screen=None, num_segment=1):
    """Get the fixed points according to the initial ``candidates`` and the parameter setting ``args``.

    "candidates" and "args" can be obtained through:

    >>> all_candidates = []
    >>> all_par1 = []
    >>> all_par2 = []
    >>> for p1 in par1_list:
    >>>   for p2 in par2_list:
    >>>     nullcline_points = _get_nullcline_points(p1, p2)
    >>>     all_candidates.append(nullcline_points)
    >>>     all_par1.append(jnp.ones_like(nullcline_points) * p1)
    >>>     all_par2.append(jnp.ones_like(nullcline_points) * p2)

    Parameters
    ----------
    candidates: np.ndarray, jnp.ndarray
      The candidate points (batched) to optimize, like the nullcline points.
    args : tuple
      The parameters (batched).
    tol_loss : float
    tol_unique : float
    loss_screen : float, optional

    Returns
    -------
    res : tuple
      The fixed point results.
    """
    logger.warning("I am trying to find fixed points by optimization ...")
    logger.warning(f"{C.prefix}There are {len(candidates)} candidates")

    candidates = jnp.asarray(candidates)
    args = tuple(jnp.asarray(a) for a in args)

    all_ids = []
    all_fps = []
    all_args = tuple([] for _ in range(len(args)))
    seg_len = int(np.ceil(len(candidates) / num_segment))
    segment_ids = np.arange(0, len(candidates), seg_len)
    selected_ids = jnp.arange(len(candidates))

    for _j, i in enumerate(segment_ids):
      if len(segment_ids) > 1:
        logger.warning(f"{C.prefix}segment {_j} ...")
      seg_fps = candidates[i: i + seg_len]
      seg_args = tuple(a[i: i + seg_len] for a in args)
      seg_ids = selected_ids[i: i + seg_len]

      if loss_screen is not None:
        # screen by the function loss
        losses = self.F_vmap_fp_aux(seg_fps, *seg_args)
        ids = jnp.where(losses < loss_screen)[0]
        seg_fps = seg_fps[ids]
        seg_args = tuple(a[ids] for a in seg_args)
        seg_ids = seg_ids[ids]
      if len(seg_fps):
        # optimization
        seg_fps = self.F_vmap_fp_opt(seg_fps, *seg_args)
        # loss
        losses = self.F_vmap_fp_aux(seg_fps.x, *seg_args)
        # valid indices
        ids = jnp.where(losses <= tol_loss)[0]
        seg_ids = seg_ids[ids]
        all_fps.append(seg_fps.x[ids])
        all_ids.append(seg_ids)
        for i in range(len(all_args)):
          all_args[i].append(seg_args[i][ids])
    all_fps = jnp.concatenate(all_fps)
    all_ids = jnp.concatenate(all_ids)
    all_args = tuple(jnp.concatenate(args) for args in all_args)
    return all_fps, all_ids, all_args


class Num3DAnalyzer(Num2DAnalyzer):
  def __init__(self, *args, **kwargs):
    super(Num3DAnalyzer, self).__init__(*args, **kwargs)
    if len(self.target_vars) < 3:
      raise errors.AnalyzerError(f'{Num1DAnalyzer.__name__} only supports dynamical system '
                                 f'with >= 3 variables. But we got {len(self.target_vars)} '
                                 f'variables in {self.model}.')
    self.z_var = self.target_var_names[2]

  @property
  def F_fz(self):
    """The function to evaluate :math:`f_y(*\mathrm{vars}, *\mathrm{pars})`."""
    if C.F_fz not in self.analyzed_results:
      variables, arguments = _get_args(self.model.F[self.z_var])
      wrapper = _std_derivative(arguments, self.target_var_names, self.target_par_names)
      f = wrapper(self.model.F[self.z_var])
      f = partial(f, **(self.pars_update + self.fixed_vars))
      self.analyzed_results[C.F_fz] = bm.jit(f, device=self.jit_device)
    return self.analyzed_results[C.F_fz]

  def fz_signs(self, pars=(), cache=False):
    xyz = tuple(self.resolutions.values())
    return utils.get_sign2(self.F_fz, *xyz, args=pars)
