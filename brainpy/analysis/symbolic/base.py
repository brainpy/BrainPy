# -*- coding: utf-8 -*-

import logging
from copy import deepcopy

import numpy as np

from brainpy import errors, math, tools
from brainpy.analysis import solver
from brainpy.analysis.symbolic import utils

try:
  import sympy
  from brainpy.integrators import analysis_by_sympy
except ModuleNotFoundError:
  sympy = None
  analysis_by_sympy = None

logger = logging.getLogger('brainpy.analysis')

__all__ = [
  'BaseAnalyzer',
  'Base1DAnalyzer',
  'Base2DAnalyzer',
]


class BaseAnalyzer(object):
  """Dynamics Analyzer for Neuron Models.

  This class is a base class which aims for analyze the analysis in
  neuron models. A neuron model is characterized by a series of dynamical
  variables and parameters:

  .. backend::

      {dF \over dt} = F(v_1, v_2, ..., p_1, p_2, ...)

  where :backend:`v_1, v_2` are variables, :backend:`p_1, p_2` are parameters.

  Parameters
  ----------
  model_or_integrals : simulation.Population, function, functions
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
  numerical_resolution : float, dict
      The resolution for numerical iterative solvers. Default is 0.1. It can set the
      numerical resolution of dynamical variables or dynamical parameters. For example,
      set ``numerical_resolution=0.1`` will generalize it to all variables and parameters;
      set ``numerical_resolution={var1: 0.1, var2: 0.2, par1: 0.1, par2: 0.05}`` will specify
      the particular resolutions to variables and parameters. Moreover, you can also set
      ``numerical_resolution={var1: np.array([...]), var2: 0.1}`` to specify the search points
      need to explore for variable `var1`. This will be useful to set sense search points at some
      inflection points.
  options : dict, optional
      The other setting parameters, which includes:

          perturbation
              float. The small perturbation used to solve the function derivative.
          sympy_solver_timeout
              float, with the unit of second. The maximum  time allowed to use sympy solver
              to get the variable relationship.
          escape_sympy_solver
              bool. Whether escape to use sympy solver, and directly use numerical optimization
              method to solve the nullcline and fixed points.
          lim_scale
              float. The axis limit scale factor. Default is 1.05. The setting means
              the axes will be clipped to ``[var_min * (1-lim_scale)/2, var_max * (var_max-1)/2]``.
  """

  def __init__(self,
               model_or_integrals,
               target_vars,
               fixed_vars=None,
               target_pars=None,
               pars_update=None,
               numerical_resolution=0.1,
               options=None):

    if (sympy is None) or (analysis_by_sympy is None):
      raise errors.PackageMissingError('"SymPy" must be installed for dynamics analysis.')

    # model
    # -----
    if isinstance(model_or_integrals, utils.DynamicModel):
      self.model = model_or_integrals
    elif (isinstance(model_or_integrals, (tuple, list)) and callable(model_or_integrals[0])) or \
        callable(model_or_integrals):
      self.model = utils.transform_integrals_to_model(model_or_integrals)
    else:
      raise ValueError

    # target variables
    # ----------------
    if not isinstance(target_vars, dict):
      raise errors.BrainPyError('"target_vars" must be a dict, with the format of '
                                '{"var1": (var1_min, var1_max)}.')
    self.target_vars = target_vars
    self.dvar_names = list(self.target_vars.keys())
    for key in self.target_vars.keys():
      if key not in self.model.variables:
        raise ValueError(f'{key} is not a dynamical variable in {self.model}.')

    # fixed variables
    # ----------------
    if not isinstance(fixed_vars, dict):
      raise errors.BrainPyError('"fixed_vars" must be a dict with the format '
                                'of {"var1": val1, "var2": val2}.')
    for key in fixed_vars.keys():
      if key not in self.model.variables:
        raise ValueError(f'{key} is not a dynamical variable in {self.model}.')
    self.fixed_vars = fixed_vars

    # check duplicate
    for key in self.fixed_vars.keys():
      if key in self.target_vars:
        raise errors.BrainPyError(f'"{key}" is defined as a target variable in "target_vars", '
                                  f'but also defined as a fixed variable in "fixed_vars".')

    # equations of dynamical variables
    # --------------------------------
    var2eq = {ana.var_name: ana for ana in self.model.analyzers}
    self.target_eqs = tools.DictPlus()
    for key in self.target_vars.keys():
      if key not in var2eq:
        raise errors.BrainPyError(f'target "{key}" is not a dynamical variable.')
      diff_eq = var2eq[key]
      sub_exprs = diff_eq.get_f_expressions(substitute_vars=list(self.target_vars.keys()))
      old_exprs = diff_eq.get_f_expressions(substitute_vars=None)
      self.target_eqs[key] = tools.DictPlus(sub_exprs=sub_exprs,
                                            old_exprs=old_exprs,
                                            diff_eq=diff_eq,
                                            func_name=diff_eq.func_name)

    # parameters to update
    # ---------------------
    if pars_update is None:
      pars_update = dict()
    if not isinstance(pars_update, dict):
      raise errors.BrainPyError('"pars_update" must be a dict with the format '
                                'of {"par1": val1, "par2": val2}.')
    # keys = set(list(pars_update.keys()))
    # keys.update(list(self.model.pars_update.keys()))
    # new_pars_update = {}
    # for key in keys:
    #   if key not in pars_update and key not in fixed_vars:
    #     new_pars_update[key] = self.model.pars_update[key]
    #   else:
    #     new_pars_update[key] = pars_update[key]
    for key in pars_update.keys():
      if (key not in self.model.scopes) and (key not in self.model.parameters):
        raise errors.BrainPyError(f'"{key}" is not a valid parameter in "{self.model}" model.')
    self.pars_update = pars_update

    # dynamical parameters
    # ---------------------
    if target_pars is None:
      target_pars = dict()
    if not isinstance(target_pars, dict):
      raise errors.BrainPyError('"target_pars" must be a dict with the format of {"par1": (val1, val2)}.')
    for key in target_pars.keys():
      if (key not in self.model.scopes) and (key not in self.model.parameters):
        raise errors.BrainPyError(f'"{key}" is not a valid parameter in "{self.model}" model.')
    self.target_pars = target_pars
    self.dpar_names = list(self.target_pars.keys())

    # check duplicate
    for key in self.pars_update.keys():
      if key in self.target_pars:
        raise errors.BrainPyError(f'"{key}" is defined as a target parameter in "target_pars", '
                                  f'but also defined as a fixed parameter in "pars_update".')

    # resolutions for numerical methods
    # ---------------------------------
    self.resolutions = dict()
    if isinstance(numerical_resolution, float):
      for key, lim in self.target_vars.items():
        self.resolutions[key] = np.arange(*lim, numerical_resolution)
      for key, lim in self.target_pars.items():
        self.resolutions[key] = np.arange(*lim, numerical_resolution)
    elif isinstance(numerical_resolution, dict):
      for key in self.dvar_names + self.dpar_names:
        if key not in numerical_resolution:
          raise errors.BrainPyError(f'Must provide the resolution setting of dynamical '
                                    f'variable/parameter "{key}", '
                                    f'but only get {numerical_resolution}.')
        resolution = numerical_resolution[key]
        if isinstance(resolution, float):
          lim = self.target_vars[key] if key in self.target_vars else self.target_pars[key]
          self.resolutions[key] = np.arange(*lim, resolution)
        elif isinstance(resolution, np.ndarray):
          if not np.ndim(resolution) == 1:
            raise errors.BrainPyError(f'resolution must be a 1D vector, but get its '
                                      f'shape with {resolution.shape}.')
          self.resolutions[key] = np.ascontiguousarray(resolution)
        else:
          raise errors.BrainPyError(f'Unknown resolution setting: {key}: {resolution}')
    else:
      raise errors.BrainPyError(f'Unknown resolution type: {type(numerical_resolution)}')

    # a dict to store the analyzed results
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
    # 'y_by_x_in_y_eq' :
    # 'x_by_y_in_y_eq' :
    # 'y_by_x_in_x_eq' :
    # 'x_by_y_in_x_eq' :
    self.analyzed_results = tools.DictPlus()

    # other settings
    # --------------
    if options is None:
      options = dict()
    self.options = tools.DictPlus()
    self.options['perturbation'] = options.get('perturbation', 1e-6)
    self.options['sympy_solver_timeout'] = options.get('sympy_solver_timeout', 5)  # s
    self.options['escape_sympy_solver'] = options.get('escape_sympy_solver', False)
    self.options['lim_scale'] = options.get('lim_scale', 1.05)


class Base1DAnalyzer(BaseAnalyzer):
  """Neuron analysis analyzer for 1D system.

  It supports the analysis of 1D dynamical system.

  .. backend::

      {dx \over dt} = f(x, t)
  """

  def __init__(self, *args, **kwargs):
    super(Base1DAnalyzer, self).__init__(*args, **kwargs)

    self.x_var = self.dvar_names[0]
    self.x_eq_group = self.target_eqs[self.x_var]

  def get_f_dx(self):
    """Get the derivative function of the first variable. """
    if 'dxdt' not in self.analyzed_results:
      scope = deepcopy(self.pars_update)
      scope.update(self.fixed_vars)
      # scope.update(analysis_by_sympy.get_mapping_scope())
      scope.update(self.x_eq_group.diff_eq.func_scope)
      scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      func_code = f'def func({argument}):\n'
      for expr in self.x_eq_group.old_exprs[:-1]:
        func_code += f'  {expr.var_name} = {expr.code}\n'
      func_code += f'  return {self.x_eq_group.old_exprs[-1].code}'
      exec(compile(func_code, '', 'exec'), scope)
      func = scope['func']
      self.analyzed_results['dxdt'] = func
    return self.analyzed_results['dxdt']

  def get_f_dfdx(self, origin=True):
    """Get the derivative of ``f`` by variable ``x``. """
    if 'dfdx' not in self.analyzed_results:
      x_var = self.dvar_names[0]
      x_symbol = sympy.Symbol(x_var, real=True)
      x_eq = self.x_eq_group.sub_exprs[-1].code
      x_eq = analysis_by_sympy.str2sympy(x_eq)

      eq_x_scope = deepcopy(self.pars_update)
      eq_x_scope.update(self.fixed_vars)
      # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_x_scope.update(self.x_eq_group['diff_eq'].func_scope)
      eq_x_scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      time_out = self.options.sympy_solver_timeout

      sympy_failed = True
      if not self.options.escape_sympy_solver and not x_eq.contain_unknown_func:
        try:
          logger.info(f'SymPy solve derivative of "{self.x_eq_group.func_name}'
                      f'({argument})" by "{x_var}", ')
          x_eq = x_eq.expr
          f = utils.timeout(time_out)(lambda: sympy.diff(x_eq, x_symbol))
          dfxdx_expr = f()

          # check
          all_vars = set(eq_x_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(analysis_by_sympy.sympy2str(dfxdx_expr), all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            sympy_failed = True
          else:
            logger.info('\tsuccess.')
            func_codes = [f'def dfdx({argument}):']
            for expr in self.x_eq_group.sub_exprs[:-1]:
              func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {analysis_by_sympy.sympy2str(dfxdx_expr)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
            dfdx = eq_x_scope['dfdx']
            sympy_failed = False
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {time_out} s timeout.')
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')

      if sympy_failed:
        scope = dict(_fx=self.get_f_dx(), perturb=self.options.perturbation, math=math)
        func_codes = [f'def dfdx({argument}):']
        if not origin:
          func_codes.append(f'origin = _fx({argument})')
        func_codes.append(f'disturb = _fx({x_var}+perturb, '
                          f'{",".join(self.dvar_names[1:] + self.dpar_names)})')
        if not origin:
          func_codes.append(f'return (disturb - origin) / perturb')
        else:
          func_codes.append(f'return disturb / perturb')
        exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
        dfdx = scope['dfdx']
      self.analyzed_results['dfdx'] = dfdx
    return self.analyzed_results['dfdx']

  def get_f_fixed_point(self):
    """Get the function to solve the fixed point.
    """

    if 'fixed_point' not in self.analyzed_results:
      x_eq = analysis_by_sympy.str2sympy(self.x_eq_group.sub_exprs[-1].code)

      scope = deepcopy(self.pars_update)
      scope.update(self.fixed_vars)
      # scope.update(analysis_by_sympy.get_mapping_scope())
      scope.update(self.x_eq_group.diff_eq.func_scope)
      scope['math'] = math

      timeout_len = self.options.sympy_solver_timeout
      argument1 = ', '.join(self.dvar_names + self.dpar_names)
      argument2 = ", ".join(self.dvar_names[1:] + self.dpar_names)

      sympy_failed = True
      if not self.options.escape_sympy_solver and not x_eq.contain_unknown_func:
        try:
          logger.info(f'SymPy solve "{self.x_eq_group.func_name}({argument1}) = 0" '
                      f'to "{self.x_var} = f({argument2})", ')

          # solver
          f = utils.timeout(timeout_len)(
            lambda: sympy.solve(x_eq.expr, sympy.Symbol(self.x_var, real=True)))
          results = f()
          for res in results:
            all_vars = set(scope.keys())
            all_vars.update(self.dvar_names + self.dpar_names)
            if utils.contain_unknown_symbol(analysis_by_sympy.sympy2str(res), all_vars):
              logger.info('\tfailed because contain unknown symbols.')
              sympy_failed = True
              break
          else:
            logger.info('\tsuccess.')
            # function codes
            func_codes = [f'def solve_x({argument2}):']
            for expr in self.x_eq_group.sub_exprs[:-1]:
              func_codes.append(f'{expr.var_name} = {expr.code}')
            result_expr = ', '.join([analysis_by_sympy.sympy2str(expr)
                                     for expr in results])
            func_codes.append(f'_res_ = {result_expr}')
            func_codes.append(f'return math.array(_res_)')

            # function compilation
            exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
            self.analyzed_results['fixed_point'] = scope['solve_x']
            sympy_failed = False
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')
          sympy_failed = True
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {timeout_len} s timeout.')
          sympy_failed = True

      if sympy_failed:
        # function codes
        func_codes = [f'def optimizer_x({argument1}):']
        for expr in self.x_eq_group.old_exprs[:-1]:
          func_codes.append(f'{expr.var_name} = {expr.code}')
        func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')

        # function compile
        optimizer = utils.jit_compile(scope, '\n  '.join(func_codes), 'optimizer_x')
        xs = self.resolutions[self.x_var]

        def f(*args):
          # `args` corresponds to `self.dvar_names[1:] + self.dpar_names`
          x_values = solver.find_root_of_1d(optimizer, xs, args)
          return np.array(x_values)

        self.analyzed_results['fixed_point'] = f

    return self.analyzed_results['fixed_point']


class Base2DAnalyzer(Base1DAnalyzer):
  """Neuron analysis analyzer for 2D system.

  It supports the analysis of 2D dynamical system.

  .. backend::

      {dx \over dt} = f(x, t, y)

      {dy \over dt} = g(y, t, x)

  Parameters
  ----------

  options : dict, optional
      The other setting parameters, which includes:

          shgo_args
              dict. Arguments of `shgo` optimization method, which can be used to set the
              fields of: constraints, n, iters, callback, minimizer_kwargs, options,
              sampling_method.
          show_shgo
              bool. whether print the shgo's value.
          fl_tol
              float. The tolerance of the function value to recognize it as a condidate of
              function root point.
          xl_tol
              float. The tolerance of the l2 norm distances between this point and previous
              points. If the norm distances are all bigger than `xl_tol` means this
              point belong to a new function root point.

  """

  def __init__(self, *args, **kwargs):
    super(Base2DAnalyzer, self).__init__(*args, **kwargs)

    self.y_var = self.dvar_names[1]
    self.y_eq_group = self.target_eqs[self.y_var]

    # options
    # ---------

    options = kwargs.get('options', dict())
    if options is None:
      options = dict()
    self.options['shgo_args'] = options.get('shgo_args', dict())
    self.options['show_shgo'] = options.get('show_shgo', False)
    self.options['fl_tol'] = options.get('fl_tol', 1e-6)
    self.options['xl_tol'] = options.get('xl_tol', 1e-4)
    for a in ['y_by_x_in_y_eq', 'y_by_x_in_x_eq', 'x_by_y_in_x_eq', 'x_by_y_in_y_eq']:
      if a in options:
        # check "subs"
        subs = options[a]
        if isinstance(subs, str):
          subs = [subs]
        elif isinstance(subs, (tuple, list)):
          subs = subs
        else:
          raise ValueError(f'Unknown setting of "{a}": {subs}')

        # check "f"
        scope = deepcopy(self.pars_update)
        scope.update(self.fixed_vars)
        scope['math'] = math
        # scope.update(analysis_by_sympy.get_mapping_scope())
        if a.endswith('y_eq'):
          scope.update(self.y_eq_group['diff_eq'].func_scope)
        else:
          scope.update(self.x_eq_group['diff_eq'].func_scope)

        # function code
        argument = ",".join(self.dvar_names[2:] + self.dpar_names)
        if a.startswith('y_by_x'):
          func_codes = [f'def func({self.x_var}, {argument}):\n']
        else:
          func_codes = [f'def func({self.y_var}, {argument}):\n']
        func_codes.extend(subs)
        func_codes.append(f'return {subs[-1].split("=")[0]}')

        # function compilation
        exec(compile("\n  ".join(func_codes), '', 'exec'), scope)
        f = scope['func']

        # results
        self.analyzed_results[a] = tools.DictPlus(status='sympy_success', subs=subs, f=f)

  def get_f_dy(self):
    """Get the derivative function of the second variable. """
    if 'dydt' not in self.analyzed_results:
      if len(self.dvar_names) < 2:
        raise errors.BrainPyError(f'Analyzer only receives {len(self.dvar_names)} '
                                  f'dynamical variables, cannot get "dy".')
      y_var = self.dvar_names[1]
      scope = deepcopy(self.pars_update)
      scope.update(self.fixed_vars)
      # scope.update(analysis_by_sympy.get_mapping_scope())
      scope.update(self.y_eq_group.diff_eq.func_scope)
      scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      func_code = f'def func({argument}):\n'
      for expr in self.y_eq_group.old_exprs[:-1]:
        func_code += f'  {expr.var_name} = {expr.code}\n'
      func_code += f'  return {self.y_eq_group.old_exprs[-1].code}'
      exec(compile(func_code, '', 'exec'), scope)
      self.analyzed_results['dydt'] = scope['func']
    return self.analyzed_results['dydt']

  def get_f_dfdy(self, origin=True):
    """Get the derivative of ``f`` by variable ``y``. """
    if 'dfdy' not in self.analyzed_results:
      x_var = self.dvar_names[0]
      y_var = self.dvar_names[1]
      y_symbol = sympy.Symbol(y_var, real=True)
      x_eq = self.target_eqs[x_var].sub_exprs[-1].code
      x_eq = analysis_by_sympy.str2sympy(x_eq)

      eq_x_scope = deepcopy(self.pars_update)
      eq_x_scope.update(self.fixed_vars)
      # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_x_scope.update(self.x_eq_group['diff_eq'].func_scope)
      eq_x_scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      time_out = self.options.sympy_solver_timeout

      sympy_failed = True
      if not self.options.escape_sympy_solver and not x_eq.contain_unknown_func:
        try:
          logger.info(f'SymPy solve derivative of "{self.x_eq_group.func_name}'
                      f'({argument})" by "{y_var}", ')
          x_eq = x_eq.expr
          f = utils.timeout(time_out)(lambda: sympy.diff(x_eq, y_symbol))
          dfxdy_expr = f()

          # check
          all_vars = set(eq_x_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(analysis_by_sympy.sympy2str(dfxdy_expr), all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            sympy_failed = True
          else:
            logger.info('\tsuccess.')
            func_codes = [f'def dfdy({argument}):']
            for expr in self.x_eq_group.sub_exprs[:-1]:
              func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {analysis_by_sympy.sympy2str(dfxdy_expr)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
            dfdy = eq_x_scope['dfdy']
            sympy_failed = False
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {time_out} s timeout.')
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')

      if sympy_failed:
        scope = dict(_fx=self.get_f_dx(), perturb=self.options.perturbation, math=math)
        func_codes = [f'def dfdy({argument}):']
        if not origin:
          func_codes.append(f'origin = _fx({argument})')
        func_codes.append(f'disturb = _fx({x_var}, {y_var}+perturb, '
                          f'{",".join(self.dvar_names[2:] + self.dpar_names)})')
        if not origin:
          func_codes.append(f'return (disturb - origin) / perturb')
        else:
          func_codes.append(f'return disturb / perturb')
        exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
        dfdy = scope['dfdy']

      self.analyzed_results['dfdy'] = dfdy
    return self.analyzed_results['dfdy']

  def get_f_dgdx(self, origin=True):
    """Get the derivative of ``g`` by variable ``x``. """
    if 'dgdx' not in self.analyzed_results:
      x_var = self.dvar_names[0]
      x_symbol = sympy.Symbol(x_var, real=True)
      y_var = self.dvar_names[1]
      y_eq = self.target_eqs[y_var].sub_exprs[-1].code
      y_eq = analysis_by_sympy.str2sympy(y_eq)

      eq_y_scope = deepcopy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_y_scope.update(self.y_eq_group['diff_eq'].func_scope)
      eq_y_scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      time_out = self.options.sympy_solver_timeout

      sympy_failed = True
      if not self.options.escape_sympy_solver and not y_eq.contain_unknown_func:
        try:
          logger.info(f'SymPy solve derivative of "{self.y_eq_group.func_name}'
                      f'({argument})" by "{x_var}", ')
          y_eq = y_eq.expr
          f = utils.timeout(time_out)(lambda: sympy.diff(y_eq, x_symbol))
          dfydx_expr = f()

          # check
          all_vars = set(eq_y_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(analysis_by_sympy.sympy2str(dfydx_expr), all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            sympy_failed = True
          else:
            logger.info('\tsuccess.')
            func_codes = [f'def dgdx({argument}):']
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {analysis_by_sympy.sympy2str(dfydx_expr)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
            dgdx = eq_y_scope['dgdx']
            sympy_failed = False
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {time_out} s timeout.')
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')

      if sympy_failed:
        scope = dict(_fy=self.get_f_dy(), perturb=self.options.perturbation, math=math)
        func_codes = [f'def dgdx({argument}):']
        if not origin:
          func_codes.append(f'origin = _fy({argument})')
        func_codes.append(f'disturb = _fy({x_var}+perturb, '
                          f'{",".join(self.dvar_names[1:] + self.dpar_names)})')
        if not origin:
          func_codes.append(f'return (disturb - origin) / perturb')
        else:
          func_codes.append(f'return disturb / perturb')
        exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
        dgdx = scope['dgdx']

      self.analyzed_results['dgdx'] = dgdx
    return self.analyzed_results['dgdx']

  def get_f_dgdy(self, origin=True):
    """Get the derivative of ``g`` by variable ``y``. """
    if 'dgdy' not in self.analyzed_results:
      x_var = self.dvar_names[0]
      y_var = self.dvar_names[1]
      y_symbol = sympy.Symbol(y_var, real=True)
      y_eq = self.target_eqs[y_var].sub_exprs[-1].code
      y_eq = analysis_by_sympy.str2sympy(y_eq)

      eq_y_scope = deepcopy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_y_scope.update(self.y_eq_group['diff_eq'].func_scope)
      eq_y_scope['math'] = math

      argument = ', '.join(self.dvar_names + self.dpar_names)
      argument2 = ', '.join(self.dvar_names[2:] + self.dpar_names)
      time_out = self.options.sympy_solver_timeout

      sympy_failed = True
      if not self.options.escape_sympy_solver and not y_eq.contain_unknown_func:
        try:
          logger.info(f'\tSymPy solve derivative of "{self.y_eq_group.func_name}'
                      f'({argument})" by "{y_var}", ')
          y_eq = y_eq.expr
          f = utils.timeout(time_out)(lambda: sympy.diff(y_eq, y_symbol))
          dfydx_expr = f()

          # check
          all_vars = set(eq_y_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(analysis_by_sympy.sympy2str(dfydx_expr), all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            sympy_failed = True
          else:
            logger.info('\tsuccess.')
            func_codes = [f'def dgdy({argument}):']
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_codes.append(f'{expr.var_name} = {expr.code}')
            func_codes.append(f'return {analysis_by_sympy.sympy2str(dfydx_expr)}')
            exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
            dgdy = eq_y_scope['dgdy']
            sympy_failed = False
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {time_out} s timeout.')
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')

      if sympy_failed:
        scope = dict(_fy=self.get_f_dy(), perturb=self.options.perturbation, math=math)
        func_codes = [f'def dgdy({argument}):']
        if not origin:
          func_codes.append(f'origin = _fy({argument})')
        func_codes.append(f'disturb = _fy({x_var}, {y_var}+perturb, {argument2})')
        if not origin:
          func_codes.append(f'return (disturb - origin) / perturb')
        else:
          func_codes.append(f'return disturb / perturb')
        exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
        dgdy = scope['dgdy']

      self.analyzed_results['dgdy'] = dgdy
    return self.analyzed_results['dgdy']

  def get_f_jacobian(self):
    """Get the function to solve jacobian matrix.
    """
    if 'jacobian' not in self.analyzed_results:
      dfdx = self.get_f_dfdx()
      dfdy = self.get_f_dfdy()
      dgdx = self.get_f_dgdx()
      dgdy = self.get_f_dgdy()

      argument = ','.join(self.dvar_names + self.dpar_names)
      scope = dict(f_dfydy=dgdy, f_dfydx=dgdx, f_dfxdy=dfdy, f_dfxdx=dfdx, math=math)
      func_codes = [f'def f_jacobian({argument}):']
      func_codes.append(f'dfxdx = f_dfxdx({argument})')
      func_codes.append(f'dfxdy = f_dfxdy({argument})')
      func_codes.append(f'dfydx = f_dfydx({argument})')
      func_codes.append(f'dfydy = f_dfydy({argument})')
      func_codes.append('return math.array([[dfxdx, dfxdy], [dfydx, dfydy]])')
      exec(compile('\n  '.join(func_codes), '', 'exec'), scope)
      self.analyzed_results['jacobian'] = scope['f_jacobian']

    return self.analyzed_results['jacobian']

  def get_f_fixed_point(self):
    """Get the function to solve the fixed point.
    """

    if 'fixed_point' not in self.analyzed_results:
      vars_and_pars = ','.join(self.dvar_names[2:] + self.dpar_names)

      eq_xy_scope = deepcopy(self.pars_update)
      eq_xy_scope.update(self.fixed_vars)
      # eq_xy_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_xy_scope.update(self.x_eq_group['diff_eq'].func_scope)
      eq_xy_scope.update(self.y_eq_group['diff_eq'].func_scope)
      eq_xy_scope['math'] = math

      # Try 1: substitute y_group to x_group
      #        y_by_x
      # ------------------------------------
      y_by_x_in_y_eq = self.get_y_by_x_in_y_eq()
      if y_by_x_in_y_eq['status'] == 'sympy_success':
        func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
        func_codes += y_by_x_in_y_eq['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.x_eq_group.old_exprs[:-1]])
        func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        optimizer = utils.jit_compile(eq_xy_scope, func_code, 'optimizer_x')

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          x_range = self.resolutions[self.x_var]
          x_values = solver.find_root_of_1d(optimizer, x_range, args)
          x_values = np.array(x_values)
          y_values = y_by_x_in_y_eq['f'](x_values, *args)
          y_values = np.array(y_values)
          return x_values, y_values

        self.analyzed_results['fixed_point'] = f
        return f

      # Try 2: substitute y_group to x_group
      #        x_by_y
      # ------------------------------------
      x_by_y_in_y_eq = self.get_x_by_y_in_y_eq()
      if x_by_y_in_y_eq['status'] == 'sympy_success':
        func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
        func_codes += x_by_y_in_y_eq['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.x_eq_group.old_exprs[:-1]])
        func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        optimizer = utils.jit_compile(eq_xy_scope, func_code, 'optimizer_y')

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          y_range = self.resolutions[self.y_var]
          y_values = solver.find_root_of_1d(optimizer, y_range, args)
          y_values = np.array(y_values)
          x_values = x_by_y_in_y_eq['f'](y_values, *args)
          x_values = np.array(x_values)
          return x_values, y_values

        self.analyzed_results['fixed_point'] = f
        return f

      # Try 3: substitute x_group to y_group
      #        y_by_x
      # ------------------------------------
      y_by_x_in_x_eq = self.get_y_by_x_in_x_eq()
      if y_by_x_in_x_eq['status'] == 'sympy_success':
        func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
        func_codes += y_by_x_in_x_eq['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.y_eq_group.old_exprs[:-1]])
        func_codes.append(f'return {self.y_eq_group.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        optimizer = utils.jit_compile(eq_xy_scope, func_code, 'optimizer_x')

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          x_range = self.resolutions[self.x_var]
          x_values = solver.find_root_of_1d(optimizer, x_range, args)
          x_values = np.array(x_values)
          y_values = y_by_x_in_x_eq['f'](x_values, *args)
          y_values = np.array(y_values)
          return x_values, y_values

        self.analyzed_results['fixed_point'] = f
        return f

      # Try 4: substitute x_group to y_group
      #        x_by_y
      # ------------------------------------
      x_by_y_in_x_eq = self.get_x_by_y_in_x_eq()
      if x_by_y_in_x_eq['status'] == 'sympy_success':
        func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
        func_codes += x_by_y_in_x_eq['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.y_eq_group.old_exprs[:-1]])
        func_codes.append(f'return {self.y_eq_group.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        optimizer = utils.jit_compile(eq_xy_scope, func_code, 'optimizer_y')

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          y_range = self.resolutions[self.y_var]
          y_values = solver.find_root_of_1d(optimizer, y_range, args)
          y_values = np.array(y_values)
          x_values = x_by_y_in_x_eq['f'](y_values, *args)
          x_values = np.array(x_values)
          return x_values, y_values

        self.analyzed_results['fixed_point'] = f
        return f

      # Try 5: numerical optimization method
      # ------------------------------------
      # f
      eq_x_scope = deepcopy(self.pars_update)
      eq_x_scope.update(self.fixed_vars)
      # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_x_scope.update(self.x_eq_group['diff_eq'].func_scope)
      eq_x_scope['math'] = math

      func_codes = [f'def f_x({",".join(self.dvar_names + self.dpar_names)}):']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.x_eq_group.old_exprs[:-1]])
      func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')
      exec(compile('\n  '.join(func_codes), '', 'exec'), eq_x_scope)
      f_x = eq_x_scope['f_x']

      # g
      eq_y_scope = deepcopy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_y_scope.update(self.y_eq_group['diff_eq'].func_scope)
      eq_y_scope['math'] = math

      func_codes = [f'def g_y({",".join(self.dvar_names + self.dpar_names)}):']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.y_eq_group.old_exprs[:-1]])
      func_codes.append(f'return {self.y_eq_group.old_exprs[-1].code}')
      exec(compile('\n  '.join(func_codes), '', 'exec'), eq_y_scope)
      g_y = eq_y_scope['g_y']

      # f**2 + g**2
      def optimizer(x, *args):
        return f_x(x[0], x[1], *args) ** 2 + g_y(x[0], x[1], *args) ** 2

      # optimization results
      def f(*args):
        # ``args`` are equal to ``vars_and_pars``
        return solver.find_root_of_2d(optimizer,
                                      x_bound=self.target_vars[self.x_var],
                                      y_bound=self.target_vars[self.y_var],
                                      args=args,
                                      shgo_args=self.options.shgo_args,
                                      fl_tol=self.options.fl_tol,
                                      xl_tol=self.options.xl_tol,
                                      verbose=self.options.show_shgo)

      self.analyzed_results['fixed_point'] = f
      return f

    return self.analyzed_results['fixed_point']

  def get_f_optimize_x_nullcline(self, coords=None):
    """Get the function to solve X nullcline by using numerical optimization method.

    Parameters
    ----------
    coords : str
        The coordination.
    """
    if coords is None:
      coords = self.x_var + '-' + self.y_var

    key = f'optimize_x_nullcline,{coords}'
    if key not in self.analyzed_results:
      # check coordinate setting
      coord_splits = [a.strip() for a in coords.strip().split('-')]
      if self.x_var not in coord_splits:
        raise ValueError(f'Variable "{self.x_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      if self.y_var not in coord_splits:
        raise ValueError(f'Variable "{self.y_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')

      # x equation scope
      eq_x_scope = deepcopy(self.pars_update)
      eq_x_scope.update(self.fixed_vars)
      # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_x_scope.update(self.x_eq_group.diff_eq.func_scope)
      eq_x_scope['math'] = math

      argument = ','.join(self.dvar_names[2:] + self.dpar_names)

      # optimization function
      func_codes = [f'def optimizer_x({self.x_var},{self.y_var},{argument}):']
      for expr in self.x_eq_group.old_exprs[:-1]:
        func_codes.append(f'{expr.var_name} = {expr.code}')
      func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      optimizer_x_by_y = utils.jit_compile(eq_x_scope, func_code, 'optimizer_x')

      func_codes = [f'def optimizer_y({self.y_var},{self.x_var},{argument}):']
      for expr in self.x_eq_group.old_exprs[:-1]:
        func_codes.append(f'{expr.var_name} = {expr.code}')
      func_codes.append(f'return {self.x_eq_group.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      optimizer_y_by_x = utils.jit_compile(eq_x_scope, func_code, 'optimizer_y')

      # optimization results
      xs = self.resolutions[self.x_var]
      ys = self.resolutions[self.y_var]

      def f1(*args):
        # ``args`` corresponds to the dynamical parameters
        x_values, y_values = [], []
        for y in ys:
          for x in solver.find_root_of_1d(optimizer_x_by_y, xs, (y,) + args):
            x_values.append(x)
            y_values.append(y)
        return np.array(x_values), np.array(y_values)

      def f2(*args):
        # ``args`` corresponds to the dynamical parameters
        x_values, y_values = [], []
        for x in xs:
          for y in solver.find_root_of_1d(optimizer_y_by_x, ys, (x,) + args):
            x_values.append(x)
            y_values.append(y)
        return np.array(x_values), np.array(y_values)

      key1 = f'optimize_x_nullcline,{self.x_var}-{self.y_var}'
      key2 = f'optimize_x_nullcline,{self.y_var}-{self.x_var}'
      self.analyzed_results[key1] = f1
      self.analyzed_results[key2] = f2

    return self.analyzed_results[key]

  def get_f_optimize_y_nullcline(self, coords=None):
    """Get the function to solve Y nullcline by using numerical optimization method.

    Parameters
    ----------
    coords : str
        The coordination.
    """
    if coords is None:
      coords = self.x_var + '-' + self.y_var

    key = f'optimize_y_nullcline,{coords}'
    if key not in self.analyzed_results:
      # check coordinate setting
      coord_splits = [a.strip() for a in coords.strip().split('-')]
      if self.x_var not in coord_splits:
        raise ValueError(f'Variable "{self.x_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')
      if self.y_var not in coord_splits:
        raise ValueError(f'Variable "{self.y_var}" must be in coordinate '
                         f'settings. But we get "{coords}".')

      # y equation scope
      eq_y_scope = deepcopy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
      eq_y_scope.update(self.y_eq_group.diff_eq.func_scope)
      eq_y_scope['math'] = math

      argument = ','.join(self.dvar_names[2:] + self.dpar_names)

      # optimization function
      func_codes = [f'def optimizer_x({self.x_var},{self.y_var},{argument}):']
      for expr in self.y_eq_group.old_exprs[:-1]:
        func_codes.append(f'{expr.var_name} = {expr.code}')
      func_codes.append(f'return {self.y_eq_group.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      optimizer_x_by_y = utils.jit_compile(eq_y_scope, func_code, 'optimizer_x')

      func_codes = [f'def optimizer_y({self.y_var},{self.x_var},{argument}):']
      for expr in self.y_eq_group.old_exprs[:-1]:
        func_codes.append(f'{expr.var_name} = {expr.code}')
      func_codes.append(f'return {self.y_eq_group.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      optimizer_y_by_x = utils.jit_compile(eq_y_scope, func_code, 'optimizer_y')

      # optimization results
      xs = self.resolutions[self.x_var]
      ys = self.resolutions[self.y_var]

      def f1(*args):
        # ``args`` corresponds to the dynamical parameters
        x_values, y_values = [], []
        for y in ys:
          for x in solver.find_root_of_1d(optimizer_x_by_y, xs, (y,) + args):
            x_values.append(x)
            y_values.append(y)
        return np.array(x_values), np.array(y_values)

      def f2(*args):
        # ``args`` corresponds to the dynamical parameters
        x_values, y_values = [], []
        for x in xs:
          for y in solver.find_root_of_1d(optimizer_y_by_x, ys, (x,) + args):
            x_values.append(x)
            y_values.append(y)
        return np.array(x_values), np.array(y_values)

      key1 = f'optimize_y_nullcline,{self.x_var}-{self.y_var}'
      key2 = f'optimize_y_nullcline,{self.y_var}-{self.x_var}'
      self.analyzed_results[key1] = f1
      self.analyzed_results[key2] = f2

    return self.analyzed_results[key]

  def get_y_by_x_in_y_eq(self):
    """Get the expression of "y_by_x_in_y_eq".

    Specifically, ``self.analyzed_results['y_by_x_in_y_eq']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'escape'
    - subs : substituted expressions (relationship) of y_by_x
    - f : function of y_by_x
    """
    if 'y_by_x_in_y_eq' not in self.analyzed_results:
      results = tools.DictPlus()
      if not self.options.escape_sympy_solver:
        y_symbol = sympy.Symbol(self.y_var, real=True)
        code = self.target_eqs[self.y_var].sub_exprs[-1].code
        y_eq = analysis_by_sympy.str2sympy(code).expr

        eq_y_scope = deepcopy(self.pars_update)
        eq_y_scope.update(self.fixed_vars)
        # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
        eq_y_scope.update(self.y_eq_group['diff_eq'].func_scope)
        eq_y_scope['math'] = math

        argument = ', '.join(self.dvar_names + self.dpar_names)
        timeout_len = self.options.sympy_solver_timeout

        try:
          logger.info(f'SymPy solve "{self.y_eq_group.func_name}({argument}) = 0" to '
                      f'"{self.y_var} = f({self.x_var}, '
                      f'{",".join(self.dvar_names[2:] + self.dpar_names)})", ')
          # solve the expression
          f = utils.timeout(timeout_len)(lambda: sympy.solve(y_eq, y_symbol))
          y_by_x_in_y_eq = f()
          if len(y_by_x_in_y_eq) > 1:
            raise NotImplementedError('Do not support multiple values.')
          y_by_x_in_y_eq = analysis_by_sympy.sympy2str(y_by_x_in_y_eq[0])

          # check
          all_vars = set(eq_y_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(y_by_x_in_y_eq, all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            results['status'] = 'sympy_failed'
          else:
            logger.info('\tsuccess.')
            # substituted codes
            subs_codes = [f'{expr.var_name} = {expr.code}'
                          for expr in self.y_eq_group.sub_exprs[:-1]]
            subs_codes.append(f'{self.y_var} = {y_by_x_in_y_eq}')

            # compile the function
            func_code = f'def func({self.x_var}, {",".join(self.dvar_names[2:] + self.dpar_names)}):\n'
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {y_by_x_in_y_eq}'
            exec(compile(func_code, '', 'exec'), eq_y_scope)

            # set results
            results['status'] = 'sympy_success'
            results['subs'] = subs_codes
            results['f'] = eq_y_scope['func']

        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')
          results['status'] = 'sympy_failed'
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {timeout_len} s timeout.')
          results['status'] = 'sympy_failed'
      else:
        results['status'] = 'escape'
      self.analyzed_results['y_by_x_in_y_eq'] = results
    return self.analyzed_results['y_by_x_in_y_eq']

  def get_y_by_x_in_x_eq(self):
    """Get the expression of "y_by_x_in_x_eq".

    Specifically, ``self.analyzed_results['y_by_x_in_x_eq']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'escape'
    - subs : substituted expressions (relationship) of y_by_x
    - f : function of y_by_x
    """
    if 'y_by_x_in_x_eq' not in self.analyzed_results:
      results = tools.DictPlus()

      if not self.options.escape_sympy_solver:
        y_symbol = sympy.Symbol(self.y_var, real=True)
        code = self.x_eq_group.sub_exprs[-1].code
        x_eq = analysis_by_sympy.str2sympy(code).expr

        eq_x_scope = deepcopy(self.pars_update)
        eq_x_scope.update(self.fixed_vars)
        # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
        eq_x_scope.update(self.x_eq_group['diff_eq'].func_scope)
        eq_x_scope['math'] = math

        argument = ', '.join(self.dvar_names + self.dpar_names)
        timeout_len = self.options.sympy_solver_timeout

        try:
          logger.info(f'SymPy solve "{self.x_eq_group.func_name}({argument}) = 0" to '
                      f'"{self.y_var} = f({self.x_var}, '
                      f'{",".join(self.dvar_names[2:] + self.dpar_names)})", ')

          # solve the expression
          f = utils.timeout(timeout_len)(lambda: sympy.solve(x_eq, y_symbol))
          y_by_x_in_x_eq = f()
          if len(y_by_x_in_x_eq) > 1:
            raise NotImplementedError('Do not support multiple values.')
          y_by_x_in_x_eq = analysis_by_sympy.sympy2str(y_by_x_in_x_eq[0])

          all_vars = set(eq_x_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(y_by_x_in_x_eq, all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            results['status'] = 'sympy_failed'
          else:
            logger.info('\tsuccess.')

            # substituted codes
            subs_codes = [f'{expr.var_name} = {expr.code}'
                          for expr in self.x_eq_group.sub_exprs[:-1]]
            subs_codes.append(f'{self.y_var} = {y_by_x_in_x_eq}')

            # compile the function
            func_code = f'def func({self.x_var}, {",".join(self.dvar_names[2:] + self.dpar_names)}):\n'
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {y_by_x_in_x_eq}'
            exec(compile(func_code, '', 'exec'), eq_x_scope)

            # set results
            results['status'] = 'sympy_success'
            results['subs'] = subs_codes
            results['f'] = eq_x_scope['func']
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')
          results['status'] = 'sympy_failed'
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {timeout_len} s timeout.')
          results['status'] = 'sympy_failed'
      else:
        results['status'] = 'escape'
      self.analyzed_results['y_by_x_in_x_eq'] = results
    return self.analyzed_results['y_by_x_in_x_eq']

  def get_x_by_y_in_y_eq(self):
    """Get the expression of "x_by_y_in_y_eq".

    Specifically, ``self.analyzed_results['x_by_y_in_y_eq']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'escape'
    - subs : substituted expressions (relationship) of x_by_y
    - f : function of x_by_y
    """
    if 'x_by_y_in_y_eq' not in self.analyzed_results:
      results = tools.DictPlus()

      if not self.options.escape_sympy_solver:
        x_symbol = sympy.Symbol(self.x_var, real=True)
        code = self.target_eqs[self.y_var].sub_exprs[-1].code
        y_eq = analysis_by_sympy.str2sympy(code).expr

        eq_y_scope = deepcopy(self.pars_update)
        eq_y_scope.update(self.fixed_vars)
        # eq_y_scope.update(analysis_by_sympy.get_mapping_scope())
        eq_y_scope.update(self.y_eq_group['diff_eq'].func_scope)
        eq_y_scope['math'] = math

        argument = ', '.join(self.dvar_names + self.dpar_names)
        timeout_len = self.options.sympy_solver_timeout

        try:
          logger.info(f'SymPy solve "{self.y_eq_group.func_name}({argument}) = 0" to '
                      f'"{self.x_var} = f({",".join(self.dvar_names[1:] + self.dpar_names)})", ')
          # solve the expression
          f = utils.timeout(timeout_len)(lambda: sympy.solve(y_eq, x_symbol))
          x_by_y_in_y_eq = f()
          if len(x_by_y_in_y_eq) > 1:
            raise NotImplementedError('Do not support multiple values.')
          x_by_y_in_y_eq = analysis_by_sympy.sympy2str(x_by_y_in_y_eq[0])

          # check
          all_vars = set(eq_y_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(x_by_y_in_y_eq, all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            results['status'] = 'sympy_failed'
          else:
            logger.info('\tsuccess.')

            # substituted codes
            subs_codes = [f'{expr.var_name} = {expr.code}'
                          for expr in self.y_eq_group.sub_exprs[:-1]]
            subs_codes.append(f'{self.x_var} = {x_by_y_in_y_eq}')

            # compile the function
            func_code = f'def func({",".join(self.dvar_names[1:] + self.dpar_names)}):\n'
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {x_by_y_in_y_eq}'
            exec(compile(func_code, '', 'exec'), eq_y_scope)

            # set results
            results['status'] = 'sympy_success'
            results['subs'] = subs_codes
            results['f'] = eq_y_scope['func']
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')
          results['status'] = 'sympy_failed'
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {timeout_len} s timeout.')
          results['status'] = 'sympy_failed'
      else:
        results['status'] = 'escape'
      self.analyzed_results['x_by_y_in_y_eq'] = results
    return self.analyzed_results['x_by_y_in_y_eq']

  def get_x_by_y_in_x_eq(self):
    """Get the expression of "x_by_y_in_x_eq".

    Specifically, ``self.analyzed_results['x_by_y_in_x_eq']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'escape'
    - subs : substituted expressions (relationship) of x_by_y
    - f : function of x_by_y
    """
    if 'x_by_y_in_x_eq' not in self.analyzed_results:
      results = tools.DictPlus()

      if not self.options.escape_sympy_solver:
        x_symbol = sympy.Symbol(self.x_var, real=True)
        code = self.x_eq_group.sub_exprs[-1].code
        x_eq = analysis_by_sympy.str2sympy(code).expr

        eq_x_scope = deepcopy(self.pars_update)
        eq_x_scope.update(self.fixed_vars)
        # eq_x_scope.update(analysis_by_sympy.get_mapping_scope())
        eq_x_scope.update(self.x_eq_group['diff_eq'].func_scope)
        eq_x_scope['math'] = math

        argument = ', '.join(self.dvar_names + self.dpar_names)
        timeout_len = self.options.sympy_solver_timeout

        try:
          logger.info(f'SymPy solve "{self.x_eq_group.func_name}({argument}) = 0" to '
                      f'"{self.x_var} = f({",".join(self.dvar_names[1:] + self.dpar_names)})", ')
          # solve the expression
          f = utils.timeout(timeout_len)(lambda: sympy.solve(x_eq, x_symbol))
          x_by_y_in_x_eq = f()
          if len(x_by_y_in_x_eq) > 1:
            raise NotImplementedError('Do not support multiple values.')
          x_by_y_in_x_eq = analysis_by_sympy.sympy2str(x_by_y_in_x_eq[0])

          # check
          all_vars = set(eq_x_scope.keys())
          all_vars.update(self.dvar_names + self.dpar_names)
          if utils.contain_unknown_symbol(x_by_y_in_x_eq, all_vars):
            logger.info('\tfailed because contain unknown symbols.')
            results['status'] = 'sympy_failed'
          else:
            logger.info('\tsuccess.')

            # substituted codes
            subs_codes = [f'{expr.var_name} = {expr.code}'
                          for expr in self.x_eq_group.sub_exprs[:-1]]
            subs_codes.append(f'{self.x_var} = {x_by_y_in_x_eq}')

            # compile the function
            func_code = f'def func({",".join(self.dvar_names[1:] + self.dpar_names)}):\n'
            for expr in self.y_eq_group.sub_exprs[:-1]:
              func_code += f'  {expr.var_name} = {expr.code}\n'
            func_code += f'  return {x_by_y_in_x_eq}'
            exec(compile(func_code, '', 'exec'), eq_x_scope)

            # set results
            results['status'] = 'sympy_success'
            results['subs'] = subs_codes
            results['f'] = eq_x_scope['func']
        except NotImplementedError:
          logger.info('\tfailed because the equation is too complex.')
          results['status'] = 'sympy_failed'
        except KeyboardInterrupt:
          logger.info(f'\tfailed because {timeout_len} s timeout.')
          results['status'] = 'sympy_failed'
      else:
        results['status'] = 'escape'
      self.analyzed_results['x_by_y_in_x_eq'] = results
    return self.analyzed_results['x_by_y_in_x_eq']
