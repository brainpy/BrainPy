# -*- coding: utf-8 -*-

import logging
from typing import List

import sympy
import numpy as np
import brainpy.math as bm
from brainpy import tools
from brainpy.analysis import constants as C
from brainpy.analysis.numeric import low_dim_analyzer, solver
from brainpy.analysis.symbolic import utils
from brainpy.integrators import analysis_by_sympy

logger = logging.getLogger('brainpy.analysis')

__all__ = [
  'LowDimAnalyzer2D',
]


def _update_scope(scope):
  scope['math'] = bm
  scope['bm'] = bm


def _dict_copy(target):
  assert isinstance(target, dict)
  return {k: v for k, v in target.items()}


def _get_substitution(substitute_var: str,
                      target_var: str,
                      eq_group: dict,
                      target_var_names: List[str],
                      target_par_names: List[str],
                      escape_sympy_solver: bool,
                      timeout_len: float,
                      eq_y_scope: dict):
  results = tools.DictPlus()
  if not escape_sympy_solver:
    y_symbol = sympy.Symbol(target_var, real=True)
    code = eq_group["sub_exprs"][-1].code
    y_eq = analysis_by_sympy.str2sympy(code).expr

    eq_y_scope.update(eq_group['diff_eq'].func_scope)
    _update_scope(eq_y_scope)

    argument = ', '.join(target_var_names + target_par_names)

    try:
      logger.warning(f'SymPy solve "{eq_group["func_name"]}({argument}) = 0" to '
                     f'"{target_var} = f({substitute_var}, {",".join(target_var_names[2:] + target_par_names)})", ')
      # solve the expression
      f = tools.timeout(timeout_len)(lambda: sympy.solve(y_eq, y_symbol))
      y_by_x_in_y_eq = f()
      if len(y_by_x_in_y_eq) > 1:
        raise NotImplementedError('Do not support multiple values.')
      y_by_x_in_y_eq = analysis_by_sympy.sympy2str(y_by_x_in_y_eq[0])

      # check
      all_vars = set(eq_y_scope.keys())
      all_vars.update(target_var_names + target_par_names)
      unknown_symbols = utils.unknown_symbol(y_by_x_in_y_eq, all_vars)
      if len(unknown_symbols):
        logger.warning(f'\tfailed because contain unknown symbols: {unknown_symbols}.')
        results['status'] = C.sympy_failed
        results['subs'] = []
        results['f'] = None
      else:
        logger.warning('\tsuccess.')
        # substituted codes
        subs_codes = [f'{expr.var_name} = {expr.code}' for expr in eq_group["sub_exprs"][:-1]]
        subs_codes.append(f'{target_var} = {y_by_x_in_y_eq}')

        # compile the function
        func_code = f'def func({substitute_var}, {",".join(target_var_names[2:] + target_par_names)}):\n'
        for expr in eq_group["sub_exprs"][:-1]:
          func_code += f'  {expr.var_name} = {expr.code}\n'
        func_code += f'  return {y_by_x_in_y_eq}'
        exec(compile(func_code, '', 'exec'), eq_y_scope)

        # set results
        results['status'] = C.sympy_success
        results['subs'] = subs_codes
        results['f'] = eq_y_scope['func']

    except NotImplementedError:
      logger.warning('\tfailed because the equation is too complex.')
      results['status'] = C.sympy_failed
      results['subs'] = []
      results['f'] = None
    except KeyboardInterrupt:
      logger.warning(f'\tfailed because {timeout_len} s timeout.')
      results['status'] = C.sympy_failed
      results['subs'] = []
      results['f'] = None
  else:
    results['status'] = C.sympy_escape
    results['subs'] = []
    results['f'] = None
  return results


class LowDimAnalyzer2D(low_dim_analyzer.LowDimAnalyzer2D):
  r"""Neuron analysis analyzer for 2D system.

  It supports the analysis of 2D dynamical system.

  .. math::

      {dx \over dt} = fx(x, t, y)

      {dy \over dt} = fy(y, t, x)
  """

  def __init__(self, *args, **kwargs):
    super(LowDimAnalyzer2D, self).__init__(*args, **kwargs)

    self.y_var = self.target_var_names[1]

    # options
    # ---------
    options = kwargs.get('options', dict())
    if options is None: options = dict()
    assert isinstance(options, dict)
    for a in [C.y_by_x_in_fy, C.y_by_x_in_fx, C.x_by_y_in_fx, C.x_by_y_in_fy]:
      if a in options:
        # check "subs"
        subs = options[a]
        if isinstance(subs, str):
          subs = [subs]
        elif isinstance(subs, (tuple, list)):
          subs = subs
          for s in subs:
            assert isinstance(s, str)
        else:
          raise ValueError(f'Unknown setting of "{a}": {subs}')

        # check "f"
        scope = _dict_copy(self.pars_update)
        scope.update(self.fixed_vars)
        _update_scope(scope)
        if a.startswith('fy::'):
          scope.update(self.fy_eqs['diff_eq'].func_scope)
        else:
          scope.update(self.fx_eqs['diff_eq'].func_scope)

        # function code
        argument = ",".join(self.target_var_names[2:] + self.target_par_names)
        if a.endswith('y=f(x)'):
          func_codes = [f'def func({self.x_var}, {argument}):\n']
        else:
          func_codes = [f'def func({self.y_var}, {argument}):\n']
        func_codes.extend(subs)
        func_codes.append(f'return {subs[-1].split("=")[0]}')

        # function compilation
        exec(compile("\n  ".join(func_codes), '', 'exec'), scope)
        f = scope['func']

        # results
        self.analyzed_results[a] = tools.DictPlus(status=C.sympy_success, subs=subs, f=f)

  @property
  def fx_eqs(self):
    return self.target_eqs[self.x_var]

  @property
  def fy_eqs(self):
    return self.target_eqs[self.y_var]

  @property
  def y_by_x_in_fy(self):
    """Get the expression of "y=f(x)" in :math:`f_y` equation.

    Specifically, ``self.analyzed_results['y_by_x_in_fy']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'sympy_escape'
    - subs : substituted expressions (relationship) of y_by_x
    - f : function of y_by_x
    """
    if C.y_by_x_in_fy not in self.analyzed_results:
      eq_y_scope = _dict_copy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      results = _get_substitution(substitute_var=self.x_var,
                                  target_var=self.y_var,
                                  eq_group=self.fy_eqs,
                                  target_var_names=self.target_var_names,
                                  target_par_names=self.target_par_names,
                                  escape_sympy_solver=self.escape_sympy_solver,
                                  timeout_len=self.sympy_solver_timeout,
                                  eq_y_scope=eq_y_scope)
      self.analyzed_results[C.y_by_x_in_fy] = results
    return self.analyzed_results[C.y_by_x_in_fy]

  @property
  def y_by_x_in_fx(self):
    """Get the expression of "y_by_x_in_fx".

    Specifically, ``self.analyzed_results['y_by_x_in_fx']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'sympy_escape'
    - subs : substituted expressions (relationship) of y_by_x
    - f : function of y_by_x
    """
    if C.y_by_x_in_fx not in self.analyzed_results:
      eq_y_scope = _dict_copy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      results = _get_substitution(substitute_var=self.x_var,
                                  target_var=self.y_var,
                                  eq_group=self.fx_eqs,
                                  target_var_names=self.target_var_names,
                                  target_par_names=self.target_par_names,
                                  escape_sympy_solver=self.escape_sympy_solver,
                                  timeout_len=self.sympy_solver_timeout,
                                  eq_y_scope=eq_y_scope)
      self.analyzed_results[C.y_by_x_in_fx] = results
    return self.analyzed_results[C.y_by_x_in_fx]

  @property
  def x_by_y_in_fy(self):
    """Get the expression of "x_by_y_in_fy".

    Specifically, ``self.analyzed_results['x_by_y_in_fy']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'sympy_escape'
    - subs : substituted expressions (relationship) of x_by_y
    - f : function of x_by_y
    """
    if C.x_by_y_in_fy not in self.analyzed_results:
      eq_y_scope = _dict_copy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      results = _get_substitution(substitute_var=self.y_var,
                                  target_var=self.x_var,
                                  eq_group=self.fy_eqs,
                                  target_var_names=self.target_var_names,
                                  target_par_names=self.target_par_names,
                                  escape_sympy_solver=self.escape_sympy_solver,
                                  timeout_len=self.sympy_solver_timeout,
                                  eq_y_scope=eq_y_scope)
      self.analyzed_results[C.x_by_y_in_fy] = results
    return self.analyzed_results[C.x_by_y_in_fy]

  @property
  def x_by_y_in_fx(self):
    """Get the expression of "x_by_y_in_fx".

    Specifically, ``self.analyzed_results['x_by_y_in_fx']`` is a Dict,
    with the following keywords:

    - status : 'sympy_success', 'sympy_failed', 'sympy_escape'
    - subs : substituted expressions (relationship) of x_by_y
    - f : function of x_by_y
    """
    if C.x_by_y_in_fx not in self.analyzed_results:
      eq_y_scope = _dict_copy(self.pars_update)
      eq_y_scope.update(self.fixed_vars)
      results = _get_substitution(substitute_var=self.y_var,
                                  target_var=self.x_var,
                                  eq_group=self.fx_eqs,
                                  target_var_names=self.target_var_names,
                                  target_par_names=self.target_par_names,
                                  escape_sympy_solver=self.escape_sympy_solver,
                                  timeout_len=self.sympy_solver_timeout,
                                  eq_y_scope=eq_y_scope)
      self.analyzed_results[C.x_by_y_in_fx] = results
    return self.analyzed_results[C.x_by_y_in_fx]

  @property
  def F_fixed_points(self, tol_unique=1e-2):
    if C.F_fixed_point not in self.analyzed_results:
      vars_and_pars = ','.join(self.target_var_names[2:] + self.target_par_names)

      eq_xy_scope = _dict_copy(self.pars_update)
      eq_xy_scope.update(self.fixed_vars)
      eq_xy_scope.update(self.fx_eqs['diff_eq'].func_scope)
      eq_xy_scope.update(self.fy_eqs['diff_eq'].func_scope)
      _update_scope(eq_xy_scope)
      
      if self.y_by_x_in_fy['status'] == C.sympy_success:
        func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
        func_codes += self.y_by_x_in_fx['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.fx_eqs['old_exprs'][:-1]])
        func_codes.append(f'return {self.fx_eqs["old_exprs"][-1].code}')
        func_code = '\n  '.join(func_codes)
        exec(compile(func_code, '', 'exec'), eq_xy_scope)
        optimizer1 = eq_xy_scope['optimizer_x']

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          x_range = self.resolutions[self.x_var]
          x_values = solver.roots_of_1d_by_x(optimizer1, x_range, args)
          x_values = np.array(x_values)
          y_values = self.y_by_x_in_fx['f'](x_values, *args)
          y_values = np.array(y_values)
          return x_values, y_values

        self.analyzed_results[C.F_fixed_point] = f 
      
      elif self.x_by_y_in_fy['status'] == C.sympy_success:
        func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
        func_codes += self.x_by_y_in_fy['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.fx_eqs.old_exprs[:-1]])
        func_codes.append(f'return {self.fx_eqs.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        exec(compile(func_code, '', 'exec'), eq_xy_scope)
        optimizer2 = eq_xy_scope['optimizer_y']

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          y_range = self.resolutions[self.y_var]
          y_values = solver.roots_of_1d_by_x(optimizer2, y_range, args)
          y_values = np.array(y_values)
          x_values = self.x_by_y_in_fy['f'](y_values, *args)
          x_values = np.array(x_values)
          return x_values, y_values

        self.analyzed_results[C.F_fixed_point] = f
      
      elif self.x_by_y_in_fx['status'] == C.sympy_success:
        func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
        func_codes += self.x_by_y_in_fx['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.fy_eqs.old_exprs[:-1]])
        func_codes.append(f'return {self.fy_eqs.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        exec(compile(func_code, '', 'exec'), eq_xy_scope)
        optimizer3 = eq_xy_scope['optimizer_y']

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          y_range = self.resolutions[self.y_var]
          y_values = solver.roots_of_1d_by_x(optimizer3, y_range, args)
          y_values = np.array(y_values)
          x_values = self.x_by_y_in_fx['f'](y_values, *args)
          x_values = np.array(x_values)
          return x_values, y_values

        self.analyzed_results[C.F_fixed_point] = f
      
      elif self.y_by_x_in_fx['status'] == C.sympy_success:
        func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
        func_codes += self.y_by_x_in_fx['subs']
        func_codes.extend([f'{expr.var_name} = {expr.code}'
                           for expr in self.fy_eqs.old_exprs[:-1]])
        func_codes.append(f'return {self.fy_eqs.old_exprs[-1].code}')
        func_code = '\n  '.join(func_codes)
        exec(compile(func_code, '', 'exec'), eq_xy_scope)
        optimizer4 = eq_xy_scope['optimizer_x']

        def f(*args):
          # ``args`` are equal to ``vars_and_pars``
          x_range = self.resolutions[self.x_var]
          x_values = solver.roots_of_1d_by_x(optimizer4, x_range, args)
          x_values = np.array(x_values)
          y_values = self.y_by_x_in_fx['f'](x_values, *args)
          y_values = np.array(y_values)
          return x_values, y_values

        self.analyzed_results[C.F_fixed_point] = f
      
      else:
        super(LowDimAnalyzer2D, self).F_fixed_points(tol_unique=tol_unique)

    return self.analyzed_results[C.F_fixed_point]
