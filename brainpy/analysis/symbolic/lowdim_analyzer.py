# -*- coding: utf-8 -*-

import logging
from typing import List

import sympy
import jax.numpy as jnp
import brainpy.math as bm
from brainpy import tools, errors
from brainpy.analysis import constants as C, utils
from brainpy.analysis.numeric import lowdim_analyzer as numeric_analyzer
from brainpy.integrators import analysis_by_sympy, constants as IC

logger = logging.getLogger('brainpy.analysis')

__all__ = [
  'Sym2DAnalyzer',
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
                      escape_sympy: bool,
                      timeout_len: float,
                      eq_y_scope: dict,
                      _jit_device=None):
  results = tools.DictPlus()
  if not escape_sympy:
    y_symbol = sympy.Symbol(target_var, real=True)
    code = eq_group["sub_exprs"][-1].code
    y_eq = analysis_by_sympy.str2sympy(code).expr

    eq_y_scope.update(eq_group['diff_eq'].func_scope)
    _update_scope(eq_y_scope)

    argument = ', '.join(target_var_names + target_par_names)

    try:
      logger.warning(f'{C.prefix}SymPy solve "{eq_group["func_name"]}({argument}) = 0" to '
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
        logger.warning(f'{C.prefix}{C.prefix}failed because contain unknown '
                       f'symbols: {unknown_symbols}.')
        results['status'] = C.sympy_failed
        results['subs'] = []
        results['f'] = None
      else:
        logger.warning(f'{C.prefix}{C.prefix}success.')
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
        results['f'] = bm.jit(eq_y_scope['func'], device=_jit_device)
        results['vmap_f'] = bm.jit(bm.vmap(eq_y_scope['func']), device=_jit_device)
        results['_non_jit_f'] = eq_y_scope['func']

    except NotImplementedError:
      logger.warning(f'{C.prefix}{C.prefix}failed because the equation is too complex.')
      results['status'] = C.sympy_failed
      results['subs'] = []
      results['f'] = None
    except KeyboardInterrupt:
      logger.warning(f'{C.prefix}{C.prefix}failed because {timeout_len} s timeout.')
      results['status'] = C.sympy_failed
      results['subs'] = []
      results['f'] = None
  else:
    results['status'] = C.sympy_escape
    results['subs'] = []
    results['f'] = None
  return results


class Sym2DAnalyzer(numeric_analyzer.Num2DAnalyzer):
  r"""Neuron analysis analyzer for 2D system.

  It supports the analysis of 2D dynamical system.

  .. math::

      {dx \over dt} = fx(x, t, y)

      {dy \over dt} = fy(y, t, x)
  """

  def __init__(self, *args, **kwargs):
    super(Sym2DAnalyzer, self).__init__(*args, **kwargs)

    self.y_var = self.target_var_names[1]
    self.model = utils.num2sym(self.model)

    if 'escape_sympy' not in self.options:
      self.options['escape_sympy'] = False
    if 'sympy_timeout' not in self.options:
      self.options['sympy_timeout'] = 5.

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
        fnames = diff_eq.func_name.split('_')[4:]
        target_eqs[key] = tools.DictPlus(sub_exprs=sub_exprs,
                                         old_exprs=old_exprs,
                                         diff_eq=diff_eq,
                                         func_name='_'.join(fnames))
      self.analyzed_results['target_eqs'] = target_eqs
    return self.analyzed_results['target_eqs']

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
                                  escape_sympy=self.options['escape_sympy'],
                                  timeout_len=self.options['sympy_timeout'],
                                  eq_y_scope=eq_y_scope,
                                  _jit_device=self.jit_device)
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
                                  escape_sympy=self.options['escape_sympy'],
                                  timeout_len=self.options['sympy_timeout'],
                                  eq_y_scope=eq_y_scope,
                                  _jit_device=self.jit_device)
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
                                  escape_sympy=self.options['escape_sympy'],
                                  timeout_len=self.options['sympy_timeout'],
                                  eq_y_scope=eq_y_scope,
                                  _jit_device=self.jit_device)
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
                                  escape_sympy=self.options['escape_sympy'],
                                  timeout_len=self.options['sympy_timeout'],
                                  eq_y_scope=eq_y_scope,
                                  _jit_device=self.jit_device)
      self.analyzed_results[C.x_by_y_in_fx] = results
    return self.analyzed_results[C.x_by_y_in_fx]

  def _get_fx_nullcline_points(self, coords=None, tol=1e-7, num_segments=1, fp_aux_filter=0.):
    coords = (self.x_var + '-' + self.y_var) if coords is None else coords
    key = C.fx_nullcline_points + ',' + coords
    if key not in self.analyzed_results:
      xs = self.resolutions[self.x_var]
      P = tuple(self.resolutions[p].value for p in self.target_par_names)
      if self.y_by_x_in_fx['status'] == 'sympy_success':
        vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((xs,) + P)))
        y_values = self.y_by_x_in_fx['vmap_f'](*vps)
        self.analyzed_results[key] = (vps[0], y_values) + vps[1:]
      else:
        ys = self.resolutions[self.y_var]
        if self.x_by_y_in_fx['status'] == 'sympy_success':
          vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((ys,) + P)))
          x_values = self.x_by_y_in_fx['vmap_f'](*vps)
          self.analyzed_results[key] = (x_values,) + vps
        else:
          super(Sym2DAnalyzer, self)._get_fx_nullcline_points(coords=coords, tol=tol,
                                                              num_segments=num_segments,
                                                              fp_aux_filter=fp_aux_filter)
    return self.analyzed_results[key]

  def _get_fy_nullcline_points(self, coords=None, tol=1e-7, num_segments=1, fp_aux_filter=0.):
    coords = (self.x_var + '-' + self.y_var) if coords is None else coords
    key = C.fy_nullcline_points + ',' + coords
    if key not in self.analyzed_results:
      xs = self.resolutions[self.x_var]
      P = tuple(self.resolutions[p].value for p in self.target_par_names)
      if self.y_by_x_in_fy['status'] == 'sympy_success':
        vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((xs,) + P)))
        y_values = self.y_by_x_in_fy['vmap_f'](*vps)
        self.analyzed_results[key] = (vps[0], y_values) + vps[1:]
      else:
        ys = self.resolutions[self.y_var]
        if self.x_by_y_in_fy['status'] == 'sympy_success':
          vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((ys,) + P)))
          x_values = self.x_by_y_in_fy['vmap_f'](*vps)
          self.analyzed_results[key] = (x_values, ) + vps
        else:
          super(Sym2DAnalyzer, self)._get_fy_nullcline_points(coords=coords, tol=tol,
                                                              num_segments=num_segments,
                                                              fp_aux_filter=fp_aux_filter)
    return self.analyzed_results[key]

  def _get_fixed_points2(self):
    # scope
    eq_xy_scope = _dict_copy(self.pars_update)
    eq_xy_scope.update(self.fixed_vars)
    eq_xy_scope.update(self.fx_eqs['diff_eq'].func_scope)
    eq_xy_scope.update(self.fy_eqs['diff_eq'].func_scope)
    _update_scope(eq_xy_scope)

    # points
    xs = self.resolutions[self.x_var]
    ys = self.resolutions[self.x_var]
    ps = tuple(self.resolutions[p].value for p in self.target_par_names)

    vars_and_pars = ','.join(self.target_var_names[2:] + self.target_par_names)
    if self.y_by_x_in_fy['status'] == C.sympy_success:
      func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
      func_codes += self.y_by_x_in_fx['subs']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.fx_eqs['old_exprs'][:-1]])
      func_codes.append(f'return {self.fx_eqs["old_exprs"][-1].code}')
      func_code = '\n  '.join(func_codes)
      exec(compile(func_code, '', 'exec'), eq_xy_scope)
      opt1 = eq_xy_scope['optimizer_x']
      vmap_opt1 = bm.jit(bm.vmap(opt1), device=self.jit_device)
      vmap_berentq_opt1 = bm.jit(bm.vmap(utils.jax_brentq(opt1)), device=self.jit_device)

      # solve
      # -------
      # ``args`` are equal to ``vars_and_pars``
      vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((xs,) + ps)))
      starts, ends, vps = utils.brentq_candidates(vmap_opt1, *vps)
      x_values, p_values = utils.brentq_roots2(vmap_berentq_opt1, starts, ends, *vps)
      y_values = vmap_opt1(x_values, *p_values)
      return jnp.stack([x_values, y_values]).T, p_values

    elif self.x_by_y_in_fy['status'] == C.sympy_success:
      func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
      func_codes += self.x_by_y_in_fy['subs']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.fx_eqs.old_exprs[:-1]])
      func_codes.append(f'return {self.fx_eqs.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      exec(compile(func_code, '', 'exec'), eq_xy_scope)
      opt2 = eq_xy_scope['optimizer_y']
      vmap_opt2 = bm.jit(bm.vmap(opt2), device=self.jit_device)
      vmap_berentq_opt2 = bm.jit(bm.vmap(utils.jax_brentq(opt2)), device=self.jit_device)

      # solve
      # -------
      # ``args`` are equal to ``vars_and_pars``
      vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((ys,) + ps)))
      starts, ends, vps = utils.brentq_candidates(vmap_opt2, *vps)
      y_values, p_values = utils.brentq_roots2(vmap_berentq_opt2, starts, ends, *vps)
      x_values = vmap_opt2(y_values, *p_values)
      return jnp.stack([x_values, y_values]).T, p_values

    elif self.x_by_y_in_fx['status'] == C.sympy_success:
      func_codes = [f'def optimizer_y({self.y_var}, {vars_and_pars}):']
      func_codes += self.x_by_y_in_fx['subs']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.fy_eqs.old_exprs[:-1]])
      func_codes.append(f'return {self.fy_eqs.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      exec(compile(func_code, '', 'exec'), eq_xy_scope)
      opt3 = eq_xy_scope['optimizer_y']
      vmap_opt3 = bm.jit(bm.vmap(opt3), device=self.jit_device)
      vmap_berentq_opt3 = bm.jit(bm.vmap(utils.jax_brentq(opt3)), device=self.jit_device)

      # solve
      # -------
      # ``args`` are equal to ``vars_and_pars``
      vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((ys,) + ps)))
      starts, ends, vps = utils.brentq_candidates(vmap_opt3, *vps)
      y_values, p_values = utils.brentq_roots2(vmap_berentq_opt3, starts, ends, *vps)
      x_values = vmap_opt3(y_values, *p_values)
      return jnp.stack([x_values, y_values]).T, p_values

    elif self.y_by_x_in_fx['status'] == C.sympy_success:
      func_codes = [f'def optimizer_x({self.x_var}, {vars_and_pars}):']
      func_codes += self.y_by_x_in_fx['subs']
      func_codes.extend([f'{expr.var_name} = {expr.code}'
                         for expr in self.fy_eqs.old_exprs[:-1]])
      func_codes.append(f'return {self.fy_eqs.old_exprs[-1].code}')
      func_code = '\n  '.join(func_codes)
      exec(compile(func_code, '', 'exec'), eq_xy_scope)
      opt4 = eq_xy_scope['optimizer_x']
      vmap_opt4 = bm.jit(bm.vmap(opt4), device=self.jit_device)
      vmap_berentq_opt4 = bm.jit(bm.vmap(utils.jax_brentq(opt4)), device=self.jit_device)

      # solve
      # -------
      # ``args`` are equal to ``vars_and_pars``
      vps = tuple(vp.value.flatten() for vp in bm.meshgrid(*((xs,) + ps)))
      starts, ends, vps = utils.brentq_candidates(vmap_opt4, *vps)
      x_values, p_values = utils.brentq_roots2(vmap_berentq_opt4, starts, ends, *vps)
      y_values = vmap_opt4(x_values, *p_values)
      return jnp.stack([x_values, y_values]).T, p_values

    else:
      return None, None
