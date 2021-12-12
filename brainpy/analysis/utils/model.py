# -*- coding: utf-8 -*-



import inspect
from pprint import pprint

import numpy as np

import brainpy.math as bm
from brainpy import errors, tools
from brainpy.integrators import analysis_by_ast
from brainpy.integrators import analysis_by_sympy
from brainpy.integrators.ode.base import ODEIntegrator
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.utils import run_model

__all__ = [
  'model_transform',
  'num2sym',
  'NumDSWrapper',
  'SymDSWrapper',
  'Trajectory',
]


def model_transform(model):
  # check integrals
  if isinstance(model, NumDSWrapper):
    return model
  elif isinstance(model, ODEIntegrator):  #
    model = [model]
  if isinstance(model, (list, tuple)):
    if len(model) == 0:
      raise errors.AnalyzerError(f'Found no integrators: {model}')
    model = tuple(model)
    for intg in model:
      if not isinstance(intg, ODEIntegrator):
        raise errors.AnalyzerError(f'Must be the instance of {ODEIntegrator}, but got {intg}.')
  elif isinstance(model, dict):
    if len(model) == 0:
      raise errors.AnalyzerError(f'Found no integrators: {model}')
    model = tuple(model.values())
    for intg in model:
      if not isinstance(intg, ODEIntegrator):
        raise errors.AnalyzerError(f'Must be the instance of {ODEIntegrator}, but got {intg}')
  elif isinstance(model, DynamicalSystem):
    model = tuple(model.ints().subset(ODEIntegrator).unique().values())
  else:
    raise errors.UnsupportedError(f'Dynamics analysis by symbolic approach only supports '
                                  f'list/tuple/dict of {ODEIntegrator} or {DynamicalSystem}, '
                                  f'but we got: {type(model)}: {str(model)}')

  # pars to update
  pars_update = set()
  for intg in model:
    pars_update.update(intg.parameters[1:])

  all_variables = set()
  all_parameters = set()
  for integral in model:
    if len(integral.variables) != 1:
      raise errors.AnalyzerError(f'Only supports one {ODEIntegrator.__name__} one variable, '
                                 f'but we got {len(integral.variables)} variables in {integral}.')
    var = integral.variables[0]
    if var in all_variables:
      raise errors.AnalyzerError(f'Variable name {var} has been defined before. '
                                 f'Please change another name.')
    all_variables.add(var)
    # parameters
    all_parameters.update(integral.parameters[1:])

  # form a dynamic model
  return NumDSWrapper(integrals=model,
                      variables=list(all_variables),
                      parameters=list(all_parameters),
                      pars_update=pars_update)


class NumDSWrapper(object):
  """The wrapper of a dynamical model."""

  def __init__(self,
               integrals,
               variables,
               parameters,
               pars_update=None):
    self.INTG = integrals  # all integrators
    self.F = {intg.variables[0]: intg.f for intg in integrals}  # all integrators
    self.variables = variables  # all variables
    self.parameters = parameters  # all parameters
    self.pars_update = pars_update  # the parameters to update


def num2sym(model):
  assert isinstance(model, NumDSWrapper)
  all_scope = dict(math=bm)
  analyzers = []
  for integral in model.INTG:
    assert isinstance(integral, ODEIntegrator)

    # code scope
    code_scope = dict()
    closure_vars = inspect.getclosurevars(integral.f)
    code_scope.update(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)
    if hasattr(integral.f, '__self__'):
      code_scope['self'] = integral.f.__self__
    # separate variables
    code = tools.deindent(inspect.getsource(integral.f))
    analysis = analysis_by_ast.separate_variables(code)
    variables_for_returns = analysis['variables_for_returns']
    expressions_for_returns = analysis['expressions_for_returns']
    for vi, (key, vars) in enumerate(variables_for_returns.items()):
      variables = []
      for v in vars:
        if len(v) > 1:
          raise ValueError(f'Cannot analyze multi-assignment code line: {vars}.')
        variables.append(v[0])
      expressions = expressions_for_returns[key]
      var_name = integral.variables[vi]
      DE = analysis_by_sympy.SingleDiffEq(var_name=var_name,
                                          variables=variables,
                                          expressions=expressions,
                                          derivative_expr=key,
                                          scope={k: v for k, v in code_scope.items()},
                                          func_name=integral.func_name)
      analyzers.append(DE)
    all_scope.update(code_scope)
  return SymDSWrapper(analyzers=analyzers, scopes=all_scope,
                      integrals=model.INTG,
                      variables=model.variables,
                      parameters=model.parameters,
                      pars_update=model.pars_update)


class SymDSWrapper(NumDSWrapper):
  def __init__(self,
               analyzers,
               scopes,

               integrals,
               variables,
               parameters,
               pars_update=None):
    super(SymDSWrapper, self).__init__(integrals=integrals,
                                       variables=variables,
                                       parameters=parameters,
                                       pars_update=pars_update)
    self.analyzers = analyzers
    self.scopes = scopes



class Trajectory(object):
  """Trajectory Class.

  Parameters
  ----------
  model : NumDSWrapper
    The instance of DynamicModel.
  size : int, tuple, list
    The network size.
  target_vars : dict
    The target variables, with the format of "{key: initial_v}".
  fixed_vars : dict
    The fixed variables, with the format of "{key: fixed_v}".
  pars_update : dict
    The parameters to update.
  """

  def __init__(self, model, size, target_vars, fixed_vars, pars_update, show_code=False):
    assert isinstance(model, NumDSWrapper), f'"model" must be an instance of {NumDSWrapper}, ' \
                                            f'while we got {model}'
    self.model = model
    self.target_vars = target_vars
    self.fixed_vars = fixed_vars
    self.pars_update = pars_update
    self.show_code = show_code
    self.scope = {k: v for k, v in model.scopes.items()}

    # check network size
    if isinstance(size, int):
      size = (size,)
    elif isinstance(size, (tuple, list)):
      assert isinstance(size[0], int)
      size = tuple(size)
    else:
      raise ValueError

    # monitors, variables, parameters
    self.mon = tools.DictPlus()
    self.vars_and_pars = tools.DictPlus()
    for key, val in target_vars.items():
      self.vars_and_pars[key] = np.ones(size) * val
      self.mon[key] = []
    for key, val in fixed_vars.items():
      self.vars_and_pars[key] = np.ones(size) * val
    for key, val in pars_update.items():
      self.vars_and_pars[key] = val
    self.scope['VP'] = self.vars_and_pars
    self.scope['MON'] = self.mon
    self.scope['_fixed_vars'] = fixed_vars

    code_lines = ['def run_func(t_and_dt):']
    code_lines.append('  _t, _dt = t_and_dt')
    for integral in self.model.INTG:
      assert isinstance(integral, ODEIntegrator)
      func_name = integral.func_name
      self.scope[func_name] = integral
      # update the step function
      assigns = [f'VP["{var}"]' for var in integral.variables]
      calls = [f'VP["{var}"]' for var in integral.variables]
      calls.append('_t')
      calls.extend([f'VP["{var}"]' for var in integral.parameters[1:]])
      code_lines.append(f'  {", ".join(assigns)} = {func_name}({", ".join(calls)})')
      # reassign the fixed variables
      for key, val in fixed_vars.items():
        code_lines.append(f'  VP["{key}"][:] = _fixed_vars["{key}"]')
    # monitor the target variables
    for key in target_vars.keys():
      code_lines.append(f'  MON["{key}"].append(VP["{key}"])')
    # compile
    code = '\n'.join(code_lines)
    if show_code:
      print(code)
      print()
      pprint(self.scope)
      print()

    # recompile
    exec(compile(code, '', 'exec'), self.scope)
    self.run_func = self.scope['run_func']

  def run(self, duration, report=0.1):
    if isinstance(duration, (int, float)):
      duration = [0, duration]
    elif isinstance(duration, (tuple, list)):
      assert len(duration) == 2
      duration = tuple(duration)
    else:
      raise ValueError

    # get the times
    times = np.arange(duration[0], duration[1], bm.get_dt())
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = []
    # run the model
    run_model(run_func=self.run_func, times=times, report=report)
    # reshape the monitor
    for key in self.mon.keys():
      self.mon[key] = np.asarray(self.mon[key])

