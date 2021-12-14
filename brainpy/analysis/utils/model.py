# -*- coding: utf-8 -*-


import inspect

import brainpy.math as bm
from brainpy import errors, tools
from brainpy.base.base import TensorCollector
from brainpy.integrators import analysis_by_ast
from brainpy.integrators import analysis_by_sympy
from brainpy.integrators.ode.base import ODEIntegrator
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.runner import StructRunner

__all__ = [
  'model_transform',
  'num2sym',
  'NumDSWrapper',
  'SymDSWrapper',
  'TrajectModel',
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


class TrajectModel(DynamicalSystem):
  def __init__(self, integrals, initial_vars, pars=None, dt=None):
    super(TrajectModel, self).__init__()

    # variables
    assert isinstance(initial_vars, dict)
    initial_vars = {k: bm.Variable(bm.asarray(v)) for k, v in initial_vars.items()}
    self.register_implicit_vars(initial_vars)
    self.all_vars = tuple(self.implicit_vars.values())

    # parameters
    pars = dict() if pars is None else pars
    assert isinstance(pars, dict)
    self.pars = [bm.asarray(v) for k, v in pars.items()]

    # integrals
    self.integrals = integrals

    # runner
    self.runner = StructRunner(self,
                               monitors=list(initial_vars.keys()),
                               dyn_vars=self.vars().unique(), dt=dt)

  def update(self, _t, _dt):
    for i, intg in enumerate(self.integrals):
      self.all_vars[i].update(intg(*self.all_vars, *self.pars, dt=_dt))

  def __getattr__(self, item):
    child_vars = super(TrajectModel, self).__getattribute__('implicit_vars')
    if item in child_vars:
      return child_vars[item]
    else:
      return super(TrajectModel, self).__getattribute__(item)

  def run(self, duration):
    self.runner.run(duration)
    return self.runner.mon

