# -*- coding: utf-8 -*-


import brainpy.math as bm
from brainpy import errors
from brainpy.integrators.ode.base import ODEIntegrator
from brainpy.building.brainobjects import DynamicalSystem
from brainpy.simulation.runner import StructRunner

__all__ = [
  'model_transform',
  'NumDSWrapper',
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
    self.name2integral = {intg.variables[0]: intg for intg in integrals}
    self.name2derivative = {intg.variables[0]: intg.f for intg in integrals}


class TrajectModel(DynamicalSystem):
  def __init__(self, integrals: dict, initial_vars: dict, pars=None, dt=None):
    super(TrajectModel, self).__init__()

    # variables
    assert isinstance(initial_vars, dict)
    initial_vars = {k: bm.Variable(bm.asarray(v)) for k, v in initial_vars.items()}
    self.register_implicit_vars(initial_vars)
    # self.all_vars = tuple(self.implicit_vars.values())

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
    all_vars = list(self.implicit_vars.values())
    for key, intg in self.integrals.items():
      self.implicit_vars[key].update(intg(*all_vars, *self.pars, dt=_dt))

  def __getattr__(self, item):
    child_vars = super(TrajectModel, self).__getattribute__('implicit_vars')
    if item in child_vars:
      return child_vars[item]
    else:
      return super(TrajectModel, self).__getattribute__(item)

  def run(self, duration):
    self.runner.run(duration)
    return self.runner.mon

