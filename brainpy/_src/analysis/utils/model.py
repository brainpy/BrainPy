# -*- coding: utf-8 -*-


from brainpy._src.math.object_transform import Variable
from brainpy._src.math.environment import get_float
from brainpy._src.math.interoperability import as_jax
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.context import share
from brainpy._src.runners import DSRunner
from brainpy._src.integrators.base import Integrator
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.base import ODEIntegrator
from brainpy._src.integrators.ode.generic import odeint
from brainpy.errors import AnalyzerError, UnsupportedError

__all__ = [
  'model_transform',
  'NumDSWrapper',
  'TrajectModel',
]


def _check_model(model):
  if isinstance(model, Integrator):
    if not isinstance(model, ODEIntegrator):
      raise AnalyzerError(f'Must be the instance of {ODEIntegrator.__name__}, but got {model}.')
  elif callable(model):
    model = odeint(model)
  else:
    raise ValueError(f'Please provide derivative function or integral function. But we got {model}')
  if isinstance(model.f, JointEq):
    return [type(model)(eq, var_type=model.var_type, dt=model.dt) for eq in model.f.eqs]
  else:
    return [model]


def model_transform(model):
  # check model
  if isinstance(model, DynamicalSystem):
    model = tuple(model.nodes(level=-1).subset(ODEIntegrator).unique().values())
  elif isinstance(model, NumDSWrapper):
    return model
  elif isinstance(model, ODEIntegrator):  #
    model = [model]
  elif callable(model):
    model = [model]
  all_models = []
  if isinstance(model, (list, tuple)):
    if len(model) == 0:
      raise AnalyzerError(f'Found no derivative/integral functions: {model}')
    for fun in tuple(model):
      all_models.extend(_check_model(fun))
  elif isinstance(model, dict):
    if len(model) == 0:
      raise AnalyzerError(f'Found no derivative/integral functions: {model}')
    for fun in tuple(model.values()):
      all_models.extend(_check_model(fun))
  else:
    raise UnsupportedError(f'Dynamics analysis by symbolic approach only supports '
                           f'derivative/integral functions or {DynamicalSystem.__name__}, '
                           f'but we got: {type(model)}: {str(model)}')

  # pars to update
  pars_update = set()
  for fun in all_models:
    pars_update.update(fun.parameters[1:])

  # variables and parameters
  all_variables = set()
  all_parameters = set()
  for integral in all_models:
    # variable
    if len(integral.variables) != 1:
      raise AnalyzerError(f'Only supports one {ODEIntegrator.__name__} one variable, '
                          f'but we got {len(integral.variables)} variables in {integral}.')
    var = integral.variables[0]
    if var in all_variables:
      raise AnalyzerError(f'Variable name {var} has been defined before. '
                          f'Please change another name.')
    all_variables.add(var)
    # parameter
    all_parameters.update(integral.parameters[1:])

  # form a dynamic model
  return NumDSWrapper(integrals=all_models,
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
    self.f_integrals = integrals  # all integrators
    self.f_derivatives = {intg.variables[0]: intg.f for intg in integrals}  # all integrators
    self.variables = variables  # all variables
    self.parameters = parameters  # all parameters
    self.pars_update = pars_update  # the parameters to update
    self.name2integral = {intg.variables[0]: intg for intg in integrals}
    self.name2derivative = {intg.variables[0]: intg.f for intg in integrals}

  def __repr__(self):
    return f'{self.__class__.__name__}(variables={self.variables}, parameters={self.parameters})'


class TrajectModel(DynamicalSystem):
  def __init__(self, integrals: dict, initial_vars: dict, pars=None, dt=None):
    super(TrajectModel, self).__init__()

    # variables
    assert isinstance(initial_vars, dict)
    initial_vars = {k: Variable(as_jax(v, dtype=get_float()))
                    for k, v in initial_vars.items()}
    self.register_implicit_vars(initial_vars)

    # parameters
    pars = dict() if pars is None else pars
    assert isinstance(pars, dict)
    self.pars = [as_jax(v, dtype=get_float()) for k, v in pars.items()]

    # integrals
    self.integrals = integrals

    # runner
    self.runner = DSRunner(self, monitors=list(initial_vars.keys()), dt=dt, progress_bar=False)

  def update(self):
    all_vars = list(self.implicit_vars.values())
    for key, intg in self.integrals.items():
      self.implicit_vars[key].update(intg(*all_vars, *self.pars, dt=share['dt']))

  def __getattr__(self, item):
    child_vars = super(TrajectModel, self).__getattribute__('implicit_vars')
    if item in child_vars:
      return child_vars[item]
    else:
      return super(TrajectModel, self).__getattribute__(item)

  def run(self, duration):
    self.runner.run(duration)
    return self.runner.mon
