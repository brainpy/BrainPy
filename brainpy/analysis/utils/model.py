# -*- coding: utf-8 -*-


import jax.numpy as jnp

import brainpy.math as bm
from brainpy import errors
from brainpy.dyn.base import DynamicalSystem
from brainpy.dyn.runners import DSRunner
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode.base import ODEIntegrator

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

  # check model types
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

  new_model = []
  for intg in model:
    if isinstance(intg.f, JointEq):
      new_model.extend([type(intg)(eq, var_type=intg.var_type, dt=intg.dt) for eq in intg.f.eqs])
    else:
      new_model.append(intg)

  # pars to update
  pars_update = set()
  for intg in new_model:
    pars_update.update(intg.parameters[1:])

  all_variables = set()
  all_parameters = set()
  for integral in new_model:
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
  return NumDSWrapper(integrals=new_model,
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
    initial_vars = {k: bm.Variable(jnp.asarray(bm.as_device_array(v), dtype=jnp.float_))
                    for k, v in initial_vars.items()}
    self.register_implicit_vars(initial_vars)

    # parameters
    pars = dict() if pars is None else pars
    assert isinstance(pars, dict)
    self.pars = [jnp.asarray(bm.as_device_array(v), dtype=jnp.float_)
                 for k, v in pars.items()]

    # integrals
    self.integrals = integrals

    # runner
    self.runner = DSRunner(self,
                           monitors=list(initial_vars.keys()),
                           dyn_vars=self.vars().unique(), dt=dt,
                           progress_bar=False)

  def update(self, t, dt):
    all_vars = list(self.implicit_vars.values())
    for key, intg in self.integrals.items():
      self.implicit_vars[key].update(intg(*all_vars, *self.pars, dt=dt))

  def __getattr__(self, item):
    child_vars = super(TrajectModel, self).__getattribute__('implicit_vars')
    if item in child_vars:
      return child_vars[item]
    else:
      return super(TrajectModel, self).__getattribute__(item)

  def run(self, duration):
    self.runner.run(duration)
    return self.runner.mon
