# -*- coding: utf-8 -*-


import abc

from brainpy import math
from brainpy.integrators import constants, utils
from brainpy.integrators.base import Integrator

__all__ = [
  'ODEIntegrator',
]


def f_names(f):
  func_name = constants.unique_name('ode')
  if f.__name__.isidentifier():
    func_name += '_' + f.__name__
  return func_name


class ODEIntegrator(Integrator):
  """Numerical Integrator for Ordinary Differential Equations (ODEs).

  Parameters
  ----------
  f : callable
    The derivative function.
  var_type: str
    The type for each variable.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.
  """

  def __init__(self, f, var_type=None, dt=None, name=None, show_code=False):

    dt = math.get_dt() if dt is None else dt
    parses = utils.get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

    # super initialization
    super(ODEIntegrator, self).__init__(name=name,
                                        variables=variables,
                                        parameters=parameters,
                                        arguments=arguments,
                                        dt=dt)

    # others
    self.show_code = show_code
    self.var_type = var_type  # variable type

    # derivative function
    self.derivative = {constants.F: f}
    self.f = f

    # code scope
    self.code_scope = {constants.F: f}

    # code lines
    self.func_name = f_names(f)
    self.code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']

  @abc.abstractmethod
  def build(self):
    raise NotImplementedError('Must implement how to build your step function.')
