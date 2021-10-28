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
  """ODE Integrator."""

  def __init__(self, f, var_type=None, dt=None, name=None, show_code=False):
    super(ODEIntegrator, self).__init__(name=name)

    # others
    self.dt = math.get_dt() if dt is None else dt
    assert isinstance(self.dt, (int, float)), f'"dt" must be a float, but got {self.dt}'
    self.show_code = show_code

    # derivative function
    self.derivative = {constants.F: f}

    # integration function
    self.integral = None

    # parse function arguments
    variables, parameters, arguments = utils.get_args(f)
    self.variables = variables  # variable names, (before 't')
    self.parameters = parameters  # parameter names, (after 't')
    self.arguments = list(arguments) + [f'{constants.DT}={self.dt}']  # function arguments
    self.var_type = var_type  # variable type

    # code scope
    self.code_scope = {constants.F: f}

    # code lines
    self.func_name = f_names(f)
    self.code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']

  @abc.abstractmethod
  def build(self):
    raise NotImplementedError('Must implement how to build your step function.')

  def __call__(self, *args, **kwargs):
    assert self.integral is not None, 'Please build the integrator first.'
    return self.integral(*args, **kwargs)
