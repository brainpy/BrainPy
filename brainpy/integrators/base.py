# -*- coding: utf-8 -*-


from brainpy.base.base import Base

from brainpy.tools.checking import check_float
from brainpy.integrators.constants import DT

__all__ = [
  'Integrator',
]


class AbstractIntegrator(Base):
  """Basic Integrator Class."""

  # func_name
  # derivative
  # code_scope
  #
  def build(self, *args, **kwargs):
    raise NotImplementedError('Implement build method by yourself.')

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Integrator(AbstractIntegrator):
  """Basic Integrator Class."""

  def __init__(self, variables, parameters, arguments, dt, name=None):
    super(Integrator, self).__init__(name=name)

    self._dt = dt
    check_float(dt, 'dt', allow_none=False, allow_int=True)
    self._variables = variables  # variables
    self._parameters = parameters # parameters
    self._arguments = list(arguments) + [f'{DT}={self.dt}'] # arguments
    self._integral = None  # integral function

  @property
  def dt(self):
    return self._dt

  @dt.setter
  def dt(self, value):
    raise ValueError('Cannot set "dt" by users.')

  @property
  def variables(self):
    return self._variables

  @variables.setter
  def variables(self, values):
    raise ValueError('Cannot set "variables" by users.')

  @property
  def parameters(self):
    return self._parameters

  @parameters.setter
  def parameters(self, values):
    raise ValueError('Cannot set "parameters" by users.')

  @property
  def arguments(self):
    return self._arguments

  @arguments.setter
  def arguments(self, values):
    raise ValueError('Cannot set "arguments" by users.')

  @property
  def integral(self):
    return self._integral

  @integral.setter
  def integral(self, f):
    self.set_integral(f)

  def set_integral(self, f):
    if not callable(f):
      raise ValueError(f'integral function must be a callable function, '
                       f'but we got {type(f)}: {f}')
    self._integral = f

  def __call__(self, *args, **kwargs):
    assert self.integral is not None, 'Please build the integrator first.'
    return self.integral(*args, **kwargs)
