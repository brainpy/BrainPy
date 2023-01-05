# -*- coding: utf-8 -*-


from typing import Dict, Sequence, Union

from brainpy._src.math.object_transform.base import BrainPyObject
from brainpy._src.math import TimeDelay, LengthDelay
from brainpy.check import is_float, is_dict_data
from brainpy.errors import DiffEqError
from .constants import DT

__all__ = [
  'Integrator',
]


class AbstractIntegrator(BrainPyObject):
  """Basic Integrator Class."""

  # func_name
  # derivative
  # code_scope
  #

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Integrator(AbstractIntegrator):
  """Basic Integrator Class."""

  def __init__(
      self,
      variables: Sequence[str],
      parameters: Sequence[str],
      arguments: Sequence[str],
      dt: float,
      name: str = None,
      state_delays: Dict[str, Union[TimeDelay, LengthDelay]] = None,
  ):
    super(Integrator, self).__init__(name=name)

    self._dt = dt
    is_float(dt, 'dt', allow_none=False, allow_int=True)
    self._variables = list(variables)  # variables
    self._parameters = list(parameters)  # parameters
    self._arguments = list(arguments) + [f'{DT}={self._dt}', ]  # arguments
    self._integral = None  # integral function
    self.arg_names = self._variables + self._parameters + [DT]

    # state delays
    self._state_delays = dict()
    if state_delays is not None:
      is_dict_data(state_delays, key_type=str, val_type=(TimeDelay, LengthDelay))
      for key, delay in state_delays.items():
        if key not in self.variables:
          raise DiffEqError(f'"{key}" is not defined in the variables: {self.variables}')
        self._state_delays[key] = delay
    self.register_implicit_nodes(self._state_delays)

  @property
  def dt(self):
    """The numerical integration precision."""
    return self._dt

  @dt.setter
  def dt(self, value):
    raise ValueError('Cannot set "dt" by users.')

  @property
  def variables(self):
    """The variables defined in the differential equation."""
    return self._variables

  @variables.setter
  def variables(self, values):
    raise ValueError('Cannot set "variables" by users.')

  @property
  def parameters(self):
    """The parameters defined in the differential equation."""
    return self._parameters

  @parameters.setter
  def parameters(self, values):
    raise ValueError('Cannot set "parameters" by users.')

  @property
  def arguments(self):
    """All arguments when calling the numer integrator of the differential equation."""
    return self._arguments

  @arguments.setter
  def arguments(self, values):
    raise ValueError('Cannot set "arguments" by users.')

  @property
  def integral(self):
    """The integral function."""
    return self._integral

  @integral.setter
  def integral(self, f):
    self.set_integral(f)

  def set_integral(self, f):
    """Set the integral function."""
    if not callable(f):
      raise ValueError(f'integral function must be a callable function, '
                       f'but we got {type(f)}: {f}')
    self._integral = f

  @property
  def state_delays(self):
    """State delays."""
    return self._state_delays

  @state_delays.setter
  def state_delays(self, value):
    raise ValueError('Cannot set "state_delays" by users.')

  def __call__(self, *args, **kwargs):
    assert self.integral is not None, 'Please build the integrator first.'

    # check arguments
    for i, arg in enumerate(args):
      kwargs[self.arg_names[i]] = arg

    # integral
    new_vars = self.integral(**kwargs)
    if len(self.variables) == 1:
      dict_vars = {self.variables[0]: new_vars}
    else:
      dict_vars = {k: new_vars[i] for i, k in enumerate(self.variables)}

    # update state delay variables
    dt = kwargs.pop(DT, self.dt)
    for key, delay in self.state_delays.items():
      if isinstance(delay, LengthDelay):
        delay.update(dict_vars[key])
      elif isinstance(delay, TimeDelay):
        delay.update(dict_vars[key])
      else:
        raise ValueError('Unknown delay variable. We only supports '
                         'brainpy.math.LengthDelay, brainpy.math.TimeDelay. '
                         f'While we got {delay}')

    return new_vars
