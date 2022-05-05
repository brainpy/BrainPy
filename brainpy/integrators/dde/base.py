# -*- coding: utf-8 -*-

from typing import Union, Callable, Dict

import brainpy.math as bm
from brainpy.errors import DiffEqError
from brainpy.integrators.base import Integrator
from brainpy.integrators.constants import F, DT, unique_name
from brainpy.integrators.utils import get_args
from brainpy.tools.checking import check_dict_data

__all__ = [
  'DDEIntegrator',
]


class DDEIntegrator(Integrator):
  """Basic numerical integrator for delay differential equations (DDEs).
  """

  def __init__(
      self,
      f: Callable,
      var_type: str = None,
      dt: Union[float, int] = None,
      name: str = None,
      show_code: bool = False,
      state_delays: Dict[str, bm.TimeDelay] = None,
      neutral_delays: Dict[str, bm.NeutralDelay] = None,
  ):
    dt = bm.get_dt() if dt is None else dt
    parses = get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

    # super initialization
    super(DDEIntegrator, self).__init__(name=name,
                                        variables=variables,
                                        parameters=parameters,
                                        arguments=arguments,
                                        dt=dt,
                                        state_delays=state_delays)

    # other settings
    self.var_type = var_type
    self.show_code = show_code

    # derivative function
    self.derivative = {F: f}
    self.f = f

    # code scope
    self.code_scope = {F: f}

    # code lines
    self.func_name = _f_names(f)
    self.code_lines = [f'def {self.func_name}({", ".join(self.arguments)}):']

    # delays
    self._neutral_delays = dict()
    if neutral_delays is not None:
      check_dict_data(neutral_delays,
                      key_type=str,
                      val_type=bm.NeutralDelay)
      for key, delay in neutral_delays.items():
        if key not in self.variables:
          raise DiffEqError(f'"{key}" is not defined in the variables: {self.variables}')
        self._neutral_delays[key] = delay
    self.register_implicit_nodes(self._neutral_delays)
    if (len(self.neutral_delays) + len(self.state_delays)) == 0:
      raise DiffEqError('There is no delay variable, it should not be '
                        'a delay differential equation, please use "brainpy.odeint()". '
                        'Or, if you forget add delay variables, please set them with '
                        '"state_delays" and "neutral_delays" arguments.')

  @property
  def neutral_delays(self):
    return self._neutral_delays

  @neutral_delays.setter
  def neutral_delays(self, value):
    raise ValueError('Cannot set "neutral_delays" by users.')

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

    dt = kwargs.pop(DT, self.dt)
    # update neutral delay variables
    if len(self.neutral_delays):
      kwargs.update(dict_vars)
      new_dvars = self.f(**kwargs)
      if len(self.variables) == 1:
        new_dvars = {self.variables[0]: new_dvars}
      else:
        new_dvars = {k: new_dvars[i] for i, k in enumerate(self.variables)}
      for key, delay in self.neutral_delays.items():
        if isinstance(delay, bm.LengthDelay):
          delay.update(new_dvars[key])
        elif isinstance(delay, bm.TimeDelay):
          delay.update(kwargs['t'] + dt, new_dvars[key])
        else:
          raise ValueError('Unknown delay variable. We only supports '
                           'brainpy.math.LengthDelay, brainpy.math.TimeDelay, '
                           'brainpy.math.NeutralDelay. '
                           f'While we got {delay}')

    # update state delay variables
    for key, delay in self.state_delays.items():
      if isinstance(delay, bm.LengthDelay):
        delay.update(dict_vars[key])
      elif isinstance(delay, bm.TimeDelay):
        delay.update(kwargs['t'] + dt, dict_vars[key])
      else:
        raise ValueError('Unknown delay variable. We only supports '
                         'brainpy.math.LengthDelay, brainpy.math.TimeDelay, '
                         'brainpy.math.NeutralDelay. '
                         f'While we got {delay}')

    return new_vars


def _f_names(f):
  func_name = unique_name('dde')
  if f.__name__.isidentifier():
    func_name += '_' + f.__name__
  return func_name
