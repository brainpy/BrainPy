# -*- coding: utf-8 -*-

__all__ = [
  'Integrator',
  'ODEIntegrator',
  'SDEIntegrator',
  'DDEIntegrator',
  'FDEIntegrator',
]


class Integrator(object):
  """The base Integrator class for differential equations.

  Numerical integrator is setup by this.

  Parameters
  ----------
  call : function
    The numerical integration function.
  variables : list of str, tuple of str
    The variables in the system.
  parameters : list of str, tuple of str
    The parameters in the system.
  var_type : str
    The variable type.
  dt : float, int
    The numerical precision.
  code : str
    The compiled code for numerical integration.
  """

  def __init__(self, call, variables, parameters, var_type, dt, code):
    self.variables = tuple(variables)
    self.parameters = tuple(parameters)
    self.var_type = var_type
    self.dt = dt
    self.call = call
    self.code = code

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)


class ODEIntegrator(Integrator):
  """Integrator class for ordinary differential equations.

  Parameters
  ----------
  f : function
    The original function.
  """

  def __init__(self, f, **kwargs):
    super(ODEIntegrator, self).__init__(**kwargs)
    self.f = f


class SDEIntegrator(Integrator):
  """Integrator class for stochastic differential equations.

  Parameters
  ----------
  f : function
    The original function representing the drift coefficient.
  g : function
    The original function representing the difussion coefficient.
  """

  def __init__(self, f, g, intg_type, wiener_type, **kwargs):
    super(SDEIntegrator, self).__init__(**kwargs)
    self.f = f
    self.g = g
    self.intg_type = intg_type
    self.wiener_type = wiener_type


class DDEIntegrator(Integrator):
  """Integrator class for delayed differential equations.

  Parameters
  ----------
  f : function
    The original function.
  """
  pass


class FDEIntegrator(Integrator):
  """Integrator class for fractional differential equations.

  Parameters
  ----------
  f : function
    The original function.
  """
  pass
