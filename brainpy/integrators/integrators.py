# -*- coding: utf-8 -*-


class Integrator(object):
  def __init__(self, call, variables, parameters, var_type, dt):
    self.variables = variables
    self.parameters = parameters
    self.var_type = var_type
    self.dt = dt
    self.call = call

  def __call__(self, *args, **kwargs):
    return self.call(*args, **kwargs)


class ODEIntegrator(Integrator):
  def __init__(self, f, **kwargs):
    super(ODEIntegrator, self).__init__(**kwargs)
    self.f = f



class SDEIntegrator(Integrator):
  def __init__(self, f, g, **kwargs):
    super(SDEIntegrator, self).__init__(**kwargs)
    self.f = f
    self.g = g


class DDEIntegrator(Integrator):
  pass


class FDEIntegrator(Integrator):
  pass
