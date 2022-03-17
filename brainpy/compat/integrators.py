# -*- coding: utf-8 -*-

import warnings

from brainpy.integrators import ode, sde

__all__ = [
  'set_default_odeint',
  'set_default_sdeint',
  'get_default_odeint',
  'get_default_sdeint',
]


def set_default_odeint(method):
  """Set default ode integrator.

  .. deprecated:: 2.1.0
     Please use "brainpy.ode.set_default_odeint" instead.
  """
  warnings.warn('Please use "brainpy.ode.set_default_odeint" instead. '
                '"brainpy.set_default_odeint" is deprecated since '
                'version 2.1.0', DeprecationWarning)
  ode.set_default_odeint(method)


def get_default_odeint():
  """Get default ode integrator.

  .. deprecated:: 2.1.0
     Please use "brainpy.ode.get_default_odeint" instead.
  """
  warnings.warn('Please use "brainpy.ode.get_default_odeint" instead. '
                '"brainpy.get_default_odeint" is deprecated since '
                'version 2.1.0', DeprecationWarning)
  ode.get_default_odeint()


def set_default_sdeint(method):
  """Set default sde integrator.

  .. deprecated:: 2.1.0
     Please use "brainpy.ode.set_default_sdeint" instead.
  """
  warnings.warn('Please use "brainpy.sde.set_default_sdeint" instead. '
                '"brainpy.set_default_sdeint" is deprecated since '
                'version 2.1.0', DeprecationWarning)
  sde.set_default_sdeint(method)


def get_default_sdeint():
  """Get default sde integrator.

  .. deprecated:: 2.1.0
     Please use "brainpy.ode.get_default_sdeint" instead.
  """
  warnings.warn('Please use "brainpy.sde.get_default_sdeint" instead. '
                '"brainpy.get_default_sdeint" is deprecated since '
                'version 2.1.0', DeprecationWarning)
  sde.get_default_sdeint()
