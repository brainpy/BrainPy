# -*- coding: utf-8 -*-

import warnings

from brainpy.dyn import runners as dyn_runner
from brainpy.integrators import runner as intg_runner

__all__ = [
  'IntegratorRunner',
  'DSRunner',
]


class IntegratorRunner(intg_runner.IntegratorRunner):
  """Integrator runner class.

  .. deprecated:: 2.1.0
     Please use "brainpy.integrators.IntegratorRunner" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.integrators.IntegratorRunner" instead. '
                  '"brainpy.IntegratorRunner" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(IntegratorRunner, self).__init__(*args, **kwargs)


class DSRunner(dyn_runner.DSRunner):
  """Dynamical system runner class.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.DSRunner" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.DSRunner" instead. '
                  '"brainpy.DSRunner" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(DSRunner, self).__init__(*args, **kwargs)
