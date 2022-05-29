# -*- coding: utf-8 -*-

import warnings

from brainpy.dyn import runners as dyn_runner
from brainpy.integrators import runner as intg_runner

__all__ = [
  'IntegratorRunner',
  'DSRunner',
  'StructRunner',
  'ReportRunner'
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


class StructRunner(dyn_runner.DSRunner):
  """Dynamical system runner class.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.StructRunner" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.StructRunner" instead. '
                  '"brainpy.StructRunner" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(StructRunner, self).__init__(*args, **kwargs)


class ReportRunner(dyn_runner.ReportRunner):
  """Dynamical system runner class.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.ReportRunner" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.dyn.ReportRunner" instead. '
                  '"brainpy.ReportRunner" is deprecated since '
                  'version 2.1.0', DeprecationWarning)
    super(ReportRunner, self).__init__(*args, **kwargs)
