# -*- coding: utf-8 -*-

import warnings

from brainpy.sim import runners

__all__ = [
  'IntegratorRunner',
  'DSRunner',
  'StructRunner',
  'ReportRunner'
]


class IntegratorRunner(runners.IntegratorRunner):
  def __init__(self, *args, **kwargs):
    super(IntegratorRunner, self).__init__(*args, **kwargs)
    warnings.warn('Please use "brainpy.sim.IntegratorRunner" instead. '
                  '"brainpy.IntegratorRunner" will be removed since version 2.1.0.',
                  DeprecationWarning)


class DSRunner(runners.DSRunner):
  def __init__(self, *args, **kwargs):
    super(DSRunner, self).__init__(*args, **kwargs)
    warnings.warn('Please use "brainpy.sim.DSRunner" instead. '
                  '"brainpy.DSRunner" will be removed since version 2.1.0.',
                  DeprecationWarning)


StructRunner = runners.StructRunner
ReportRunner = runners.ReportRunner
