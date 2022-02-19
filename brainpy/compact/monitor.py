# -*- coding: utf-8 -*-
import warnings

from brainpy.sim import monitor

__all__ = [
  'Monitor'
]


class Monitor(monitor.Monitor):
  def __init__(self, *args, **kwargs):
    super(Monitor, self).__init__(*args, **kwargs)
    warnings.warn('Please use "brainpy.sim.Monitor" instead. '
                  '"brainpy.Monitor" will be removed since version 2.1.0.',
                  DeprecationWarning)
