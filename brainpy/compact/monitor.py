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
                  '"brainpy.Monitor" is deprecated since version 2.0.3.',
                  DeprecationWarning)
