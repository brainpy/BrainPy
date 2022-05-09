# -*- coding: utf-8 -*-
import warnings

from brainpy.running import monitor

__all__ = [
  'Monitor'
]


class Monitor(monitor.Monitor):
  """Monitor class.

  .. deprecated:: 2.1.0
     Please use "brainpy.running.Monitor" instead.
  """
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.running.Monitor" instead. '
                  '"brainpy.Monitor" is deprecated since version 2.1.0.',
                  DeprecationWarning)
    super(Monitor, self).__init__(*args, **kwargs)
