# -*- coding: utf-8 -*-

import types

from brainpy.base.collector import TensorCollector
from brainpy.errors import MonitorError, RunningError
from brainpy.tools.checking import check_dict_data
from .monitor import Monitor

__all__ = [
  'Runner'
]


class Runner(object):
  """Base Runner.

  Parameters
  ----------
  target: Any
    The target model.
  monitors: None, list of str, tuple of str, Monitor
    Variables to monitor.
  jit: bool
  progress_bar: bool
  dyn_vars: Optional, dict
  numpy_mon_after_run : bool
  """
  def __init__(self, target, monitors=None, fun_monitors=None,
               jit=True, progress_bar=True, dyn_vars=None,
               numpy_mon_after_run=True):
    # target model, while implement __call__ function
    self.target = target

    # jit instruction
    assert isinstance(jit, bool), 'Must be a boolean variable.'
    self.jit = jit

    # monitors
    if monitors is None:
      self.mon = Monitor(variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise MonitorError(f'"monitors" only supports list/tuple/dict/ '
                         f'instance of Monitor, not {type(monitors)}.')
    self.mon.build()  # build the monitor

    # extra monitors
    if fun_monitors is None:
      fun_monitors = dict()
    check_dict_data(fun_monitors, key_type=str, val_type=types.FunctionType)
    self.fun_monitors = fun_monitors
    for key in self.fun_monitors.keys():
      self.mon.item_names.append(key)
      self.mon.item_contents[key] = []

    # progress bar
    assert isinstance(progress_bar, bool), 'Must be a boolean variable.'
    self.progress_bar = progress_bar
    self._pbar = None

    # dynamical changed variables
    if dyn_vars is None:
      dyn_vars = dict()
    if isinstance(dyn_vars, (list, tuple)):
      dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
    if not isinstance(dyn_vars, dict):
      raise RunningError(f'"dyn_vars" must be a dict, but we got {type(dyn_vars)}')
    self.dyn_vars = TensorCollector(dyn_vars)

    # numpy mon after run
    self.numpy_mon_after_run = numpy_mon_after_run

