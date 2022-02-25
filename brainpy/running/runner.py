# -*- coding: utf-8 -*-

from brainpy.errors import MonitorError, RunningError
from .monitor import Monitor

__all__ = [
  'Runner'
]


class Runner(object):
  def __init__(self, target, monitors=None, jit=True, progress_bar=True, dyn_vars=None):
    # target model, while implement __call__ function
    self.target = target

    # jit instruction
    assert isinstance(jit, bool), 'Must be a boolean variable.'
    self.jit = jit

    # monitors
    if monitors is None:
      self.mon = Monitor(target=self, variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(target=self, variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise MonitorError(f'"monitors" only supports list/tuple/dict/ '
                         f'instance of Monitor, not {type(monitors)}.')
    self.mon.build()  # build the monitor

    # progress bar
    assert isinstance(progress_bar, bool), 'Must be a boolean variable.'
    self.progress_bar = progress_bar
    self._pbar = None

    # dynamical changed variables
    if dyn_vars is None:
      dyn_vars = self.target.vars().unique()
    if isinstance(dyn_vars, (list, tuple)):
      dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
    if not isinstance(dyn_vars, dict):
      raise RunningError(f'"dyn_vars" must be a dict, but we got {type(dyn_vars)}')
    self.dyn_vars = dyn_vars

