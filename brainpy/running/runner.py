# -*- coding: utf-8 -*-

import gc
import types
from typing import Callable, Dict, Sequence, Union

import numpy as np

from brainpy import math as bm
from brainpy.base import Base
from brainpy.base.collector import TensorCollector
from brainpy.errors import MonitorError, RunningError
from brainpy.tools.checking import check_dict_data
from brainpy.tools.others import DotDict
from . import constants as C

__all__ = [
  'Runner',
]


class Runner(object):
  """Base Runner.

  Parameters
  ----------
  target: Any
    The target model.

  monitors: None, sequence of str, dict, Monitor
    Variables to monitor.

    - A list of string. Like `monitors=['a', 'b', 'c']`
    - A list of string with index specification. Like `monitors=[('a', 1), ('b', [1,3,5]), 'c']`
    - A dict with the explicit monitor target, like: `monitors={'a': model.spike, 'b': model.V}`
    - A dict with the index specification, like: `monitors={'a': (model.spike, 0), 'b': (model.V, [1,2])}`

  fun_monitors: dict
    Monitoring variables by callable functions. Should be a dict.
    The `key` should be a string for later retrieval by `runner.mon[key]`.
    The `value` should be a callable function which receives two arguments: `t` and `dt`.

  jit: bool, dict
    The JIT settings.

  progress_bar: bool
    Use progress bar to report the running progress or not?

  dyn_vars: Optional, dict
    The dynamically changed variables. Instance of :py:class:`~.Variable`.

  numpy_mon_after_run : bool
    When finishing the network running, transform the JAX arrays into numpy ndarray or not?
  """

  mon: DotDict
  jit: Dict[str, bool]
  target: Base

  def __init__(
      self,
      target: Base,
      monitors: Union[Sequence, Dict] = None,
      fun_monitors: Dict[str, Callable] = None,
      jit: Union[bool, Dict[str, bool]] = True,
      progress_bar: bool = True,
      dyn_vars: Union[Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      numpy_mon_after_run: bool = True
  ):
    # target model, while implement __call__ function
    self.target = target

    # jit instruction
    self.jit = dict()
    if isinstance(jit, bool):
      self.jit = {C.PREDICT_PHASE: jit}
    elif isinstance(jit, dict):
      for k, v in jit.items():
        self.jit[k] = v
      self.jit[C.PREDICT_PHASE] = jit.pop(C.PREDICT_PHASE, True)
    else:
      raise ValueError(f'Unknown "jit" setting: {jit}')

    if monitors is None:
      monitors = dict()
    elif isinstance(monitors, (list, tuple)):
      # format string monitors
      monitors = self._format_seq_monitors(monitors)
      # get monitor targets
      monitors = self._find_monitor_targets(monitors)
    elif isinstance(monitors, dict):
      _monitors = dict()
      for key, val in monitors.items():
        if not isinstance(key, str):
          raise MonitorError('Expect the key of the dict "monitors" must be a string. But got '
                             f'{type(key)}: {key}')
        if isinstance(val, bm.Variable):
          val = (val, None)
        if isinstance(val, (tuple, list)):
          if not isinstance(val[0], bm.Variable):
            raise MonitorError('Expect the format of (variable, index) in the monitor setting. '
                               f'But we got {val}')
          if len(val) == 1:
            _monitors[key] = (val[0], None)
          elif len(val) == 2:
            if isinstance(val[1], (int, np.integer)):
              idx = bm.array([val[1]])
            else:
              idx = None if val[1] is None else bm.asarray(val[1])
            _monitors[key] = (val[0], idx)
          else:
            raise MonitorError('Expect the format of (variable, index) in the monitor setting. '
                               f'But we got {val}')
        else:
          raise MonitorError('Expect the format of (variable, index) in the monitor setting. '
                             f'But we got {val}')
      monitors = _monitors
    else:
      raise MonitorError(f'We only supports a format of list/tuple/dict of '
                         f'"vars", while we got {type(monitors)}.')
    self.monitors = monitors

    # extra monitors
    if fun_monitors is None:
      fun_monitors = dict()
    check_dict_data(fun_monitors, key_type=str, val_type=types.FunctionType)
    self.fun_monitors = fun_monitors

    # monitor for user access
    self.mon = DotDict()
    self.mon['var_names'] = tuple(self.monitors.keys()) + tuple(self.fun_monitors.keys())

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

  def format_monitors(self):
    return_with_idx = dict()
    return_without_idx = dict()
    for key, (variable, idx) in self.monitors.items():
      if idx is None:
        return_without_idx[key] = variable
      else:
        return_with_idx[key] = (variable, bm.asarray(idx))
    return return_without_idx, return_with_idx

  def _format_seq_monitors(self, monitors):
    if not isinstance(monitors, (tuple, list)):
      raise TypeError(f'Must be a sequence, but we got {type(monitors)}')
    _monitors = []
    for mon in monitors:
      if isinstance(mon, str):
        _monitors.append((mon, None))
      elif isinstance(mon, (tuple, list)):
        if isinstance(mon[0], str):
          if len(mon) == 1:
            _monitors.append((mon[0], None))
          elif len(mon) == 2:
            if isinstance(mon[1], (int, np.integer)):
              idx = bm.array([mon[1]])
            else:
              idx = None if mon[1] is None else bm.asarray(mon[1])
            _monitors.append((mon[0], idx))
          else:
            raise MonitorError(f'We expect the monitor format with (name, index). But we got {mon}')
        else:
          raise MonitorError(f'We expect the monitor format with (name, index). But we got {mon}')
      else:
        raise MonitorError(f'We do not support monitor with {type(mon)}: {mon}')
    return _monitors

  def _find_monitor_targets(self, _monitors):
    if not isinstance(_monitors, (tuple, list)):
      raise TypeError(f'Must be a sequence, but we got {type(_monitors)}')
    # get monitor targets
    monitors = {}
    name2node = {node.name: node for node in list(self.target.nodes(level=-1).unique().values())}
    for mon in _monitors:
      key, index = mon[0], mon[1]
      splits = key.split('.')
      if len(splits) == 1:
        if not hasattr(self.target, splits[0]):
          raise RunningError(f'{self.target} does not has variable {key}.')
        monitors[key] = (getattr(self.target, splits[-1]), index)
      else:
        if not hasattr(self.target, splits[0]):
          if splits[0] not in name2node:
            raise MonitorError(f'Cannot find target {key} in monitor of {self.target}, please check.')
          else:
            master = name2node[splits[0]]
            assert len(splits) == 2
            monitors[key] = (getattr(master, splits[-1]), index)
        else:
          master = self.target
          for s in splits[:-1]:
            try:
              master = getattr(master, s)
            except KeyError:
              raise MonitorError(f'Cannot find {key} in {master}, please check.')
          monitors[key] = (getattr(master, splits[-1]), index)
    return monitors

  def build_monitors(self, return_without_idx, return_with_idx, shared_args) -> Callable:
    raise NotImplementedError

  def __del__(self):
    if hasattr(self, 'mon'):
      for key in tuple(self.mon.keys()):
        del self.mon[key]
    for key in tuple(self.__dict__.keys()):
      del self.__dict__[key]
    gc.collect()
