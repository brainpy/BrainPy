# -*- coding: utf-8 -*-

import gc
import types
import warnings
from typing import Callable, Dict, Sequence, Union

import numpy as np

from brainpy import math as bm, check
from brainpy._src.math.object_transform.base import BrainPyObject
from brainpy.errors import MonitorError, RunningError
from brainpy.tools import DotDict
from . import constants as C

__all__ = [
  'Runner',
]


class Runner(BrainPyObject):
  """Base Runner.

  Parameters
  ----------
  target: Any
    The target model.

  monitors: None, sequence of str, dict, Monitor
    Variables to monitor.

    - A list of string. Like ``monitors=['a', 'b', 'c']``
    - A list of string with index specification. Like ``monitors=[('a', 1), ('b', [1,3,5]), 'c']``
    - A dict with the explicit monitor target, like: ``monitors={'a': model.spike, 'b': model.V}``
    - A dict with the index specification, like: ``monitors={'a': (model.spike, 0), 'b': (model.V, [1,2])}``
    - A dict with the callable function, like ``monitors={'a': lambda tdi: model.spike[:5]}``

    .. versionchanged:: 2.3.1
       ``func_monitors`` are merged into ``monitors``.

  fun_monitors: dict
    Monitoring variables by a dict of callable functions.
    The `key` should be a string for later retrieval by `runner.mon[key]`.
    The `value` should be a callable function which receives two arguments: `t` and `dt`.

    .. deprecated:: 2.3.1
       Use ``monitors`` instead.
  jit: bool, dict
    The JIT settings.

  progress_bar: bool
    Use progress bar to report the running progress or not?

  dyn_vars: Optional, Variable, sequence of Variable, dict
    The dynamically changed variables. Instance of :py:class:`~.Variable`.

  numpy_mon_after_run : bool
    When finishing the network running, transform the JAX arrays into numpy ndarray or not?
  """

  mon: DotDict
  '''Monitor data.'''

  jit: Dict[str, bool]
  '''Flag to denote whether to use JIT.'''

  def __init__(
      self,
      target: BrainPyObject,
      monitors: Union[Sequence, Dict] = None,
      fun_monitors: Dict[str, Callable] = None,
      jit: Union[bool, Dict[str, bool]] = True,
      progress_bar: bool = True,
      dyn_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      numpy_mon_after_run: bool = True
  ):
    super().__init__()
    # target model, while implement __call__ function
    self.target = target

    # jit instruction
    self._origin_jit = jit
    self.jit = dict()
    if isinstance(jit, bool):
      self.jit = {C.PREDICT_PHASE: jit}
    elif isinstance(jit, dict):
      for k, v in jit.items():
        self.jit[k] = v
      self.jit[C.PREDICT_PHASE] = jit.pop(C.PREDICT_PHASE, True)
    else:
      raise ValueError(f'Unknown "jit" setting: {jit}')

    # monitor construction
    if monitors is None:
      monitors = dict()
    elif isinstance(monitors, (list, tuple)):
      # format string monitors
      monitors = self._format_seq_monitors(monitors)
      # get monitor targets
      monitors = self._find_seq_monitor_targets(monitors)
    elif isinstance(monitors, dict):
      # format string monitors
      monitors = self._format_dict_monitors(monitors)
      # get monitor targets
      monitors = self._find_dict_monitor_targets(monitors)
    else:
      raise MonitorError(f'We only supports a format of list/tuple/dict of '
                         f'"vars", while we got {type(monitors)}.')
    self._monitors: dict = monitors

    # deprecated func_monitors
    if fun_monitors is not None:
      if isinstance(fun_monitors, dict):
        warnings.warn("`fun_monitors` is deprecated since version 2.3.1. "
                      "Define `func_monitors` in `monitors`")
      check.is_dict_data(fun_monitors, key_type=str, val_type=types.FunctionType)
      self._monitors.update(fun_monitors)

    # monitor for user access
    self.mon = DotDict()

    # progress bar
    assert isinstance(progress_bar, bool), 'Must be a boolean variable.'
    self.progress_bar = progress_bar
    self._pbar = None

    # dynamical changed variables
    self._dyn_vars = check.is_all_vars(dyn_vars, out_as='dict')

    # numpy mon after run
    self.numpy_mon_after_run = numpy_mon_after_run

  def _format_seq_monitors(self, monitors):
    if not isinstance(monitors, (tuple, list)):
      raise TypeError(f'Must be a tuple/list, but we got {type(monitors)}')
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

  def _format_dict_monitors(self, monitors):
    if not isinstance(monitors, dict):
      raise TypeError(f'Must be a dict, but we got {type(monitors)}')
    _monitors = dict()
    for key, val in monitors.items():
      if not isinstance(key, str):
        raise MonitorError('Expect the key of the dict "monitors" must be a string. But got '
                           f'{type(key)}: {key}')
      if isinstance(val, (bm.Variable, str)):
        val = (val, None)

      if isinstance(val, (tuple, list)):
        if not isinstance(val[0], (bm.Variable, str)):
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
      elif callable(val):
        _monitors[key] = val
      else:
        raise MonitorError('The value of dict monitor expect a sequence with (variable, index) '
                           f'or a callable function. But we got {val}')
    return _monitors

  def _find_seq_monitor_targets(self, _monitors):
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

  def _find_dict_monitor_targets(self, _monitors):
    if not isinstance(_monitors, dict):
      raise TypeError(f'Must be a dict, but we got {type(_monitors)}')
    # get monitor targets
    monitors = {}
    name2node = None
    for _key, _mon in _monitors.items():
      if isinstance(_mon, str):
        if name2node is None:
          name2node = {node.name: node for node in list(self.target.nodes(level=-1).unique().values())}

        key, index = _mon[0], _mon[1]
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
      else:
        monitors[_key] = _mon
    return monitors

  def __del__(self):
    if hasattr(self, 'mon'):
      for key in tuple(self.mon.keys()):
        del self.mon[key]
    for key in tuple(self.__dict__.keys()):
      del self.__dict__[key]
    # gc.collect()
