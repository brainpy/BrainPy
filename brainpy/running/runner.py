# -*- coding: utf-8 -*-

import types
from typing import Callable, Dict, Sequence, Union

from brainpy.base import Base
from brainpy.base.collector import TensorCollector
from brainpy.errors import MonitorError, RunningError
from brainpy.tools.checking import check_dict_data
from brainpy import math as bm
from .monitor import Monitor

__all__ = [
  'Runner',
]


class Runner(object):
  """Base Runner.

  Parameters
  ----------
  target: Any
    The target model.
  monitors: None, list of str, tuple of str, Monitor
    Variables to monitor.
  jit: bool, dict
  progress_bar: bool
  dyn_vars: Optional, dict
  numpy_mon_after_run : bool
  """

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
      self.jit = {'predict': jit}
    elif isinstance(jit, dict):
      for k, v in jit.items():
        self.jit[k] = v
      self.jit = {'predict': jit.pop('predict', True)}
    else:
      raise ValueError(f'Unknown "jit" setting: {jit}')

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
    monitors = check_and_format_monitors(host=self.target, mon=self.mon)
    return_with_idx = dict()
    return_without_idx = dict()
    for key, target, variable, idx, interval in monitors:
      if interval is not None:
        raise ValueError(f'Running with "{self.__class__.__name__}" does '
                         f'not support "interval" in the monitor.')
      data = target
      for k in variable.split('.'):
        data = getattr(data, k)
      if not isinstance(data, bm.Variable):
        raise RunningError(f'"{key}" in {target} is not a dynamically changed Variable, '
                           f'its value will not change, we think there is no need to '
                           f'monitor its trajectory.')
      if idx is None:
        return_without_idx[key] = data
      else:
        return_with_idx[key] = (data, bm.asarray(idx))

    return return_without_idx, return_with_idx

  def build_monitors(self, return_without_idx, return_with_idx) -> Callable:
    raise NotImplementedError


def check_and_format_monitors(host, mon):
  """Return a formatted monitor items:

  >>> [(node, key, target, variable, idx, interval),
  >>>  ...... ]

  """
  assert isinstance(host, Base)
  assert isinstance(mon, Monitor)

  formatted_mon_items = []

  # master node:
  #    Check whether the input target node is accessible,
  #    and check whether the target node has the attribute
  name2node = {node.name: node for node in list(host.nodes().unique().values())}
  for key, idx, interval in zip(mon.item_names, mon.item_indices, mon.item_intervals):
    # target and variable
    splits = key.split('.')
    if len(splits) == 1:
      if not hasattr(host, splits[0]):
        raise RunningError(f'{host} does not has variable {key}.')
      target = host
      variable = splits[-1]
    else:
      if not hasattr(host, splits[0]):
        if splits[0] not in name2node:
          raise RunningError(f'Cannot find target {key} in monitor of {host}, please check.')
        else:
          target = name2node[splits[0]]
          assert len(splits) == 2
          variable = splits[-1]
      else:
        target = host
        for s in splits[:-1]:
          try:
            target = getattr(target, s)
          except KeyError:
            raise RunningError(f'Cannot find {key} in {host}, please check.')
        variable = splits[-1]

    # idx
    if isinstance(idx, int): idx = bm.array([idx])

    # interval
    if interval is not None:
      if not isinstance(interval, float):
        raise RunningError(f'"interval" must be a float (denotes time), but we got {interval}')

    # append
    formatted_mon_items.append((key, target, variable, idx, interval,))

  return formatted_mon_items
