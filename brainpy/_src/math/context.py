"""
Context for brainpy computation.

This context defines all shared data used in all modules in a computation.
"""

from typing import Dict, Any

from brainpy._src.tools.dicts import DotDict
from .delayvars import DelayVariable
from .object_transform.base import BrainPyObject
from .environment import get_dt as _get_dt_

__all__ = [
  'share',
]


class DelayEntry:
  def __init__(self, target: str, time=None, step=None):
    if time is None and step is None:
      raise ValueError('Please provide time or step.')
    self.target = target
    self.time = time
    self.step = step


class _ShareContext(BrainPyObject):
  def __init__(self):
    super().__init__()

    # Shared data across all nodes at current time step.
    # -------------

    self._arguments = DotDict()
    self._delays: Dict[str, DelayVariable] = DotDict()
    self._delay_entries: Dict[str, str] = DotDict()
    self._identifiers = set()

  @property
  def dt(self):
    if 'dt' in self._arguments:
      return self._arguments['dt']
    else:
      return _get_dt_()

  def load(self, key):
    """Get the shared data by the ``key``.

    Args:
      key (str): the key to indicate the data.
    """
    if key in self._arguments:
      return self._arguments[key]
    if key in self._delays:
      return self._delays[key]
    if key in self._delay_entries:
      entry = key
      delay = self._delay_entries[entry]
      return self._delays[delay].at(entry)
    raise KeyError(f'Cannot found shared data of {key}.')

  def save(self, identifier: str, data: Any) -> None:
    """Save shared arguments in the global context."""
    assert isinstance(identifier, str)

    if identifier in self._identifiers:
      raise ValueError(f'{identifier} has been used. Please assign another name.')
    if isinstance(data, DelayVariable):
      self._delays[identifier] = data
    elif isinstance(data, DelayEntry):
      if isinstance(data.target, DelayVariable):
        delay_key = f'delay{id(data)}'
        self.save(delay_key, data.target)
        delay = data.target
      elif isinstance(data.target, str):
        if data.target not in self._delays:
          raise ValueError(f'Delay target {data.target} has not been registered.')
        delay = self._delays[data.target]
        delay_key = data.target
      else:
        raise ValueError(f'Unknown delay target. {type(data.target)}')
      delay.register_entry(identifier, delay_time=data.time, delay_step=data.step)
      self._delay_entries[identifier] = delay_key
    else:
      self._arguments[identifier] = data
    self._identifiers.add(identifier)

  def get_shargs(self) -> DotDict:
    """Get all shared arguments in the global context."""
    return self._arguments.copy()

  def remove_shargs(self, *args) -> None:
    """Clear all shared arguments in the global context."""
    if len(args) > 0:
      for a in args:
        self._arguments.pop(a)
    else:
      self._arguments.clear()

  def clear(self) -> None:
    """Clear all shared data in this computation context."""
    self._arguments.clear()
    self._delays.clear()
    self._delay_entries.clear()
    self._identifiers.clear()

  def update(self):
    for delay in self._delays.values():
      delay.update()

  def reset_state(self, batch_axis: int = None):
    for delay in self._delays.values():
      delay.reset_state(batch_axis)


share = _ShareContext()
