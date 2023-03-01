"""
Context for brainpy computation.

This context defines all shared data used in all modules in a computation.
"""

from typing import Dict, Any, Union

from brainpy._src.tools.dicts import DotDict
from brainpy._src.dyn.delay import Delay
from brainpy._src.math.environment import get_dt
from brainpy._src.math.object_transform.base import BrainPyObject, dyn_dict

__all__ = [
  'share',
]


class _ShareContext(BrainPyObject):
  def __init__(self):
    super().__init__()

    # Shared data across all nodes at current time step.
    # -------------

    self._arguments = DotDict()
    self._delays: Dict[str, Delay] = dyn_dict()

  @property
  def dt(self):
    if 'dt' in self._arguments:
      return self._arguments['dt']
    else:
      return get_dt()

  @dt.setter
  def dt(self, dt):
    self.set_dt(dt)

  def set_dt(self, dt: Union[int, float]):
    self._arguments['dt'] = dt

  def load(self, key, value: Any = None):
    """Get the shared data by the ``key``.

    Args:
      key (str): the key to indicate the data.
      value (Any): the default value when ``key`` is not defined in the shared.
    """
    if key == 'dt':
      return self.dt
    if key in self._arguments:
      return self._arguments[key]
    if key in self._delays:
      return self._delays[key]
    if value is None:
      raise KeyError(f'Cannot found shared data of {key}.')
    else:
      return value

  def save(self, *args, **kwargs) -> None:
    """Save shared arguments in the global context."""
    assert len(args) % 2 == 0
    for i in range(0, len(args), 2):
      identifier = args[i * 2]
      data = args[i * 2 + 1]
      if isinstance(data, Delay):
        if identifier in self._delays:
          raise ValueError(f'{identifier} has been used. Please assign another name.')
        self._delays[identifier] = data
      else:
        self._arguments[identifier] = data
    for identifier, data in kwargs.items():
      if isinstance(data, Delay):
        if identifier in self._delays:
          raise ValueError(f'{identifier} has been used. Please assign another name.')
        self._delays[identifier] = data
      else:
        self._arguments[identifier] = data

  def get_shargs(self) -> DotDict:
    """Get all shared arguments in the global context."""
    return self._arguments.copy()

  def clear_delays(self, *delays) -> None:
    """Clear all delay variables in this global context."""
    if len(delays):
      for d in delays:
        self._delays.pop(d)
    else:
      self._delays.clear()

  def clear_shargs(self, *args) -> None:
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

  def __call__(self, *args, **kwargs):
    return self.update(*args, **kwargs)

  def update(self, *args, **kwargs):
    for delay in self._delays.values():
      delay.update()

  def reset(self, batch_size: int = None):
    self.reset_state(batch_size=batch_size)

  def reset_state(self, batch_size: int = None):
    for delay in self._delays.values():
      delay.reset_state(batch_size)


share = _ShareContext()
