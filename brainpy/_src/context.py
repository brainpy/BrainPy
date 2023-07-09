"""
Context for brainpy computation.

This context defines all shared data used in all modules in a computation.
"""

from typing import Any, Union

from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.math.environment import get_dt
from brainpy._src.tools.dicts import DotDict

__all__ = [
  'share',
]


class _ShareContext(DynamicalSystem):
  def __init__(self):
    super().__init__()

    # Shared data across all nodes at current time step.
    # -------------

    self._arguments = DotDict()

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
    if value is None:
      raise KeyError(f'Cannot found shared data of {key}. '
                     f'Please define it with "brainpy.share.save({key}=<?>)". ')
    else:
      return value

  def save(self, *args, **kwargs) -> None:
    """Save shared arguments in the global context."""
    assert len(args) % 2 == 0
    for i in range(0, len(args), 2):
      identifier = args[i * 2]
      data = args[i * 2 + 1]
      self._arguments[identifier] = data
    for identifier, data in kwargs.items():
      self._arguments[identifier] = data

  def __setitem__(self, key, value):
    """Enable setting the shared item by ``bp.share[key] = value``."""
    self.save(key, value)

  def __getitem__(self, item):
    """Enable loading the shared parameter by ``bp.share[key]``."""
    return self.load(item)

  def get_shargs(self) -> DotDict:
    """Get all shared arguments in the global context."""
    return self._arguments.copy()

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

  def __call__(self, *args, **kwargs):
    pass

  def update(self, *args, **kwargs):
    pass

  def reset(self, batch_size: int = None):
    pass

  def reset_state(self, batch_size: int = None):
    pass


share = _ShareContext()
