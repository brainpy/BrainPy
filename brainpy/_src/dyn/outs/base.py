from typing import Optional

from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.mixin import ParamDesc, BindCondData

__all__ = [
  'SynOut'
]


class SynOut(DynamicalSystem, ParamDesc, BindCondData):
  """Base class for synaptic outputs.

  :py:class:`~.SynOut` is also subclass of :py:class:`~.ParamDesc` and :pu:class:`~.BindCondData`.
  """
  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

  def __call__(self, *args, **kwargs):
    if self._conductance is None:
      raise ValueError(f'Please first pack conductance data at the current step using '
                       f'".{BindCondData.bind_cond.__name__}(data)". {self}')
    ret = self.update(self._conductance, *args, **kwargs)
    return ret
