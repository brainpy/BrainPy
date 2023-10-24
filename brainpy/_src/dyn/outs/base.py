from typing import Optional

import brainpy.math as bm
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.mixin import ParamDesc, BindCondData

__all__ = [
  'SynOut'
]


class SynOut(DynamicalSystem, ParamDesc, BindCondData):
  """Base class for synaptic outputs.

  :py:class:`~.SynOut` is also subclass of :py:class:`~.ParamDesc` and :pu:class:`~.BindCondData`.
  """
  def __init__(self,
               name: Optional[str] = None,
               scaling: Optional[bm.Scaling] = None):
    super().__init__(name=name)
    self._conductance = None
    if scaling is None:
      self.scaling = bm.get_membrane_scaling()
    else:
      self.scaling = scaling

  def __call__(self, *args, **kwargs):
    if self._conductance is None:
      raise ValueError(f'Please first pack conductance data at the current step using '
                       f'".{BindCondData.bind_cond.__name__}(data)". {self}')
    ret = self.update(self._conductance, *args, **kwargs)
    return ret

  def reset_state(self, *args, **kwargs):
    pass

  def offset_scaling(self, x, bias=None, scale=None):
    s = self.scaling.offset_scaling(x, bias=bias, scale=scale)
    if isinstance(x, bm.Array):
      x.value = s
      return x
    return s

  def std_scaling(self, x, scale=None):
    s = self.scaling.std_scaling(x, scale=scale)
    if isinstance(x, bm.Array):
      x.value = s
      return x
    return s

  def inv_scaling(self, x, scale=None):
    s = self.scaling.inv_scaling(x, scale=scale)
    if isinstance(x, bm.Array):
      x.value = s
      return x
    return s
