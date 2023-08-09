from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy._src.dyn.base import IonChaDyn
from brainpy._src.initialize import Initializer, parameter
from brainpy.types import Shape, ArrayType
from .base import Ion

__all__ = [
  'Sodium',
  'SodiumFixed',
]


class Sodium(Ion):
  pass


class SodiumFixed(Sodium):
  """Fixed Sodium dynamics.

  This calcium model has no dynamics. It holds fixed reversal
  potential :math:`E` and concentration :math:`C`.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, ArrayType, Initializer, Callable] = 50.,
      C: Union[float, ArrayType, Initializer, Callable] = 0.0400811,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      **channels
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     method=method,
                     name=name,
                     mode=mode,
                     **channels)
    self.E = parameter(E, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)

  def reset_state(self, V, C=None, E=None, batch_size=None):
    C = self.C if C is None else C
    E = self.E if E is None else E
    nodes = self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values()
    self.check_hierarchies(type(self), *tuple(nodes))
    for node in nodes:
      node.reset_state(V, C, E, batch_size)
