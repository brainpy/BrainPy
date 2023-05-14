
from typing import Callable
import brainpy.math as bm
from .base import Initializer


class Clip(Initializer):
  def __init__(self, init: Callable, min=None, max=None):
    self.min = min
    self.max = max
    self.init = init

  def __call__(self, shape, dtype=None):
    x = self.init(shape, dtype)
    if self.min is not None:
      x = bm.maximum(self.min, x)
    if self.max is not None:
      x = bm.minimum(self.max, x)
    return x

  def __repr__(self):
    return f'{self.__class__.__name__}({self.init}, min={self.min}, max={self.max})'




