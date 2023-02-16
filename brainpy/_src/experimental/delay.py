# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional, Dict

import jax

from brainpy import math as bm
from brainpy._src.dyn.base import DynamicalSystem, not_pass_shargs
from brainpy._src.math.delayvars import DelayVariable, ROTATE_UPDATE, CONCAT_UPDATE


class Delay(DynamicalSystem, DelayVariable):
  """Delay for dynamical systems which has a fixed delay length.

  Detailed docstring please see :py:class:`~.DelayVariable`.
  """

  def __init__(
      self,
      target: bm.Variable,
      length: int = 0,
      before_t0: Union[float, int, bool, bm.Array, jax.Array, Callable] = None,
      entries: Optional[Dict] = None,
      method: str = ROTATE_UPDATE,
      mode: bm.Mode = None,
      name: str = None,
  ):
    DynamicalSystem.__init__(self, mode=mode)
    if method is None:
      if self.mode.is_a(bm.NonBatchingMode):
        method = ROTATE_UPDATE
      elif self.mode.is_parent_of(bm.TrainingMode):
        method = CONCAT_UPDATE
      else:
        method = ROTATE_UPDATE
    DelayVariable.__init__(self,
                           target=target,
                           length=length,
                           before_t0=before_t0,
                           entries=entries,
                           method=method,
                           name=name)

  @not_pass_shargs
  def update(self, *args, **kwargs):
    return DelayVariable.update(self, *args, **kwargs)

