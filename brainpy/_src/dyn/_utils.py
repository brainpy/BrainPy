# -*- coding: utf-8 -*-

from typing import Optional

from brainpy._src.math.ndarray import Variable
from brainpy._src.math.object_transform.base import BrainPyObject

__all__ = [
  'get_output_var',
]


def get_output_var(
    out_var: Optional[str],
    target: BrainPyObject
) -> Optional[Variable]:
  if out_var is not None:
    assert isinstance(out_var, str)
    if not hasattr(target, out_var):
      raise ValueError(f'{target} does not has variable {out_var}')
    out_var = getattr(target, out_var)
    if not isinstance(out_var, Variable):
      raise ValueError(f'{target} does not has variable {out_var}')
  return out_var
