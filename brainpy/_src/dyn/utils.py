from typing import Optional, Union
import brainpy.math as bm

__all__ = [
  'get_spk_type',
]


def get_spk_type(spk_type: Optional[type] = None, mode: Optional[bm.Mode] = None):
  if mode is None:
    return bm.bool
  elif isinstance(mode, bm.TrainingMode):
    return bm.float_ if (spk_type is None) else spk_type
  else:
    assert isinstance(mode, bm.Mode)
    return bm.bool if (spk_type is None) else spk_type
