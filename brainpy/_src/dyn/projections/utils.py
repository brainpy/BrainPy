from brainpy import math as bm
from brainpy._src.mixin import ReturnInfo


def _get_return(return_info):
  if isinstance(return_info, bm.Variable):
    return return_info.value
  elif isinstance(return_info, ReturnInfo):
    return return_info.get_data()
  else:
    raise NotImplementedError

