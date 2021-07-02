# -*- coding: utf-8 -*-


from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'jit',
  'Vectorize',
  'Parallel',
]

_NUMBA_FUNC_NO = 0


def jit(ds_or_func, *args, **kwargs):
  if isinstance(ds_or_func, DynamicSystem):
    pass

  elif callable(ds_or_func):
    pass

  else:
    raise errors.ModelUseError(f'Only support instance of '
                               f'{DynamicSystem.__name__}, '
                               f'or a callable function, '
                               f'but we got {type(ds_or_func)}.')



class Parallel(DynamicSystem):
  target_backend = 'numba'


class Vectorize(DynamicSystem):
  target_backend = 'numba'
