# -*- coding: utf-8 -*-


from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'jit',
  'Vectorize',
  'Parallel',
]

_NUMPY_FUNC_NO = 0



def jit(ds_or_func, *args, **kwargs):
  if isinstance(ds_or_func, DynamicSystem):
    return ds_or_func

  elif callable(ds_or_func):
    return ds_or_func

  else:
    raise errors.ModelUseError(f'Only support instance of '
                               f'{DynamicSystem.__name__}, '
                               f'or a callable function, '
                               f'but we got {type(ds_or_func)}.')



class Vectorize(DynamicSystem):
  target_backend = 'numpy'


class Parallel(DynamicSystem):
  target_backend = 'numpy'
