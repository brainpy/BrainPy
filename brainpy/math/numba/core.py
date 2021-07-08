# -*- coding: utf-8 -*-


from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'jit',
  'vectorize',
  'parallel',
]


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



def vectorize(f):
  return f

def parallel(f):
  return f

