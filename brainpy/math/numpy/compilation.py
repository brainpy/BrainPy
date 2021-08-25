# -*- coding: utf-8 -*-


from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'jit',
  'vmap',
  'pmap',
]


def jit(ds_or_func, *args, **kwargs):
  print('JIT compilation in numpy backend '
        'can not be available right now.')
  return ds_or_func


def vmap(f):
  raise NotImplementedError('Vectorize compilation in numpy backend '
                            'can not be available right now.')


def pmap(f):
  raise NotImplementedError('Parallel compilation in numpy backend '
                            'can not be available right now.')
