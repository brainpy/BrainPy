# -*- coding: utf-8 -*-


__all__ = [
  'transform_brainpy_array'
]


def transform_brainpy_array(array):
  if hasattr(array, 'is_brainpy_array'):
    if array.is_brainpy_array:
      return array.value
  return array
