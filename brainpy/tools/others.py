# -*- coding: utf-8 -*-

__all__ = [
  'size2num'
]


def size2num(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError
