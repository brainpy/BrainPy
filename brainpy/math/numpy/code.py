# -*- coding: utf-8 -*-


__all__ = [
  'control_transform',
]


def control_transform(f=None, show_code=False):
  if f is None:
    return lambda func: func
  else:
    return f



