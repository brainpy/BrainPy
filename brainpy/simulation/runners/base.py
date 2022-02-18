# -*- coding: utf-8 -*-

__all__ = [
  'Runner'
]


class Runner(object):
  """Basic Runner Class."""

  def __call__(self, *args, **kwargs):
    raise NotImplementedError

