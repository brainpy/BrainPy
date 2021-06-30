# -*- coding: utf-8 -*-

from brainpy import errors


__all__ = [
  'add'
]


object2name = dict()


def add(name, obj):
  if name in object2name:
    if id(object2name[name]) != id(obj):
      raise errors.UniqueNameError(name, object2name[name], obj)


