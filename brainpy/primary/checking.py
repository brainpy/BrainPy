# -*- coding: utf-8 -*-

from brainpy import errors

__all__ = [
  'check_name',
  'get_name',
]

object2name = dict()
_typed_names = {}


def check_name(name, obj):
  if not name.isidentifier():
    raise errors.ModelUseError(f'"{name}" isn\'t a valid identifier '
                               f'according to Python language definition. '
                               f'Please choose another name.')
  if name in object2name:
    if id(object2name[name]) != id(obj):
      raise errors.UniqueNameError(name, object2name[name], obj)


def get_name(type):
  if type not in _typed_names:
    _typed_names[type] = 0
  name = f'{type}{_typed_names[type]}'
  _typed_names[type] += 1
  return name
