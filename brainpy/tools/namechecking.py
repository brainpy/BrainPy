# -*- coding: utf-8 -*-

from brainpy import errors

__all__ = [
  'check_name',
  'get_name',
]

_name2id = dict()
_typed_names = {}


def check_name(name, obj):
  if not name.isidentifier():
    raise errors.BrainPyError(f'"{name}" isn\'t a valid identifier '
                              f'according to Python language definition. '
                              f'Please choose another name.')
  if name in _name2id:
    if _name2id[name] != id(obj):
      raise errors.UniqueNameError(f'In BrainPy, each object should have a unique name. '
                                   f'However, we detect that {obj} has a used name "{name}".')
  else:
    _name2id[name] = id(obj)


def get_name(type):
  if type not in _typed_names:
    _typed_names[type] = 0
  name = f'{type}{_typed_names[type]}'
  _typed_names[type] += 1
  return name
