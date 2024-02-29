# -*- coding: utf-8 -*-

import warnings

from brainpy import errors

__all__ = [
  'clear_name_cache',
]


_name2id = dict()
_typed_names = {}


def check_name_uniqueness(name, obj):
  """Check the uniqueness of the name for the object type."""
  if not name.isidentifier():
    raise errors.BrainPyError(f'"{name}" isn\'t a valid identifier '
                              f'according to Python language definition. '
                              f'Please choose another name.')
  if name in _name2id:
    if _name2id[name] != id(obj):
      raise errors.UniqueNameError(
        f'In BrainPy, each object should have a unique name. '
        f'However, we detect that {obj} has a used name "{name}". \n'
        f'If you try to run multiple trials, you may need \n\n'
        f'>>> brainpy.brainpy_object.clear_name_cache() \n\n'
        f'to clear all cached names. '
      )
  else:
    _name2id[name] = id(obj)


def get_unique_name(type_: str):
  """Get the unique name for the given object type."""
  if type_ not in _typed_names:
    _typed_names[type_] = 0
  name = f'{type_}{_typed_names[type_]}'
  _typed_names[type_] += 1
  return name


def clear_name_cache(ignore_warn=False):
  """Clear the cached names."""
  _name2id.clear()
  _typed_names.clear()
  if not ignore_warn:
    warnings.warn(f'All named models and their ids are cleared.', UserWarning)


_fun2stack = dict()


def cache_stack(func, stack):
  _fun2stack[func] = stack


def clear_stack_cache():
  for k in tuple(_fun2stack.keys()):
    del _fun2stack[k]


def get_stack_cache(func):
  if func in _fun2stack:
    return _fun2stack[func]
  else:
    return None

