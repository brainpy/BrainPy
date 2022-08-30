# -*- coding: utf-8 -*-

import numpy as np
from jax.tree_util import register_pytree_node
from jax.util import safe_zip

__all__ = [
  'DotDict',
]


class DotDict(dict):
  """Python dictionaries with advanced dot notation access.

  For example:

  >>> d = DotDict({'a': 10, 'b': 20})
  >>> d.a
  10
  >>> d['a']
  10
  >>> d.c  # this will raise a KeyError
  KeyError: 'c'
  >>> d.c = 30  # but you can assign a value to a non-existing item
  >>> d.c
  30
  """

  '''Used to exclude variables that '''
  excluded_vars = ('excluded_vars', 'var_names')

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self
    self.var_names = ()

  def keys(self):
    """Retrieve all keys in the dict, excluding ignored keys."""
    keys = []
    for k in super(DotDict, self).keys():
      if k not in self.excluded_vars:
        keys.append(k)
    return tuple(keys)

  def values(self):
    """Retrieve all values in the dict, excluding values of ignored keys."""
    values = []
    for k, v in super(DotDict, self).items():
      if k not in self.excluded_vars:
        values.append(v)
    return tuple(values)

  def items(self):
    """Retrieve all items in the dict, excluding ignored items."""
    items = []
    for k, v in super(DotDict, self).items():
      if k not in self.excluded_vars:
        items.append((k, v))
    return items

  def to_numpy(self):
    """Change all values to numpy arrays."""
    for key in tuple(self.keys()):
      self[key] = np.asarray(self[key])

  def add_excluded_var(self, *args):
    """Add excluded variable names. """
    for arg in args:
      if not isinstance(arg, str):
        raise TypeError('Only support string.')
    self.excluded_vars += args


def flatten_func(x: DotDict):
  keys = []
  values = []
  for k, v in x.items():
    if k not in x.excluded_vars:
      values.append(v)
      keys.append(k)
  return tuple(values), tuple(keys)


register_pytree_node(
  DotDict,
  flatten_func,
  lambda keys, values: DotDict(safe_zip(keys, values))
)
