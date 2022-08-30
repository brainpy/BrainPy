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

  In general, all attributes will be included ad keys in the dict.
  For example, if you add an attribute to specify what variable names
  you have:

  >>> d.names = ('a', 'b')

  This attribute `names` will cause error when you treat the object as
  a PyTree.

  >>> from jax import jit
  >>> f = jit(lambda x: x)
  >>> f(d)
  TypeError: Argument 'a' of type <class 'str'> is not a valid JAX type.

  At this moment, you can label this attribute `names` as not a key in the dictionary
  by using the syntax::

  >>> d.add_attr_not_key('names')
  >>> f(d)
  {'a': DeviceArray(10, dtype=int32, weak_type=True),
   'b': DeviceArray(20, dtype=int32, weak_type=True),
   'c': DeviceArray(30, dtype=int32, weak_type=True)}

  """

  '''Used to exclude variables that '''
  attrs_not_keys = ('attrs_not_keys', 'var_names')

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self
    self.var_names = ()

  def keys(self):
    """Retrieve all keys in the dict, excluding ignored keys."""
    keys = []
    for k in super(DotDict, self).keys():
      if k not in self.attrs_not_keys:
        keys.append(k)
    return tuple(keys)

  def values(self):
    """Retrieve all values in the dict, excluding values of ignored keys."""
    values = []
    for k, v in super(DotDict, self).items():
      if k not in self.attrs_not_keys:
        values.append(v)
    return tuple(values)

  def items(self):
    """Retrieve all items in the dict, excluding ignored items."""
    items = []
    for k, v in super(DotDict, self).items():
      if k not in self.attrs_not_keys:
        items.append((k, v))
    return items

  def to_numpy(self):
    """Change all values to numpy arrays."""
    for key in tuple(self.keys()):
      self[key] = np.asarray(self[key])

  def add_attr_not_key(self, *args):
    """Add excluded attribute when retrieving dictionary keys. """
    for arg in args:
      if not isinstance(arg, str):
        raise TypeError('Only support string.')
    self.attrs_not_keys += args


register_pytree_node(
  DotDict,
  lambda x: (x.values(), x.keys()),
  lambda keys, values: DotDict(safe_zip(keys, values))
)
