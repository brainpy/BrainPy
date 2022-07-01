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

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self

  def to_numpy(self):
    for key in tuple(self.keys()):
      self[key] = np.asarray(self[key])


register_pytree_node(
  DotDict,
  lambda x: (tuple(x.values()), tuple(x.keys())),
  lambda keys, values: DotDict(safe_zip(keys, values))
)
