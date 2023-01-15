# -*- coding: utf-8 -*-


from typing import Union, Dict, Sequence

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

  def update(self, *args, **kwargs):
    super().update(*args, **kwargs)
    return self

  def __add__(self, other):
    """Merging two dicts.

    Parameters
    ----------
    other: dict
      The other dict instance.

    Returns
    -------
    gather: Collector
      The new collector.
    """
    gather = type(self)(self)
    gather.update(other)
    return gather

  def __sub__(self, other: Union[Dict, Sequence]):
    """Remove other item in the collector.

    Parameters
    ----------
    other: dict, sequence
      The items to remove.

    Returns
    -------
    gather: Collector
      The new collector.
    """
    if not isinstance(other, (dict, tuple, list)):
      raise ValueError(f'Only support dict/tuple/list, but we got {type(other)}.')
    gather = type(self)(self)
    if isinstance(other, dict):
      for key, val in other.items():
        if key in gather:
          if id(val) != id(gather[key]):
            raise ValueError(f'Cannot remove {key}, because we got two different values: '
                             f'{val} != {gather[key]}')
          gather.pop(key)
        else:
          raise ValueError(f'Cannot remove {key}, because we do not find it '
                           f'in {self.keys()}.')
    elif isinstance(other, (list, tuple)):
      id_to_keys = {}
      for k, v in self.items():
        id_ = id(v)
        if id_ not in id_to_keys:
          id_to_keys[id_] = []
        id_to_keys[id_].append(k)

      keys_to_remove = []
      for key in other:
        if isinstance(key, str):
          keys_to_remove.append(key)
        else:
          keys_to_remove.extend(id_to_keys[id(key)])

      for key in set(keys_to_remove):
        if key in gather:
          gather.pop(key)
        else:
          raise ValueError(f'Cannot remove {key}, because we do not find it '
                           f'in {self.keys()}.')
    else:
      raise KeyError(f'Unknown type of "other". Only support dict/tuple/list, but we got {type(other)}')
    return gather

  def subset(self, var_type):
    """Get the subset of the (key, value) pair.

    ``subset()`` can be used to get a subset of some class:

    >>> import brainpy as bp
    >>>
    >>> some_collector = Collector()
    >>>
    >>> # get all trainable variables
    >>> some_collector.subset(bp.math.TrainVar)
    >>>
    >>> # get all Variable
    >>> some_collector.subset(bp.math.Variable)

    or, it can be used to get a subset of integrators:

    >>> # get all ODE integrators
    >>> some_collector.subset(bp.ode.ODEIntegrator)

    Parameters
    ----------
    var_type : type
      The type/class to match.
    """
    gather = type(self)()
    for key, value in self.items():
      if isinstance(value, var_type):
        gather[key] = value
    return gather

  def unique(self):
    """Get a new type of collector with unique values.

    If one value is assigned to two or more keys,
    then only one pair of (key, value) will be returned.
    """
    gather = type(self)()
    seen = set()
    for k, v in self.items():
      if id(v) not in seen:
        seen.add(id(v))
        gather[k] = v
    return gather


register_pytree_node(
  DotDict,
  lambda x: (x.values(), x.keys()),
  lambda keys, values: DotDict(safe_zip(keys, values))
)
