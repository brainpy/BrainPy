from typing import Sequence, Dict, Union

from jax.tree_util import register_pytree_node
from jax.util import safe_zip

from .variables import Variable

__all__ = [
  'Collector', 'ArrayCollector', 'TensorCollector',
]


class Collector(dict):
  """A Collector is a dictionary (name, var) with some additional methods to make manipulation
  of collections of variables easy. A Collector is ordered by insertion order. It is the object
  returned by BrainPyObject.vars() and used as input in many Collector instance: optimizers, jit, etc..."""

  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""
    if key in self:
      if id(self[key]) != id(value):
        raise ValueError(f'Name "{key}" conflicts: same name for {value} and {self[key]}.')
    dict.__setitem__(self, key, value)

  def replace(self, key, new_value):
    """Replace the original key with the new value."""
    self.pop(key)
    self[key] = new_value

  def update(self, other, **kwargs):
    assert isinstance(other, (dict, list, tuple))
    if isinstance(other, dict):
      for key, value in other.items():
        self[key] = value
    elif isinstance(other, (tuple, list)):
      num = len(self)
      for i, value in enumerate(other):
        self[f'_var{i + num}'] = value
    else:
      raise ValueError(f'Only supports dict/list/tuple, but we got {type(other)}')
    for key, value in kwargs.items():
      self[key] = value
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

  def not_subset(self, var_type):
    gather = type(self)()
    for key, value in self.items():
      if not isinstance(value, var_type):
        gather[key] = value
    return gather

  def include(self, *types):
    gather = type(self)()
    for key, value in self.items():
      if value.__class__ in types:
        gather[key] = value
    return gather

  def exclude(self, *types):
    gather = type(self)()
    for key, value in self.items():
      if value.__class__ not in types:
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


class ArrayCollector(Collector):
  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""

    assert isinstance(value, Variable), type(value)
    if key in self:
      if id(self[key]) != id(value):
        raise ValueError(f'Name "{key}" conflicts: same name for {value} and {self[key]}.')
    dict.__setitem__(self, key, value)

  def dict(self):
    """Get a dict with the key and the value data.
    """
    gather = dict()
    for k, v in self.items():
      gather[k] = v.value
    return gather

  def data(self):
    """Get all data in each value."""
    return [x.value for x in self.values()]


TensorCollector = ArrayCollector

register_pytree_node(
  ArrayCollector,
  lambda x: (x.values(), x.keys()),
  lambda keys, values: ArrayCollector(safe_zip(keys, values))
)
