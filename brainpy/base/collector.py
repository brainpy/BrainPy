# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp
from contextlib import contextmanager

math = None

__all__ = [
  'Collector',
  'TensorCollector',
]


class Collector(dict):
  """A Collector is a dictionary (name, var) with some additional methods to make manipulation
  of collections of variables easy. A Collector is ordered by insertion order. It is the object
  returned by Base.vars() and used as input in many Collector instance: optimizers, jit, etc..."""

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
    # dict.__setitem__(self, key, new_value)

  def update(self, other, **kwargs):
    assert isinstance(other, dict)
    for key, value in other.items():
      self[key] = value
    for key, value in kwargs.items():
      self[key] = value

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

  def __sub__(self, other):
    """Remove other item in the collector.

    Parameters
    ----------
    other: dict
      The items to remove.

    Returns
    -------
    gather: Collector
      The new collector.
    """
    if not isinstance(other, dict):
      raise ValueError(f'Only support dict, but we got {type(other)}.')
    gather = type(self)()
    for key, val in self.items():
      if key in other:
        if id(val) != id(other[key]):
          raise ValueError(f'Cannot remove {key}, because we got two different values: '
                           f'{val} != {other[key]}')
      else:
        gather[key] = val
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
    >>> # get all JaxArray
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


class TensorCollector(Collector):
  """A ArrayCollector is a dictionary (name, var)
  with some additional methods to make manipulation
  of collections of variables easy. A Collection
  is ordered by insertion order. It is the object
  returned by DynamicalSystem.vars() and used as input
  in many DynamicalSystem instance: optimizers, Jit, etc..."""

  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""
    global math
    if math is None: from brainpy import math

    assert isinstance(value, math.ndarray)
    if key in self:
      if id(self[key]) != id(value):
        raise ValueError(f'Name "{key}" conflicts: same name for {value} and {self[key]}.')
    dict.__setitem__(self, key, value)

  def assign(self, inputs):
    """Assign data to all values.

    Parameters
    ----------
    inputs : dict
      The data for each value in this collector.
    """
    if len(self) != len(inputs):
      raise ValueError(f'The target has {len(inputs)} data, while we got '
                       f'an input value with the length of {len(inputs)}.')
    for key, v in self.items():
      v.value = inputs[key]

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
