# -*- coding: utf-8 -*-

from contextlib import contextmanager

from brainpy import errors
from brainpy import math

__all__ = [
  'Collector',
  'ArrayCollector',
]


class Collector(dict):
  """A Collector is a dictionary (name, var)
  with some additional methods to make manipulation
  of collections of variables easy. A Collection
  is ordered by insertion order. It is the object
  returned by DynamicSystem.vars() and used as input
  in many DynamicSystem instance: optimizers, Jit, etc..."""

  def __add__(self, other):
    """Overloaded add operator to merge two VarCollectors together."""
    gather = type(self)(self)
    gather.update(other)
    return gather

  def subset(self, type_):
    """Get the subset of the (key, value) pair.

    Parameters
    ----------
    type_ : Any
      The type/class to match.
    """
    gather = type(self)()
    if type(type_) == type:
      for key, value in self.items():
        if isinstance(value, type_):
          gather[key] = value
    elif type(type_) == str:
      for key, value in self.items():
        if value.__name__.startswith(type_):
          gather[key] = value
    else:
      raise errors.UnsupportedError(f'BrainPy do not support subset {type(type_)}. '
                                    f'You should provide a class name, or a str.')
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

  def dict(self):
    gather = dict()
    for k, v in self.items():
      gather[k] = v.value
    return gather


class ArrayCollector(Collector):
  """A ArrayCollector is a dictionary (name, var)
  with some additional methods to make manipulation
  of collections of variables easy. A Collection
  is ordered by insertion order. It is the object
  returned by DynamicSystem.vars() and used as input
  in many DynamicSystem instance: optimizers, Jit, etc..."""

  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""
    if key in self:
      raise ValueError(f'Name "{key}" conflicts when appending to Collection')
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

  def data(self):
    """Get all data in each value."""
    return [x.value for x in self.values()]

  @contextmanager
  def replicate(self):
    """A context manager to use in a with statement that replicates
    the variables in this collection to multiple devices. This is
    used typically prior to call to objax.Parallel, so that all
    variables have a copy on each device.

    Important: replicating also updates the random state in order
    to have a new one per device.
    """
    try:
      import jax
      import jax.numpy as jnp
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError('"Collection.replicate()" is only available in '
                                'JAX backend, while JAX is not installed.') from e

    replicated, saved_states = [], []
    x = jnp.zeros((jax.local_device_count(), 1), dtype=math.float_)
    sharded_x = jax.pmap(lambda x: x, axis_name='device')(x)
    devices = [b.device() for b in sharded_x.device_buffers]
    ndevices = len(devices)
    for d in self.values():
      if isinstance(d, RandomState):
        replicated.append(jax.api.device_put_sharded([shard for shard in d.split(ndevices)], devices))
        saved_states.append(d.value)
      else:
        replicated.append(jax.api.device_put_replicated(d.value, devices))
    self.assign(replicated)
    yield
    visited = set()
    saved_states.reverse()
    for k, d in self.items():
      if id(d) not in visited:  # Careful not to reduce twice in case of a variable and a reference to it.
        if isinstance(d, RandomState):
          d.assign(saved_states.pop())
        else:
          d.reduce(d.value)
        visited.add(id(d))
