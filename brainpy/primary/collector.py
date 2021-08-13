# -*- coding: utf-8 -*-

from contextlib import contextmanager

from brainpy import math

__all__ = [
  'Collector',
  'ArrayCollector',
]


def _size2len(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    a = 1
    for b in size:
      a *= b
    return a
  else:
    raise ValueError


class Collector(dict):
  """A ArrayCollector is a dictionary (name, var)
  with some additional methods to make manipulation
  of collections of variables easy. A Collection
  is ordered by insertion order. It is the object
  returned by DynamicSystem.vars() and used as input
  in many DynamicSystem instance: optimizers, Jit, etc..."""

  def __add__(self, other):
    """Overloaded add operator to merge two VarCollectors together."""
    vc = type(self)(self)
    vc.update(other)
    return vc

  def subset(self, type_):
    seen = set()
    gather = type(self)()
    for key, value in self.items():
      if id(value) not in seen and isinstance(value, type_):
        seen.add(id(value))
        gather[key] = value
    return gather

  def unique_values(self):
    seen = set()
    values = []
    for v in self.values():
      if id(v) not in seen:
        seen.add(id(v))
        values.append(v)
    return values

  def pretty_print(self, max_width=100):
    """Pretty print the contents of the VarCollection."""
    text = []
    total = count = 0
    longest_string = max((len(x) for x in self.keys()), default=20)
    longest_string = min(max_width, max(longest_string, 20))
    for name, v in self.items():
      size = _size2len(v.value.shape)
      total += size
      count += 1
      text.append(f'{name:{longest_string}} {size:8d} {v.value.shape}')
    text.append(f'{f"+Total({count})":{longest_string}} {total:8d}')
    return '\n'.join(text)


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

  def update(self, other):
    """Overload dict.update method to catch potential conflicts during assignment."""
    if not isinstance(other, Collector):
      other = list(other)
    else:
      other = other.items()
    conflicts = set()
    for k, v in other:
      if k in self:  # if "key" and "value" are same, skip
        if self[k] is not v:
          conflicts.add(k)
      else:  # if do not have "key", update
        self[k] = v
    if conflicts:
      raise ValueError(f'Name conflicts when combining collection {sorted(conflicts)}')

  def unique_data(self):
    """Return the list of values for this collection.
    Similarly to the assign method, each variable value is
    reported only once and in the order following the
    iter(self) iterator.

    Returns:
        A new ArrayCollector containing the subset of variables.
    """
    return [x.value for x in self.unique_values()]

  def unique_assign(self, extra_data):
    """Assign tensors to the variables in the VarCollection.
    Each variable is assigned only once and in the order
    following the iter(self) iterator.

    Args:
        extra_data: the list of tensors used to update variables values.
    """
    self_values = self.unique_values()
    # self_values = list(self.values())
    if len(self_values) != len(extra_data):
      raise ValueError(f'The target has {len(self_values)} data, while we got '
                       f'a "extra_data" with the length of {len(extra_data)}.')
    for v1, data in zip(self_values, extra_data):
      v1.value = data

  def assign(self, extra_data):
    values = self.values()
    # self_values = list(self.values())
    if len(values) != len(extra_data):
      raise ValueError(f'The target has {len(values)} data, while we got '
                       f'a "extra_data" with the length of {len(extra_data)}.')
    for v1, data in zip(values, extra_data):
      v1.value = data

  def data(self):
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
    for d in self.unique_values():
      if isinstance(d, RandomState):
        replicated.append(jax.api.device_put_sharded([shard for shard in d.split(ndevices)], devices))
        saved_states.append(d.value)
      else:
        replicated.append(jax.api.device_put_replicated(d.value, devices))
    self.unique_assign(replicated)
    yield
    visited = set()
    saved_states.reverse()
    for k, d in self.items():
      if id(d) not in visited:  # Careful not to reduce twice in case of a variable and a reference to it.
        if isinstance(d, RandomState):
          d.unique_assign(saved_states.pop())
        else:
          d.reduce(d.value)
        visited.add(id(d))
