# -*- coding: utf-8 -*-

from contextlib import contextmanager

from brainpy import errors

__all__ = [
  'Collector',
  'ArrayCollector',
]

math = None


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

  def update(self, other, **kwargs):
    assert isinstance(other, dict)
    for key, value in other.items():
      self[key] = value
    for key, value in kwargs.items():
      self[key] = value

  def __add__(self, other):
    gather = type(self)(self)
    gather.update(other)
    return gather

  def subset(self, type_, judge_func=None):
    """Get the subset of the (key, value) pair.

    ``subset()`` can be used to get a subset of some class:

    >>> import brainpy as bp
    >>>
    >>> # get all trainable variables
    >>> some_collector.subset(bp.TrainVar)
    >>>
    >>> # get all JaxArray
    >>> some_collector.subset(bp.math.JaxArray)

    or, it can be used to get a subset of integrators:

    >>> # get all ODE integrators
    >>> some_collector.subset(bp.integrators.ODE_INT)

    Parameters
    ----------
    type_ : Any
      The type/class to match.
    """
    global math
    if math is None:
      from brainpy import math

    gather = type(self)()
    if type(type_) == type:
      judge_func = lambda v: isinstance(v, type_) if judge_func is None else judge_func
      for key, value in self.items():
        if judge_func(value):
          gather[key] = value
    elif isinstance(type_, str):
      judge_func = lambda v: v.__name__.startswith(type_) if judge_func is None else judge_func
      for key, value in self.items():
        if judge_func(value):
          gather[key] = value
    elif isinstance(type_, math.Variable):
      judge_func = lambda v: type_.issametype(v) if judge_func is None else judge_func
      for key, value in self.items():
        if judge_func(value):
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
    """Get a dict with the key and the value data.
    """
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
    global math
    if math is None:
      from brainpy import math

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

  def data(self):
    """Get all data in each value."""
    return [x.value for x in self.values()]

  @contextmanager
  def replicate(self):
    """A context manager to use in a with statement that replicates
    the variables in this collection to multiple devices.

    Important: replicating also updates the random state in order
    to have a new one per device.
    """
    global math
    if math is None:
      from brainpy import math

    try:
      import jax
      import jax.numpy as jnp
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError('"ArrayCollector.replicate()" is only available in '
                                'JAX backend, while JAX is not installed.') from e

    replicated, saved_states = {}, {}
    x = jnp.zeros((jax.local_device_count(), 1), dtype=math.float_)
    sharded_x = jax.pmap(lambda x: x, axis_name='device')(x)
    devices = [b.device() for b in sharded_x.device_buffers]
    num_device = len(devices)
    for k, d in self.items():
      if isinstance(d, math.random.RandomState):
        replicated[k] = jax.api.device_put_sharded([shard for shard in d.split(num_device)], devices)
        saved_states[k] = d.value
      else:
        replicated[k] = jax.api.device_put_replicated(d.value, devices)
    self.assign(replicated)
    yield
    visited = set()
    for k, d in self.items():
      # Careful not to reduce twice in case of
      # a variable and a reference to it.
      if id(d) not in visited:
        if isinstance(d, math.random.RandomState):
          d.value = saved_states[k]
        else:
          d.value = reduce_func(d)
        visited.add(id(d))
