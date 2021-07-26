# -*- coding: utf-8 -*-


from brainpy.dnn.imports import jax
from brainpy.math.jax.ndarray import ndarray

__all__ = [
  'TrainVar',
]


class TrainVar(ndarray):
  __slots__ = "_value"
  _registered = False

  def __new__(cls, *args, **kwargs):
    if not cls._registered:
      flatten = lambda t: ((t.value,), None)
      unflatten = lambda aux_data, children: TrainVar(*children)
      jax.tree_util.register_pytree_node(TrainVar, flatten, unflatten)
      cls._registered = True
    return super().__new__(cls)

  def __init__(self, value):
    if isinstance(value, ndarray):
      value = value.value
    super(TrainVar, self).__init__(value)
