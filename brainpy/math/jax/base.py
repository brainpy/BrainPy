# -*- coding: utf-8 -*-

from jax.tree_util import register_pytree_node

__all__ = [
  'Pointer',
  'ndarray',
]


class Pointer(object):
  __slots__ = "_value"

  def __init__(self, value):
    self._value = value

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value.value if isinstance(value, Pointer) else value


register_pytree_node(Pointer,
                     lambda t: ((t.value,), None),
                     lambda aux_data, children: Pointer(*children))

ndarray = Pointer
