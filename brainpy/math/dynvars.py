# -*- coding: utf-8 -*-

from jax.tree_util import register_pytree_node
from jax.util import safe_zip
from .ndarray import Variable

__all__ = [
  'Tuple', 'List', 'Dict',
]



class Tuple(tuple):
  pass


class List(list):
  pass


class Dict(dict):
  def update(self, *args, **kwargs) -> 'Dict':
    super().update(*args, **kwargs)
    return self


register_pytree_node(Tuple,
                     lambda x: (x, ()),
                     lambda keys, values: Tuple(values))

register_pytree_node(List,
                     lambda x: (tuple(x), ()),
                     lambda keys, values: List(values))

register_pytree_node(Dict,
                     lambda x: (tuple(x.values()), tuple(x.keys())),
                     lambda keys, values: Dict(safe_zip(keys, values)))


dynamical_types = [Variable, Tuple, List, Dict]


def register_dynamical_types(t):
  if not isinstance(t, type):
    raise TypeError
  dynamical_types.append(t)

