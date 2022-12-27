# -*- coding: utf-8 -*-

from typing import Callable, Sequence, Dict, Union, TypeVar

from brainpy.base.base import BrainPyObject


Variable = TypeVar('Variable')


__all__ = [
  'FunAsObject',
]


class FunAsObject(BrainPyObject):
  """Transform a Python function as a :py:class:`~.BrainPyObject`.

  Parameters
  ----------
  f : callable
    The function to wrap.
  child_objs : optional, BrainPyObject, sequence of BrainPyObject, dict
    The nodes in the defined function ``f``.
  dyn_vars : optional, Variable, sequence of Variable, dict
    The dynamically changed variables.
  name : optional, str
    The function name.
  """

  def __init__(self,
               f: Callable,
               child_objs: Union[BrainPyObject, Sequence[BrainPyObject], Dict[dict, BrainPyObject]] = None,
               dyn_vars: Union[Variable, Sequence[Variable], Dict[dict, Variable]] = None,
               name: str = None):
    super(FunAsObject, self).__init__(name=name)
    self._f = f
    if child_objs is not None:
      self.register_implicit_nodes(child_objs)
    if dyn_vars is not None:
      self.register_implicit_vars(dyn_vars)

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)

  def __repr__(self) -> str:
    from brainpy.tools import repr_context
    name = self.__class__.__name__
    indent = " " * (len(name) + 1)
    indent2 = indent + " " * len('nodes=')
    nodes = [repr_context(str(n), indent2) for n in self.implicit_nodes.values()]
    node_string = ", \n".join(nodes)
    return (f'{name}(nodes=[{node_string}],\n' +
            " " * (len(name) + 1) + f'num_of_vars={len(self.implicit_vars)})')
