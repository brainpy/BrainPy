# -*- coding: utf-8 -*-

import jax

from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.backend.bpjax.core.function import Function


__all__ = [
  'JIT'
]


class JIT(DynamicSystem):
  """JIT (Just-In-Time) module takes a function
  or a module and compiles it for faster execution."""

  def __init__(self, f, all_vars=None, static_argnums=None):
    """Jit constructor.

    Parameters
    ----------
    f : function, DynamicSystem
      The function or the module to compile.
    all_vars : dict, Collection
      The Collection of variables used by the function or module. This argument is required for functions.
    static_argnums : optional, list, int
      Tuple of indexes of f's input arguments to treat as static (constants)).
      A new graph is compiled for each different combination of values for such inputs.
    """
    self.static_argnums = static_argnums
    if not isinstance(f, DynamicSystem):
      if all_vars is None:
        raise ValueError('You must supply the variables used by the function f.')
      f = Function(f, all_vars)

    def jit(all_data, kwargs, *args):
      try:
        self.all_vars.assign(all_data)
        return f(*args, **kwargs), self.all_vars.tensors()
      finally:
        pass

    self.all_vars = f.vars() if all_vars is None else all_vars
    self._call = jax.jit(jit, static_argnums=tuple(x + 2 for x in sorted(static_argnums or ())))
    self.__wrapped__ = f

  def __call__(self, *args, **kwargs):
    """Call the compiled version of the function or module."""
    output, changes = self._call(self.all_vars.tensors(), kwargs, *args)
    self.all_vars.assign(changes)
    return output

  def __repr__(self):
    return f'{self.__class__.__name__}(f={self.__wrapped__}, static_argnums={self.static_argnums})'
