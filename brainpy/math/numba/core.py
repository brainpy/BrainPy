# -*- coding: utf-8 -*-


from brainpy.math import numpy
from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'Function',
  'JIT',
  'Vectorize',
  'Parallel',
]

_NUMBA_FUNC_NO = 0


class Function(numpy.Function):
  """Turn a function into a DynamicSystem."""
  target_backend = 'numba'

  def __init__(self, f, VIN, name=None, monitors=None):
    """Function constructor.

    Parameters
    ----------
    f : function
      The function or the module to represent.
    VIN : list of Collector, tuple of Collector
      The collection of variables, integrators, and nodes.
    """
    if name is None:
      global _NUMBA_FUNC_NO
      name = f'NBFunc{_NUMBA_FUNC_NO}'
      _NUMBA_FUNC_NO += 1

    super(Function, self).__init__(f=f, VIN=VIN, name=name, monitors=monitors)


class JIT(DynamicSystem):
  target_backend = 'numba'


class Parallel(DynamicSystem):
  target_backend = 'numba'


class Vectorize(DynamicSystem):
  target_backend = 'numba'
