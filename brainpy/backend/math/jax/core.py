# -*- coding: utf-8 -*-

import jax

import functools
from brainpy import errors
from brainpy.backend.math import numpy
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.tools.collector import Collector
from brainpy.tools.codes import func_name

__all__ = [
  'Function',
  'JIT',
  'Vectorize',
  'Parallel',
]

_JAX_FUNC_NO = 0
_JAX_JIT_NO = 0


class Function(numpy.Function):
  """Turn a function into a DynamicSystem."""

  target_backend = 'jax'

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
      global _JAX_FUNC_NO
      name = f'JaxFunc{_JAX_FUNC_NO}'
      _JAX_FUNC_NO += 1

    super(Function, self).__init__(f=f, VIN=VIN, name=name, monitors=monitors)


class JIT(DynamicSystem):
  """JIT (Just-In-Time) module takes a function
  or a module and compiles it for faster execution."""
  target_backend = 'jax'

  def __init__(self, ds, VIN=None, static_argnums=None, name=None, monitors=None):
    self.static_argnums = static_argnums

    if not isinstance(ds, DynamicSystem):
      if VIN is None:
        raise ValueError('You must supply the VIN used by the function f.')
      ds = Function(ds, VIN, name=name)

    self.raw = ds
    self.all_vars = ds.vars() if VIN is None else VIN[0]
    self.all_ints = ds.ints() if VIN is None else VIN[1]
    self.all_nodes = ds.nodes() if VIN is None else VIN[2]

    # monitors
    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    # name
    if name is None:
      global _JAX_JIT_NO
      name = f'JaxJIT{_JAX_JIT_NO}'
      _JAX_JIT_NO += 1

    steps = {}
    for key, func in ds.steps.items():
      @functools.partial(jax.jit, static_argnums=tuple(x + 2 for x in sorted(static_argnums or ())))
      def jit(all_data, _t, _i):
        self.all_vars.assign(all_data)
        return func(_t, _i), self.all_vars.all_data()

      @func_name(name=key)
      def call(_t, _i):
        output, changes = jit(self.all_vars.all_data(), _t, _i)
        self.all_vars.assign(changes)
        return output

      steps[key] = call

    super(JIT, self).__init__(steps=steps, name=name, monitors=monitors)

  def vars(self, prefix=''):
    """Return the Collection of the variables used by the function."""
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_vars.items())
    else:
      return Collector(self.all_vars)

  # def ints(self, prefix=''):
  #   if prefix:
  #     return Collector((prefix + k, v) for k, v in self.all_ints.items())
  #   else:
  #     return Collector(self.all_ints)

  # def nodes(self, prefix=''):
  #   if prefix:
  #     return Collector((prefix + k, v) for k, v in self.all_nodes.items())
  #   else:
  #     return Collector(self.all_nodes)

  def __repr__(self):
    return f'{self.__class__.__name__}(f={self.raw}, static_argnums={self.static_argnums or None})'



class Parallel(object):
  target_backend = 'jax'

  pass


class Vectorize(DynamicSystem):
  """Vectorize module takes a function or a module and
  compiles it for running in parallel on a single device.

  Parameters
  ----------

  f : DynamicSystem, function
    The function or the module to compile for vectorization.
  all_vars : Collector
    The Collection of variables used by the function or module.
    This argument is required for functions.
  batch_axis : tuple of int, int, tuple of None
    Tuple of int or None for each of f's input arguments:
    the axis to use as batch during vectorization. Use
    None to automatically broadcast.
  """
  target_backend = 'jax'
