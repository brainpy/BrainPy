# -*- coding: utf-8 -*-

import jax
from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.tools.collector import Collector


__all__ = [
  'Function',
  'JIT',
  'Parallel',
  'Vectorize',
]

_JAX_FUNC_NO = 0


class Function(DynamicSystem):
  """Turn a function into a Module by keeping the vars it uses."""

  def __init__(self, f, all_vars, name=None, monitors=None):
    """Function constructor.

    Args:
        f: the function or the module to represent.
        all_vars: the Collection of variables used by the function.
    """
    if hasattr(f, '__name__'):
      self.all_vars = Collector((f'{{{f.__name__}}}{k}', v)
                                for k, v in all_vars.items())
    else:
      self.all_vars = Collector(all_vars)
    self.__wrapped__ = f

    # monitors
    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    # name
    if name is None:
      global _JAX_FUNC_NO
      name = f'JaxFunc{_JAX_FUNC_NO}'
      _JAX_FUNC_NO += 1

    super(Function, self).__init__(steps={'update': self.update},
                                   name=name,
                                   monitors=None)

  def update(self, *args, **kwargs):
    """Call the the function."""
    return self.__wrapped__(*args, **kwargs)

  def vars(self, prefix=''):
    """Return the Collection of the variables used by the function."""
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_vars.items())
    return Collector(self.all_vars)

  @staticmethod
  def with_vars(all_vars):
    """Decorator which turns a function into a module using provided variable collection.

    Parameters
    ----------
    all_vars : dict
      The Collection of variables used by the function.
    """

    def from_function(f):
      return Function(f, all_vars)

    return from_function

  def __repr__(self):
    return f'{self.__class__.__name__}(f={str(self.__wrapped__)})'


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

  def update(self, *args, **kwargs):
    """Call the compiled version of the function or module."""
    output, changes = self._call(self.all_vars.tensors(), kwargs, *args)
    self.all_vars.assign(changes)
    return output

  def __repr__(self):
    return f'{self.__class__.__name__}(f={self.__wrapped__}, static_argnums={self.static_argnums})'

class Parallel(object):
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

    def __init__(self, f, all_vars=None, batch_axis=(0,)):
        if not isinstance(f, DynamicSystem):
            if all_vars is None:
                raise ValueError('You must supply the VarCollection used by the function f.')
            f = Function(f, all_vars)

        def vmap(all_data, random_list, *args):
          self.all_vars.assign(all_data)
          self.all_vars.subset(RandomState).assign(random_list)
          return f(*args), self.all_vars.all_data()

        fargs = positional_args_names(f)
        assert len(batch_axis) >= len(fargs), f'The batched argument must be specified for all of {f} arguments {fargs}'
        self.batch_axis = batch_axis
        self.batch_axis_argnums = [(x, v) for x, v in enumerate(batch_axis) if v is not None]
        assert self.batch_axis_argnums, f'No arguments to function {f} are vectorizable'
        self.all_vars = all_vars or f.vars()
        self._call = jax.vmap(vmap, (None, 0) + batch_axis)
        self.__wrapped__ = f

    def update(self, *args):
        """Call the vectorized version of the function or module."""
        assert len(args) == len(self.batch_axis), f'Number of arguments passed {len(args)} must match ' \
                                                  f'batched {len(self.batch_axis)}'
        nsplits = args[self.batch_axis_argnums[0][0]].shape[self.batch_axis_argnums[0][1]]
        output, changes = self._call(self.all_vars.all_data(),
                                     [v.split(nsplits) for v in self.all_vars.subset(RandomState)],
                                     *args)
        for v, u in zip(self.all_vars, changes):
            v.reduce(u)
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(f={self.__wrapped__}, batch_axis={self.batch_axis})'
