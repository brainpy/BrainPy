# -*- coding: utf-8 -*-

import functools

import jax

from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.collector import ArrayCollector
from brainpy.tools.codes import func_name

__all__ = [
  'jit',
  'Vectorize',
  'Parallel',
]

def convert(f):
  pass


def jit(ds_or_func, static_argnums=None, **kwargs):
  """JIT (Just-In-Time) Compilation.

  Takes a function or an instance of DynamicSystem
  and compiles it for faster execution.

  Parameters
  ----------
  ds_or_func : DynamicSystem, function
    The instance of DynamicSystem or a function.
  static_argnums: An optional int or collection of ints that specify which
    positional arguments to treat as static (compile-time constant).

  Returns
  -------
  ds_of_func : DynamicSystem, function
    A wrapped version of DynamicSystem or function,
    set up for just-in-time compilation.
  """

  if isinstance(ds_or_func, DynamicSystem):
    # all variables
    all_vars = ds_or_func.vars()

    # jit step functions
    steps = {}
    static_argnums = tuple(x + 1 for x in sorted(static_argnums or ()))
    for key, func in ds_or_func.steps.items():
      @functools.partial(jax.jit, static_argnums=static_argnums)
      def jitted_func(all_data, *args, **kwargs):
        all_vars.assign(all_data)
        return func(*args, **kwargs), all_vars.unique_data()

      @func_name(name=key)
      def call(*args, **kwargs):
        output, changed_data = jitted_func(all_vars.unique_data(), *args, **kwargs)
        all_vars.assign(changed_data)
        return output

      steps[key] = call

    # update step functions
    ds_or_func.steps.update(steps)

    return ds_or_func

  elif callable(ds_or_func):
    return jax.jit(ds_or_func, static_argnums=static_argnums, **kwargs)

  else:
    raise errors.ModelUseError(f'Only support instance of '
                               f'{DynamicSystem.__name__}, '
                               f'or a callable function, '
                               f'but we got {type(ds_or_func)}.')


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
  all_vars : ArrayCollector
    The Collection of variables used by the function or module.
    This argument is required for functions.
  batch_axis : tuple of int, int, tuple of None
    Tuple of int or None for each of f's input arguments:
    the axis to use as batch during vectorization. Use
    None to automatically broadcast.
  """
  target_backend = 'jax'
