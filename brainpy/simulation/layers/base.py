# -*- coding: utf-8 -*-

import inspect

from brainpy import errors
from brainpy.base.collector import Collector
from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Module',
  'Sequential',
]


def _check_args(args):
  if args is None:
    return tuple()
  elif isinstance(args, tuple):
    return args
  else:
    return (args,)


class Module(DynamicalSystem):
  """Basic module class for DNN networks."""
  target_backend = 'jax'


class Sequential(Module):
  """Basic sequential object to control data flow.

  Parameters
  ----------
  arg_ds
    The modules without name specifications.
  name : str, optional
    The name of the sequential module.
  kwarg_ds
    The modules with name specifications.
  """

  def __init__(self, *arg_ds, monitors=None, name=None, **kwarg_ds):
    super(Sequential, self).__init__(monitors=monitors, name=name)

    self.implicit_nodes = Collector()
    # check "args"
    for ds in arg_ds:
      if not isinstance(ds, DynamicalSystem):
        raise errors.BrainPyError(f'Only support {DynamicalSystem.__name__}, '
                                  f'but we got {type(ds)}: {str(ds)}.')
      self.implicit_nodes[ds.name] = ds

    # check "kwargs"
    for key, ds in kwarg_ds.items():
      if not isinstance(ds, DynamicalSystem):
        raise errors.BrainPyError(f'Only support {DynamicalSystem.__name__}, '
                                  f'but we got {type(ds)}: {str(ds)}.')
      self.implicit_nodes[key] = ds

    # all update functions
    self._return_kwargs = ['kwargs' in inspect.signature(ds.update).parameters.keys()
                           for ds in self.implicit_nodes.values()]

  def _check_kwargs(self, i, kwargs):
    return kwargs if self._return_kwargs[i] else dict()

  def update(self, *args, **kwargs):
    """Functional call.

    Parameters
    ----------
    args : list, tuple
      The *args arguments.
    kwargs : dict
      The config arguments. The configuration used across modules.
      If the "__call__" function in submodule receives "config" arguments,
      This "config" parameter will be passed into this function.
    """
    ds = list(self.implicit_nodes.values())
    # first layer
    args = ds[0].update(*args, **self._check_kwargs(0, kwargs))
    # other layers
    for i in range(1, len(self.implicit_nodes)):
      args = ds[i].update(*_check_args(args=args), **self._check_kwargs(i, kwargs))
    return args
