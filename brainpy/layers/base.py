# -*- coding: utf-8 -*-

import inspect
import warnings

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import Collector

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


class Module(Base):
  """Basic module class."""

  @staticmethod
  def get_param(param, size):
    return bm.TrainVar(Module.init_param(param, size))

  @staticmethod
  def init_param(param, size):
    if param is None:
      return None
    if callable(param):
      param = param(size)
    elif isinstance(param, onp.ndarray):
      param = bm.asarray(param)
    elif isinstance(param, (bm.JaxArray, jnp.ndarray)):
      pass
    else:
      raise ValueError(f'Unknown param type {type(param)}: {param}')
    assert param.shape == size, f'"param.shape" is not the required size {size}'
    return param

  def __init__(self, name=None):  # initialize parameters
    warnings.warn('Please use "brainpy.rnns.Module" instead. '
                  '"brainpy.layers.Module" will be removed since '
                  'version 2.1.0.', DeprecationWarning)
    super(Module, self).__init__(name=name)

  def __call__(self, *args, **kwargs):  # initialize variables
    return self.call(*args, **kwargs)

  def call(self, *args, **kwargs):
    raise NotImplementedError


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

  def __init__(self, *arg_ds, name=None, **kwarg_ds):
    super(Sequential, self).__init__(name=name)

    self.implicit_nodes = Collector()
    # check "args"
    for ds in arg_ds:
      if not isinstance(ds, Module):
        raise errors.BrainPyError(f'Only support {Module.__name__}, '
                                  f'but we got {type(ds)}: {str(ds)}.')
      self.implicit_nodes[ds.name] = ds

    # check "kwargs"
    for key, ds in kwarg_ds.items():
      if not isinstance(ds, Module):
        raise errors.BrainPyError(f'Only support {Module.__name__}, '
                                  f'but we got {type(ds)}: {str(ds)}.')
      self.implicit_nodes[key] = ds

    # all update functions
    self._return_kwargs = ['kwargs' in inspect.signature(ds.call).parameters.keys()
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
    args = ds[0].call(*args, **self._check_kwargs(0, kwargs))
    # other layers
    for i in range(1, len(self.implicit_nodes)):
      args = ds[i].call(*_check_args(args=args), **self._check_kwargs(i, kwargs))
    return args

  def __getitem__(self, key: int):
    return list(self.implicit_nodes.values())[key]
