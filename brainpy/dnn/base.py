# -*- coding: utf-8 -*-

import inspect

from brainpy import errors
from brainpy.base.base import Base

__all__ = [
  'Module', 'Sequential',
]


def _check_args(args):
  return (args,) if not isinstance(args, tuple) else args


def _check_config(f, config):
  pars_in_f = inspect.signature(f).parameters
  if 'config' in pars_in_f.keys():
    return {'config': config}
  else:
    return {}


class Module(Base):
  """Basic DNN module.

  Parameters
  ----------
  name : str, optional
    The name of the module.
  """

  def __init__(self, name=None):
    super(Module, self).__init__(name=name)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError('Must customize your own "__call__" method.')


class Sequential(Module):
  """Basic DNN sequential object.

  Parameters
  ----------
  arg_modules
    The modules without name specifications.
  name : str, optional
    The name of the sequential module.
  kwarg_modules
    The modules with name specifications.
  """

  def __init__(self, *arg_modules, name=None, **kwarg_modules):
    self.children_modules = dict()
    # check "args"
    for module in arg_modules:
      if not isinstance(module, Module):
        raise errors.BrainPyError(f'Only support {Module.__name__}, '
                                  f'but we got {type(module)}.')
      self.children_modules[module.name] = module

    # check "kwargs"
    for key, module in kwarg_modules.items():
      if not isinstance(module, Module):
        raise errors.BrainPyError(f'Only support {Module.__name__}, '
                                  f'but we got {type(module)}.')
      self.children_modules[key] = module

    # initialize base class
    Module.__init__(self, name=name)

  def __call__(self, *args, config=dict()):
    """Functional call.

    Parameters
    ----------
    args : list, tuple
      The *args arguments.
    config : dict
      The config arguments. The configuration used across modules.
      If the "__call__" function in submodule receives "config" arguments,
      This "config" parameter will be passed into this function.
    """
    keys = list(self.children_modules.keys())
    calls = list(self.children_modules.values())

    # module 0
    try:
      args = calls[0](*args, **_check_config(calls[0], config))
    except Exception as e:
      raise type(e)(f'Sequential [{keys[0]}] {calls[0]} {e}')

    # other modules
    for i in range(1, len(self.children_modules)):
      try:
        args = calls[i](*_check_args(args=args), **_check_config(calls[i], config))
      except Exception as e:
        raise type(e)(f'Sequential [{keys[i]}] {calls[i]} {e}')
    return args

  def vars(self, method='absolute'):
    """Collect all the variables (and their names) contained
    in the list and its children instance of DynamicSystem.

    Parameters
    ----------
    method : str
      string to prefix to the variable names.

    Returns
    -------
    gather collector.ArrayCollector
        A collection of all the variables.
    """
    gather = self._vars_in_container(self.children_modules, method=method)
    gather.update(super(Sequential, self).vars(method=method))
    return gather

  def nodes(self, method='absolute', _paths=None):
    if _paths is None:
      _paths = set()
    gather = self._nodes_in_container(self.children_modules, method=method, _paths=_paths)
    gather.update(super(Sequential, self).nodes(method=method))
    return gather
