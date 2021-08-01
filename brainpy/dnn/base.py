# -*- coding: utf-8 -*-

import inspect

from brainpy import errors
from brainpy.primary.base import Primary
from brainpy.primary.collector import ArrayCollector, Collector

__all__ = [
  # abstract class
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


class Module(Primary):
  """Basic DNN module.

  Parameters
  ----------
  name : str, optional
    The name of the module.
  """

  def __init__(self, name=None):
    super(Module, self).__init__(name=name)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Sequential(Module, dict):
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
    all_systems = dict()
    # check "args"
    for module in arg_modules:
      if not isinstance(module, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(module)}.')
      all_systems[module.name] = module

    # check "kwargs"
    for key, module in kwarg_modules.items():
      if not isinstance(module, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(module)}.')
      all_systems[key] = module

    # initialize base class
    Module.__init__(self, name=name)
    dict.__init__(self, all_systems)

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
    keys = list(self.keys())
    calls = list(self.values())

    # module 0
    try:
      args = calls[0](*args, **_check_config(calls[0], config))
    except Exception as e:
      raise type(e)(f'Sequential [{keys[0]}] {calls[0]} {e}')

    # other modules
    for i in range(1, len(self)):
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
    gather = ArrayCollector()
    if method == 'relative':
      for k, v in self.items():
        for k2, v2 in v.vars(method=method).items():
          gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.items():
        gather.update(v.vars(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    gather.update(super(Sequential, self).vars(method=method))
    return gather

  def nodes(self, method='absolute'):
    gather = Collector()
    if method == 'relative':
      for k, v in self.items():
        gather[k] = v
        for k2, v2 in v.nodes(method=method).items():
          gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.items():
        gather[v.name] = v
        gather.update(v.nodes(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    gather.update(super(Sequential, self).nodes(method=method))
    return gather
