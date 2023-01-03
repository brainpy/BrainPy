# -*- coding: utf-8 -*-

"""
This file defines the basic classes for BrainPy object-oriented transformations.
These transformations include JAX's JIT, autograd, vectorization, parallelization, etc.
"""

import os
import warnings
from collections import namedtuple
from typing import Any, Tuple, Callable, Sequence, Dict, Union, Optional

import jax
import numpy as np
from jax._src.tree_util import _registry
from jax.tree_util import register_pytree_node
from jax.tree_util import register_pytree_node_class
from jax.util import safe_zip

from brainpy import errors
from brainpy._src.math.ndarray import (Array, Variable, VariableView, TrainVar)
from brainpy._src.tools.naming import (get_unique_name, check_name_uniqueness)

StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])

__all__ = [
  # objects
  'BrainPyObject', 'Base', 'FunAsObject',

  # dynamical containers
  'dyn_seq',
  'dyn_dict',
  'Collector',
  'ArrayCollector',
  'TensorCollector',
]


class BrainPyObject(object):
  """The BrainPyObject class for whole BrainPy ecosystem.

  The subclass of BrainPyObject includes:

  - ``DynamicalSystem`` in *brainpy.dyn.base.py*
  - ``Integrator`` in *brainpy.integrators.base.py*
  - ``FunAsObject`` in *brainpy.brainpy_object.function.py*
  - ``Optimizer`` in *brainpy.optimizers.py*
  - ``Scheduler`` in *brainpy.optimizers.py*
  - and others.
  """

  _excluded_vars = ()

  def __init__(self, name=None):
    super().__init__()
    cls = self.__class__
    if cls not in _registry:
      register_pytree_node_class(cls)

    # check whether the object has a unique name.
    self._name = None
    self._name = self.unique_name(name=name)
    check_name_uniqueness(name=self._name, obj=self)

    # Used to wrap the implicit variables
    # which cannot be accessed by self.xxx
    self.implicit_vars = ArrayCollector()

    # Used to wrap the implicit children nodes
    # which cannot be accessed by self.xxx
    self.implicit_nodes = Collector()

  def __setattr__(self, key: str, value: Any) -> None:
    """Overwrite `__setattr__` method for change Variable values.

    .. versionadded:: 2.3.1

    Parameters
    ----------
    key: str
      The attribute.
    value: Any
      The value.
    """
    if key in self.__dict__:
      val = self.__dict__[key]
      if isinstance(val, Variable):
        val.value = value
        return
    super().__setattr__(key, value)

  def tree_flatten(self):
    """Flattens the object as a PyTree.

    The flattening order is determined by attributes added order.

    .. versionadded:: 2.3.1

    Returns
    -------
    res: tuple
      A tuple of dynamical values and static values.
    """
    dts = (BrainPyObject,) + tuple(dynamical_types)
    dynamic_names = []
    dynamic_values = []
    static_names = []
    static_values = []
    for k, v in self.__dict__.items():
      if isinstance(v, dts):
        dynamic_names.append(k)
        dynamic_values.append(v)
      else:
        static_values.append(v)
        static_names.append(k)
    return tuple(dynamic_values), (tuple(dynamic_names),
                                   tuple(static_names),
                                   tuple(static_values))

  @classmethod
  def tree_unflatten(cls, aux, dynamic_values):
    """

    .. versionadded:: 2.3.1

    Parameters
    ----------
    aux
    dynamic_values

    Returns
    -------

    """
    self = cls.__new__(cls)
    dynamic_names, static_names, static_values = aux
    for name, value in zip(dynamic_names, dynamic_values):
      object.__setattr__(self, name, value)
    for name, value in zip(static_names, static_values):
      object.__setattr__(self, name, value)
    return self

  @property
  def name(self):
    """Name of the model."""
    return self._name

  @name.setter
  def name(self, name: str = None):
    self._name = self.unique_name(name=name)
    check_name_uniqueness(name=self._name, obj=self)

  def register_implicit_vars(self, *variables, **named_variables):
    for variable in variables:
      if isinstance(variable, Variable):
        self.implicit_vars[f'var{id(variable)}'] = variable
      elif isinstance(variable, (tuple, list)):
        for v in variable:
          if not isinstance(v, Variable):
            raise ValueError(f'Must be instance of {Variable.__name__}, but we got {type(v)}')
          self.implicit_vars[f'var{id(v)}'] = v
      elif isinstance(variable, dict):
        for k, v in variable.items():
          if not isinstance(v, Variable):
            raise ValueError(f'Must be instance of {Variable.__name__}, but we got {type(v)}')
          self.implicit_vars[k] = v
      else:
        raise ValueError(f'Unknown type: {type(variable)}')
    for key, variable in named_variables.items():
      if not isinstance(variable, Variable):
        raise ValueError(f'Must be instance of {Variable.__name__}, but we got {type(variable)}')
      self.implicit_vars[key] = variable

  def register_implicit_nodes(self, *nodes, node_cls: type = None, **named_nodes):
    if node_cls is None:
      node_cls = BrainPyObject
    for node in nodes:
      if isinstance(node, node_cls):
        self.implicit_nodes[node.name] = node
      elif isinstance(node, (tuple, list)):
        for n in node:
          if not isinstance(n, node_cls):
            raise ValueError(f'Must be instance of {node_cls.__name__}, but we got {type(n)}')
          self.implicit_nodes[n.name] = n
      elif isinstance(node, dict):
        for k, n in node.items():
          if not isinstance(n, node_cls):
            raise ValueError(f'Must be instance of {node_cls.__name__}, but we got {type(n)}')
          self.implicit_nodes[k] = n
      else:
        raise ValueError(f'Unknown type: {type(node)}')
    for key, node in named_nodes.items():
      if not isinstance(node, node_cls):
        raise ValueError(f'Must be instance of {node_cls.__name__}, but we got {type(node)}')
      self.implicit_nodes[key] = node

  def vars(self,
           method: str = 'absolute',
           level: int = -1,
           include_self: bool = True,
           exclude_types: Tuple[type, ...] = None):
    """Collect all variables in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the variables.
    level: int
      The hierarchy level to find variables.
    include_self: bool
      Whether include the variables in the self.
    exclude_types: tuple of type
      The type to exclude.

    Returns
    -------
    gather : ArrayCollector
      The collection contained (the path, the variable).
    """
    if exclude_types is None:
      exclude_types = (VariableView,)
    nodes = self.nodes(method=method, level=level, include_self=include_self)
    gather = ArrayCollector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        include = False
        if isinstance(v, Variable):
          include = True
          if len(exclude_types) and isinstance(v, exclude_types):
            include = False
        if include:
          if k not in node._excluded_vars:
            gather[f'{node_path}.{k}' if node_path else k] = v
      gather.update({f'{node_path}.{k}': v for k, v in node.implicit_vars.items()})
    return gather

  def train_vars(self, method='absolute', level=-1, include_self=True):
    """The shortcut for retrieving all trainable variables.

    Parameters
    ----------
    method : str
      The method to access the variables. Support 'absolute' and 'relative'.
    level: int
      The hierarchy level to find TrainVar instances.
    include_self: bool
      Whether include the TrainVar instances in the self.

    Returns
    -------
    gather : ArrayCollector
      The collection contained (the path, the trainable variable).
    """
    return self.vars(method=method, level=level, include_self=include_self).subset(TrainVar)

  def _find_nodes(self, method='absolute', level=-1, include_self=True, _lid=0, _paths=None):
    if _paths is None:
      _paths = set()
    gather = Collector()
    if include_self:
      if method == 'absolute':
        gather[self.name] = self
      elif method == 'relative':
        gather[''] = self
      else:
        raise ValueError(f'No support for the method of "{method}".')
    if (level > -1) and (_lid >= level):
      return gather
    if method == 'absolute':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, BrainPyObject):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[v.name] = v
            nodes.append(v)
      for node in self.implicit_nodes.values():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[node.name] = node
          nodes.append(node)
      for v in nodes:
        gather.update(v._find_nodes(method=method,
                                    level=level,
                                    _lid=_lid + 1,
                                    _paths=_paths,
                                    include_self=include_self))

    elif method == 'relative':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, BrainPyObject):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[k] = v
            nodes.append((k, v))
      for key, node in self.implicit_nodes.items():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[key] = node
          nodes.append((key, node))
      for k1, v1 in nodes:
        for k2, v2 in v1._find_nodes(method=method,
                                     _paths=_paths,
                                     _lid=_lid + 1,
                                     level=level,
                                     include_self=include_self).items():
          if k2: gather[f'{k1}.{k2}'] = v2

    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def nodes(self, method='absolute', level=-1, include_self=True):
    """Collect all children nodes.

    Parameters
    ----------
    method : str
      The method to access the nodes.
    level: int
      The hierarchy level to find nodes.
    include_self: bool
      Whether include the self.

    Returns
    -------
    gather : Collector
      The collection contained (the path, the node).
    """
    return self._find_nodes(method=method, level=level, include_self=include_self)

  def unique_name(self, name=None, type_=None):
    """Get the unique name for this object.

    Parameters
    ----------
    name : str, optional
      The expected name. If None, the default unique name will be returned.
      Otherwise, the provided name will be checked to guarantee its uniqueness.
    type_ : str, optional
      The name of this class, used for object naming.

    Returns
    -------
    name : str
      The unique name for this object.
    """
    if name is None:
      if type_ is None:
        return get_unique_name(type_=self.__class__.__name__)
      else:
        return get_unique_name(type_=type_)
    else:
      check_name_uniqueness(name=name, obj=self)
      return name

  def state_dict(self):
    """Returns a dictionary containing a whole state of the module.

    Returns
    -------
    out: dict
      A dictionary containing a whole state of the module.
    """
    return self.vars().unique().dict()

  def load_state_dict(self, state_dict: Dict[str, Any], warn: bool = True):
    """Copy parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Parameters
    ----------
    state_dict: dict
      A dict containing parameters and persistent buffers.
    warn: bool
      Warnings when there are missing keys or unexpected keys in the external ``state_dict``.

    Returns
    -------
    out: StateLoadResult
      ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:

      * **missing_keys** is a list of str containing the missing keys
      * **unexpected_keys** is a list of str containing the unexpected keys
    """
    variables = self.vars().unique()
    keys1 = set(state_dict.keys())
    keys2 = set(variables.keys())
    unexpected_keys = list(keys1 - keys2)
    missing_keys = list(keys2 - keys1)
    for key in keys2.intersection(keys1):
      variables[key].value = state_dict[key]
    if warn:
      if len(unexpected_keys):
        warnings.warn(f'Unexpected keys in state_dict: {unexpected_keys}', UserWarning)
      if len(missing_keys):
        warnings.warn(f'Missing keys in state_dict: {missing_keys}', UserWarning)
    return StateLoadResult(missing_keys, unexpected_keys)

  def load_states(self, filename, verbose=False):
    """Load the model states.

    Parameters
    ----------
    filename : str
      The filename which stores the model states.
    verbose: bool
      Whether report the load progress.
    """
    from brainpy._src.checkpoints import io
    if not os.path.exists(filename):
      raise errors.BrainPyError(f'Cannot find the file path: {filename}')
    elif filename.endswith('.hdf5') or filename.endswith('.h5'):
      io.load_by_h5(filename, target=self, verbose=verbose)
    elif filename.endswith('.pkl'):
      io.load_by_pkl(filename, target=self, verbose=verbose)
    elif filename.endswith('.npz'):
      io.load_by_npz(filename, target=self, verbose=verbose)
    elif filename.endswith('.mat'):
      io.load_by_mat(filename, target=self, verbose=verbose)
    else:
      raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {io.SUPPORTED_FORMATS}')

  def save_states(self, filename, variables=None, **setting):
    """Save the model states.

    Parameters
    ----------
    filename : str
      The file name which to store the model states.
    variables: optional, dict, ArrayCollector
      The variables to save. If not provided, all variables retrieved by ``~.vars()`` will be used.
    """
    if variables is None:
      variables = self.vars(method='absolute', level=-1)

    from brainpy._src.checkpoints import io
    if filename.endswith('.hdf5') or filename.endswith('.h5'):
      io.save_as_h5(filename, variables=variables)
    elif filename.endswith('.pkl') or filename.endswith('.pickle'):
      io.save_as_pkl(filename, variables=variables)
    elif filename.endswith('.npz'):
      io.save_as_npz(filename, variables=variables, **setting)
    elif filename.endswith('.mat'):
      io.save_as_mat(filename, variables=variables)
    else:
      raise errors.BrainPyError(f'Unknown file format: {filename}. We only supports {io.SUPPORTED_FORMATS}')

  def to(self, device: Optional[Any]):
    """Moves all variables into the given device.

    Args:
      device: The device.
    """
    for var in self.vars().unique().values():
      var.value = jax.device_put(var.value, device=device)

  def cpu(self):
    """Move all variable into the CPU device."""
    self.to(device=jax.devices('cpu')[0])

  def cuda(self):
    """Move all variables into the GPU device."""
    self.to(device=jax.devices('gpu')[0])

  def tpu(self):
    """Move all variables into the TPU device."""
    self.to(device=jax.devices('tpu')[0])


Base = BrainPyObject


class FunAsObject(BrainPyObject):
  """Transform a Python function as a :py:class:`~.BrainPyObject`.

  Parameters
  ----------
  target : callable
    The function to wrap.
  child_objs : optional, BrainPyObject, sequence of BrainPyObject, dict
    The nodes in the defined function ``f``.
  dyn_vars : optional, Variable, sequence of Variable, dict
    The dynamically changed variables.
  name : optional, str
    The function name.
  """

  def __init__(
      self,
      target: Callable,
      child_objs: Union[BrainPyObject, Sequence[BrainPyObject], Dict[dict, BrainPyObject]] = None,
      dyn_vars: Union[Variable, Sequence[Variable], Dict[dict, Variable]] = None,
      name: str = None
  ):
    super(FunAsObject, self).__init__(name=name)
    self.target = target
    if child_objs is not None:
      self.register_implicit_nodes(child_objs)
    if dyn_vars is not None:
      self.register_implicit_vars(dyn_vars)

  def __call__(self, *args, **kwargs):
    return self.target(*args, **kwargs)

  def __repr__(self) -> str:
    from brainpy.tools import repr_context
    name = self.__class__.__name__
    indent = " " * (len(name) + 1)
    indent2 = indent + " " * len('nodes=')
    nodes = [repr_context(str(n), indent2) for n in self.implicit_nodes.values()]
    node_string = ", \n".join(nodes)
    return (f'{name}(nodes=[{node_string}],\n' +
            " " * (len(name) + 1) + f'num_of_vars={len(self.implicit_vars)})')


class dyn_seq(list):
  """A list to represent a dynamically changed numerical
  sequence in which its element can be changed during JIT compilation.

  .. note::
     The element must be numerical, like ``bool``, ``int``, ``float``,
     ``jax.Array``, ``numpy.ndarray``, ``brainpy.math.Array``.
  """

  def append(self, element):
    if not isinstance(element, (bool, int, float, jax.Array, Array, np.ndarray)):
      raise TypeError(f'Each element should be a numerical value.')

  def extend(self, iterable) -> None:
    for element in iterable:
      self.append(element)


register_pytree_node(dyn_seq,
                     lambda x: (tuple(x), ()),
                     lambda _, values: dyn_seq(values))


class dyn_dict(dict):
  """A dict to represent a dynamically changed numerical
  dictionary in which its element can be changed during JIT compilation.

  .. note::
     Each key must be a string, and each value must be numerical, including
     ``bool``, ``int``, ``float``, ``jax.Array``, ``numpy.ndarray``,
     ``brainpy.math.Array``.
  """

  def update(self, *args, **kwargs) -> 'dyn_dict':
    super().update(*args, **kwargs)
    return self


register_pytree_node(dyn_dict,
                     lambda x: (tuple(x.values()), tuple(x.keys())),
                     lambda keys, values: dyn_dict(safe_zip(keys, values)))


class Collector(dict):
  """A Collector is a dictionary (name, var) with some additional methods to make manipulation
  of collections of variables easy. A Collector is ordered by insertion order. It is the object
  returned by BrainPyObject.vars() and used as input in many Collector instance: optimizers, jit, etc..."""

  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""
    if key in self:
      if id(self[key]) != id(value):
        raise ValueError(f'Name "{key}" conflicts: same name for {value} and {self[key]}.')
    dict.__setitem__(self, key, value)

  def replace(self, key, new_value):
    """Replace the original key with the new value."""
    self.pop(key)
    self[key] = new_value

  def update(self, other, **kwargs):
    assert isinstance(other, (dict, list, tuple))
    if isinstance(other, dict):
      for key, value in other.items():
        self[key] = value
    elif isinstance(other, (tuple, list)):
      num = len(self)
      for i, value in enumerate(other):
        self[f'_var{i + num}'] = value
    else:
      raise ValueError(f'Only supports dict/list/tuple, but we got {type(other)}')
    for key, value in kwargs.items():
      self[key] = value
    return self

  def __add__(self, other):
    """Merging two dicts.

    Parameters
    ----------
    other: dict
      The other dict instance.

    Returns
    -------
    gather: Collector
      The new collector.
    """
    gather = type(self)(self)
    gather.update(other)
    return gather

  def __sub__(self, other: Union[Dict, Sequence]):
    """Remove other item in the collector.

    Parameters
    ----------
    other: dict, sequence
      The items to remove.

    Returns
    -------
    gather: Collector
      The new collector.
    """
    if not isinstance(other, (dict, tuple, list)):
      raise ValueError(f'Only support dict/tuple/list, but we got {type(other)}.')
    gather = type(self)(self)
    if isinstance(other, dict):
      for key, val in other.items():
        if key in gather:
          if id(val) != id(gather[key]):
            raise ValueError(f'Cannot remove {key}, because we got two different values: '
                             f'{val} != {gather[key]}')
          gather.pop(key)
        else:
          raise ValueError(f'Cannot remove {key}, because we do not find it '
                           f'in {self.keys()}.')
    elif isinstance(other, (list, tuple)):
      id_to_keys = {}
      for k, v in self.items():
        id_ = id(v)
        if id_ not in id_to_keys:
          id_to_keys[id_] = []
        id_to_keys[id_].append(k)

      keys_to_remove = []
      for key in other:
        if isinstance(key, str):
          keys_to_remove.append(key)
        else:
          keys_to_remove.extend(id_to_keys[id(key)])

      for key in set(keys_to_remove):
        if key in gather:
          gather.pop(key)
        else:
          raise ValueError(f'Cannot remove {key}, because we do not find it '
                           f'in {self.keys()}.')
    else:
      raise KeyError(f'Unknown type of "other". Only support dict/tuple/list, but we got {type(other)}')
    return gather

  def subset(self, var_type):
    """Get the subset of the (key, value) pair.

    ``subset()`` can be used to get a subset of some class:

    >>> import brainpy as bp
    >>>
    >>> some_collector = Collector()
    >>>
    >>> # get all trainable variables
    >>> some_collector.subset(bp.math.TrainVar)
    >>>
    >>> # get all Variable
    >>> some_collector.subset(bp.math.Variable)

    or, it can be used to get a subset of integrators:

    >>> # get all ODE integrators
    >>> some_collector.subset(bp.ode.ODEIntegrator)

    Parameters
    ----------
    var_type : type
      The type/class to match.
    """
    gather = type(self)()
    for key, value in self.items():
      if isinstance(value, var_type):
        gather[key] = value
    return gather

  def unique(self):
    """Get a new type of collector with unique values.

    If one value is assigned to two or more keys,
    then only one pair of (key, value) will be returned.
    """
    gather = type(self)()
    seen = set()
    for k, v in self.items():
      if id(v) not in seen:
        seen.add(id(v))
        gather[k] = v
    return gather


class ArrayCollector(Collector):
  """A ArrayCollector is a dictionary (name, var)
  with some additional methods to make manipulation
  of collections of variables easy. A Collection
  is ordered by insertion order. It is the object
  returned by DynamicalSystem.vars() and used as input
  in many DynamicalSystem instance: optimizers, Jit, etc..."""

  def __setitem__(self, key, value):
    """Overload bracket assignment to catch potential conflicts during assignment."""

    assert isinstance(value, Array)
    if key in self:
      if id(self[key]) != id(value):
        raise ValueError(f'Name "{key}" conflicts: same name for {value} and {self[key]}.')
    dict.__setitem__(self, key, value)

  def assign(self, inputs):
    """Assign data to all values.

    Parameters
    ----------
    inputs : dict
      The data for each value in this collector.
    """
    if len(self) != len(inputs):
      raise ValueError(f'The target has {len(inputs)} data, while we got '
                       f'an input value with the length of {len(inputs)}.')
    for key, v in self.items():
      v.value = inputs[key]

  def dict(self):
    """Get a dict with the key and the value data.
    """
    gather = dict()
    for k, v in self.items():
      gather[k] = v.value
    return gather

  def data(self):
    """Get all data in each value."""
    return [x.value for x in self.values()]

  @classmethod
  def from_other(cls, other: Union[Sequence, Dict]):
    if isinstance(other, (tuple, list)):
      return cls({id(o): o for o in other})
    elif isinstance(other, dict):
      return cls(other)
    else:
      raise TypeError


TensorCollector = ArrayCollector

register_pytree_node(
  ArrayCollector,
  lambda x: (x.values(), x.keys()),
  lambda keys, values: ArrayCollector(safe_zip(keys, values))
)

dynamical_types = [Variable, dyn_seq, dyn_dict]
