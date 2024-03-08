# -*- coding: utf-8 -*-

"""
This file defines the basic classes for BrainPy object-oriented transformations.
These transformations include JAX's JIT, autograd, vectorization, parallelization, etc.
"""

import numbers
import warnings
from collections import namedtuple
from typing import Any, Tuple, Callable, Sequence, Dict, Union, Optional

import jax
import numpy as np
from jax.tree_util import register_pytree_node_class

from brainpy._src.math import defaults
from brainpy._src.math.modes import Mode
from brainpy._src.math.ndarray import (Array, )
from brainpy._src.math.object_transform.collectors import (ArrayCollector, Collector)
from brainpy._src.math.object_transform.naming import (get_unique_name,
                                                       check_name_uniqueness)
from brainpy._src.math.object_transform.variables import (Variable, VariableView, TrainVar,
                                                          VarList, VarDict)
from brainpy._src.math.sharding import BATCH_AXIS

variable_ = None
StateLoadResult = namedtuple('StateLoadResult', ['missing_keys', 'unexpected_keys'])
registered = set()

__all__ = [
  'BrainPyObject', 'Base', 'FunAsObject', 'ObjectTransform',

  'NodeDict', 'node_dict', 'NodeList', 'node_list',
]


class BrainPyObject(object):
  """The BrainPyObject class for the whole BrainPy ecosystem.

  The subclass of BrainPyObject includes but not limited to:

  - ``DynamicalSystem`` in *brainpy.dyn.base.py*
  - ``Integrator`` in *brainpy.integrators.base.py*
  - ``Optimizer`` in *brainpy.optimizers.py*
  - ``Scheduler`` in *brainpy.optimizers.py*

  .. note::
    Note a variable created in the ``BrainPyObject`` will never be replaced.

    For example, if here we create an object which has an attribute ``a``:

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class MyObj(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>     super().__init__()
    >>>     self.a = bm.Variable(bm.ones(1))
    >>>
    >>>   def reset1(self):
    >>>     self.a = bm.asarray([10.])
    >>>
    >>>   def reset2(self):
    >>>     self.a = 1.
    >>>
    >>> ob = MyObj()
    >>> id(ob.a)
    2643434845056

    After we call ``ob.reset1()`` function, ``ob.a`` is still the original Variable.
    what's change is its value.

    >>> ob.reset1()
    >>> id(ob.a)
    2643434845056

    What's really happend when we call ``self.a = bm.asarray([10.])`` is
    ``self.a.value = bm.asarray([10.])``.  Therefore we when call ``ob.reset2()``,
    there will be an error.

    >>> ob.reset2()
    brainpy.errors.MathError: The shape of the original data is (1,), while we got () with batch_axis=None.


  """

  _excluded_vars = ()

  def __init__(self, name=None):
    super().__init__()

    if defaults.bp_object_as_pytree:
      if self.__class__ not in registered:
        register_pytree_node_class(self.__class__)
        registered.add(self.__class__)

    # check whether the object has a unique name.
    self._name = None
    self._name = self.unique_name(name=name)
    check_name_uniqueness(name=self._name, obj=self)

    # Used to wrap the implicit variables
    # which cannot be accessed by self.xxx
    self._implicit_vars: Optional[ArrayCollector] = None

    # Used to wrap the implicit children nodes
    # which cannot be accessed by self.xxx
    self._implicit_nodes: Optional[Collector] = None

  @property
  def implicit_vars(self):
    if self._implicit_vars is None:
      self._implicit_vars = ArrayCollector()
    return self._implicit_vars

  @property
  def implicit_nodes(self):
    if self._implicit_nodes is None:
      self._implicit_nodes = Collector()
    return self._implicit_nodes

  def setattr(self, key: str, value: Any) -> None:
    super().__setattr__(key, value)

  def tracing_variable(
      self,
      name: str,
      init: Union[Callable, Array, jax.Array],
      shape: Union[int, Sequence[int]],
      batch_or_mode: Union[int, bool, Mode] = None,
      batch_axis: int = 0,
      axis_names: Optional[Sequence[str]] = None,
      batch_axis_name: Optional[str] = BATCH_AXIS,
  ) -> Variable:
    """Initialize the variable which can be traced during computations and transformations.

    Although this function is designed to initialize tracing variables during computation or compilation,
    it can also be used for the initialization of variables before computation and compilation.

    - If the variable has not been instantiated, a :py:class:`~.Variable` will be instantiated.
    - If the variable has been created, the further call of this function will return the created variable.

    Here is the usage example::

       class Example(bm.BrainPyObject):
         def fun(self):
           # The first time of calling `.fun()`, this line will create a Variable instance.
           # If users repeatedly call `.fun()` function, this line will not initialize variables again.
           # Instead, it will return the variable has been created.
           self.tracing_variable('a', bm.zeros, (10,))

           # The created variable can be accessed with self.xxx
           self.a.value = bm.ones(10)

           # Calling this function again will not reinitialize the
           # variable again, Instead, it will return the variable
           # that has been created.
           a = self.tracing_variable('a', bm.zeros, (10,))

    .. versionadded:: 2.4.5

    Args:
      name: str. The variable name.
      init: callable, Array. The data to be initialized as a ``Variable``.
      batch_or_mode: int, bool, Mode. This is used to specify the batch size of this variable.
        If it is a boolean or an instance of ``Mode``, the batch size will be 1.
        If it is None, the variable has no batch axis.
      shape: int, sequence of int. The shape of the variable.
      batch_axis: int. The batch axis, if batch size is given.
      axis_names: sequence of str. The name for each axis. These names should match the given ``axes``.
      batch_axis_name: str. The name for the batch axis. The name will be used
        if ``batch_or_mode`` is given. Default is ``brainpy.math.sharding.BATCH_AXIS``.

    Returns:
      The instance of :py:class:`~.Variable`.
    """
    # the variable has been created
    if hasattr(self, name):
      var = getattr(self, name)
      if isinstance(var, Variable):
        return var
        # if var.shape != value.shape:
        #   raise ValueError(
        #     f'"{name}" has been used in this class with the shape of {var.shape} (!= {value.shape}). '
        #     f'Please assign another name for the initialization of variables '
        #     f'tracing during computation and compilation.'
        #   )
        # if var.dtype != value.dtype:
        #   raise ValueError(
        #     f'"{name}" has been used in this class with the dtype of {var.dtype} (!= {value.dtype}). '
        #     f'Please assign another name for the initialization of variables '
        #     f'tracing during computation and compilation.'
        #   )

    global variable_
    if variable_ is None:
      from brainpy.initialize import variable_
    with jax.ensure_compile_time_eval():
      value = variable_(init, shape, batch_or_mode, batch_axis, axis_names, batch_axis_name)
      value.ready_to_trace = True
    self.setattr(name, value)
    return value

  def __setattr__(self, key: str, value: Any) -> None:
    """Overwrite `__setattr__` method for changing :py:class:`~.Variable` values.

    .. versionadded:: 2.3.1

    Args:
      key: str. The attribute.
      value: Any. The value.
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
    dynamic_names = []
    dynamic_values = []
    static_names = []
    static_values = []
    for k, v in self.__dict__.items():
      if isinstance(v, (BrainPyObject, Variable, NodeList, NodeDict, VarList, VarDict)):
        # if isinstance(v, (BrainPyObject, Variable)):
        dynamic_names.append(k)
        dynamic_values.append(v)
      else:
        static_values.append(v)
        static_names.append(k)
    return tuple(dynamic_values), (tuple(dynamic_names), tuple(static_names), tuple(static_values))

  @classmethod
  def tree_unflatten(cls, aux, dynamic_values):
    """Unflatten the data to construct an object of this class.

    .. versionadded:: 2.3.1
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

  def register_implicit_vars(self, *variables, var_cls: type = None, **named_variables):
    if var_cls is None:
      var_cls = (Variable, VarList, VarDict)

    for variable in variables:
      if isinstance(variable, var_cls):
        self.implicit_vars[f'var{id(variable)}'] = variable
      elif isinstance(variable, (tuple, list)):
        for v in variable:
          if not isinstance(v, var_cls):
            raise ValueError(f'Must be instance of {var_cls}, but we got {type(v)}')
          self.implicit_vars[f'var{id(v)}'] = v
      elif isinstance(variable, dict):
        for k, v in variable.items():
          if not isinstance(v, var_cls):
            raise ValueError(f'Must be instance of {var_cls}, but we got {type(v)}')
          self.implicit_vars[k] = v
      else:
        raise ValueError(f'Unknown type: {type(variable)}')
    for key, variable in named_variables.items():
      if not isinstance(variable, var_cls):
        raise ValueError(f'Must be instance of {var_cls}, but we got {type(variable)}')
      self.implicit_vars[key] = variable

  def register_implicit_nodes(self, *nodes, node_cls: type = None, **named_nodes):
    if node_cls is None:
      node_cls = (BrainPyObject, NodeList, NodeDict)

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

  def vars(
      self,
      method: str = 'absolute',
      level: int = -1,
      include_self: bool = True,
      exclude_types: Tuple[type, ...] = None
  ):
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
      for k in node.__dict__.keys():
        if k in node._excluded_vars:
          continue
        v = getattr(node, k)
        if isinstance(v, Variable) and not isinstance(v, exclude_types):
          gather[f'{node_path}.{k}' if node_path else k] = v
        elif isinstance(v, VarList):
          for i, vv in enumerate(v):
            if not isinstance(vv, exclude_types):
              gather[f'{node_path}.{k}-{i}' if node_path else k] = vv
        elif isinstance(v, VarDict):
          for kk, vv in v.items():
            if not isinstance(vv, exclude_types):
              gather[f'{node_path}.{k}-{kk}' if node_path else k] = vv
      # implicit vars
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
          _add_node2(self, v, _paths, gather, nodes)
        elif isinstance(v, NodeList):
          for v2 in v:
            _add_node2(self, v2, _paths, gather, nodes)
        elif isinstance(v, NodeDict):
          for v2 in v.values():
            if isinstance(v2, BrainPyObject):
              _add_node2(self, v2, _paths, gather, nodes)

      # implicit nodes
      for node in self.implicit_nodes.values():
        _add_node2(self, node, _paths, gather, nodes)

      # finding nodes recursively
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
          _add_node1(self, k, v, _paths, gather, nodes)
        elif isinstance(v, NodeList):
          for i, v2 in enumerate(v):
            _add_node1(self, k + '-' + str(i), v2, _paths, gather, nodes)
        elif isinstance(v, NodeDict):
          for k2, v2 in v.items():
            if isinstance(v2, BrainPyObject):
              _add_node1(self, k + '.' + k2, v2, _paths, gather, nodes)

      # implicit nodes
      for key, node in self.implicit_nodes.items():
        _add_node1(self, key, node, _paths, gather, nodes)

      # finding nodes recursively
      for k1, v1 in nodes:
        for k2, v2 in v1._find_nodes(method=method,
                                     _paths=_paths,
                                     _lid=_lid + 1,
                                     level=level,
                                     include_self=include_self).items():
          if k2:
            gather[f'{k1}.{k2}'] = v2

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

  def save_state(self, **kwargs) -> Dict:
    """Save states as a dictionary. """
    return self.__save_state__(**kwargs)

  def load_state(self, state_dict: Dict, **kwargs) -> Optional[Tuple[Sequence[str], Sequence[str]]]:
    """Load states from a dictionary."""
    return self.__load_state__(state_dict, **kwargs)

  def __save_state__(self, **kwargs) -> Dict:
    """Save states. """
    return self.vars(include_self=True, level=0).unique().dict()

  def __load_state__(self, state_dict: Dict, **kwargs) -> Optional[Tuple[Sequence[str], Sequence[str]]]:
    """Load states from the external objects."""
    variables = self.vars(include_self=True, level=0).unique()
    keys1 = set(state_dict.keys())
    keys2 = set(variables.keys())
    for key in keys2.intersection(keys1):
      variables[key].value = jax.numpy.asarray(state_dict[key])
    unexpected_keys = list(keys1 - keys2)
    missing_keys = list(keys2 - keys1)
    return unexpected_keys, missing_keys

  def state_dict(self, **kwargs) -> dict:
    """Returns a dictionary containing a whole state of the module.

    Returns
    -------
    out: dict
      A dictionary containing a whole state of the module.
    """
    nodes = self.nodes()  # retrieve all nodes
    return {key: node.save_state(**kwargs) for key, node in nodes.items()}

  def load_state_dict(
      self,
      state_dict: Dict[str, Any],
      warn: bool = True,
      compatible: str = 'v2',
      **kwargs,
  ):
    """Copy parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Parameters
    ----------
    state_dict: dict
      A dict containing parameters and persistent buffers.
    warn: bool
      Warnings when there are missing keys or unexpected keys in the external ``state_dict``.
    compatible: bool
      The version of API for compatibility.

    Returns
    -------
    out: StateLoadResult
      ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:

      * **missing_keys** is a list of str containing the missing keys
      * **unexpected_keys** is a list of str containing the unexpected keys
    """
    if compatible == 'v1':
      variables = self.vars().unique()
      keys1 = set(state_dict.keys())
      keys2 = set(variables.keys())
      unexpected_keys = list(keys1 - keys2)
      missing_keys = list(keys2 - keys1)
      for key in keys2.intersection(keys1):
        variables[key].value = jax.numpy.asarray(state_dict[key])
    elif compatible == 'v2':
      nodes = self.nodes()
      missing_keys = []
      unexpected_keys = []
      for name, node in nodes.items():
        r = node.load_state(state_dict[name] if name in state_dict else {}, **kwargs)
        if r is not None:
          missing, unexpected = r
          missing_keys.extend([f'{name}.{key}' for key in missing])
          unexpected_keys.extend([f'{name}.{key}' for key in unexpected])
    else:
      raise ValueError(f'Unknown compatible version: {compatible}')
    if warn:
      if len(unexpected_keys):
        warnings.warn(f'Unexpected keys in state_dict: {unexpected_keys}', UserWarning)
      if len(missing_keys):
        warnings.warn(f'Missing keys in state_dict: {missing_keys}', UserWarning)
    return StateLoadResult(missing_keys, unexpected_keys)

  def to(self, device: Optional[Any]):
    """Moves all variables into the given device.

    Args:
      device: The device.
    """
    for key, var in self.state_dict().items():
      if isinstance(var, Array):
        var.value = jax.device_put(var.value, device=device)
      else:
        setattr(self, key, jax.device_put(var, device=device))
    return self

  def cpu(self):
    """Move all variable into the CPU device."""
    return self.to(device=jax.devices('cpu')[0])

  def cuda(self):
    """Move all variables into the GPU device."""
    return self.to(device=jax.devices('gpu')[0])

  def tpu(self):
    """Move all variables into the TPU device."""
    return self.to(device=jax.devices('tpu')[0])


def _add_node2(self, v, _paths, gather, nodes):
  path = (id(self), id(v))
  if path not in _paths:
    _paths.add(path)
    gather[v.name] = v
    nodes.append(v)


def _add_node1(self, k, v, _paths, gather, nodes):
  path = (id(self), id(v))
  if path not in _paths:
    _paths.add(path)
    gather[k] = v
    nodes.append((k, v))


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
    from brainpy._src.tools import repr_context
    name = self.__class__.__name__
    indent = " " * (len(name) + 1)
    indent2 = indent + " " * len('nodes=')
    nodes = [repr_context(str(n), indent2) for n in self.implicit_nodes.values()]
    node_string = ", \n".join(nodes)
    return (f'{name}(nodes=[{node_string}],\n' +
            " " * (len(name) + 1) + f'num_of_vars={len(self.implicit_vars)})')


def _check_elem(elem):
  if not isinstance(elem, (numbers.Number, jax.Array, Array, np.ndarray, BrainPyObject)):
    raise TypeError(f'Element should be a numerical value or a BrainPyObject.')
  return elem


def _check_num_elem(elem):
  if not isinstance(elem, (numbers.Number, jax.Array, Array, np.ndarray)):
    raise TypeError(f'Element should be a numerical value.')
  return elem


def _check_obj_elem(elem):
  if not isinstance(elem, BrainPyObject):
    raise TypeError(f'Element should be a {BrainPyObject.__class__.__name__}.')
  return elem


class ObjectTransform(BrainPyObject):
  """Object-oriented JAX transformation for BrainPy computation.
  """

  def __init__(self, name: str = None):
    super().__init__(name=name)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  def __repr__(self):
    return self.__class__.__name__


class NodeList(list):
  """A sequence of :py:class:`~.BrainPyObject`, which is compatible with
  :py:func:`~.vars()` and :py:func:`~.nodes()` operations in a :py:class:`~.BrainPyObject`.

  That is to say, any nodes that are wrapped into :py:class:`~.NodeList` will be automatically
  retieved when using :py:func:`~.nodes()` function.

  >>> import brainpy as bp
  >>> l = bm.node_list([bp.dnn.Dense(1, 2),
  >>>                   bp.dnn.LSTMCell(2, 3)])
  """

  def __init__(self, seq=()):
    super().__init__()
    self.extend(seq)

  def append(self, element) -> 'NodeList':
    # if not isinstance(element, BrainPyObject):
    #   raise TypeError(f'element must be an instance of {BrainPyObject.__name__}.')
    super().append(element)
    return self

  def extend(self, iterable) -> 'NodeList':
    for element in iterable:
      self.append(element)
    return self


node_list = NodeList


class NodeDict(dict):
  """A dictionary of :py:class:`~.BrainPyObject`, which is compatible with
  :py:func:`.vars()` operation in a :py:class:`~.BrainPyObject`.
  """

  # def _check_elem(self, elem):
  #   if not isinstance(elem, BrainPyObject):
  #     raise TypeError(f'Element should be {BrainPyObject.__name__}, but got {type(elem)}.')
  #   return elem

  def __init__(self, *args, check_unique: bool = False, **kwargs):
    super().__init__()
    self.check_unique = check_unique
    self.update(*args, **kwargs)

  def update(self, *args, **kwargs) -> 'NodeDict':
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.items():
          self[k] = v
      elif isinstance(arg, tuple):
        assert len(arg) == 2
        self[arg[0]] = args[1]
    for k, v in kwargs.items():
      self[k] = v
    return self

  def __setitem__(self, key, value) -> 'NodeDict':
    if self.check_unique:
      exist = self.get(key, None)
      if id(exist) != id(value):
        raise KeyError(f'Duplicate usage of key "{key}". "{key}" has been used for {value}.')
    super().__setitem__(key, value)
    return self


node_dict = NodeDict
