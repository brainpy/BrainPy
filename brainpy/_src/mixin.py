import numbers
import sys
from dataclasses import dataclass
from typing import Union, Dict, Callable, Sequence, Optional, TypeVar
from typing import (_SpecialForm, _type_check, _remove_dups_flatten)

import jax
import jax.numpy as jnp
import numpy as np

from brainpy import math as bm, tools
from brainpy._src.math.object_transform.naming import get_unique_name
from brainpy._src.initialize import parameter
from brainpy.types import ArrayType

if sys.version_info.minor > 8:
  from typing import (_UnionGenericAlias)
else:
  from typing import (_GenericAlias, _tp_cache)

DynamicalSystem = None

__all__ = [
  'MixIn',
  'ParamDesc',
  'ParamDescInit',
  'AlignPost',
  'AutoDelaySupp',
  'NoSH',
  'Container',
  'TreeNode',
  'BindCondData',
  'JointType',
]

global_delay_data = dict()


class MixIn(object):
  """Base MixIn object."""
  pass


class ParamDesc(MixIn):
  """:py:class:`~.MixIn` indicates the function for describing initialization parameters.

  This mixin enables the subclass has a classmethod ``desc``, which
  produces an instance of :py:class:`~.ParamDescInit`.

  Note this MixIn can be applied in any Python object.
  """

  not_desc_params: Optional[Sequence[str]] = None

  @classmethod
  def desc(cls, *args, **kwargs) -> 'ParamDescInit':
    return ParamDescInit(cls, *args, **kwargs)


class ParamDescInit(object):
  """Delayed initialization for parameter describers.
  """

  def __init__(self, cls: type, *desc_tuple, **desc_dict):
    self.cls = cls

    # arguments
    self.args = desc_tuple
    self.kwargs = desc_dict

    # identifier
    if isinstance(cls, _JointGenericAlias):
      name = str(cls)
      repr_kwargs = {k: v for k, v in desc_dict.items()}
    else:
      assert isinstance(cls, type)
      if issubclass(cls, ParamDesc) and (cls.not_desc_params is not None):
        repr_kwargs = {k: v for k, v in desc_dict.items() if k not in cls.not_desc_params}
      else:
        repr_kwargs = {k: v for k, v in desc_dict.items()}
      name = cls.__name__
    for k in tuple(repr_kwargs.keys()):
      if isinstance(repr_kwargs[k], bm.Variable):
        repr_kwargs[k] = id(repr_kwargs[k])
    repr_args = tools.repr_dict(repr_kwargs)
    if len(desc_tuple):
      repr_args = f"{', '.join([repr(arg) for arg in desc_tuple])}, {repr_args}"
    self._identifier = f'{name}({repr_args})'

  def __call__(self, *args, **kwargs):
    return self.cls(*self.args, *args, **self.kwargs, **kwargs)

  def init(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  def __instancecheck__(self, instance):
    if not isinstance(instance, ParamDescInit):
      return False
    if not issubclass(instance.cls, self.cls):
      return False
    return True

  @classmethod
  def __class_getitem__(cls, item: type):
    return ParamDescInit(item)

  @property
  def identifier(self):
    return self._identifier

  @identifier.setter
  def identifier(self, value):
    self._identifier = value


class AlignPost(MixIn):
  """Align post MixIn.

  This class provides a ``add_current()`` function for
  add external currents.
  """

  def add_current(self, *args, **kwargs):
    raise NotImplementedError


@dataclass
class ReturnInfo:
  size: Sequence[int]
  axis_names: Optional[Sequence[str]] = None
  batch_or_mode: Optional[Union[int, bm.Mode]] = None
  data: Union[Callable, bm.Array, jax.Array] = bm.zeros

  def get_data(self):
    if isinstance(self.data, Callable):
      if isinstance(self.batch_or_mode, int):
        size = (self.batch_or_mode,) + tuple(self.size)
      elif isinstance(self.batch_or_mode, bm.NonBatchingMode):
        size = tuple(self.size)
      elif isinstance(self.batch_or_mode, bm.BatchingMode):
        size = (self.batch_or_mode.batch_size,) + tuple(self.size)
      else:
        size = tuple(self.size)
      init = self.data(size)
    elif isinstance(self.data, (bm.Array, jax.Array)):
      init = self.data
    else:
      raise ValueError
    return init


class AutoDelaySupp(MixIn):
  """``MixIn`` to support the automatic delay in synaptic projection :py:class:`~.SynProj`."""

  def return_info(self) -> Union[bm.Variable, ReturnInfo]:
    raise NotImplementedError('Must implement the "return_info()" function.')


class NoSH(MixIn):
  """``MixIn`` to indicate that no shared parameters should be passed into the ``update()`` function."""

  def __init__(self, *args, **kwargs):
    self._pass_shared_args = False


class Container(MixIn):
  """Container :py:class:`~.MixIn` which wrap a group of objects.
  """
  children: bm.node_dict

  def __getitem__(self, item):
    """Overwrite the slice access (`self['']`). """
    if item in self.children:
      return self.children[item]
    else:
      raise ValueError(f'Unknown item {item}, we only found {list(self.children.keys())}')

  def __getattr__(self, item):
    """Overwrite the dot access (`self.`). """
    if item == 'children':
      return super().__getattribute__('children')
    else:
      children = super().__getattribute__('children')
      if item in children:
        return children[item]
      else:
        return super().__getattribute__(item)

  def __repr__(self):
    cls_name = self.__class__.__name__
    indent = ' ' * len(cls_name)
    child_str = [tools.repr_context(repr(val), indent) for val in self.children.values()]
    string = ", \n".join(child_str)
    return f'{cls_name}({string})'

  def __get_elem_name(self, elem):
    if isinstance(elem, bm.BrainPyObject):
      return elem.name
    else:
      return get_unique_name('ContainerElem')

  def format_elements(self, child_type: type, *children_as_tuple, **children_as_dict):
    res = dict()

    # add tuple-typed components
    for module in children_as_tuple:
      if isinstance(module, child_type):
        res[self.__get_elem_name(module)] = module
      elif isinstance(module, (list, tuple)):
        for m in module:
          if not isinstance(m, child_type):
            raise ValueError(f'Should be instance of {child_type.__name__}. '
                             f'But we got {type(m)}')
          res[self.__get_elem_name(m)] = m
      elif isinstance(module, dict):
        for k, v in module.items():
          if not isinstance(v, child_type):
            raise ValueError(f'Should be instance of {child_type.__name__}. '
                             f'But we got {type(v)}')
          res[k] = v
      else:
        raise ValueError(f'Cannot parse sub-systems. They should be {child_type.__name__} '
                         f'or a list/tuple/dict of {child_type.__name__}.')
    # add dict-typed components
    for k, v in children_as_dict.items():
      if not isinstance(v, child_type):
        raise ValueError(f'Should be instance of {child_type.__name__}. '
                         f'But we got {type(v)}')
      res[k] = v
    return res

  def add_elem(self, **elements):
    """Add new elements.

    >>> obj = Container()
    >>> obj.add_elem(a=1.)

    Args:
      elements: children objects.
    """
    # self.check_hierarchies(type(self), **elements)
    self.children.update(self.format_elements(object, **elements))


class TreeNode(MixIn):
  """Tree node. """

  master_type: type

  def check_hierarchies(self, root, *leaves, **named_leaves):
    global DynamicalSystem
    if DynamicalSystem is None:
      from brainpy._src.dynsys import DynamicalSystem

    for leaf in leaves:
      if isinstance(leaf, DynamicalSystem):
        self.check_hierarchy(root, leaf)
      elif isinstance(leaf, (list, tuple)):
        self.check_hierarchies(root, *leaf)
      elif isinstance(leaf, dict):
        self.check_hierarchies(root, **leaf)
      else:
        raise ValueError(f'Do not support {type(leaf)}.')
    for leaf in named_leaves.values():
      if not isinstance(leaf, DynamicalSystem):
        raise ValueError(f'Do not support {type(leaf)}. Must be instance of {DynamicalSystem.__name__}')
      self.check_hierarchy(root, leaf)

  def check_hierarchy(self, root, leaf):
    if hasattr(leaf, 'master_type'):
      master_type = leaf.master_type
    else:
      raise ValueError('Child class should define "master_type" to '
                       'specify the type of the root node. '
                       f'But we did not found it in {leaf}')
    if not issubclass(root, master_type):
      raise TypeError(f'Type does not match. {leaf} requires a master with type '
                      f'of {leaf.master_type}, but the master now is {root}.')


class DelayRegister(MixIn):
  local_delay_vars: bm.node_dict

  def register_delay(
      self,
      identifier: str,
      delay_step: Optional[Union[int, ArrayType, Callable]],
      delay_target: bm.Variable,
      initial_delay_data: Union[Callable, ArrayType, numbers.Number] = None,
  ):
    """Register delay variable.

    Parameters
    ----------
    identifier: str
      The delay variable name.
    delay_step: Optional, int, ArrayType, callable, Initializer
      The number of the steps of the delay.
    delay_target: Variable
      The target variable for delay.
    initial_delay_data: float, int, ArrayType, callable, Initializer
      The initializer for the delay data.

    Returns
    -------
    delay_step: int, ArrayType
      The number of the delay steps.
    """
    # delay steps
    if delay_step is None:
      delay_type = 'none'
    elif isinstance(delay_step, (int, np.integer, jnp.integer)):
      delay_type = 'homo'
    elif isinstance(delay_step, (bm.ndarray, jnp.ndarray, np.ndarray)):
      if delay_step.size == 1 and delay_step.ndim == 0:
        delay_type = 'homo'
      else:
        delay_type = 'heter'
        delay_step = bm.asarray(delay_step)
    elif callable(delay_step):
      delay_step = parameter(delay_step, delay_target.shape, allow_none=False)
      delay_type = 'heter'
    else:
      raise ValueError(f'Unknown "delay_steps" type {type(delay_step)}, only support '
                       f'integer, array of integers, callable function, brainpy.init.Initializer.')
    if delay_type == 'heter':
      if delay_step.dtype not in [bm.int32, bm.int64]:
        raise ValueError('Only support delay steps of int32, int64. If your '
                         'provide delay time length, please divide the "dt" '
                         'then provide us the number of delay steps.')
      if delay_target.shape[0] != delay_step.shape[0]:
        raise ValueError(f'Shape is mismatched: {delay_target.shape[0]} != {delay_step.shape[0]}')
    if delay_type != 'none':
      max_delay_step = int(bm.max(delay_step))

    # delay target
    if delay_type != 'none':
      if not isinstance(delay_target, bm.Variable):
        raise ValueError(f'"delay_target" must be an instance of Variable, but we got {type(delay_target)}')

    # delay variable
    # TODO
    if delay_type != 'none':
      if identifier not in global_delay_data:
        delay = bm.LengthDelay(delay_target, max_delay_step, initial_delay_data)
        global_delay_data[identifier] = (delay, delay_target)
        self.local_delay_vars[identifier] = delay
      else:
        delay = global_delay_data[identifier][0]
        if delay is None:
          delay = bm.LengthDelay(delay_target, max_delay_step, initial_delay_data)
          global_delay_data[identifier] = (delay, delay_target)
          self.local_delay_vars[identifier] = delay
        elif delay.num_delay_step - 1 < max_delay_step:
          global_delay_data[identifier][0].reset(delay_target, max_delay_step, initial_delay_data)
    else:
      if identifier not in global_delay_data:
        global_delay_data[identifier] = (None, delay_target)
    return delay_step

  def get_delay_data(
      self,
      identifier: str,
      delay_step: Optional[Union[int, bm.Array, jax.Array]],
      *indices: Union[int, slice, bm.Array, jax.Array],
  ):
    """Get delay data according to the provided delay steps.

    Parameters
    ----------
    identifier: str
      The delay variable name.
    delay_step: Optional, int, ArrayType
      The delay length.
    indices: optional, int, slice, ArrayType
      The indices of the delay.

    Returns
    -------
    delay_data: ArrayType
      The delay data at the given time.
    """
    if delay_step is None:
      return global_delay_data[identifier][1].value

    if identifier in global_delay_data:
      if bm.ndim(delay_step) == 0:
        return global_delay_data[identifier][0](delay_step, *indices)
      else:
        if len(indices) == 0:
          indices = (bm.arange(delay_step.size),)
        return global_delay_data[identifier][0](delay_step, *indices)

    elif identifier in self.local_delay_vars:
      if bm.ndim(delay_step) == 0:
        return self.local_delay_vars[identifier](delay_step)
      else:
        if len(indices) == 0:
          indices = (bm.arange(delay_step.size),)
        return self.local_delay_vars[identifier](delay_step, *indices)

    else:
      raise ValueError(f'{identifier} is not defined in delay variables.')

  def update_local_delays(self, nodes: Union[Sequence, Dict] = None):
    """Update local delay variables.

    This function should be called after updating neuron groups or delay sources.
    For example, in a network model,


    Parameters
    ----------
    nodes: sequence, dict
      The nodes to update their delay variables.
    """
    global DynamicalSystem
    if DynamicalSystem is None:
      from brainpy._src.dynsys import DynamicalSystem

    # update delays
    if nodes is None:
      nodes = tuple(self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().values())
    elif isinstance(nodes, dict):
      nodes = tuple(nodes.values())
    if not isinstance(nodes, (tuple, list)):
      nodes = (nodes,)
    for node in nodes:
      for name in node.local_delay_vars:
        delay = global_delay_data[name][0]
        target = global_delay_data[name][1]
        delay.update(target.value)

  def reset_local_delays(self, nodes: Union[Sequence, Dict] = None):
    """Reset local delay variables.

    Parameters
    ----------
    nodes: sequence, dict
      The nodes to Reset their delay variables.
    """
    global DynamicalSystem
    if DynamicalSystem is None:
      from brainpy._src.dynsys import DynamicalSystem

    # reset delays
    if nodes is None:
      nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().values()
    elif isinstance(nodes, dict):
      nodes = nodes.values()
    for node in nodes:
      for name in node.local_delay_vars:
        delay = global_delay_data[name][0]
        target = global_delay_data[name][1]
        delay.reset(target.value)

  def get_delay_var(self, name):
    return global_delay_data[name]


class BindCondData(MixIn):
  """Bind temporary conductance data.
  """

  def __init__(self, *args, **kwargs):
    self._conductance = None

  def bind_cond(self, conductance):
    self._conductance = conductance

  def unbind_cond(self):
    self._conductance = None


T = TypeVar('T')


def get_type(types):
  class NewType(type):
    def __instancecheck__(self, other):
      cls_of_other = other.__class__
      return all([issubclass(cls_of_other, cls) for cls in types])

  return NewType


class _MetaUnionType(type):
  def __new__(cls, name, bases, dct):
    if isinstance(bases, type):
      bases = (bases,)
    elif isinstance(bases, (list, tuple)):
      bases = tuple(bases)
      for base in bases:
        assert isinstance(base, type), f'Must be type. But got {base}'
    else:
      raise TypeError(f'Must be type. But got {bases}')
    return super().__new__(cls, name, bases, dct)

  def __instancecheck__(self, other):
    cls_of_other = other.__class__
    return all([issubclass(cls_of_other, cls) for cls in self.__bases__])

  def __subclasscheck__(self, subclass):
    return all([issubclass(subclass, cls) for cls in self.__bases__])


class UnionType2(MixIn):
  """Union type for multiple types.

  >>> import brainpy as bp
  >>>
  >>> isinstance(bp.dyn.Expon(1), JointType[bp.DynamicalSystem, bp.mixin.ParamDesc, bp.mixin.AutoDelaySupp])
  """

  @classmethod
  def __class_getitem__(cls, types: Union[type, Sequence[type]]) -> type:
    return _MetaUnionType('UnionType', types, {})


if sys.version_info.minor > 8:
  class _JointGenericAlias(_UnionGenericAlias, _root=True):
    def __subclasscheck__(self, subclass):
      return all([issubclass(subclass, cls) for cls in set(self.__args__)])


  @_SpecialForm
  def JointType(self, parameters):
    """Joint type; JointType[X, Y] means both X and Y.

    To define a union, use e.g. Union[int, str].

    Details:
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by `type(None)`.
    - Unions of unions are flattened, e.g.::

        JointType[JointType[int, str], float] == JointType[int, str, float]

    - Unions of a single argument vanish, e.g.::

        JointType[int] == int  # The constructor actually returns int

    - Redundant arguments are skipped, e.g.::

        JointType[int, str, int] == JointType[int, str]

    - When comparing unions, the argument order is ignored, e.g.::

        JointType[int, str] == JointType[str, int]

    - You cannot subclass or instantiate a union.
    - You can use Optional[X] as a shorthand for JointType[X, None].
    """
    if parameters == ():
      raise TypeError("Cannot take a Joint of no types.")
    if not isinstance(parameters, tuple):
      parameters = (parameters,)
    msg = "JointType[arg, ...]: each arg must be a type."
    parameters = tuple(_type_check(p, msg) for p in parameters)
    parameters = _remove_dups_flatten(parameters)
    if len(parameters) == 1:
      return parameters[0]
    return _JointGenericAlias(self, parameters)

else:
  class _JointGenericAlias(_GenericAlias, _root=True):
    def __subclasscheck__(self, subclass):
      return all([issubclass(subclass, cls) for cls in set(self.__args__)])


  class _SpecialForm2(_SpecialForm, _root=True):
    @_tp_cache
    def __getitem__(self, parameters):
      if self._name == 'JointType':
        if parameters == ():
          raise TypeError("Cannot take a Joint of no types.")
        if not isinstance(parameters, tuple):
          parameters = (parameters,)
        msg = "JointType[arg, ...]: each arg must be a type."
        parameters = tuple(_type_check(p, msg) for p in parameters)
        parameters = _remove_dups_flatten(parameters)
        if len(parameters) == 1:
          return parameters[0]
        return _JointGenericAlias(self, parameters)
      else:
        return super().__getitem__(parameters)


  JointType = _SpecialForm2(
    'JointType',
    doc="""Joint type; JointType[X, Y] means both X and Y.
  
    To define a union, use e.g. JointType[int, str].  
    
    Details:
    
    - The arguments must be types and there must be at least one.
    - None as an argument is a special case and is replaced by `type(None)`.
    - Unions of unions are flattened, e.g.::
  
        JointType[JointType[int, str], float] == JointType[int, str, float]
  
    - Unions of a single argument vanish, e.g.::
  
        JointType[int] == int  # The constructor actually returns int
  
    - Redundant arguments are skipped, e.g.::
  
        JointType[int, str, int] == JointType[int, str]
  
    - When comparing unions, the argument order is ignored, e.g.::
  
        JointType[int, str] == JointType[str, int]
  
    - You cannot subclass or instantiate a union.
    - You can use Optional[X] as a shorthand for JointType[X, None].
    """
  )
