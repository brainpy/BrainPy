from contextlib import contextmanager
from typing import Optional, Any, List, Callable, Sequence, Union, Dict, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from brainpy._src.math.ndarray import Array
from brainpy._src.math.sharding import BATCH_AXIS
from brainpy.errors import MathError

__all__ = [
  'Variable',
  'TrainVar',
  'Parameter',
  'VariableView',

  'VarList', 'var_list',
  'VarDict', 'var_dict',
]


class VariableStack(dict):
  """Variable stack, for collecting all :py:class:`~.Variable` used in the program.

  :py:class:`~.VariableStack` supports all features of python dict.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._values = dict()

  def add(self, var: 'Variable'):
    """Add a new :py:class:`~.Variable`."""
    assert isinstance(var, Variable), f'must be instance of {Variable}'
    id_ = id(var)
    if id_ not in self:
      self[id_] = var
      self._values[id_] = var._value

  def collect_values(self):
    """Collect the value of each variable once again."""
    for id_, var in self.items():
      self._values[id_] = var._value

  def assign_org_values(self):
    """Assign the original value for each variable."""
    for id_, var in self.items():
      if id_ in self._values:
        var._value = self._values[id_]

  def assign(self, data: Union[Dict, Sequence], check: bool = True):
    """Assign the value for each :math:`~.Variable` according to the given ``data``.

    Args:
      data: dict, list, tuple. The data of all variables
      check: bool. Check whether the shape and type of the given data are consistent with original data.
    """
    if isinstance(data, dict):
      assert len(data) == len(self), 'Data length mismatch. '
      if check:
        for id_, elem in self.items():
          elem.value = data[id_]
      else:
        for id_, elem in self.items():
          elem._value = data[id_]
    elif isinstance(data, (tuple, list)):
      assert len(data) == len(self), 'Data length mismatch. '
      if check:
        for i, elem in enumerate(self.values()):
          elem.value = data[i]
      else:
        for i, elem in enumerate(self.values()):
          elem._value = data[i]
    else:
      raise TypeError

  def call_on_subset(self, cond: Callable, call: Callable) -> dict:
    """Call a function on the subset of this :py:class:`~VariableStack`.

    >>> import brainpy.math as bm
    >>> stack = VariableStack(a=bm.Variable(1), b=bm.random.RandomState(1))
    >>> stack.call_on_subset(lambda a: isinstance(a, bm.random.RandomState),
    >>>                      lambda a: a.split_key())
    {'b': Array([3819641963, 2025898573], dtype=uint32)}

    Args:
      cond: The function to determine whether the element belongs to the wanted subset.
      call: The function to call if the element belongs to the wanted subset.

    Returns:
      A dict containing the results of ``call`` function for each element in the ``cond`` constrained subset.
    """
    res = dict()
    for id_, elem in self.items():
      if cond(elem):
        res[id_] = call(elem)
    return res

  def separate_by_instance(self, cls: type) -> Tuple['VariableStack', 'VariableStack']:
    """Separate all variables into two groups: (variables that are instances of the given ``cls``,
    variables that are not instances of the given ``cls``).

    >>> import brainpy.math as bm
    >>> stack = VariableStack(a=bm.Variable(1), b=bm.random.RandomState(1))
    >>> stack.separate_by_instance(bm.random.RandomState)
    ({'b': RandomState(key=([0, 1], dtype=uint32))},
     {'a': Variable(value=Array([0.]), dtype=float32)})
    >>> stack.separate_by_instance(bm.Variable)
    ({'a': Variable(value=Array([0.]), dtype=float32),
      'b': RandomState(key=([0, 1], dtype=uint32))},
     {})

    Args:
      cls: The class type.

    Returns:
      A tuple with two elements:

      - VariableStack of variables that are instances of the given ``cls``
      - VariableStack of variables that are not instances of the given ``cls``
    """
    is_instances = type(self)()
    not_instances = type(self)()
    for id_, elem in self.items():
      if isinstance(elem, cls):
        is_instances[id_] = elem
      else:
        not_instances[id_] = elem
    return is_instances, not_instances

  def subset_by_instance(self, cls: type) -> 'VariableStack':
    """Collect all variables which are instances of the given class type."""
    new_dict = type(self)()
    for id_, elem in self.items():
      if isinstance(elem, cls):
        new_dict[id_] = elem
    return new_dict

  def subset_by_not_instance(self, cls: type) -> 'VariableStack':
    """Collect all variables which are not instance of the given class type."""
    new_dict = type(self)()
    for id_, elem in self.items():
      if not isinstance(elem, cls):
        new_dict[id_] = elem
    return new_dict

  instance_of = subset_by_instance
  not_instance_of = subset_by_not_instance

  def dict_data_of_subset(self, subset_cond: Callable) -> dict:
    """Get data of the given subset constrained by function ``subset_cond``.

    Args:
      subset_cond: A function to determine whether the element is in the subset wanted.

    Returns:
      A dict of data for elements of the wanted subset.
    """
    res = dict()
    for id_, elem in self.items():
      if subset_cond(elem):
        res[id_] = elem.value
    return res

  def dict_data(self) -> dict:
    """Get all data in the collected variables with a python dict structure."""
    new_dict = dict()
    for id_, elem in tuple(self.items()):
      new_dict[id_] = elem.value
    return new_dict

  def list_data(self) -> list:
    """Get all data in the collected variables with a python list structure."""
    new_list = list()
    for elem in tuple(self.values()):
      new_list.append(elem.value if isinstance(elem, Array) else elem)
    return new_list

  def remove_by_id(self, *ids, error_when_absent=False):
    """Remove or pop variables in the stack by the given ids."""
    if error_when_absent:
      for id_ in ids:
        self.pop(id_)
    else:
      for id_ in ids:
        self.pop(id_, None)

  remove_var_by_id = remove_by_id

  def __enter__(self) -> 'VariableStack':
    self.collect_values()  # recollect the original value of each variable
    var_stack_list.append(self)
    return self

  def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
    var_stack_list.pop()
    self.assign_org_values()  # reassign the original value for each variable
    self._values.clear()

  def __add__(self, other: dict):
    new_dict = VariableStack(self)
    new_dict.update(other)
    new_dict._values.update(self._values)
    if isinstance(other, VariableStack):
      new_dict._values.update(other._values)
    return new_dict


var_stack_list: List[VariableStack] = []
transform_stack: List[Callable] = []


@contextmanager
def new_transform(transform: Any):
  transform_stack.append(transform)
  try:
    yield
  finally:
    transform_stack.pop()


def outermost_stack():
  if len(var_stack_list):
    return var_stack_list[0]
  else:
    return None


def outermost_transform():
  if len(transform_stack):
    return transform_stack[0]
  else:
    return None


def current_transform_number():
  return len(transform_stack)


def _stack_add_read(var: 'Variable'):
  pass


def _stack_add_write(var: 'Variable'):
  pass


@register_pytree_node_class
class Variable(Array):
  """The pointer to specify the dynamical variable.

  Initializing an instance of ``Variable`` by two ways:

  >>> import brainpy.math as bm
  >>> # 1. init a Variable by the concreate data
  >>> v1 = bm.Variable(bm.zeros(10))
  >>> # 2. init a Variable by the data shape
  >>> v2 = bm.Variable(10)

  Note that when initializing a `Variable` by the data shape,
  all values in this `Variable` will be initialized as zeros.

  Args:
    value_or_size: Shape, Array, int. The value or the size of the value.
    dtype: Any. The type of the data.
    batch_axis: optional, int. The batch axis.
    axis_names: sequence of str. The name for each axis.
  """

  __slots__ = ('_value', '_batch_axis', 'ready_to_trace', 'axis_names')

  def __init__(
      self,
      value_or_size: Any,
      dtype: type = None,
      batch_axis: int = None,
      *,
      axis_names: Optional[Sequence[str]] = None,
      ready_to_trace: bool = None
  ):
    if isinstance(value_or_size, int):
      value = jnp.zeros(value_or_size, dtype=dtype)
    elif isinstance(value_or_size, (tuple, list)) and all([isinstance(s, int) for s in value_or_size]):
      value = jnp.zeros(value_or_size, dtype=dtype)
    else:
      value = value_or_size

    super().__init__(value, dtype=dtype)

    # check batch axis
    if isinstance(value, Variable):
      if value.batch_axis is not None and batch_axis is not None:
        if batch_axis != value.batch_axis:
          raise ValueError(f'"batch_axis" is not consistent. Got batch_axis in the given value '
                           f'is {value.batch_axis}, but the specified batch_axis is {batch_axis}')
      batch_axis = value.batch_axis

    # assign batch axis
    self._batch_axis = batch_axis
    if batch_axis is not None:
      if batch_axis >= np.ndim(self._value):
        raise MathError(f'This variables has {np.ndim(self._value)} dimension, '
                        f'but the batch axis is set to be {batch_axis}.')

    # ready to trace the variable
    if ready_to_trace is None:
      if len(var_stack_list) == 0:
        self.ready_to_trace = True
      else:
        self.ready_to_trace = False
    else:
      self.ready_to_trace = ready_to_trace
    if axis_names is not None:
      if len(axis_names) + 1 == self.ndim:
        axis_names = list(axis_names)
        axis_names.insert(self.batch_axis, BATCH_AXIS)
      assert len(axis_names) == self.ndim
      axis_names = tuple(axis_names)
    self.axis_names = axis_names

  @property
  def size_without_batch(self):
    if self.batch_axis is None:
      return self.size
    else:
      sizes = self.size
      return sizes[:self.batch_size] + sizes[self.batch_axis + 1:]

  @property
  def batch_axis(self) -> Optional[int]:
    return self._batch_axis

  @batch_axis.setter
  def batch_axis(self, val):
    raise ValueError(f'Cannot set "batch_axis" after creating a {self.__class__.__name__} instance.')

  @property
  def batch_size(self) -> Optional[int]:
    if self.batch_axis is None:
      return None
    else:
      return self.shape[self.batch_axis]

  @batch_size.setter
  def batch_size(self, val):
    raise ValueError(f'Cannot set "batch_size" manually.')

  @property
  def value(self):
    self._append_to_stack()
    return self._value

  @value.setter
  def value(self, v):
    _value = self.value
    ext_shape = jnp.shape(v)
    int_shape = jnp.shape(_value)
    if self._batch_axis is not None:
      ext_shape = ext_shape[:self._batch_axis] + ext_shape[self._batch_axis + 1:]
      int_shape = int_shape[:self._batch_axis] + int_shape[self._batch_axis + 1:]
    if ext_shape != int_shape:
      error = f"The shape of the original data is {int_shape}, while we got {ext_shape}"
      error += f' with batch_axis={self._batch_axis}.'
      raise MathError(error)
    ext_dtype = _get_dtype(v)
    int_dtype = self.dtype
    if ext_dtype != int_dtype:
      raise MathError(f"The dtype of the original data is {int_dtype}, "
                      f"while we got {ext_dtype}.")
    self._append_to_stack()
    if isinstance(v, Array):
      v = v.value
    elif isinstance(v, np.ndarray):
      v = jnp.asarray(v)
    else:
      v = v
    self._value = v

  def _append_to_stack(self):
    if self.ready_to_trace:
      for stack in var_stack_list:
        stack.add(self)

  def tree_flatten(self):
    """Flattens this variable.

    Returns:
      A pair where the first element is a list of leaf values
      and the second element is a treedef representing the
      structure of the flattened tree.
    """
    return (self._value,), None

  @classmethod
  def tree_unflatten(cls, aux_data, flat_contents):
    """Reconstructs a variable from the aux_data and the leaves.

    Args:
      aux_data:
      flat_contents:

    Returns:
      The variable.
    """
    return cls(*flat_contents, ready_to_trace=False)

  def clone(self) -> 'Variable':
    """Clone the variable. """
    r = type(self)(jnp.array(self.value, copy=True), batch_axis=self.batch_axis)
    r.ready_to_trace = self.ready_to_trace
    return r


def _get_dtype(v):
  if hasattr(v, 'dtype'):
    dtype = v.dtype
  else:
    dtype = canonicalize_dtype(type(v))
  return dtype


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


@register_pytree_node_class
class TrainVar(Variable):
  """The pointer to specify the trainable variable.
  """

  def __init__(
      self,
      value_or_size: Any,
      dtype: type = None,
      batch_axis: int = None,
      *,
      axis_names: Optional[Sequence[str]] = None,
      ready_to_trace: bool = True
  ):
    super().__init__(
      value_or_size,
      dtype=dtype,
      batch_axis=batch_axis,
      ready_to_trace=ready_to_trace,
      axis_names=axis_names,
    )


@register_pytree_node_class
class Parameter(Variable):
  """The pointer to specify the parameter.
  """

  def __init__(
      self,
      value_or_size: Any,
      dtype: type = None,
      batch_axis: int = None,
      *,
      axis_names: Optional[Sequence[str]] = None,
      ready_to_trace: bool = True
  ):
    super().__init__(
      value_or_size,
      dtype=dtype,
      batch_axis=batch_axis,
      ready_to_trace=ready_to_trace,
      axis_names=axis_names,
    )


class VariableView(Variable):
  """A view of a Variable instance.

  This class is used to create a subset view of ``brainpy.math.Variable``.

  >>> import brainpy.math as bm
  >>> bm.random.seed(123)
  >>> origin = bm.Variable(bm.random.random(5))
  >>> view = bm.VariableView(origin, slice(None, 2, None))  # origin[:2]
  VariableView([0.02920651, 0.19066381], dtype=float32)

  ``VariableView`` can be used to update the subset of the original
  Variable instance, and make operations on this subset of the Variable.

  >>> view[:] = 1.
  >>> view
  VariableView([1., 1.], dtype=float32)
  >>> origin
  Variable([1.       , 1.       , 0.5482849, 0.6564884, 0.8446237], dtype=float32)
  >>> view + 10
  Array([11., 11.], dtype=float32)
  >>> view *= 10
  VariableView([10., 10.], dtype=float32)

  The above example demonstrates that the updating of an ``VariableView`` instance
  is actually made in the original ``Variable`` instance.

  Moreover, it's worthy to note that ``VariableView`` is not a PyTree.
  """
  _need_record = False

  def __init__(
      self,
      value: Variable,
      index: Any,
  ):
    self.index = jax.tree_util.tree_map(_as_jax_array_, index, is_leaf=lambda a: isinstance(a, Array))
    if not isinstance(value, Variable):
      raise ValueError('Must be instance of Variable.')
    super().__init__(value.value, batch_axis=value.batch_axis, ready_to_trace=False)
    self._value = value

  def __repr__(self) -> str:
    print_code = repr(self._value)
    prefix = f'{self.__class__.__name__}'
    blank = " " * (len(prefix) + 1)
    lines = print_code.split("\n")
    lines[0] = prefix + "(" + lines[0]
    for i in range(1, len(lines)):
      lines[i] = blank + lines[i]
    lines[-1] += ","
    lines.append(blank + f'index={self.index})')
    print_code = "\n".join(lines)
    return print_code

  @property
  def value(self):
    return self._value[self.index]

  @value.setter
  def value(self, v):
    int_shape = self.shape
    if self.batch_axis is None:
      ext_shape = v.shape
    else:
      ext_shape = v.shape[:self.batch_axis] + v.shape[self.batch_axis + 1:]
      int_shape = int_shape[:self.batch_axis] + int_shape[self.batch_axis + 1:]
    if ext_shape != int_shape:
      error = f"The shape of the original data is {self.shape}, while we got {v.shape}"
      if self.batch_axis is None:
        error += '. Do you forget to set "batch_axis" when initialize this variable?'
      else:
        error += f' with batch_axis={self.batch_axis}.'
      raise MathError(error)
    if v.dtype != self._value.dtype:
      raise MathError(f"The dtype of the original data is {self._value.dtype}, "
                      f"while we got {v.dtype}.")
    self._value[self.index] = v.value if isinstance(v, Array) else v


@register_pytree_node_class
class VarList(list):
  """A sequence of :py:class:`~.Variable`, which is compatible with
  :py:func:`.vars()` operation in a :py:class:`~.BrainPyObject`.

  Actually, :py:class:`~.VarList` is a python list.

  :py:class:`~.VarList` is specifically designed to store Variable instances.

  """

  def __init__(self, seq=()):
    super().__init__()
    self.extend(seq)

  def append(self, element) -> 'VarList':
    if not isinstance(element, Variable):
      raise TypeError(f'element must be an instance of {Variable.__name__}.')
    super().append(element)
    return self

  def extend(self, iterable) -> 'VarList':
    for element in iterable:
      self.append(element)
    return self

  def __setitem__(self, key, value) -> 'VarList':
    """Override the item setting.

    This function ensures that the Variable appended in the :py:class:`~.VarList` will not be overridden,
    and only the value can be changed for each element.

    >>> import brainpy.math as bm
    >>> l = bm.var_list([bm.Variable(1), bm.Variable(2)])
    >>> print(id(l[0]), id(l[1]))
    2077748389472 2077748389552
    >>> l[1] = bm.random.random(2)
    >>> l[0] = bm.random.random(1)
    >>> print(id(l[0]), id(l[1]))  # still the original Variable instances
    2077748389472 2077748389552
    """
    if isinstance(key, int):
      self[key].value = value
    else:
      super().__setitem__(key, value)
    return self

  def tree_flatten(self):
    return tuple(self), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(children)


var_list = VarList


@register_pytree_node_class
class VarDict(dict):
  """A dictionary of :py:class:`~.Variable`, which is compatible with
  :py:func:`.vars()` operation in a :py:class:`~.BrainPyObject`.

  Actually, :py:class:`~.VarDict` is a python dict.

  :py:class:`~.VarDict` is specifically designed to store Variable instances.

  """

  def _check_elem(self, elem):
    if not isinstance(elem, Variable):
      raise TypeError(f'Element should be {Variable.__name__}, but got {type(elem)}.')
    return elem

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.update(*args, **kwargs)

  def update(self, *args, **kwargs) -> 'VarDict':
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

  def __setitem__(self, key, value) -> 'VarDict':
    """Override the item setting.

    This function ensures that the Variable appended in the :py:class:`~.VarList` will not be overridden.

    >>> import brainpy.math as bm
    >>> d = bm.var_dict({'a': bm.Variable(1), 'b': bm.Variable(2)})
    >>> print(id(d['a']), id(d['b']))
    2077667833504 2077748488176
    >>> d['b'] = bm.random.random(2)
    >>> d['a'] = bm.random.random(1)
    >>> print(id(d['a']), id(d['b']))  # still the original Variable instances
    2077667833504 2077748488176
    """
    if key in self:
      self[key].value = value
    else:
      super().__setitem__(key, self._check_elem(value))
    return self

  def tree_flatten(self):
    return tuple(self.values()), tuple(self.keys())

  @classmethod
  def tree_unflatten(cls, keys, values):
    return cls(jax.util.safe_zip(keys, values))


var_dict = VarDict
