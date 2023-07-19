# -*- coding: utf-8 -*-

from functools import wraps, partial
from typing import Union, Sequence, Dict, Callable, Tuple, Type, Optional, Any

import jax
import numpy as np
import numpy as onp
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap
from jax.lax import cond

conn = None
init = None
var_obs = None

Array = None
BrainPyObject = None

__all__ = [
  'is_checking',
  'turn_on',
  'turn_off',

  'is_shape_consistency',
  'is_shape_broadcastable',
  'check_shape_except_batch',
  'check_shape',
  'is_dict_data',
  'is_callable',
  'is_initializer',
  'is_connector',
  'is_float',
  'is_integer',
  'is_string',
  'is_sequence',
  'is_subclass',
  'is_instance',
  'is_elem_or_seq_or_dict',
  'is_all_vars',
  'is_all_objs',
  'jit_error',
  'jit_error_checking',
  'jit_error2',

  'serialize_kwargs',
]

_check = True
_name_check = True


def is_checking():
  """Whether the checking is turn on."""
  return _check


def turn_on():
  """Turn on the checking."""
  global _check
  _check = True


def turn_off():
  """Turn off the checking."""
  global _check
  _check = False


# def turn_off_name_check

def is_shape_consistency(shapes, free_axes=None, return_format_shapes=False):
  assert isinstance(shapes, (tuple, list)), f'Must be a sequence of shape. While we got {shapes}.'
  for shape in shapes:
    assert isinstance(shapes, (tuple, list)), (f'Must be a sequence of shape. While '
                                               f'we got one element is {shape}.')
  dims = onp.unique([len(shape) for shape in shapes])
  if len(dims) > 1:
    raise ValueError(f'The provided shape dimensions are not consistent. ')
  if free_axes is None:
    type_ = 'none'
    free_axes = ()
  elif isinstance(free_axes, (tuple, list)):
    type_ = 'seq'
    free_axes = tuple(free_axes)
  elif isinstance(free_axes, int):
    type_ = 'int'
    free_axes = (free_axes,)
  else:
    raise ValueError
  free_axes = [(dims[0] + axis if axis < 0 else axis) for axis in free_axes]
  all_shapes = []
  for shape in shapes:
    assert isinstance(shapes, (tuple, list)), (f'Must be a sequence of shape. While '
                                               f'we got one element is {shape}.')
    shape = tuple([sh for i, sh in enumerate(shape) if i not in free_axes])
    all_shapes.append(shape)
  unique_shape = tuple(set(all_shapes))
  if len(unique_shape) > 1:
    if len(free_axes):
      raise ValueError(f'The provided shape (without axes of {free_axes}) are not consistent.')
    else:
      raise ValueError(f'The provided shape are not consistent.')
  if return_format_shapes:
    if type_ == 'int':
      free_shapes = tuple([shape[free_axes[0]] for shape in shapes])
    elif type_ == 'seq':
      free_shapes = tuple([tuple([shape[axis] for axis in free_axes]) for shape in shapes])
    else:
      free_shapes = None
    return unique_shape[0], free_shapes


def is_shape_broadcastable(shapes, free_axes=(), return_format_shapes=False):
  """Check whether the given shapes are broadcastable.

  See https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
  for more details.

  Parameters
  ----------
  shapes
  free_axes
  return_format_shapes

  Returns
  -------

  """
  max_dim = max([len(shape) for shape in shapes])
  shapes = [[1] * (max_dim - len(s)) + list(s) for s in shapes]
  return is_shape_consistency(shapes, free_axes, return_format_shapes)


def check_shape_except_batch(shape1, shape2, batch_idx=0, mode='raise'):
  """Check whether two shapes are compatible except the batch size axis."""
  assert mode in ['raise', 'bool']
  if len(shape2) != len(shape1):
    if mode == 'raise':
      raise ValueError(f'Dimension mismatch between two shapes. '
                       f'{shape1} != {shape2}')
    else:
      return False
  new_shape1 = list(shape1)
  new_shape2 = list(shape2)
  new_shape1.pop(batch_idx)
  new_shape2.pop(batch_idx)
  if new_shape1 != new_shape2:
    if mode == 'raise':
      raise ValueError(f'Two shapes {new_shape1} and {new_shape2} are not '
                       f'consistent when excluding the batch axis '
                       f'{batch_idx}')
    else:
      return False
  return True


def check_shape(all_shapes, free_axes: Union[Sequence[int], int] = -1):
  # check "all_shapes"
  if isinstance(all_shapes, dict):
    all_shapes = tuple(all_shapes.values())
  elif isinstance(all_shapes, (tuple, list)):
    all_shapes = tuple(all_shapes)
  else:
    raise ValueError
  # maximum number of dimension
  max_dim = max([len(shape) for shape in all_shapes])
  all_shapes = [[1] * (max_dim - len(s)) + list(s) for s in all_shapes]
  # check "free_axes"
  type_ = 'seq'
  if isinstance(free_axes, int):
    free_axes = (free_axes,)
    type_ = 'int'
  elif isinstance(free_axes, (tuple, list)):
    free_axes = tuple(free_axes)
  assert isinstance(free_axes, tuple)
  free_axes = [(axis + max_dim if axis < 0 else axis) for axis in free_axes]
  fixed_axes = [i for i in range(max_dim) if i not in free_axes]
  # get all free shapes
  if type_ == 'int':
    free_shape = [shape[free_axes[0]] for shape in all_shapes]
  else:
    free_shape = [[shape[axis] for axis in free_axes] for shape in all_shapes]
  # get all assumed fixed shapes
  fixed_shapes = [[shape[axis] for shape in all_shapes] for axis in fixed_axes]
  max_fixed_shapes = [max(shape) for shape in fixed_shapes]
  # check whether they can broadcast compatible
  for i, shape in enumerate(fixed_shapes):
    if len(set(shape) - {1, max_fixed_shapes[i]}):
      raise ValueError(f'Shapes out of axes {free_axes} are not '
                       f'broadcast compatible: \n'
                       f'{all_shapes}')
  return free_shape, max_fixed_shapes


def is_dict_data(a_dict: Dict,
                 key_type: Union[Type, Tuple[Type, ...]] = None,
                 val_type: Union[Type, Tuple[Type, ...]] = None,
                 name: str = None,
                 allow_none: bool = True):
  """Check the dictionary data.
  """
  if allow_none and a_dict is None:
    return None
  name = '' if (name is None) else f'"{name}"'
  if not isinstance(a_dict, dict):
    raise ValueError(f'{name} must be a dict, while we got {type(a_dict)}')
  for key, value in a_dict.items():
    if (key_type is not None) and (not isinstance(key, key_type)):
      raise ValueError(f'{name} must be a dict of ({key_type}, {val_type}), '
                       f'while we got ({type(key)}, {type(value)})')
    if (val_type is not None) and (not isinstance(value, val_type)):
      raise ValueError(f'{name} must be a dict of ({key_type}, {val_type}), '
                       f'while we got ({type(key)}, {type(value)})')
  return a_dict


def is_callable(fun: Callable,
                name: str = None,
                allow_none: bool = False):
  name = '' if name is None else name
  if fun is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'{name} must be a callable function, but we got None.')
  if not callable(fun):
    raise ValueError(f'{name} should be a callable function. While we got {type(fun)}')
  return fun


def is_initializer(
    initializer,
    name: str = None,
    allow_none: bool = False
):
  """Check the initializer.
  """
  global Array
  if Array is None: from brainpy._src.math.ndarray import Array

  global init
  if init is None:
    from brainpy import initialize
    init = initialize

  name = '' if name is None else name
  if initializer is None:
    if allow_none:
      return
    else:
      raise ValueError(f'{name} must be an initializer, but we got None.')
  if isinstance(initializer, init.Initializer):
    return initializer
  elif isinstance(initializer, (Array, jax.Array)):
    return initializer
  elif callable(initializer):
    return initializer
  else:
    raise ValueError(f'{name} should be an instance of brainpy.init.Initializer, '
                     f'tensor or callable function. While we got {type(initializer)}')


def is_connector(
    connector,
    name: str = None,
    allow_none: bool = False
):
  """Check the connector.
  """
  global Array
  if Array is None:
    from brainpy._src.math.ndarray import Array
  global conn
  if conn is None: from brainpy import connect as conn

  name = '' if name is None else name
  if connector is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'{name} must be an initializer, but we got None.')
  if isinstance(connector, conn.Connector):
    return connector
  elif isinstance(connector, (Array, jax.Array)):
    return connector
  elif callable(connector):
    return connector
  else:
    raise ValueError(f'{name} should be an instance of brainpy.conn.Connector, '
                     f'tensor or callable function. While we got {type(connector)}')


def is_sequence(
    value: Sequence,
    name: str = None,
    elem_type: Union[type, Sequence[type]] = None,
    allow_none: bool = True
):
  if name is None: name = ''
  if value is None:
    if allow_none:
      return
    else:
      raise ValueError(f'{name} must be a sequence, but got None')
  if not isinstance(value, (tuple, list)):
    raise ValueError(f'{name} should be a sequence, but we got a {type(value)}')
  if elem_type is not None:
    for v in value:
      if not isinstance(v, elem_type):
        raise ValueError(f'Elements in {name} should be {elem_type}, '
                         f'but we got {type(elem_type)}: {v}')
  return value


def is_float(
    value: float,
    name: str = None,
    min_bound: float = None,
    max_bound: float = None,
    allow_none: bool = False,
    allow_int: bool = True
) -> float:
  """Check float type.

  Parameters
  ----------
  value: Any
  name: optional, str
  min_bound: optional, float
    The allowed minimum value.
  max_bound: optional, float
    The allowed maximum value.
  allow_none: bool
    Whether allow the value is None.
  allow_int: bool
    Whether allow the value be an integer.
  """
  if name is None: name = ''
  if value is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'{name} must be a float, but got None')
  if allow_int:
    if not isinstance(value, (float, int, np.integer, np.floating)):
      raise ValueError(f'{name} must be a float, but got {type(value)}')
  else:
    if not isinstance(value, (float, np.floating)):
      raise ValueError(f'{name} must be a float, but got {type(value)}')
  if min_bound is not None:
    jit_error2(value < min_bound,
               ValueError(f"{name} must be a float bigger than {min_bound}, "
                          f"while we got {value}"))

  if max_bound is not None:
    jit_error2(value > max_bound,
               ValueError(f"{name} must be a float smaller than {max_bound}, "
                          f"while we got {value}"))
  return value


def is_integer(value: int, name=None, min_bound=None, max_bound=None, allow_none=False):
  """Check integer type.

  Parameters
  ----------
  value: int, optional
  name: optional, str
  min_bound: optional, int
    The allowed minimum value.
  max_bound: optional, int
    The allowed maximum value.
  allow_none: bool
    Whether allow the value is None.
  """
  if name is None: name = ''
  if value is None:
    if allow_none:
      return
    else:
      raise ValueError(f'{name} must be an int, but got None')
  if not isinstance(value, (int, np.integer)):
    if hasattr(value, '__array__'):
      if not (np.issubdtype(value.dtype, np.integer) and value.ndim == 0 and value.size == 1):
        raise ValueError(f'{name} must be an int, but got {value}')
    else:
      raise ValueError(f'{name} must be an int, but got {value}')
  if min_bound is not None:
    jit_error2(jnp.any(value < min_bound),
               ValueError(f"{name} must be an int bigger than {min_bound}, "
                          f"while we got {value}"))
  if max_bound is not None:
    jit_error2(jnp.any(value > max_bound),
               ValueError(f"{name} must be an int smaller than {max_bound}, "
                          f"while we got {value}"))
  return value


def is_string(value: str, name: str = None, candidates: Sequence[str] = None, allow_none=False):
  """Check string type.
  """
  if name is None: name = ''
  if value is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'{name} must be a str, but got None')
  if candidates is not None:
    if value not in candidates:
      raise ValueError(f'{name} must be a str in {candidates}, '
                       f'but we got {value}')
  return value


def serialize_kwargs(shared_kwargs: Optional[Dict]):
  """Serialize kwargs."""
  shared_kwargs = dict() if shared_kwargs is None else shared_kwargs
  is_dict_data(shared_kwargs,
               key_type=str,
               val_type=(bool, float, int, complex, str),
               name='shared_kwargs')
  shared_kwargs = {key: shared_kwargs[key] for key in sorted(shared_kwargs.keys())}
  return str(shared_kwargs)


def is_subclass(
    instance: Any,
    supported_types: Union[Type, Sequence[Type]],
    name: str = ''
) -> None:
  r"""Check whether the instance is in the inheritance tree of the supported types.

  This function is used to check whether the given ``instance`` is an instance of
  parent types in the inheritance hierarchy of the given ``supported_types``.


  Here we have the following inheritance hierarchy::

           A
         /   \
        B     C
       / \   / \
      D   E F   G

  If ``supported_types`` is ``[E, F]``, then

  - the instance of ``D`` or ``G`` will fail to pass the check.
  - the instance of ``E`` or ``F`` will success to pass the check.
  - the instance of ``B`` or ``C`` will also success to pass the check.
  - the instance of ``A`` will success to pass the check too.

  Parameters
  ----------
  instance: Any
    The instance in the inheritance hierarchy tree.
  supported_types: type, list of type, tuple of type
    All types that are supported.
  name: str
    The checking target name.
  """
  mode_type = type(instance)
  if isinstance(supported_types, type):
    supported_types = (supported_types,)
  if not isinstance(supported_types, (tuple, list)):
    raise TypeError(f'supported_types must be a tuple/list of type. But wwe got {type(supported_types)}')
  for smode in supported_types:
    if not isinstance(smode, type):
      raise TypeError(f'supported_types must be a tuple/list of type. But wwe got {smode}')
  checking = [issubclass(smode, mode_type) for smode in supported_types]
  if any(checking):
    return instance
  else:
    raise NotImplementedError(f"{name} does not support {instance}. We only support "
                              f"{', '.join([mode.__name__ for mode in supported_types])}. ")


def is_instance(
    instance: Any,
    supported_types: Union[Type, Sequence[Type]],
    name: str = ''
):
  r"""Check whether the ``instance`` is the instance of the given types.

  This function is used to check whether the given ``instance`` is an instance of
  the given ``supported_types``.

  Here we have the following inheritance hierarchy::

           A
         /   \
        B     C
       / \   / \
      D   E F   G

  If ``supported_types`` is ``[B, F]``, then

  - the instance of ``A`` or ``C`` or ``G`` will fail to pass the check.
  - the instance of ``B`` or ``D`` or ``E`` or ``F`` will success to pass the check.

  Parameters
  ----------
  instance: Any
    The instance in the inheritance hierarchy tree.
  supported_types: type, list of type, tuple of type
    All types that are supported.
  name: str
    The checking target name.
  """
  if not name:
    name = 'We'
  if not isinstance(instance, supported_types):
    raise NotImplementedError(f"{name} expect to get an instance of {supported_types}."
                              f"But we got {type(instance)}. ")
  return instance


def is_elem_or_seq_or_dict(targets: Any,
                           elem_type: Union[type, Tuple[type, ...]],
                           out_as: str = 'tuple'):
  assert out_as in ['tuple', 'list', 'dict', None], 'Only support to output as tuple/list/dict/None'

  if targets is None:
    keys = []
    vals = []
  elif isinstance(targets, elem_type):
    keys = [id(targets)]
    vals = [targets]
  elif isinstance(targets, (list, tuple)):
    is_leaf = [isinstance(l, elem_type) for l in targets]
    if not all(is_leaf):
      raise ValueError(f'Only support {elem_type}, sequence of {elem_type}, or dict of {elem_type}.')
    keys = [id(v) for v in targets]
    vals = list(targets)
  elif isinstance(targets, dict):
    is_leaf = [isinstance(l, elem_type) for l in targets.values()]
    if not all(is_leaf):
      raise ValueError(f'Only support {elem_type}, sequence of {elem_type}, or dict of {elem_type}.')
    keys = list(targets.keys())
    vals = list(targets.values())
  else:
    raise ValueError(f'Only support {elem_type}, sequence of {elem_type}, or dict of {elem_type}.')

  if out_as is None:
    return targets
  elif out_as == 'list':
    return vals
  elif out_as == 'tuple':
    return tuple(vals)
  elif out_as == 'dict':
    return dict(zip(keys, vals))
  else:
    raise KeyError


def is_all_vars(dyn_vars: Any, out_as: str = 'tuple'):
  global var_obs
  if var_obs is None:
    from brainpy.math import Variable, VarList, VarDict
    var_obs = (VarList, VarDict, Variable)

  return is_elem_or_seq_or_dict(dyn_vars, var_obs, out_as)


def is_all_objs(targets: Any, out_as: str = 'tuple'):
  global BrainPyObject
  if BrainPyObject is None:
    from brainpy._src.math.object_transform.base import BrainPyObject
  return is_elem_or_seq_or_dict(targets, BrainPyObject, out_as)


def _err_jit_true_branch(err_fun, x):
  id_tap(err_fun, x)
  return


def _err_jit_false_branch(x):
  return


def _cond(err_fun, pred, err_arg):
  from brainpy._src.math.remove_vmap import remove_vmap

  @wraps(err_fun)
  def true_err_fun(arg, transforms):
    err_fun(arg)

  cond(remove_vmap(pred),
       partial(_err_jit_true_branch, true_err_fun),
       _err_jit_false_branch,
       err_arg)


def jit_error(pred, err_fun, err_arg=None):
  """Check errors in a jit function.

  Parameters
  ----------
  pred: bool
    The boolean prediction.
  err_fun: callable
    The error function, which raise errors.
  err_arg: any
    The arguments which passed into `err_f`.
  """
  from brainpy._src.math.interoperability import as_jax
  partial(_cond, err_fun)(as_jax(pred), err_arg)


jit_error_checking = jit_error


def jit_error2(pred: bool, err: Exception):
  """Check errors in a jit function.

  Parameters
  ----------
  pred: bool
    The boolean prediction.
  err: Exception
    The error.
  """
  from brainpy._src.math.remove_vmap import remove_vmap
  from brainpy._src.math.interoperability import as_jax

  assert isinstance(err, Exception), 'Must be instance of Exception.'

  def true_err_fun(arg, transforms):
    raise err

  cond(remove_vmap(as_jax(pred)),
       lambda: id_tap(true_err_fun, None),
       lambda: None)

