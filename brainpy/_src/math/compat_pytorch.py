from typing import Union, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .ndarray import Array, _as_jax_array_, _return, _check_out
from .compat_numpy import (
  concatenate, shape
)

__all__ = [
  'Tensor',
  'flatten',
  'cat',
  'abs',
  'absolute',
  'acos',
  'arccos',
  'acosh',
  'arccosh',
  'add',
  'addcdiv',
  'addcmul',
  'angle',
  'asin',
  'arcsin',
  'asinh',
  'arcsin',
  'atan',
  'arctan',
  'atan2',
  'atanh',
]


Tensor = Array
cat = concatenate


def flatten(input: Union[jax.Array, Array],
            start_dim: Optional[int] = None,
            end_dim: Optional[int] = None) -> jax.Array:
  """Flattens input by reshaping it into a one-dimensional tensor.
  If ``start_dim`` or ``end_dim`` are passed, only dimensions starting
  with ``start_dim`` and ending with ``end_dim`` are flattened.
  The order of elements in input is unchanged.

  .. note::
     Flattening a zero-dimensional tensor will return a one-dimensional view.

  Parameters
  ----------
  input: Array
    The input array.
  start_dim: int
    the first dim to flatten
  end_dim: int
    the last dim to flatten

  Returns
  -------
  out: Array
  """
  input = _as_jax_array_(input)
  shape = input.shape
  ndim = input.ndim
  if ndim == 0:
    ndim = 1
  if start_dim is None:
    start_dim = 0
  elif start_dim < 0:
    start_dim = ndim + start_dim
  if end_dim is None:
    end_dim = ndim - 1
  elif end_dim < 0:
    end_dim = ndim + end_dim
  end_dim += 1
  if start_dim < 0 or start_dim > ndim:
    raise ValueError(f'start_dim {start_dim} is out of size.')
  if end_dim < 0 or end_dim > ndim:
    raise ValueError(f'end_dim {end_dim} is out of size.')
  new_shape = shape[:start_dim] + (np.prod(shape[start_dim: end_dim], dtype=int), ) + shape[end_dim:]
  return jnp.reshape(input, new_shape)


def unsqueeze(input: Union[jax.Array, Array], dim: int) -> Array:
    """Returns a new tensor with a dimension of size one inserted at the specified position.
  The returned tensor shares the same underlying data with this tensor.
  A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used.
  Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
  Parameters
  ----------
  input: Array
    The input Array
  dim: int
    The index at which to insert the singleton dimension

  Returns
  -------
  out: Array
  """
    input = _as_jax_array_(input)
    return Array(jnp.expand_dims(input, dim))


# Math operations
def abs(input: Union[jax.Array, Array],
        *, out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.abs(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

absolute = abs

def acos(input: Union[jax.Array, Array],
         *, out: Optional[Union[Array,jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arccos(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arccos = acos

def acosh(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array,jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arccosh(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arccosh = acosh

def add(input: Union[jax.Array, Array, jnp.number],
        other: Union[jax.Array, Array, jnp.number],
        *, alpha: Optional[jnp.number] = 1,
        out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  other = _as_jax_array_(other)
  other = jnp.multiply(alpha, other)
  r = jnp.add(input, other)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

def addcdiv(input: Union[jax.Array, Array, jnp.number],
            tensor1: Union[jax.Array, Array, jnp.number],
            tensor2: Union[jax.Array, Array, jnp.number],
            *, value: jnp.number = 1,
            out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  tensor1 = _as_jax_array_(tensor1)
  tensor2 = _as_jax_array_(tensor2)
  other = jnp.divide(tensor1, tensor2)
  return add(input, other, alpha=value, out=out)

def addcmul(input:  Union[jax.Array, Array, jnp.number],
            tensor1: Union[jax.Array, Array, jnp.number],
            tensor2: Union[jax.Array, Array, jnp.number],
            *, value: jnp.number = 1,
            out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  tensor1 = _as_jax_array_(tensor1)
  tensor2 = _as_jax_array_(tensor2)
  other = jnp.multiply(tensor1, tensor2)
  return add(input, other, alpha=value, out=out)

def angle(input: Union[jax.Array, Array, jnp.number],
          *, out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.angle(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

def asin(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array,jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arcsin(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arcsin = asin

def asinh(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array,jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arcsinh(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arcsinh = asinh

def atan(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array,jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arctan(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arctan = atan

def atanh(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arctanh(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arctanh = atanh

def atan2(input: Union[jax.Array, Array],
          *, out: Optional[Union[Array, jax.Array, np.ndarray]] = None) -> Optional[Array]:
  input = _as_jax_array_(input)
  r = jnp.arctan2(input)
  if out is None:
    return _return(r)
  else:
    _check_out(out)
    out.value = r

arctan2 = atan2