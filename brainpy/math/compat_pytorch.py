# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Union, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .compat_numpy import (concatenate, minimum, maximum, )
from .ndarray import Array, _as_jax_array_, _return, _check_out

__all__ = [
    'Tensor',
    'flatten',
    'unflatten',
    'unsqueeze',
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
    'clamp_max',
    'clamp_min',
    'arctan2',
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

    Parameters::

    input: Array
      The input array.
    start_dim: int
      the first dim to flatten
    end_dim: int
      the last dim to flatten

    Returns::

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
    new_shape = shape[:start_dim] + (np.prod(shape[start_dim: end_dim], dtype=int),) + shape[end_dim:]
    return jnp.reshape(input, new_shape)


def unflatten(x: Union[jax.Array, Array], dim: int, sizes: Sequence[int]) -> Array:
    """
    Expands a dimension of the input tensor over multiple dimensions.

    Args:
      x: input tensor.
      dim: Dimension to be unflattened, specified as an index into ``x.shape``.
      sizes: New shape of the unflattened dimension. One of its elements can be -1
          in which case the corresponding output dimension is inferred.
          Otherwise, the product of ``sizes`` must equal ``input.shape[dim]``.

    Returns:
      A tensor with the same data as ``input``, but with ``dim`` split into multiple dimensions.
      The returned tensor has one more dimension than the input tensor.
      The returned tensor shares the same underlying data with this tensor.
    """
    assert x.ndim > dim, ('The dimension to be unflattened should be less than the tensor dimension. '
                          f'Got {dim} and {x.ndim}.')
    x = _as_jax_array_(x)
    shape = x.shape
    new_shape = shape[:dim] + tuple(sizes) + shape[dim + 1:]
    r = jnp.reshape(x, new_shape)
    return _return(r)


def unsqueeze(x: Union[jax.Array, Array], dim: int) -> Array:
    """Returns a new tensor with a dimension of size one inserted at the specified position.

    The returned tensor shares the same underlying data with this tensor.
    A dim value within the range ``[-input.dim() - 1, input.dim() + 1)`` can be used.
    Negative dim will correspond to unsqueeze() applied at ``dim = dim + input.dim() + 1``.

    Parameters::

    x: Array
      The input Array
    dim: int
      The index at which to insert the singleton dimension

    Returns::

    out: Array
    """
    x = _as_jax_array_(x)
    r = jnp.expand_dims(x, dim)
    return _return(r)


# Math operations
def abs(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.abs(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


absolute = abs


def acos(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arccos(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arccos = acos


def acosh(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arccosh(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arccosh = acosh


def add(
    x: Union[jax.Array, Array, jnp.number],
    y: Union[jax.Array, Array, jnp.number],
    *,
    alpha: Optional[jnp.number] = 1,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    r"""
    Adds ``other``, scaled by ``alpha``, to ``input``.

    .. math::

        \text { out }_i=\text { input }_i+\text { alpha } \times \text { other }_i

    """
    x = _as_jax_array_(x)
    y = _as_jax_array_(y)
    y = jnp.multiply(alpha, y)
    r = jnp.add(x, y)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


def addcdiv(
    x: Union[jax.Array, Array, jnp.number],
    tensor1: Union[jax.Array, Array, jnp.number],
    tensor2: Union[jax.Array, Array, jnp.number],
    *,
    value: jnp.number = 1,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    tensor1 = _as_jax_array_(tensor1)
    tensor2 = _as_jax_array_(tensor2)
    other = jnp.divide(tensor1, tensor2)
    return add(x, other, alpha=value, out=out)


def addcmul(
    x: Union[jax.Array, Array, jnp.number],
    tensor1: Union[jax.Array, Array, jnp.number],
    tensor2: Union[jax.Array, Array, jnp.number],
    *,
    value: jnp.number = 1,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    tensor1 = _as_jax_array_(tensor1)
    tensor2 = _as_jax_array_(tensor2)
    other = jnp.multiply(tensor1, tensor2)
    return add(x, other, alpha=value, out=out)


def angle(
    x: Union[jax.Array, Array, jnp.number],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.angle(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


def asin(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arcsin(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arcsin = asin


def asinh(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arcsinh(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arcsinh = asinh


def atan(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arctan(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arctan = atan


def atanh(
    x: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x = _as_jax_array_(x)
    r = jnp.arctanh(x)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arctanh = atanh


def atan2(
    x1: Union[jax.Array, Array],
    x2: Union[jax.Array, Array],
    *,
    out: Optional[Union[Array, jax.Array, np.ndarray]] = None
) -> Optional[Array]:
    x1 = _as_jax_array_(x1)
    x2 = _as_jax_array_(x2)
    r = jnp.arctan2(x1, x2)
    if out is None:
        return _return(r)
    else:
        _check_out(out)
        out.value = r


arctan2 = atan2
clamp_max = minimum
clamp_min = maximum
