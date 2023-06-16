from functools import partial
from typing import Dict, Optional, Any, Union, Sequence

import jax
import numpy as np
from jax._src.sharding_impls import UnspecifiedValue, UNSPECIFIED
from jax.sharding import PartitionSpec, Mesh, NamedSharding, Sharding

from .ndarray import Array

__all__ = [
  'set',
  'get_sharding',
  'partition_by_axname',
  'partition_by_sharding',
]

_mesh: Optional[Mesh] = None


def set(
    mesh: Optional[Mesh] = None,
    mesh_shape: Optional[Sequence[int]] = None,
    mesh_axes: Optional[Sequence[str]] = None,
):
  global _mesh

  if mesh_axes is not None:
    assert mesh_axes is not None, 'Provide both "mesh_axes" and "mesh_shape".'
    assert mesh is None, 'Provide either "mesh" or "mesh_axes" + "mesh_shape".'
    assert len(mesh_axes) == len(mesh_shape)
    mesh = Mesh(np.asarray(jax.devices()).reshape(*mesh_shape), axis_names=mesh_axes)
    _mesh = mesh
  else:
    if mesh is not None:
      _mesh = mesh
      assert mesh_shape is None and mesh_axes is None, 'Provide either "mesh" or "mesh_axes" + "mesh_shape".'
    else:
      _mesh = None


def _device_put(x: Union[Array, jax.Array, np.ndarray],
                named_shard: NamedSharding):
  if isinstance(x, Array):
    x.value = jax.device_put(x, device=named_shard)
  return x


def get_sharding(
    axis_names: Optional[Sequence[str]] = None,
    mesh: Optional[Mesh] = None
) -> Union[UnspecifiedValue, NamedSharding]:
  """Get sharding according to the given axes information.

  Args:
    axis_names: list of str, or tuple of str. The name for each axis in the array.
    mesh: Mesh. The given device mesh.

  Returns:
    The instance of NamedSharding.
  """
  if axis_names is None:
    return UNSPECIFIED
  if mesh is None:
    mesh = _mesh
  if mesh is None:
    return UNSPECIFIED
  else:
    axis_names = [(name if name in mesh.axis_names else None) for name in axis_names]
    return NamedSharding(mesh, PartitionSpec(*axis_names))


def partition_by_axname(
    x: Any,
    axis_names: Optional[Sequence[str]] = None,
    mesh: Optional[Mesh] = None
):
  """Put the given arrays into the mesh devices.

  Args:
    x: any. Any array.
    axis_names: list of str, or tuple of str. The name for each axis in the array.
    mesh: Mesh. The given device mesh.

  Returns:
    The re-sharded arrays.
  """
  if axis_names is None:
    return x
  if mesh is None:
    if _mesh is None:
      return x
    mesh = _mesh
  shard = get_sharding(axis_names, mesh)
  if shard is None:
    return x
  else:
    f = partial(_device_put, named_shard=shard)
    return jax.tree_util.tree_map(f, x, is_leaf=lambda a: isinstance(a, Array))


def partition_by_sharding(
    x: Any,
    sharding: Optional[Sharding] = None,
):
  if sharding is None:
    return x
  else:
    assert isinstance(sharding, Sharding)
    f = partial(_device_put, named_shard=sharding)
    return jax.tree_util.tree_map(f, x, is_leaf=lambda a: isinstance(a, Array))

