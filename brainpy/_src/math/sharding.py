from functools import partial
from typing import Optional, Any, Union, Sequence
from contextlib import contextmanager

import jax
import numpy as np
from jax._src.sharding_impls import UnspecifiedValue, UNSPECIFIED
from jax.sharding import PartitionSpec, Mesh, NamedSharding, Sharding

from .ndarray import Array

__all__ = [
  'device_mesh',
  'get_sharding',
  'partition_by_axname',
  'partition_by_sharding',
  'partition',

  'NEU_AXIS',
  'PRE_AXIS',
  'POST_AXIS',
  'SYN_AXIS',
  'BATCH_AXIS',
  'TIME_AXIS',
]

NEU_AXIS = 'neuron'

PRE_AXIS = 'pre'

POST_AXIS = 'post'

SYN_AXIS = 'synapse'

BATCH_AXIS = 'batch'

TIME_AXIS = 'time'

_default_mesh: Optional[Mesh] = None


@contextmanager
def device_mesh(
    devices: Any,
    axis_names: Sequence[str],
):
  global _default_mesh
  _old_mesh = _default_mesh

  devices = np.asarray(devices)
  assert devices.ndim == len(axis_names)
  mesh = Mesh(devices, axis_names=axis_names)

  _default_mesh = mesh

  try:
    yield
  finally:
    _default_mesh = _old_mesh


def _device_put(x: Union[Array, jax.Array, np.ndarray],
                device: Union[None, jax.Device, Sharding] = None):
  if isinstance(x, Array):
    x.value = jax.device_put(x, device=device)
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
    mesh = _default_mesh
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
    axis_names: sequence of str. The name for each axis in the array.
    mesh: Mesh. The given device mesh.

  Returns:
    The re-sharded arrays.
  """
  if axis_names is None:
    return x
  else:
    for _leaf in jax.tree_util.tree_leaves(x, is_leaf=lambda a: isinstance(a, Array)):
      assert np.ndim(_leaf) == len(axis_names)
  if mesh is None:
    if _default_mesh is None:
      return x
    mesh = _default_mesh
  sharding = get_sharding(axis_names, mesh)
  if sharding is None:
    return x
  else:
    f = partial(_device_put, device=sharding)
    return jax.tree_util.tree_map(f, x, is_leaf=lambda a: isinstance(a, Array))


def partition_by_sharding(
    x: Any,
    sharding: Optional[Sharding] = None,
):
  """Partition inputs with the given sharding strategy."""
  if sharding is None:
    return x
  else:
    assert isinstance(sharding, Sharding)
    if isinstance(x, (Array, jax.Array)):
      return _device_put(x, device=sharding)
    return jax.tree_util.tree_map(partial(_device_put, device=sharding),
                                  x,
                                  is_leaf=lambda a: isinstance(a, Array))


def partition(
    x: Any,
    sharding: Optional[Union[Sequence[str], jax.Device, Sharding]] = None,
):
  if sharding is None:
    return x
  elif isinstance(sharding, (jax.Device, Sharding)):
    if isinstance(x, (Array, jax.Array)):
      return _device_put(x, device=sharding)
    return jax.tree_util.tree_map(partial(_device_put, device=sharding),
                                  x,
                                  is_leaf=lambda a: isinstance(a, Array))
  elif isinstance(sharding, (tuple, list)) and any([isinstance(s, str) for s in sharding]):
    return partition_by_axname(x, sharding)
  else:
    raise TypeError
