# -*- coding: utf-8 -*-
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
from contextlib import contextmanager
from functools import partial
from typing import Optional, Any, Union, Sequence

import jax
import numpy as np
from jax.sharding import PartitionSpec, Mesh, NamedSharding, Sharding

from .ndarray import ShardedArray, Array

__all__ = [
    'device_mesh',
    'get_sharding',
    'partition_by_axname',
    'partition_by_sharding',
    'partition',
    'keep_constraint',

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


def is_bp_array(x):
    return isinstance(x, Array)


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
        yield _default_mesh
    finally:
        _default_mesh = _old_mesh


def _device_put(x: Union[Array, jax.Array, np.ndarray],
                device: Union[None, jax.Device, Sharding] = None):
    """Transfers ``x`` to ``device``.

    Note that this function can only transfer ``brainpy.math.Array``, ``jax.Array``,
    and ``numpy.ndarray``. Other value will be directly returned.

    Args:
      x: The input array.
      device: The given device.

    Returns:
      A copy of ``x`` that resides on ``device``.
    """
    if isinstance(x, Array):
        x.value = jax.device_put(x.value, device=device)
        return x
    else:
        if isinstance(x, (jax.Array, np.ndarray)):
            # wrap the data as brainpy.math.Array is important (experimental)
            return ShardedArray(jax.device_put(x, device=device), keep_sharding=True)
        else:
            return x


def get_sharding(
    axis_names: Optional[Sequence[str]] = None,
    mesh: Optional[Mesh] = None
) -> Optional[NamedSharding]:
    """Get sharding according to the given axes information.

    Args:
      axis_names: list of str, or tuple of str. The name for each axis in the array.
      mesh: Mesh. The given device mesh.

    Returns:
      The instance of NamedSharding.
    """
    if axis_names is None:
        return None
    if mesh is None:
        mesh = _default_mesh
    if mesh is None:
        return None
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
        for _leaf in jax.tree_util.tree_leaves(x, is_leaf=is_bp_array):
            if np.ndim(_leaf) != len(axis_names):
                raise ValueError(f'The input array shape is {np.shape(_leaf)}, '
                                 f'while the given axis names are {axis_names}. '
                                 f'Dimensions are mismatch.')
    if mesh is None:
        if _default_mesh is None:
            return x
        mesh = _default_mesh
    sharding = get_sharding(axis_names, mesh)
    if sharding is None:
        return x
    else:
        return jax.tree_util.tree_map(partial(_device_put, device=sharding),
                                      x, is_leaf=is_bp_array)


def partition_by_sharding(
    x: Any,
    sharding: Optional[Sharding] = None,
):
    """Partition inputs with the given sharding strategy.

    Args:
      x: The input arrays. It can be a pyTree of arrays.
      sharding: The `jax.sharding.Sharding` instance.

    Returns:
      The sharded ``x``, which has been partitioned by the given sharding stragety.
    """
    if sharding is None:
        return x
    else:
        if not isinstance(sharding, Sharding):
            raise TypeError(f'sharding must be instance of jax.sharding.Sharding. While we got {sharding}.')
        return jax.tree_util.tree_map(partial(_device_put, device=sharding),
                                      x,
                                      is_leaf=is_bp_array)


def partition(
    x: Any,
    sharding: Optional[Union[Sequence[str], jax.Device, Sharding]] = None,
):
    """Partition the input arrays onto devices by the given sharding strategies.

    Args:
      x: Any input arrays. It can also be a PyTree of arrays.
      sharding: The sharding strategy.

    Returns:
      The partitioned arrays.
      Notably, the
    """
    if sharding is None:
        return x
    elif isinstance(sharding, (jax.Device, Sharding)):
        return jax.tree_util.tree_map(partial(_device_put, device=sharding),
                                      x, is_leaf=is_bp_array)
    elif isinstance(sharding, (tuple, list)) and any([isinstance(s, str) for s in sharding]):
        return partition_by_axname(x, sharding)
    else:
        raise TypeError('"sharding" only supports jax.sharding.Sharding or a sequence of axis names. \n'
                        f'But we got {sharding}')


def _keep_constraint(x: Any):
    if isinstance(x, Array):
        x = x.value
    if isinstance(x, jax.Array):
        if hasattr(x, 'sharding'):
            if x.sharding is not None:
                return jax.lax.with_sharding_constraint(x, x.sharding)
        return x
    else:
        return x


def keep_constraint(x: Any):
    """Keep the sharding constraint of the given inputs during computation.

    Args:
      x: Any.

    Returns:
      constraint_x: Same as ``x``.
    """
    return jax.tree_util.tree_map(_keep_constraint, x, is_leaf=is_bp_array)
