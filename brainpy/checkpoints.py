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
from typing import Dict, Any, Optional

import braintools
import jax
from braintools.file import msgpack_register_serialization, AsyncManager

from brainpy.math.ndarray import Array
from brainpy.types import PyTree

__all__ = [
    'save_pytree', 'load_pytree', 'AsyncManager',
]

AsyncManager = braintools.file.AsyncManager


def _array_dict_state(x: Array) -> Dict[str, jax.Array]:
    return x.value


def _restore_array(x, state_dict: jax.Array) -> Array:
    x.value = state_dict
    return x


msgpack_register_serialization(Array, _array_dict_state, _restore_array)


def save_pytree(
    filename: str,
    target: PyTree,
    overwrite: bool = True,
    async_manager: Optional[AsyncManager] = None,
    verbose: bool = True,
) -> None:
    """Save a checkpoint of the model. Suitable for single-host.

    In this method, every JAX process saves the checkpoint on its own. Do not
    use it if you have multiple processes and you intend for them to save data
    to a common directory (e.g., a GCloud bucket). To save multi-process
    checkpoints to a shared storage or to save `GlobalDeviceArray`s, use
    `multiprocess_save()` instead.

    Pre-emption safe by writing to temporary before a final rename and cleanup
    of past files. However, if async_manager is used, the final
    commit will happen inside an async callback, which can be explicitly waited
    by calling `async_manager.wait_previous_save()`.

    Parameters::

    filename: str
      str or pathlib-like path to store checkpoint files in.
    target: Any
      serializable flax object, usually a flax optimizer.
    overwrite: bool
      overwrite existing checkpoint files if a checkpoint at the
      current or a later step already exits (default: False).
    async_manager: optional, AsyncManager
      if defined, the save will run without blocking the main
      thread. Only works for single host. Note that an ongoing save will still
      block subsequent saves, to make sure overwrite/keep logic works correctly.
    verbose: bool
      Whether output the print information.

    Returns::

    out: str
      Filename of saved checkpoint.
    """
    return braintools.file.msgpack_save(
        filename,
        target,
        overwrite=overwrite,
        async_manager=async_manager,
        verbose=verbose,
    )


def load_pytree(
    filename: str,
    target: Optional[Any] = None,
    parallel: bool = True,
) -> PyTree:
    """Load the checkpoint from the given checkpoint path.

    Parameters::

    filename: str
      checkpoint file or directory of checkpoints to restore from.
    parallel: bool
      whether to load seekable checkpoints in parallel, for speed.

    Returns::

    out: Any
      Restored `target` updated from checkpoint file, or if no step specified and
      no checkpoint files present, returns the passed-in `target` unchanged.
      If a file path is specified and is not found, the passed-in `target` will be
      returned. This is to match the behavior of the case where a directory path
      is specified but the directory has not yet been created.
    """
    return braintools.file.msgpack_load(filename, target=target, parallel=parallel)
