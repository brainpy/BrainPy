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
"""Tests for ``brainpy/running/jax_multiprocessing.py``.

Covers :func:`jax_vectorize_map` (``jax.vmap`` chunking) and
:func:`jax_parallelize_map` (``jax.pmap`` chunking), including:

- the chunked vmap path with a trailing partial chunk and with dict-form args;
- the length-mismatch guard;
- (regression for P15-H3) ``jax_parallelize_map`` with a task count that is *not*
  a multiple of the device count, which produces a trailing partial chunk sharded
  on a *subset* of devices. The faulty version cached one ``pmap`` and then crashed
  in the closing ``bm.concatenate`` with "Received incompatible devices for jitted
  computation". The multi-device sub-test runs in a subprocess (devices must be
  configured before JAX initialises) and is skipped if extra host devices cannot be
  spun up.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import brainpy.math as bm
from brainpy.running.jax_multiprocessing import jax_vectorize_map, jax_parallelize_map


def _double(x):
    return x * 2.0


# --------------------------------------------------------------------------- #
# jax_vectorize_map (vmap)
# --------------------------------------------------------------------------- #

def test_vectorize_map_partial_chunk():
    # 5 tasks, chunk size 2 -> chunks of 2, 2, 1 (trailing partial chunk).
    args = [np.arange(5.0)]
    r = np.asarray(jax_vectorize_map(_double, args, num_parallel=2))
    np.testing.assert_allclose(r, np.arange(5.0) * 2.0)


def test_vectorize_map_partial_chunk_clear_buffer():
    args = [np.arange(5.0)]
    r = np.asarray(jax_vectorize_map(_double, args, num_parallel=2, clear_buffer=True))
    np.testing.assert_allclose(r, np.arange(5.0) * 2.0)


def test_vectorize_map_dict_args():
    def add(x, y):
        return x + y

    args = {'x': np.arange(4.0), 'y': np.arange(4.0) * 10}
    r = np.asarray(jax_vectorize_map(add, args, num_parallel=3))
    np.testing.assert_allclose(r, np.arange(4.0) * 11)


def test_vectorize_map_length_mismatch_raises():
    with pytest.raises(ValueError):
        jax_vectorize_map(_double, [np.arange(4.0), np.arange(3.0)], num_parallel=2)


def test_vectorize_map_bad_arguments_type_raises():
    with pytest.raises(TypeError):
        jax_vectorize_map(_double, 42, num_parallel=2)


# --------------------------------------------------------------------------- #
# jax_parallelize_map (pmap)
# --------------------------------------------------------------------------- #

def test_parallelize_map_single_device():
    # On a single device num_parallel must be 1; chunks of size 1 each.
    args = [np.arange(3.0)]
    r = np.asarray(jax_parallelize_map(_double, args, num_parallel=1))
    np.testing.assert_allclose(r, np.arange(3.0) * 2.0)


def test_parallelize_map_length_mismatch_raises():
    with pytest.raises(ValueError):
        jax_parallelize_map(_double, [np.arange(2.0), np.arange(1.0)], num_parallel=1)


# Regression for P15-H3: trailing partial chunk across multiple devices.
_MULTI_DEVICE_SNIPPET = r"""
import numpy as np
import jax
assert jax.local_device_count() == 4, jax.local_device_count()
from brainpy.running.jax_multiprocessing import jax_parallelize_map
# 6 tasks, num_parallel == 4 devices -> chunks of 4 then 2 (partial, subset of devices).
r = jax_parallelize_map(lambda x: x * 2.0, [np.arange(6.0)], num_parallel=4)
r = np.asarray(r)
expected = np.arange(6.0) * 2.0
assert np.allclose(r, expected), (r, expected)
print('OK')
"""


def test_parallelize_map_partial_chunk_multi_device():
    env = dict(os.environ)
    env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=4').strip()
    env.setdefault('JAX_PLATFORMS', 'cpu')
    proc = subprocess.run(
        [sys.executable, '-c', _MULTI_DEVICE_SNIPPET],
        env=env, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        # Could not spin up 4 host devices in this environment -> skip rather than fail.
        if 'AssertionError' in proc.stderr and 'local_device_count' in proc.stderr:
            pytest.skip('Could not configure 4 host devices for the pmap test.')
        pytest.fail(f'multi-device pmap run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
    assert 'OK' in proc.stdout
