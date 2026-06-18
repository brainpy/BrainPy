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
"""Line-coverage tests for ``brainpy/running/native_multiprocessing.py``.

Exercises the :func:`process_pool` and :func:`process_pool_lock` wrappers around
``multiprocessing.Pool``:

- list/tuple parameters -> ``apply_async(args=...)``.
- dict parameters -> ``apply_async(kwds=...)``.
- the lock variant injecting a shared ``Manager().Lock()`` for both the
  sequence and dict parameter forms.
- the ``ValueError`` branch for an unsupported parameter type.

Worker functions are module-level (hence picklable) and deliberately do *no*
JAX work, to avoid the ``os.fork()`` + multithreaded-JAX deadlock. Process
counts are kept at 1 with tiny iterables.
"""

import multiprocessing

import pytest

from brainpy.running.native_multiprocessing import process_pool, process_pool_lock


# --------------------------------------------------------------------------- #
# module-level (picklable) worker functions
# --------------------------------------------------------------------------- #

def _add(a, b):
    return a + b


def _add_kw(x=0, y=0):
    return x + y


def _add_lock(a, b, lock):
    # exercise the lock object that process_pool_lock injects as the last arg
    lock.acquire()
    try:
        return a + b
    finally:
        lock.release()


def _add_lock_kw(x=0, y=0, lock=None):
    assert lock is not None
    lock.acquire()
    try:
        return x + y
    finally:
        lock.release()


# Guard: spawned children must be able to import this test module. Under the
# default 'fork' start method (Linux) this is fine; if the platform default is
# 'spawn', the module is re-imported in the child which is also fine since the
# workers are top-level. We keep num_process=1 throughout.


def test_process_pool_with_sequence_params():
    results = process_pool(_add, [(1, 2), (3, 4)], num_process=1)
    assert sorted(results) == [3, 7]


def test_process_pool_with_dict_params():
    results = process_pool(_add_kw, [{'x': 1, 'y': 2}, {'x': 10, 'y': 20}], num_process=1)
    assert sorted(results) == [3, 30]


def test_process_pool_unknown_param_type_raises():
    with pytest.raises(ValueError):
        process_pool(_add, [42], num_process=1)


def test_process_pool_lock_with_sequence_params():
    results = process_pool_lock(_add_lock, [(1, 2), (5, 6)], num_process=1)
    assert sorted(results) == [3, 11]


def test_process_pool_lock_with_dict_params():
    results = process_pool_lock(_add_lock_kw, [{'x': 1, 'y': 2}], num_process=1)
    assert results == [3]


def test_process_pool_lock_unknown_param_type_raises():
    with pytest.raises(ValueError):
        process_pool_lock(_add_lock, [42], num_process=1)
