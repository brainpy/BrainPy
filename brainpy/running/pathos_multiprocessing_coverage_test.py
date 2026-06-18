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
"""Line-coverage tests for ``brainpy/running/pathos_multiprocessing.py``.

Exercises :func:`cpu_ordered_parallel`, :func:`cpu_unordered_parallel` and the
private :func:`_parallel` driver.

``pathos`` is an *optional* dependency and is not installed in the test
environment, so the module imports with ``ProcessPool is None`` /
``cpu_count is None``. We cover two regimes:

1. The real (unpatched) code path, which raises :class:`PackageMissingError`
   because ``pathos`` is missing.
2. The internal logic of ``_parallel`` (``num_process`` resolution, dict vs
   sequence argument handling, ordered vs unordered map selection, task-count
   inference, ``pool.clear`` teardown) by monkeypatching in a lightweight fake
   ``ProcessPool`` and ``cpu_count``. This lets us cover the lines that are
   otherwise unreachable without the heavyweight optional dependency.

Notes
-----
The ``win32`` + Python >= 3.11 ``NotImplementedError`` guard
(``pathos_multiprocessing._parallel`` lines ~82-84) is platform-specific and
unreachable on Linux/macOS CI; it is intentionally not covered here.
"""

import sys

import pytest

import brainpy as bp
from brainpy._errors import PackageMissingError
from brainpy.running import pathos_multiprocessing as pm

_WIN_UNSUPPORTED = sys.platform == 'win32' and sys.version_info.minor >= 11


# --------------------------------------------------------------------------- #
# real path: pathos missing -> PackageMissingError
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(pm.ProcessPool is not None,
                    reason='pathos installed; missing-package branch not hit')
@pytest.mark.skipif(_WIN_UNSUPPORTED, reason='win32 + py>=3.11 raises earlier')
def test_cpu_ordered_parallel_missing_pathos():
    with pytest.raises(PackageMissingError):
        bp.running.cpu_ordered_parallel(lambda x: x, [[1, 2, 3]])


@pytest.mark.skipif(pm.ProcessPool is not None,
                    reason='pathos installed; missing-package branch not hit')
@pytest.mark.skipif(_WIN_UNSUPPORTED, reason='win32 + py>=3.11 raises earlier')
def test_cpu_unordered_parallel_missing_pathos():
    with pytest.raises(PackageMissingError):
        bp.running.cpu_unordered_parallel(lambda x: x, [[1, 2, 3]])


# --------------------------------------------------------------------------- #
# fake-pathos path: drive the internals of _parallel
# --------------------------------------------------------------------------- #

class _FakePool:
    """A minimal stand-in for ``pathos.multiprocessing.ProcessPool``.

    Runs the mapped function serially in-process so no real subprocesses (and
    no JAX/fork interaction) are spawned. Records ``clear`` so teardown is
    asserted.
    """

    cleared = False

    def __init__(self, nodes=None):
        self.nodes = nodes

    def imap(self, fn, *iterables):
        return map(fn, *iterables)

    # unordered map: same serial behaviour is fine for testing
    uimap = imap

    def clear(self):
        type(self).cleared = True


@pytest.fixture()
def fake_pathos(monkeypatch):
    """Install a fake ProcessPool + cpu_count for the duration of a test."""
    _FakePool.cleared = False
    monkeypatch.setattr(pm, 'ProcessPool', _FakePool)
    monkeypatch.setattr(pm, 'cpu_count', lambda: 4)
    if _WIN_UNSUPPORTED:
        pytest.skip('win32 + py>=3.11 raises before reaching the patched code')
    return _FakePool


def test_ordered_with_sequence_arguments(fake_pathos):
    out = bp.running.cpu_ordered_parallel(lambda x: x * 2, [[1, 2, 3]], num_process=2)
    assert out == [2, 4, 6]
    assert fake_pathos.cleared is True


def test_unordered_with_sequence_arguments(fake_pathos):
    out = bp.running.cpu_unordered_parallel(lambda x: x + 1, [[1, 2, 3]], num_process=2)
    assert sorted(out) == [2, 3, 4]
    assert fake_pathos.cleared is True


def test_dict_arguments_dispatch_by_keyword(fake_pathos):
    # dict arguments -> run_f maps positional iterables back to keyword args
    out = bp.running.cpu_ordered_parallel(
        lambda a, b: a + b,
        {'a': [1, 2, 3], 'b': [10, 20, 30]},
        num_process=1,
    )
    assert out == [11, 22, 33]


def test_num_process_none_uses_cpu_count(fake_pathos):
    # num_process=None -> resolved to cpu_count() (=4) without error
    out = bp.running.cpu_ordered_parallel(lambda x: x, [[1, 2]], num_process=None)
    assert out == [1, 2]


def test_num_process_float_fraction(fake_pathos):
    # float -> int(round(fraction * cpu_count())); 0.5 * 4 == 2
    out = bp.running.cpu_ordered_parallel(lambda x: x, [[7, 8]], num_process=0.5)
    assert out == [7, 8]


def test_num_task_explicit(fake_pathos):
    # explicit num_task is honoured (drives the tqdm total) and does not break
    out = bp.running.cpu_ordered_parallel(lambda x: x, [[1, 2, 3]],
                                          num_process=1, num_task=3)
    assert out == [1, 2, 3]


def test_arguments_without_len_infers_no_total(fake_pathos):
    # a non-Sized iterable -> lengths == [] -> num_task stays None (no crash)
    def gen():
        yield 1
        yield 2

    out = bp.running.cpu_ordered_parallel(lambda x: x, [gen()], num_process=1)
    assert out == [1, 2]


def test_invalid_num_process_type_raises(fake_pathos):
    with pytest.raises(ValueError):
        bp.running.cpu_ordered_parallel(lambda x: x, [[1, 2]], num_process='bad')


def test_invalid_arguments_type_raises(fake_pathos):
    # arguments that are neither dict nor tuple/list -> TypeError
    with pytest.raises(TypeError):
        bp.running.cpu_ordered_parallel(lambda x: x, 12345, num_process=1)
