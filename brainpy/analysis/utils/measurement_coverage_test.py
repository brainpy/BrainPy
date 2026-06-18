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
"""Coverage tests for :mod:`brainpy.analysis.utils.measurement`.

Exercises:
- ``find_indexes_of_limit_cycle_max`` for both the "found a cycle" and
  "no cycle" paths;
- ``euclidean_distance`` for both ndarray and dict inputs plus the missing
  ``num_point`` error;
- ``euclidean_distance_jax`` for both ndarray and dict inputs plus the missing
  ``num_point`` error.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from brainpy.analysis.utils import measurement as m


# --------------------------------------------------------------------------- #
# find_indexes_of_limit_cycle_max
# --------------------------------------------------------------------------- #
def test_find_indexes_limit_cycle_found():
    # Construct a clean periodic signal whose two consecutive maxima have
    # nearly identical heights and a span well above 1e-3.
    t = np.linspace(0, 6 * np.pi, 600)
    arr = np.sin(t)
    idx = m.find_indexes_of_limit_cycle_max(arr, tol=0.1)
    assert idx[0] != -1
    assert idx[1] > idx[0]


def test_find_indexes_limit_cycle_not_found():
    # Monotonically increasing signal: no two qualifying maxima -> [-1, -1].
    arr = np.linspace(0., 5., 200)
    idx = m.find_indexes_of_limit_cycle_max(arr, tol=0.001)
    assert idx[0] == -1 and idx[1] == -1


# --------------------------------------------------------------------------- #
# euclidean_distance (numba/numpy)
# --------------------------------------------------------------------------- #
def test_euclidean_distance_array_matches_scipy():
    rng = np.random.RandomState(0)
    pts = rng.randn(5, 3)
    got = m.euclidean_distance(pts)
    expected = squareform(pdist(pts, metric='euclidean'))
    assert np.allclose(got, expected)


# NOTE (defect): ``euclidean_distance`` is wrapped in ``@numba_jit`` and its
# dict-input branch (lines 73-81) cannot be reached: numba raises a
# ``TypingError`` ("non-precise type pyobject") before the Python body runs
# when a dict-of-arrays is passed.  The ``ValueError('Please provide num_point')``
# guard inside that branch is therefore dead code.  The dict path is only
# usable through the pure-jax ``euclidean_distance_jax`` mirror (tested below).
def test_euclidean_distance_dict_is_unreachable_under_numba():
    from numba.core.errors import TypingError
    with pytest.raises((TypingError, ValueError)):
        m.euclidean_distance({'a': np.zeros((3, 2)), 'b': np.zeros((3, 2))}, num_point=3)


# --------------------------------------------------------------------------- #
# euclidean_distance_jax
# --------------------------------------------------------------------------- #
def test_euclidean_distance_jax_array():
    rng = np.random.RandomState(2)
    pts = rng.randn(5, 3)
    got = np.asarray(m.euclidean_distance_jax(pts))
    expected = squareform(pdist(pts, metric='euclidean'))
    assert np.allclose(got, expected, atol=1e-5)


def test_euclidean_distance_jax_dict():
    rng = np.random.RandomState(3)
    a = rng.randn(4, 2)
    b = rng.randn(4, 2)
    got = np.asarray(m.euclidean_distance_jax({'a': a, 'b': b}, num_point=4))
    stacked = np.hstack([a, b])
    expected = squareform(pdist(stacked, metric='euclidean'))
    assert np.allclose(got, expected, atol=1e-5)


def test_euclidean_distance_jax_dict_requires_num_point():
    with pytest.raises(ValueError):
        m.euclidean_distance_jax({'a': np.zeros((3, 2))})
