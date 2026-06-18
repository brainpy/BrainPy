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
"""Coverage tests for :mod:`brainpy.analysis.utils.others`.

Exercises:
- ``Segment`` iteration with int and per-target segment counts;
- ``check_initials`` (success + assertion branches);
- ``check_plot_durations`` (None / flat / per-initial branches);
- ``get_sign`` and ``get_sign2`` sign-field helpers;
- ``keep_unique`` (array + dict, tolerance<=0, single-point, drop-duplicate,
  and the all-dropped fallback) and ``keep_unique_jax`` mirror paths;
- ``rescale``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.analysis.utils import others as o


# --------------------------------------------------------------------------- #
# Segment
# --------------------------------------------------------------------------- #
def test_segment_int_num_segments():
    targets = (np.arange(10), np.arange(10) * 2.)
    seg = o.Segment(targets, num_segments=2)
    chunks = list(seg)
    assert len(chunks) > 0
    # each yielded item has one slice per target
    for item in chunks:
        assert len(item) == len(targets)


def test_segment_sequence_num_segments():
    targets = (np.arange(6),)
    seg = o.Segment(targets, num_segments=[3])
    chunks = list(seg)
    # union of chunks covers all elements
    seen = np.concatenate([c[0] for c in chunks])
    assert set(seen.tolist()) == set(range(6))


# --------------------------------------------------------------------------- #
# check_initials
# --------------------------------------------------------------------------- #
def test_check_initials_ok():
    out = o.check_initials({'x': [1., 2.], 'y': [3., 4.]}, ['x', 'y'])
    assert set(out.keys()) == {'x', 'y'}
    assert len(out['x']) == 2


def test_check_initials_missing_var():
    with pytest.raises(AssertionError):
        o.check_initials({'x': [1., 2.]}, ['x', 'y'])


def test_check_initials_mismatched_lengths():
    with pytest.raises(AssertionError):
        o.check_initials({'x': [1., 2.], 'y': [3.]}, ['x', 'y'])


# --------------------------------------------------------------------------- #
# check_plot_durations
# --------------------------------------------------------------------------- #
def test_check_plot_durations_none():
    initials = {'x': [0., 0., 0.]}
    out = o.check_plot_durations(None, 100., initials['x'])
    assert len(out) == 3
    assert all(d == (0., 100.) for d in out)


def test_check_plot_durations_flat_pair():
    initials = [0., 0.]
    out = o.check_plot_durations([10., 50.], 100., initials)
    assert len(out) == 2
    assert all(tuple(d) == (10., 50.) for d in out)


def test_check_plot_durations_per_initial():
    initials = [0., 0.]
    out = o.check_plot_durations([(0., 10.), (5., 20.)], 100., initials)
    assert out == [(0., 10.), (5., 20.)]


# --------------------------------------------------------------------------- #
# get_sign / get_sign2
# --------------------------------------------------------------------------- #
def test_get_sign():
    f = lambda x, y: x + y
    xs = jnp.linspace(-1., 1., 5)
    ys = jnp.linspace(-1., 1., 5)
    out = np.asarray(o.get_sign(f, xs, ys))
    assert out.shape == (5, 5)
    assert set(np.unique(out).tolist()).issubset({-1., 0., 1.})


def test_get_sign_with_bm_array():
    f = lambda x, y: x - y
    xs = bm.linspace(-1., 1., 4)
    ys = bm.linspace(-1., 1., 4)
    out = np.asarray(o.get_sign(f, xs, ys))
    assert out.shape == (4, 4)


# NOTE (defect): ``get_sign2`` is broken and has no callers in the codebase.
# It builds ``in_axes=tuple(range(len(xyz)))`` for the vmapped function, but
# after ``jnp.meshgrid(...).flatten()`` every coordinate array is 1D, so vmap
# over axis ``i>0`` fails ("axis i is out of bounds"); with a single input the
# subsequent ``jnp.moveaxis(v, 1, 0)`` fails for the same reason.  No valid
# input shape reaches ``jnp.sign(...)``.  We assert it raises to document this.
def test_get_sign2_is_broken():
    f = lambda x, y: x * y
    xs = jnp.linspace(-1., 1., 3)
    ys = jnp.linspace(-1., 1., 3)
    with pytest.raises((ValueError, IndexError, TypeError)):
        o.get_sign2(f, xs, ys)


# --------------------------------------------------------------------------- #
# keep_unique (array)
# --------------------------------------------------------------------------- #
def test_keep_unique_array_drops_duplicates():
    pts = np.array([[0., 0.], [0., 0.001], [5., 5.]])
    fps, ids = o.keep_unique(pts, tolerance=2.5e-2)
    # first two collapse to one
    assert fps.shape[0] == 2
    assert 0 in ids and 2 in ids and 1 not in ids


def test_keep_unique_array_tolerance_nonpositive():
    pts = np.array([[0., 0.], [0., 0.], [1., 1.]])
    fps, ids = o.keep_unique(pts, tolerance=0.0)
    assert fps.shape[0] == 3
    assert list(ids) == [0, 1, 2]


def test_keep_unique_single_point():
    pts = np.array([[1., 2.]])
    fps, ids = o.keep_unique(pts, tolerance=1e-2)
    assert fps.shape[0] == 1
    assert list(ids) == [0]


def test_keep_unique_all_collapse_to_one():
    # All points within tolerance: only the first is kept.
    pts = np.array([[0., 0.], [0.001, 0.], [0., 0.001]])
    fps, ids = o.keep_unique(pts, tolerance=1.0)
    assert fps.shape[0] == 1
    assert list(ids) == [0]


def test_keep_unique_dict():
    cands = {'a': np.array([[0.], [0.001], [5.]]),
             'b': np.array([[0.], [0.0], [5.]])}
    fps, ids = o.keep_unique(cands, tolerance=2.5e-2)
    assert isinstance(fps, dict)
    assert fps['a'].shape[0] == 2


# --------------------------------------------------------------------------- #
# keep_unique_jax
# --------------------------------------------------------------------------- #
def test_keep_unique_jax_drops_duplicates():
    pts = jnp.array([[0., 0.], [0., 0.001], [5., 5.]])
    fps, ids = o.keep_unique_jax(pts, tolerance=2.5e-2)
    assert np.asarray(fps).shape[0] == 2


def test_keep_unique_jax_tolerance_nonpositive():
    pts = jnp.array([[0., 0.], [0., 0.], [1., 1.]])
    fps, ids = o.keep_unique_jax(pts, tolerance=0.0)
    assert np.asarray(fps).shape[0] == 3


def test_keep_unique_jax_single_point():
    pts = jnp.array([[1., 2.]])
    fps, ids = o.keep_unique_jax(pts, tolerance=1e-2)
    assert np.asarray(fps).shape[0] == 1


# --------------------------------------------------------------------------- #
# rescale
# --------------------------------------------------------------------------- #
def test_rescale():
    lo, hi = o.rescale((0., 10.), scale=0.1)
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(11.0)
