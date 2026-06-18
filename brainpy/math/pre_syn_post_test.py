# -*- coding: utf-8 -*-
"""Regression tests for ``brainpy/math/pre_syn_post.py``.

Targets the event-driven CSR routing of ``pre2post_event_sum`` (which delegates
to ``event.csrmv(transpose=True)``) and the empty-group / structural guards of
``syn2post_mean`` and ``syn2post_softmax``.
"""

import numpy as np

import brainpy.math as bm
from brainpy.math.pre_syn_post import (
    pre2post_event_sum,
    syn2post_sum,
    syn2post_mean,
    syn2post_softmax,
)


# pre_num=3, post_num=4 CSR: pre0 -> {1,3}, pre1 -> {0}, pre2 -> {1,3}
_INDICES = np.array([1, 3, 0, 1, 3])
_INDPTR = np.array([0, 2, 3, 5])
_POST_NUM = 4


def test_pre2post_event_sum_scalar_value():
    events = np.array([True, False, True])  # pre 0 and 2 fire
    out = np.asarray(pre2post_event_sum(events, (_INDICES, _INDPTR), _POST_NUM, values=1.))
    np.testing.assert_array_equal(out, [0., 2., 0., 2.])


def test_pre2post_event_sum_vector_value():
    events = np.array([True, False, True])
    vals = np.array([10., 20., 30., 40., 50.])
    out = np.asarray(pre2post_event_sum(events, (_INDICES, _INDPTR), _POST_NUM, values=vals))
    # pre0: post1+=10, post3+=20; pre2: post1+=40, post3+=50
    np.testing.assert_array_equal(out, [0., 50., 0., 70.])


def test_pre2post_event_sum_matches_dense_transpose():
    # equivalent dense Aᵀ @ events, with A (pre_num x post_num) of all-ones weights
    events = np.array([True, True, False])
    A = np.zeros((3, _POST_NUM), dtype=np.float32)
    for pre in range(3):
        for j in range(_INDPTR[pre], _INDPTR[pre + 1]):
            A[pre, _INDICES[j]] = 1.0
    out = np.asarray(pre2post_event_sum(events, (_INDICES, _INDPTR), _POST_NUM, values=1.))
    np.testing.assert_allclose(out, A.T @ events.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_syn2post_sum_matches_reference():
    syn = np.array([1., 2., 3., 4.])
    post_ids = np.array([0, 0, 2, 2])
    out = np.asarray(syn2post_sum(syn, post_ids, 3))
    np.testing.assert_array_equal(out, [3., 0., 7.])


def test_syn2post_mean_empty_group_is_zero_not_nan():
    syn = np.array([2., 4., 6.])
    post_ids = np.array([0, 0, 2])   # group 1 is empty
    out = np.asarray(syn2post_mean(syn, post_ids, 3))
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out, [3., 0., 6.], rtol=1e-6, atol=1e-6)


def test_syn2post_softmax_normalizes_per_group():
    syn = np.array([1., 2., 3., 4.])
    post_ids = np.array([0, 0, 1, 1])
    out = np.asarray(syn2post_softmax(syn, post_ids, 2))
    # within each post group the softmax weights sum to 1
    np.testing.assert_allclose(out[:2].sum(), 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out[2:].sum(), 1.0, rtol=1e-5, atol=1e-5)
    # values match a manual softmax of [1,2] and [3,4]
    s01 = np.exp([1., 2.] - np.max([1., 2.])); s01 /= s01.sum()
    np.testing.assert_allclose(out[:2], s01, rtol=1e-5, atol=1e-5)
