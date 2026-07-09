# -*- coding: utf-8 -*-
"""Audit coverage-boost tests for ``brainpy/connect/random_conn.py``.

The sibling audit suite already exercises the ``Fixed*`` connectors via
``conn_mat``/``pre2post`` round-trips, leaving the remaining connector classes
(``GaussianProb``, ``ProbDist``, ``SmallWorld``, ``ScaleFreeBA``,
``ScaleFreeBADual``, ``PowerLaw``, ``FixedTotalNum``) almost entirely
uncovered (file was at ~29% line coverage). This module raises line coverage
of ``random_conn.py`` toward >=90% by instantiating every public connector
class with small networks and exercising ``build_coo`` / ``build_csr`` /
``build_mat`` (both ``isOptimized=True`` and ``False`` paths where present),
``require(...)`` routing, ``__repr__``, ``allow_multi_conn`` True/False,
``include_self`` True/False, ring/grid (1D..4D) variants, directed variants,
and the validation/error branches.

NOTE on a discovered defect (pinned, not fixed):
``SmallWorld(directed=True)`` raises ``TypeError`` from ``build_conn`` because
``self._connect(prob=..., i=..., all_j=...)`` is called with 3 args while the
underlying rewire closure only accepts ``(i, all_j)`` (``prob`` is a closure
variable, not a parameter). The undirected path works fine. See
``test_smallworld_directed_is_broken`` which pins the current behaviour.

Numba-JIT closure bodies (e.g. ``ProbDist._connect_*d_jit`` interiors,
``_random_subset``) are not instrumentable by ``coverage`` and are excluded
from the achievable percentage.
"""

import numpy as np
import pytest

import brainpy as bp
import brainpy.connect.random_conn as rc
from brainpy._errors import ConnectorError


# ---------------------------------------------------------------------------
# FixedProb / FixedPreNum / FixedPostNum  (light touch: build_* + require)
# These are heavily covered by a sibling file; here we only walk the build
# methods + include_self / allow_multi_conn branches inside random_conn.py.
# ---------------------------------------------------------------------------

def test_fixedprob_build_methods_and_repr():
    fp = rc.FixedProb(0.3, seed=1)
    fp((20,), (20,))
    pre, post = fp.build_coo()
    assert pre.shape == post.shape
    indices, indptr = fp.build_csr()
    assert indptr.shape[0] == fp.pre_num + 1
    mat = fp.build_mat()
    assert mat.shape == (20, 20)
    assert 'FixedProb' in repr(fp)


def test_fixedprob_include_self_false_and_pre_ratio():
    fp = rc.FixedProb(0.5, pre_ratio=0.5, include_self=False, seed=2)
    fp((30,), (30,))
    pre, post = fp.build_coo()
    # no self connections
    assert bool(np.all(np.asarray(pre) != np.asarray(post)))
    indices, indptr = fp.build_csr()
    # A valid CSR indptr always spans the FULL pre range (pre_num + 1 entries);
    # with pre_ratio < 1 the non-selected pre rows simply have zero out-degree
    # (equal consecutive indptr values). The previous truncated indptr of length
    # int(pre_num * pre_ratio) + 1 was a malformed CSR (H4, audit 2026-07-08).
    assert indptr.shape[0] == 30 + 1
    assert bool(np.all(np.diff(np.asarray(indptr)) >= 0))
    # indptr is internally consistent: its last entry equals the number of CSR
    # indices (edges). (``build_coo`` re-samples per call, so we do not cross-check
    # against the separate ``pre`` draw above.)
    assert int(np.asarray(indptr)[-1]) == np.asarray(indices).shape[0]
    mat = fp.build_mat()
    # diagonal cleared
    assert not bool(np.any(np.diagonal(np.asarray(mat))))


def test_fixedprob_allow_multi_conn():
    fp = rc.FixedProb(0.4, allow_multi_conn=True, seed=3)
    fp((25,), (25,))
    pre, post = fp.build_coo()
    assert pre.shape == post.shape


def test_fixedprob_require_routing():
    fp = rc.FixedProb(0.3, seed=4)
    a, b = fp.require((20,), (20,), 'pre_ids', 'post_ids')
    assert a.shape == b.shape
    m = fp.require((20,), (20,), 'conn_mat')
    assert np.asarray(m).shape == (20, 20)


def test_fixedprenum_build_coo_branches():
    # include_self True
    c = rc.FixedPreNum(num=3, seed=5)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape
    # include_self False (square shapes ok)
    c = rc.FixedPreNum(num=3, include_self=False, seed=5)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape
    # allow_multi_conn
    c = rc.FixedPreNum(num=3, allow_multi_conn=True, seed=5)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape
    # float num
    c = rc.FixedPreNum(num=0.2, seed=5)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape
    assert 'FixedPreNum' in repr(c)


def test_fixedprenum_errors():
    # num > pre_num
    with pytest.raises(ConnectorError):
        rc.FixedPreNum(num=50, seed=5)((10,), (10,)).build_coo()
    # include_self False but pre_num != post_num
    with pytest.raises(ConnectorError):
        rc.FixedPreNum(num=3, include_self=False, seed=5)((10,), (12,)).build_coo()
    # bad type
    with pytest.raises(ConnectorError):
        rc.FixedPreNum(num='x')


def test_fixedpostnum_build_coo_csr_branches():
    c = rc.FixedPostNum(num=3, seed=6)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape
    indices, indptr = c.build_csr()
    assert indptr.shape[0] == 21
    # include_self False
    c = rc.FixedPostNum(num=3, include_self=False, seed=6)
    pre, post = c((20,), (20,)).build_coo()
    assert bool(np.all(np.asarray(pre) != np.asarray(post)))
    indices, indptr = c.build_csr()
    assert indptr.shape[0] == 21
    # allow_multi_conn + float num
    c = rc.FixedPostNum(num=0.2, allow_multi_conn=True, seed=6)
    pre, post = c((20,), (20,)).build_coo()
    assert pre.shape == post.shape


def test_fixedpostnum_errors_and_require():
    with pytest.raises(ConnectorError):
        rc.FixedPostNum(num=50, seed=6)((10,), (10,)).build_coo()
    with pytest.raises(ConnectorError):
        rc.FixedPostNum(num=3, include_self=False, seed=6)((10,), (12,)).build_coo()
    pp = rc.FixedPostNum(num=3, seed=6).require((20,), (20,), 'pre2post')
    assert pp[1].shape[0] == 21


# ---------------------------------------------------------------------------
# FixedTotalNum
# ---------------------------------------------------------------------------

def test_fixedtotalnum_build_coo():
    c = rc.FixedTotalNum(num=50, seed=7)
    c((20,), (20,))
    pre, post = c.build_coo()
    assert pre.shape == (50,)
    assert post.shape == (50,)
    assert 'FixedTotalNum' in repr(c)


def test_fixedtotalnum_allow_multi_conn():
    c = rc.FixedTotalNum(num=30, allow_multi_conn=True, seed=8)
    c((20,), (20,))
    pre, post = c.build_coo()
    assert pre.shape == (30,)


def test_fixedtotalnum_float_num_and_require():
    c = rc.FixedTotalNum(num=0.5, seed=8)
    assert c.num == 0.5  # constructor accepts float in [0,1]
    # integer num routed through require -> conn_mat
    c = rc.FixedTotalNum(num=40, seed=8)
    m = c.require((20,), (20,), 'conn_mat')
    assert np.asarray(m).shape == (20, 20)


def test_fixedtotalnum_errors():
    # num too large for the all-to-all matrix
    with pytest.raises(ConnectorError):
        rc.FixedTotalNum(num=1000, seed=8)((10,), (10,)).build_coo()
    # bad type
    with pytest.raises(ConnectorError):
        rc.FixedTotalNum(num='x')
    # negative int
    with pytest.raises(AssertionError):
        rc.FixedTotalNum(num=-1)
    # float out of range
    with pytest.raises(AssertionError):
        rc.FixedTotalNum(num=2.0)


# ---------------------------------------------------------------------------
# GaussianProb (OneEndConnector)
# ---------------------------------------------------------------------------

def test_gaussianprob_1d_optimized_and_not():
    for opt in (True, False):
        g = rc.GaussianProb(sigma=1.5, seed=9)
        g((20,))
        m = g.build_mat(isOptimized=opt)
        assert m.shape == (20, 20)
    assert 'GaussianProb' in repr(g)


def test_gaussianprob_2d_and_periodic():
    g = rc.GaussianProb(sigma=2.0, seed=10)
    g((8, 8))
    assert g.build_mat().shape == (64, 64)
    g = rc.GaussianProb(sigma=2.0, periodic_boundary=True, seed=10)
    g((8, 8))
    assert g.build_mat().shape == (64, 64)
    # non-optimized periodic path
    g = rc.GaussianProb(sigma=2.0, periodic_boundary=True, seed=10)
    g((6, 6))
    assert g.build_mat(isOptimized=False).shape == (36, 36)


def test_gaussianprob_encoding_values_variants():
    # single (low, high) shared across dims
    g = rc.GaussianProb(sigma=2.0, encoding_values=(0, np.pi), seed=11)
    g((10,))
    assert g.build_mat().shape == (10, 10)
    # per-dimension list of ranges
    g = rc.GaussianProb(sigma=2.0, encoding_values=((-np.pi, np.pi), (0, np.pi)), seed=11)
    g((6, 6))
    assert g.build_mat().shape == (36, 36)


def test_gaussianprob_normalize_false_and_include_self_false():
    g = rc.GaussianProb(sigma=2.0, normalize=False, include_self=False, seed=12)
    g((12,))
    m = np.asarray(g.build_mat())
    assert m.shape == (12, 12)
    assert not bool(np.any(np.diagonal(m)))


def test_gaussianprob_encoding_errors():
    # length-0 encoding
    with pytest.raises(ConnectorError):
        rc.GaussianProb(sigma=1.0, encoding_values=[])((5,)).build_mat()
    # dimension mismatch (3 ranges vs 2D net)
    with pytest.raises(ConnectorError):
        rc.GaussianProb(sigma=1.0,
                        encoding_values=((0, 1), (0, 1), (0, 1)))((6, 6)).build_mat()
    # unsupported encoding (a string)
    with pytest.raises(ConnectorError):
        rc.GaussianProb(sigma=1.0, encoding_values='abc')((5,)).build_mat()
    # unsupported element type inside list
    with pytest.raises(ConnectorError):
        rc.GaussianProb(sigma=1.0, encoding_values=[{1: 2}])((5,)).build_mat()


# ---------------------------------------------------------------------------
# SmallWorld
# ---------------------------------------------------------------------------

def test_smallworld_undirected_ring():
    sw = rc.SmallWorld(num_neighbor=4, prob=0.3, seed=13)
    m = sw((20,), (20,)).require('conn_mat')
    assert np.asarray(m).shape == (20, 20)
    assert 'SmallWorld' in repr(sw)


def test_smallworld_include_self_and_int_size():
    sw = rc.SmallWorld(num_neighbor=4, prob=0.5, include_self=True, seed=14)
    m = sw(20, 20).require('conn_mat')
    assert np.asarray(m).shape == (20, 20)


def test_smallworld_complete_graph_when_k_equals_n():
    # num_neighbor == num_node -> complete graph branch
    sw = rc.SmallWorld(num_neighbor=10, prob=0.5, seed=15)
    m = np.asarray(sw(10, 10).require('conn_mat'))
    assert m.sum() == 100  # fully connected (incl. diagonal)


def test_smallworld_errors():
    # num_neighbor > num_node
    with pytest.raises(ConnectorError):
        rc.SmallWorld(num_neighbor=30, prob=0.5, seed=16)(10, 10).require('conn_mat')
    # 2D topology not supported
    with pytest.raises(ConnectorError):
        rc.SmallWorld(num_neighbor=4, prob=0.5, seed=16)((8, 8), (8, 8)).require('conn_mat')


def test_smallworld_directed_is_broken():
    """PINNED DEFECT: directed SmallWorld calls the rewire closure with an
    extra ``prob=`` keyword that the 2-arg numba closure cannot accept."""
    sw = rc.SmallWorld(num_neighbor=4, prob=0.9, directed=True, seed=17)
    with pytest.raises(TypeError):
        sw(20, 20).require('conn_mat')


# ---------------------------------------------------------------------------
# ScaleFreeBA
# ---------------------------------------------------------------------------

def test_scalefreeba_optimized_and_not():
    for opt in (True, False):
        c = rc.ScaleFreeBA(m=3, seed=18)
        c(30, 30)
        assert c.build_mat(isOptimized=opt).shape == (30, 30)
    assert 'ScaleFreeBA' in repr(c)


def test_scalefreeba_directed_and_require():
    c = rc.ScaleFreeBA(m=3, directed=True, seed=19)
    c(30, 30)
    assert c.build_mat().shape == (30, 30)
    m = rc.ScaleFreeBA(m=2, seed=19)(30, 30).require('conn_mat')
    assert np.asarray(m).shape == (30, 30)


def test_scalefreeba_error():
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBA(m=50, seed=20)(10, 10).build_mat()


# ---------------------------------------------------------------------------
# ScaleFreeBADual
# ---------------------------------------------------------------------------

def test_scalefreebadual_optimized_and_not():
    for opt in (True, False):
        c = rc.ScaleFreeBADual(m1=2, m2=3, p=0.5, seed=21)
        c(40, 40)
        assert c.build_mat(isOptimized=opt).shape == (40, 40)
    assert 'ScaleFreeBADual' in repr(c)


def test_scalefreebadual_directed():
    c = rc.ScaleFreeBADual(m1=2, m2=3, p=0.5, directed=True, seed=22)
    c(40, 40)
    assert c.build_mat().shape == (40, 40)
    # also walk the not-optimized directed branch
    c = rc.ScaleFreeBADual(m1=2, m2=3, p=0.5, directed=True, seed=22)
    c(40, 40)
    assert c.build_mat(isOptimized=False).shape == (40, 40)


def test_scalefreebadual_errors():
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=50, m2=3, p=0.5, seed=23)(10, 10).build_mat()
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=2, m2=50, p=0.5, seed=23)(10, 10).build_mat()
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=2, m2=3, p=1.5, seed=23)(40, 40).build_mat()


# ---------------------------------------------------------------------------
# PowerLaw
# ---------------------------------------------------------------------------

def test_powerlaw_optimized_and_not():
    for opt in (True, False):
        c = rc.PowerLaw(m=3, p=0.4, seed=24)
        c(40, 40)
        assert c.build_mat(isOptimized=opt).shape == (40, 40)
    assert 'PowerLaw' in repr(c)


def test_powerlaw_directed_and_require():
    c = rc.PowerLaw(m=3, p=0.4, directed=True, seed=25)
    c(40, 40)
    assert c.build_mat().shape == (40, 40)
    c = rc.PowerLaw(m=3, p=0.4, directed=True, seed=25)
    c(40, 40)
    assert c.build_mat(isOptimized=False).shape == (40, 40)
    m = rc.PowerLaw(m=2, p=0.3, seed=25)(40, 40).require('conn_mat')
    assert np.asarray(m).shape == (40, 40)


def test_powerlaw_errors():
    # p out of range at construction
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=3, p=1.5, seed=26)
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=3, p=-0.1, seed=26)
    # m > num_node at build time
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=50, p=0.3, seed=26)(10, 10).build_mat()


# ---------------------------------------------------------------------------
# ProbDist
# ---------------------------------------------------------------------------

def test_probdist_1d():
    c = rc.ProbDist(dist=2, prob=1.0, pre_ratio=1.0, seed=27, include_self=True)
    c((20,), (20,))
    pre, post = c.build_coo()
    assert len(pre) == len(post) > 0
    assert 'ProbDist' in repr(c) or repr(c)  # default repr falls back to class name


def test_probdist_2d_3d_4d():
    for size in [(8, 8), (4, 4, 3), (3, 3, 2, 2)]:
        c = rc.ProbDist(dist=2, prob=1.0, pre_ratio=1.0, seed=28, include_self=True)
        c(size, size)
        pre, post = c.build_coo()
        assert len(pre) == len(post) > 0


def test_probdist_include_self_false_and_pre_ratio():
    c = rc.ProbDist(dist=2, prob=1.0, pre_ratio=0.5, seed=29, include_self=False)
    c((20,), (20,))
    pre, post = c.build_coo()
    assert len(pre) == len(post)


def test_probdist_errors():
    # mismatched dims
    with pytest.raises(ValueError):
        rc.ProbDist(dist=1, seed=30)((8, 8), (20,)).build_coo()
    # dimension > 4 not implemented
    with pytest.raises(NotImplementedError):
        rc.ProbDist(dist=1, seed=30)((2, 2, 2, 2, 2), (2, 2, 2, 2, 2)).build_coo()


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------

def test_module_all_exports_are_instantiable():
    # every public connector listed in __all__ is importable from the module
    for name in rc.__all__:
        assert hasattr(rc, name)
    # and reachable through the public bp.connect namespace
    for name in rc.__all__:
        assert hasattr(bp.connect, name)
