# -*- coding: utf-8 -*-
"""Line-coverage tests for ``brainpy/connect/random_conn.py``.

``boost_connect_test.py`` already walks every public connector with numba
*enabled* -- but when numba is installed (``SUPPORT_NUMBA is True``) the
connectors route through ``@numba_jit``-compiled closures
(``FixedProb._iii.single_conn``, ``FixedPreNum``/``FixedPostNum``
``single_conn``, ``SmallWorld._smallworld_rewire``, the BA / PowerLaw
``_random_subset`` samplers, and the four ``ProbDist._connect_*d_jit``
kernels). Numba compiles those function bodies, so the Python interpreter
never executes their source lines and ``coverage`` cannot instrument them.
That left ~20% of the file (the closure interiors plus the
``SUPPORT_NUMBA is False`` fallback branches) permanently dark under the
default configuration.

This module closes the gap by **forcing the pure-Python fallback**: a fixture
monkeypatches ``random_conn.SUPPORT_NUMBA -> False`` and neutralises
``numba_jit`` / ``numba_seed`` so the very same closures run as ordinary
Python and become instrumentable. ``random_conn.pos2ind`` (decorated at import
time) is swapped for a behaviourally identical pure-Python copy so the
``ProbDist`` kernels resolve it without njit. Each connector is then exercised
on tiny networks (<=20 nodes) across its meaningful branches:
``include_self`` True/False, ``allow_multi_conn`` True/False, float vs int
``num``, ``pre_ratio < 1``, ``isOptimized`` True/False, 1D..4D ``ProbDist``,
and the validation/error branches.

PINNED DEFECT (not fixed, mirrors ``boost_connect_test.py``):
``SmallWorld(directed=True).build_conn`` calls ``self._connect(prob=..., i=...,
all_j=...)`` but the rewire closure only accepts ``(i, all_j)`` -- ``prob`` is
captured, not a parameter -- so it raises ``TypeError``. This holds in both
the numba and the pure-Python configurations. See
``test_smallworld_directed_defect_pure_python``.

NOTE: ``pos2ind`` (lines 1077-1081) is njit-compiled at import time and is
swapped out wholesale here, so its *own* source lines are not instrumented by
these tests; they remain numba-only and are excluded from the achievable
percentage.
"""

import numpy as np
import pytest

import brainpy as bp
import brainpy.connect.random_conn as rc
from brainpy._errors import ConnectorError


# ---------------------------------------------------------------------------
# Fixture: force the pure-Python (numba-disabled) code paths so the closure
# bodies are executed by the interpreter and become coverage-instrumentable.
# ---------------------------------------------------------------------------

def _pure_pos2ind(pos, size):
    idx = 0
    for i, p in enumerate(pos):
        idx += p * np.prod(size[i + 1:])
    return idx


@pytest.fixture
def no_numba(monkeypatch):
    monkeypatch.setattr(rc, 'SUPPORT_NUMBA', False)
    monkeypatch.setattr(
        rc, 'numba_jit',
        lambda f=None, **kw: (f if f is not None else (lambda g: g)))
    monkeypatch.setattr(rc, 'numba_seed', lambda s: None)
    monkeypatch.setattr(rc, 'pos2ind', _pure_pos2ind)
    yield


# ===========================================================================
# FixedProb -- pure-Python single_conn + else-branch (allow_multi_conn False)
# ===========================================================================

def test_fixedprob_single_conn_pure_python(no_numba):
    fp = rc.FixedProb(0.3, seed=1)((12,), (12,))
    pre, post = fp.build_coo()
    assert pre.shape == post.shape
    indices, indptr = fp.build_csr()
    assert indptr.shape[0] == fp.pre_num + 1


def test_fixedprob_include_self_false_pure_python(no_numba):
    fp = rc.FixedProb(0.5, include_self=False, seed=2)((12,), (12,))
    pre, post = fp.build_coo()
    assert bool(np.all(np.asarray(pre) != np.asarray(post)))
    # build_csr include_self=False branch
    indices, indptr = fp.build_csr()
    assert bool(np.all(np.diff(np.asarray(indptr)) >= 0))


def test_fixedprob_pre_ratio_pure_python(no_numba):
    fp = rc.FixedProb(0.5, pre_ratio=0.5, seed=3)((10,), (10,))
    indices, indptr = fp.build_csr()
    # only int(10 * 0.5) = 5 selected pre rows -> 6 indptr entries
    assert indptr.shape[0] == int(10 * 0.5) + 1


def test_fixedprob_allow_multi_conn_jax_path():
    # allow_multi_conn skips numba entirely (jax randint); no fixture needed.
    fp = rc.FixedProb(0.4, allow_multi_conn=True, seed=4)((10,), (10,))
    pre, post = fp.build_coo()
    assert pre.shape == post.shape


# ===========================================================================
# FixedPreNum / FixedPostNum -- pure-Python single_conn bodies
# ===========================================================================

def test_fixedprenum_single_conn_pure_python(no_numba):
    c = rc.FixedPreNum(num=3, seed=5)((12,), (12,))
    pre, post = c.build_coo()
    assert pre.shape == post.shape
    # include_self False (square) branch
    c = rc.FixedPreNum(num=3, include_self=False, seed=5)((12,), (12,))
    pre, post = c.build_coo()
    assert pre.shape == post.shape
    # float num branch
    c = rc.FixedPreNum(num=0.25, seed=5)((12,), (12,))
    assert c.build_coo()[0].shape == c.build_coo()[1].shape


def test_fixedpostnum_single_conn_pure_python(no_numba):
    c = rc.FixedPostNum(num=3, seed=6)((12,), (12,))
    pre, post = c.build_coo()
    assert pre.shape == post.shape
    indices, indptr = c.build_csr()
    assert indptr.shape[0] == 13
    # include_self False branch through build_csr
    c = rc.FixedPostNum(num=3, include_self=False, seed=6)((12,), (12,))
    indices, indptr = c.build_csr()
    assert indptr.shape[0] == 13


# ===========================================================================
# SmallWorld -- pure-Python _smallworld_rewire closure
# ===========================================================================

def test_smallworld_undirected_rewire_pure_python(no_numba):
    # high prob -> rewire body runs (non_connected length checks, choice loop)
    sw = rc.SmallWorld(num_neighbor=4, prob=0.95, include_self=False, seed=7)
    m = sw(12, 12).require('conn_mat')
    assert np.asarray(m).shape == (12, 12)


def test_smallworld_include_self_true_pure_python(no_numba):
    sw = rc.SmallWorld(num_neighbor=4, prob=0.95, include_self=True, seed=8)
    m = sw(12, 12).require('conn_mat')
    assert np.asarray(m).shape == (12, 12)


def test_smallworld_complete_graph_pure_python(no_numba):
    # num_neighbor == num_node -> complete-graph short circuit
    sw = rc.SmallWorld(num_neighbor=10, prob=0.5, seed=9)
    m = np.asarray(sw(10, 10).require('conn_mat'))
    assert m.sum() == 100


def test_smallworld_dense_ring_non_connected_branch_pure_python(no_numba):
    # num_neighbor just below num_node -> each node is connected to almost all
    # others, so the rewire closure's ``non_connected`` candidate set is tiny
    # and the ``len(non_connected) <= 1 -> return -1`` early-out branch fires.
    sw = rc.SmallWorld(num_neighbor=8, prob=1.0, include_self=False, seed=99)
    m = np.asarray(sw(10, 10).require('conn_mat'))
    assert m.shape == (10, 10)


def test_smallworld_errors_pure_python(no_numba):
    with pytest.raises(ConnectorError):
        rc.SmallWorld(num_neighbor=30, prob=0.5, seed=10)(10, 10).require('conn_mat')
    with pytest.raises(ConnectorError):
        rc.SmallWorld(num_neighbor=4, prob=0.5, seed=10)((6, 6), (6, 6)).require('conn_mat')


def test_smallworld_directed_defect_pure_python(no_numba):
    """PINNED DEFECT: the directed rewire path calls the 2-arg closure with an
    extra ``prob=`` keyword -> TypeError. Holds with numba disabled too."""
    sw = rc.SmallWorld(num_neighbor=4, prob=0.9, directed=True, seed=11)
    with pytest.raises(TypeError):
        sw(12, 12).require('conn_mat')


# ===========================================================================
# ScaleFreeBA -- pure-Python _random_subset sampler
# ===========================================================================

def test_scalefreeba_pure_python_opt_and_not(no_numba):
    for opt in (True, False):
        c = rc.ScaleFreeBA(m=3, seed=12)(20, 20)
        assert c.build_mat(isOptimized=opt).shape == (20, 20)


def test_scalefreeba_directed_pure_python(no_numba):
    c = rc.ScaleFreeBA(m=3, directed=True, seed=13)(20, 20)
    assert c.build_mat().shape == (20, 20)
    assert c.build_mat(isOptimized=False).shape == (20, 20)


def test_scalefreeba_error(no_numba):
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBA(m=50, seed=14)(10, 10).build_mat()


# ===========================================================================
# ScaleFreeBADual -- pure-Python _random_subset sampler
# ===========================================================================

def test_scalefreebadual_pure_python_opt_and_not(no_numba):
    for opt in (True, False):
        c = rc.ScaleFreeBADual(m1=2, m2=3, p=0.5, seed=15)(20, 20)
        assert c.build_mat(isOptimized=opt).shape == (20, 20)


def test_scalefreebadual_directed_pure_python(no_numba):
    c = rc.ScaleFreeBADual(m1=2, m2=3, p=0.5, directed=True, seed=16)(20, 20)
    assert c.build_mat().shape == (20, 20)
    assert c.build_mat(isOptimized=False).shape == (20, 20)


def test_scalefreebadual_errors(no_numba):
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=50, m2=3, p=0.5, seed=17)(10, 10).build_mat()
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=2, m2=50, p=0.5, seed=17)(10, 10).build_mat()
    with pytest.raises(ConnectorError):
        rc.ScaleFreeBADual(m1=2, m2=3, p=1.5, seed=17)(20, 20).build_mat()


# ===========================================================================
# PowerLaw -- pure-Python _random_subset sampler + clustering step
# ===========================================================================

def test_powerlaw_pure_python_opt_and_not(no_numba):
    for opt in (True, False):
        # high p increases the chance the clustering ("triangle") branch runs
        c = rc.PowerLaw(m=3, p=0.8, seed=18)(20, 20)
        assert c.build_mat(isOptimized=opt).shape == (20, 20)


def test_powerlaw_directed_pure_python(no_numba):
    c = rc.PowerLaw(m=3, p=0.8, directed=True, seed=19)(20, 20)
    assert c.build_mat().shape == (20, 20)
    assert c.build_mat(isOptimized=False).shape == (20, 20)


def test_powerlaw_errors(no_numba):
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=3, p=1.5, seed=20)
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=3, p=-0.1, seed=20)
    with pytest.raises(ConnectorError):
        rc.PowerLaw(m=50, p=0.3, seed=20)(10, 10).build_mat()


# ===========================================================================
# ProbDist -- pure-Python _connect_{1,2,3,4}d_jit kernels + pos2ind
# ===========================================================================

def test_probdist_1d_pure_python(no_numba):
    c = rc.ProbDist(dist=2, prob=1.0, pre_ratio=1.0, seed=21, include_self=True)((12,), (12,))
    pre, post = c.build_coo()
    assert len(pre) == len(post) > 0


def test_probdist_2d_3d_4d_pure_python(no_numba):
    for size in [(5, 5), (3, 3, 3), (2, 2, 2, 2)]:
        c = rc.ProbDist(dist=2, prob=1.0, pre_ratio=1.0, seed=22, include_self=True)(size, size)
        pre, post = c.build_coo()
        assert len(pre) == len(post) > 0


def test_probdist_include_self_false_and_pre_ratio_pure_python(no_numba):
    c = rc.ProbDist(dist=2, prob=0.8, pre_ratio=0.5, seed=23, include_self=False)((12,), (12,))
    pre, post = c.build_coo()
    assert len(pre) == len(post)


def test_probdist_include_self_false_multidim_pure_python(no_numba):
    # include_self=False forces the ``d == 0 -> continue`` self-skip branch in
    # the 2D / 3D / 4D kernels (the ``i == j (== k (== l))`` diagonal cell).
    for size in [(5, 5), (3, 3, 3), (2, 2, 2, 2)]:
        c = rc.ProbDist(dist=3, prob=1.0, pre_ratio=1.0, seed=33, include_self=False)(size, size)
        pre, post = c.build_coo()
        # no self connection survives
        assert bool(np.all(np.asarray(pre) != np.asarray(post)))


def test_probdist_errors_pure_python(no_numba):
    with pytest.raises(ValueError):
        rc.ProbDist(dist=1, seed=24)((6, 6), (12,)).build_coo()
    with pytest.raises(NotImplementedError):
        rc.ProbDist(dist=1, seed=24)((2, 2, 2, 2, 2), (2, 2, 2, 2, 2)).build_coo()


# ===========================================================================
# FixedTotalNum -- non-numba (jax / np.random) branches; no fixture needed
# ===========================================================================

def test_fixedtotalnum_branches():
    c = rc.FixedTotalNum(num=40, seed=25)((10,), (10,))
    pre, post = c.build_coo()
    assert pre.shape == (40,)
    c = rc.FixedTotalNum(num=30, allow_multi_conn=True, seed=26)((10,), (10,))
    assert c.build_coo()[0].shape == (30,)
    with pytest.raises(ConnectorError):
        rc.FixedTotalNum(num=1000, seed=27)((10,), (10,)).build_coo()


# ===========================================================================
# GaussianProb -- pure numpy (no numba); exercise optimized & not + repr
# ===========================================================================

def test_gaussianprob_optimized_and_not():
    for opt in (True, False):
        g = rc.GaussianProb(sigma=1.5, seed=28)((12,))
        assert g.build_mat(isOptimized=opt).shape == (12, 12)
    g = rc.GaussianProb(sigma=2.0, periodic_boundary=True, seed=28)((6, 6))
    assert g.build_mat(isOptimized=False).shape == (36, 36)
    assert 'GaussianProb' in repr(g)
