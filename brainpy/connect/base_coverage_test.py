# -*- coding: utf-8 -*-
"""Line-coverage tests for ``brainpy/connect/base.py``.

The sibling suites (``custom_conn_test.py``, ``random_conn_test.py``,
``boost_connect_test.py``) exercise concrete connectors but leave most of the
generic ``base.Connector`` / ``TwoEndConnector`` / ``OneEndConnector``
machinery uncovered (file was at ~74% line coverage). This module fills the
remaining gaps by driving the conversion-format plumbing directly:

* ``TwoEndConnector.__init__`` with ``pre``/``post`` (int + tuple forms) and
  ``is_version2_style``;
* the structure router ``require`` -- empty input, the missing-``pre_num``
  guard, every ``len(structures) == 1`` and ``len(structures) == 2`` fast
  path, and the multi-structure ``_make_returns`` fan-out through
  ``_return_by_coo`` / ``_return_by_csr`` / ``_return_by_mat``;
* the legacy ``build_conn`` dict/tuple contract plus its error branches;
* ``_check`` validation errors;
* ``OneEndConnector.__call__`` / ``_reset_conn`` (equal/tuple/None-post and
  the mismatch error);
* the free conversion helpers ``mat2csr/mat2coo/mat2csc``,
  ``csr2mat/csr2coo/csr2csc``, ``coo2mat`` (NumPy *and* JAX array paths),
  ``coo2csc`` with a ``data`` payload, ``coo2mat_num`` / ``mat2mat_num``,
  and ``set_default_dtype`` / ``get_idx_type`` / ``visualizeMat``.

Tiny networks (<=10 nodes) are used throughout. Custom ``TwoEndConnector``
subclasses with hand-written ``build_coo`` / ``build_csr`` / ``build_mat`` /
``build_conn`` provide deterministic connectivity so the format conversions
can be asserted exactly.

NOTES on the handful of lines that stay uncovered:

* ``visualizeMat``'s heatmap body (``sns.heatmap`` ... ``plt.show``, lines
  ~815-818) is only reachable when ``seaborn`` is installed; in this
  environment seaborn is absent, so only the ImportError-guard branch runs.
* ``_make_returns`` lines 384/386 are DEAD CODE: the guard reads
  ``if (PRE_IDS in structures) and (PRE_IDS not in structures)`` -- a value can
  never be simultaneously in and not-in the same collection, so the body never
  executes. (Pinned defect; almost certainly meant ``... not in all_data``.)
* ``require`` line 507 (``raise ValueError``) is contradictory/unreachable: it
  is the ``else`` of a block guarded by "at least one ``build_*`` is
  customized", so the fall-through can never be taken.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
import brainpy.connect.base as base
from brainpy.connect.base import (
    CONN_MAT, PRE_IDS, POST_IDS, PRE2POST, POST2PRE,
    PRE2SYN, POST2SYN, COO, CSR, CSC,
)
from brainpy._errors import ConnectorError


# ---------------------------------------------------------------------------
# Deterministic custom connectors used to drive the format machinery.
#
# Connectivity (pre -> post) on a 3x3 network:
#   0 -> 1
#   1 -> 0, 1 -> 2
#   2 -> 2
# COO  : pre=[0,1,1,2]   post=[1,0,2,2]
# CSR  : indices=[1,0,2,2]  indptr=[0,1,3,4]
# MAT  : [[0,1,0],[1,0,1],[0,0,1]]
# ---------------------------------------------------------------------------

_COO_PRE = np.array([0, 1, 1, 2], dtype=np.int32)
_COO_POST = np.array([1, 0, 2, 2], dtype=np.int32)
_CSR_IND = np.array([1, 0, 2, 2], dtype=np.int32)
_CSR_PTR = np.array([0, 1, 3, 4], dtype=np.int32)
_MAT = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=bool)


class _CooConn(base.TwoEndConnector):
    def build_coo(self):
        return _COO_PRE.copy(), _COO_POST.copy()


class _CsrConn(base.TwoEndConnector):
    def build_csr(self):
        return _CSR_IND.copy(), _CSR_PTR.copy()


class _MatConn(base.TwoEndConnector):
    def build_mat(self):
        return _MAT.copy()


class _FullConn(base.TwoEndConnector):
    """Implements all three builders so the 2-structure fast paths fire."""

    def build_coo(self):
        return _COO_PRE.copy(), _COO_POST.copy()

    def build_csr(self):
        return _CSR_IND.copy(), _CSR_PTR.copy()

    def build_mat(self):
        return _MAT.copy()


_ALL_STRUCTS = (CONN_MAT, PRE_IDS, POST_IDS, PRE2POST, POST2PRE,
                PRE2SYN, POST2SYN, COO, CSR, CSC)


# ---------------------------------------------------------------------------
# __init__ / is_version2_style / __repr__
# ---------------------------------------------------------------------------

def test_init_with_int_and_tuple_sizes():
    c = base.TwoEndConnector(pre=5, post=(2, 3))
    assert c.pre_size == (5,)
    assert c.post_size == (2, 3)
    assert c.pre_num == 5
    assert c.post_num == 6
    # __repr__ on the bare base falls back to the class name
    assert repr(c) == 'TwoEndConnector'


def test_init_with_tuple_pre_and_int_post():
    c = base.TwoEndConnector(pre=(2, 4), post=8)
    assert c.pre_size == (2, 4)
    assert c.pre_num == 8
    assert c.post_size == (8,)
    assert c.post_num == 8


def test_is_version2_style():
    # bare base: none of build_coo/csr/mat customized -> v1 style
    assert base.TwoEndConnector().is_version2_style is False
    # a custom connector implementing build_coo -> v2 style
    assert _CooConn().is_version2_style is True


def test_two_end_reset_conn():
    c = _MatConn()
    c._reset_conn(4, 6)
    assert c.pre_size == (4,)
    assert c.post_size == (6,)
    assert c.pre_num == 4
    assert c.post_num == 6


# ---------------------------------------------------------------------------
# require() guards & empty input
# ---------------------------------------------------------------------------

def test_require_empty_returns_empty_tuple():
    c = _MatConn()(3, 3)
    assert c.require() == tuple()


def test_require_without_sizes_raises():
    c = _MatConn()  # pre_num / post_num never set
    with pytest.raises(ConnectorError):
        c.require('conn_mat')


def test_check_unknown_structure_raises():
    c = _MatConn()(3, 3)
    with pytest.raises(ConnectorError):
        c.require('not_a_real_structure')


def test_check_none_and_empty_directly():
    # ``require()`` short-circuits before ``_check`` on empty input, so drive
    # the None / empty-list guard branches of ``_check`` directly.
    c = _MatConn()(3, 3)
    with pytest.raises(ConnectorError):
        c._check(None)
    with pytest.raises(ConnectorError):
        c._check([])


def test_check_accepts_bare_string():
    # ``_check`` wraps a bare string structure into a single-element list
    # (the ``structures = [structures]`` branch) before validating it.
    c = _MatConn()(3, 3)
    c._check(CONN_MAT)  # valid -> no raise
    with pytest.raises(ConnectorError):
        c._check('bogus_string')


def test_not_customized_build_stubs_return_none():
    # The @tools.not_customized decorator only tags the method; its body still
    # runs when invoked directly, so calling the stubs covers their ``pass``.
    c = base.TwoEndConnector()
    assert c.build_conn() is None
    assert c.build_mat() is None
    assert c.build_csr() is None
    assert c.build_coo() is None


def test_return_by_mat_direct_non_conn_mat_request():
    # ``_make_returns`` pre-populates CONN_MAT before delegating to
    # ``_return_by_mat``, so the in-method CONN_MAT branch is only reached by
    # calling ``_return_by_mat`` directly with CONN_MAT not yet in all_data.
    c = _MatConn()(3, 3)
    all_data = {}
    c._return_by_mat([CONN_MAT, COO], mat=_MAT.copy(), all_data=all_data)
    assert CONN_MAT in all_data
    assert COO in all_data
    assert np.asarray(all_data[CONN_MAT]).sum() == 4


def test_require_accepts_sizes_then_structure():
    c = _MatConn()
    m = c.require(3, 3, CONN_MAT)
    assert np.asarray(m).shape == (3, 3)
    # pre-size only (post stays None unless caller supplies it) is not enough
    # for a 2D-requiring connector, but here the matrix is built from sizes set
    # earlier; re-require with both sizes works.
    m2 = c.require((3,), (3,), CONN_MAT)
    assert np.asarray(m2).shape == (3, 3)


# ---------------------------------------------------------------------------
# require() len==1 fast paths
# ---------------------------------------------------------------------------

def test_require_single_fastpaths_full_connector():
    c = _FullConn()(3, 3)
    assert isinstance(c.require(COO), tuple)
    assert c.require(PRE_IDS).shape == (4,)
    assert c.require(POST_IDS).shape == (4,)
    assert np.asarray(c.require(CONN_MAT)).shape == (3, 3)
    pp = c.require(PRE2POST)
    assert pp[1].shape[0] == 4  # indptr length = pre_num + 1
    csr = c.require(CSR)
    assert csr[1].shape[0] == 4


def test_require_single_csr_only_connector():
    c = _CsrConn()(3, 3)
    pp = c.require(PRE2POST)
    assert pp[1].shape[0] == 4
    csr = c.require(CSR)
    assert csr[1].shape[0] == 4


# ---------------------------------------------------------------------------
# require() len==2 fast paths (both orderings)
# ---------------------------------------------------------------------------

def test_require_pair_pre_post_ids_both_orders():
    c = _FullConn()(3, 3)
    a, b = c.require(PRE_IDS, POST_IDS)
    assert np.array_equal(np.asarray(a), _COO_PRE)
    assert np.array_equal(np.asarray(b), _COO_POST)
    a, b = c.require(POST_IDS, PRE_IDS)
    assert np.array_equal(np.asarray(a), _COO_POST)
    assert np.array_equal(np.asarray(b), _COO_PRE)


def test_require_pair_csr_and_coo_both_orders():
    c = _FullConn()(3, 3)
    coo, csr = c.require(COO, CSR)
    assert isinstance(coo, tuple) and isinstance(csr, tuple)
    csr, coo = c.require(CSR, COO)
    assert isinstance(coo, tuple) and isinstance(csr, tuple)
    # PRE2POST also satisfies the "csr-ish" predicate
    pp, coo = c.require(PRE2POST, COO)
    assert isinstance(pp, tuple)


def test_require_pair_csr_and_mat_both_orders():
    c = _FullConn()(3, 3)
    csr, mat = c.require(CSR, CONN_MAT)
    assert np.asarray(mat).shape == (3, 3)
    mat, csr = c.require(CONN_MAT, CSR)
    assert np.asarray(mat).shape == (3, 3)


def test_require_pair_coo_and_mat_both_orders():
    c = _FullConn()(3, 3)
    coo, mat = c.require(COO, CONN_MAT)
    assert np.asarray(mat).shape == (3, 3)
    mat, coo = c.require(CONN_MAT, COO)
    assert np.asarray(mat).shape == (3, 3)


# ---------------------------------------------------------------------------
# require() multi-structure fan-out through _make_returns / _return_by_*
# ---------------------------------------------------------------------------

def test_require_all_structures_from_coo():
    c = _CooConn()(3, 3)
    res = c.require(*_ALL_STRUCTS)
    assert len(res) == len(_ALL_STRUCTS)
    out = dict(zip(_ALL_STRUCTS, res))
    assert np.asarray(out[CONN_MAT]).shape == (3, 3)
    assert np.array_equal(np.asarray(out[PRE_IDS]), _COO_PRE)
    assert np.array_equal(np.asarray(out[POST_IDS]), _COO_POST)
    # CSR indptr has pre_num + 1 entries
    assert out[CSR][1].shape[0] == 4
    # PRE2SYN syn_seq length equals number of edges
    assert out[PRE2SYN][0].shape[0] == 4


def test_require_all_structures_from_csr():
    c = _CsrConn()(3, 3)
    res = c.require(*_ALL_STRUCTS)
    assert len(res) == len(_ALL_STRUCTS)
    out = dict(zip(_ALL_STRUCTS, res))
    assert np.asarray(out[CONN_MAT]).shape == (3, 3)
    # POST_IDS are the csr indices
    assert np.array_equal(np.asarray(out[POST_IDS]), _CSR_IND)
    assert out[POST2SYN][0].shape[0] == 4


def test_require_structures_from_mat():
    c = _MatConn()(3, 3)
    res = c.require(CONN_MAT, PRE_IDS, POST_IDS, PRE2POST, COO, CSR, CSC, POST2PRE)
    assert len(res) == 8
    mat = np.asarray(res[0])
    assert mat.shape == (3, 3)
    assert mat.sum() == 4  # four edges


# ---------------------------------------------------------------------------
# legacy build_conn contract (dict + tuple) and its error branches
# ---------------------------------------------------------------------------

def test_legacy_build_conn_tuple_mat():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return 'mat', _MAT.copy()

    m = C()(3, 3).require(CONN_MAT)
    assert np.asarray(m).shape == (3, 3)


def test_legacy_build_conn_tuple_csr():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return 'csr', (_CSR_IND.copy(), _CSR_PTR.copy())

    m = C()(3, 3).require(CONN_MAT)
    assert np.asarray(m).sum() == 4


def test_legacy_build_conn_tuple_coo_alias_ij():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return 'ij', (_COO_PRE.copy(), _COO_POST.copy())

    coo = C()(3, 3).require(COO)
    assert coo[0].shape[0] == 4


def test_legacy_build_conn_dict():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return dict(mat=_MAT.copy(), csr=None, coo=None)

    m = C()(3, 3).require(CONN_MAT)
    assert np.asarray(m).shape == (3, 3)


def test_legacy_build_conn_bad_tuple_key():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return 'bogus', None

    with pytest.raises(ConnectorError):
        C()(3, 3).require(CONN_MAT)


def test_legacy_build_conn_unknown_type():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return 42

    with pytest.raises(ConnectorError):
        C()(3, 3).require(CONN_MAT)


def test_legacy_build_conn_all_none():
    class C(base.TwoEndConnector):
        def build_conn(self):
            return dict(mat=None, csr=None, coo=None)

    with pytest.raises(ConnectorError):
        C()(3, 3).require(CONN_MAT)


# ---------------------------------------------------------------------------
# OneEndConnector
# ---------------------------------------------------------------------------

class _OneEnd(base.OneEndConnector):
    def build_mat(self):
        return np.zeros((self.pre_num, self.post_num), dtype=bool)


def test_one_end_call_post_none_and_tuple():
    o = _OneEnd()(6)  # post_size defaults to pre_size
    assert o.pre_size == (6,)
    assert o.post_size == (6,)
    o2 = _OneEnd()((3, 4), (3, 4))  # tuple path, equal
    assert o2.pre_size == (3, 4)
    assert o2.pre_num == 12


def test_one_end_size_mismatch_raises():
    with pytest.raises(ConnectorError):
        _OneEnd()(5, 6)


def test_one_end_reset_conn():
    o = _OneEnd()(4)
    o._reset_conn(7)  # re-inits then re-calls with post=None
    assert o.pre_size == (7,)
    assert o.post_size == (7,)


# ---------------------------------------------------------------------------
# Free conversion helpers -- NumPy paths
# ---------------------------------------------------------------------------

def test_mat_to_csr_coo_csc_numpy():
    ind, indptr = base.mat2csr(_MAT)
    assert np.array_equal(ind, _CSR_IND)
    assert np.array_equal(indptr, _CSR_PTR)
    pre, post = base.mat2coo(_MAT)
    assert np.array_equal(pre, _COO_PRE)
    assert np.array_equal(post, _COO_POST)
    cind, cindptr = base.mat2csc(_MAT)
    assert cindptr.shape[0] == _MAT.shape[1] + 1


def test_csr_roundtrips_numpy():
    m = base.csr2mat((_CSR_IND, _CSR_PTR), 3, 3)
    assert np.asarray(m).sum() == 4
    pre, ind = base.csr2coo((_CSR_IND, _CSR_PTR))
    assert np.array_equal(pre, _COO_PRE)
    csc = base.csr2csc((_CSR_IND, _CSR_PTR), 3)
    assert len(csc) == 2


def test_coo2mat_numpy():
    m = base.coo2mat((_COO_PRE, _COO_POST), 3, 3)
    assert np.asarray(m).sum() == 4


def test_coo2csc_with_data_payload_numpy():
    data = np.arange(_CSR_IND.size)
    pre_new, indptr_new, data_new = base.coo2csc(
        base.csr2coo((_CSR_IND, _CSR_PTR)), 3, data=data)
    assert data_new.shape == data.shape
    assert indptr_new.shape[0] == 4


# ---------------------------------------------------------------------------
# Free conversion helpers -- JAX array paths
# ---------------------------------------------------------------------------

def test_mat_conversions_jax():
    jmat = jnp.asarray(_MAT)
    ind, indptr = base.mat2csr(jmat)
    assert np.array_equal(np.asarray(ind), _CSR_IND)
    pre, post = base.mat2coo(jmat)
    assert np.array_equal(np.asarray(pre), _COO_PRE)
    cind, cindptr = base.mat2csc(jmat)
    assert np.asarray(cindptr).shape[0] == _MAT.shape[1] + 1


def test_csr_and_coo_conversions_jax():
    jind = jnp.asarray(_CSR_IND)
    jptr = jnp.asarray(_CSR_PTR)
    m = base.csr2mat((jind, jptr), 3, 3)
    assert np.asarray(m).sum() == 4
    pre, ind = base.csr2coo((jind, jptr))
    assert np.array_equal(np.asarray(pre), _COO_PRE)
    m2 = base.coo2mat((jnp.asarray(_COO_PRE), jnp.asarray(_COO_POST)), 3, 3)
    assert np.asarray(m2).sum() == 4
    # coo2csc jax path with data payload
    data = jnp.arange(_CSR_IND.size)
    out = base.coo2csc((jnp.asarray(_COO_PRE), jnp.asarray(_COO_POST)), 3, data=data)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# coo2mat_num / mat2mat_num (FixedTotalNum helpers) -- JAX input so .at works
# ---------------------------------------------------------------------------

def test_coo2mat_num_adds_multi_connections():
    ij = (jnp.array([0, 1, 2]), jnp.array([0, 1, 2]))
    mat = base.coo2mat_num(ij, 3, 3, num=5, seed=1)
    # 3 base diagonal entries + 2 extra -> total weight 5
    assert int(np.asarray(mat).sum()) == 5


def test_mat2mat_num_adds_multi_connections():
    mat0 = jnp.asarray(np.eye(3, dtype=bool))
    mat = base.mat2mat_num(mat0, num=6, seed=1)
    assert int(np.asarray(mat).sum()) == 6


# ---------------------------------------------------------------------------
# set_default_dtype / get_idx_type / visualizeMat
# ---------------------------------------------------------------------------

def test_set_default_dtype_roundtrip():
    old_mat, old_idx = base.MAT_DTYPE, base.IDX_DTYPE
    try:
        base.set_default_dtype(mat_dtype=np.float32, idx_dtype=np.int64)
        assert base.MAT_DTYPE == np.float32
        assert base.IDX_DTYPE == np.int64
        assert base.get_idx_type() == np.int64
        # None args are no-ops (branch coverage)
        base.set_default_dtype()
        assert base.MAT_DTYPE == np.float32
    finally:
        base.MAT_DTYPE, base.IDX_DTYPE = old_mat, old_idx
    assert base.MAT_DTYPE == old_mat


def test_visualizemat_without_seaborn_is_graceful():
    # seaborn is not installed in this environment, so this hits the
    # ImportError guard and returns without raising.
    base.visualizeMat(np.eye(4), description='a very long title ' * 8)


# ---------------------------------------------------------------------------
# A FixedProb integration smoke test through the public surface, to exercise
# _make_returns end-to-end on a randomly generated connector (v2-style coo).
# ---------------------------------------------------------------------------

def test_fixedprob_require_all_structures_integration():
    conn = bp.connect.FixedProb(prob=0.4, seed=123)(8, 8)
    res = conn.require(CONN_MAT, PRE_IDS, POST_IDS, PRE2POST, POST2PRE, COO, CSR, CSC)
    out = dict(zip((CONN_MAT, PRE_IDS, POST_IDS, PRE2POST, POST2PRE, COO, CSR, CSC), res))
    assert np.asarray(out[CONN_MAT]).shape == (8, 8)
    assert out[PRE_IDS].shape == out[POST_IDS].shape
