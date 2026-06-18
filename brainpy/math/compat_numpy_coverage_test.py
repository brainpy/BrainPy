# -*- coding: utf-8 -*-
"""Supplementary coverage tests for ``brainpy/math/compat_numpy.py``.

``math_compat_fixes_test.py`` already exercises the bulk of the numpy-compat
shims.  This module mops up the remaining uncovered functions / branches:

* ``common_type`` integer-dtype branch and the non-numeric ``TypeError`` guard.
* ``putmask`` shape-mismatch ``ValueError``.
* ``safe_eval`` -- PINS a NumPy-2.0 defect (``np.safe_eval`` was removed).
* ``savetxt`` and ``savez_compressed`` round-trips (Array operands are coerced
  to numpy first).
* ``matrix`` / ``asmatrix`` dimension handling: the ``ndim > 2`` ``ValueError``
  and the ``ndim < 2`` expand-dims branch.

The ``array`` / ``asarray`` ``TypeError`` fallbacks (lines 167-171 / 179-183)
are version-dependent defensive paths: on the installed jaxlib ``jnp.array``
accepts brainpy ``Array`` leaves directly, so the ``except TypeError`` branch is
not reachable through the public API and is left documented rather than forced.
"""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.math import compat_numpy as cn
from brainpy.math.ndarray import Array


# ---------------------------------------------------------------------------
# common_type
# ---------------------------------------------------------------------------

def test_common_type_integer_promotes_to_double():
    # integer arrays use precision 2 -> double
    t = cn.common_type(np.array([1, 2, 3], dtype=np.int32))
    assert t is np.double


def test_common_type_complex_branch():
    t = cn.common_type(np.array([1 + 2j], dtype=np.csingle))
    # complex single -> csingle
    assert t is np.csingle


def test_common_type_non_numeric_raises():
    with pytest.raises(TypeError):
        cn.common_type(np.array(['a', 'b']))


# ---------------------------------------------------------------------------
# putmask
# ---------------------------------------------------------------------------

def test_putmask_non_array_raises():
    with pytest.raises(ValueError):
        cn.putmask(jnp.arange(6), jnp.arange(6) > 2, jnp.arange(6))


def test_putmask_shape_mismatch_raises():
    a = bm.asarray(np.arange(6))
    with pytest.raises(ValueError):
        cn.putmask(a, a > 2, bm.asarray(np.arange(3)))   # values shape != a shape


def test_putmask_happy_path_full_mask():
    # same-shape values + all-True mask -> the masked assignment writes through.
    a = bm.asarray(np.arange(6))
    vals = bm.asarray(np.arange(6) * 10)
    cn.putmask(a, jnp.ones(6, dtype=bool), vals)
    np.testing.assert_array_equal(np.asarray(bm.as_jax(a)), np.arange(6) * 10)


# ---------------------------------------------------------------------------
# safe_eval -- DEFECT under NumPy 2.0
# ---------------------------------------------------------------------------

def test_safe_eval_numpy2_defect():
    # NOTE: DEFECT -- ``cn.safe_eval`` delegates to ``np.safe_eval`` which was
    # removed in NumPy 2.0, so any call raises ``AttributeError`` rather than
    # returning the parsed literal.  Pinning the current broken behaviour.
    with pytest.raises(AttributeError):
        cn.safe_eval('[1, 2, 3]')


# ---------------------------------------------------------------------------
# savetxt / savez_compressed (Array operands coerced to numpy)
# ---------------------------------------------------------------------------

def test_savetxt_writes_file():
    d = tempfile.mkdtemp()
    fn = os.path.join(d, 'arr.txt')
    cn.savetxt(fn, bm.asarray(np.arange(6.).reshape(2, 3)))
    assert os.path.exists(fn)
    loaded = np.loadtxt(fn)
    np.testing.assert_allclose(loaded, np.arange(6.).reshape(2, 3))


def test_savez_compressed_writes_file():
    d = tempfile.mkdtemp()
    fn = os.path.join(d, 'arrs.npz')
    cn.savez_compressed(fn, a=bm.asarray(np.arange(3.)), b=np.arange(2))
    assert os.path.exists(fn)
    with np.load(fn) as data:
        np.testing.assert_allclose(data['a'], np.arange(3.))
        np.testing.assert_allclose(data['b'], np.arange(2))


# ---------------------------------------------------------------------------
# matrix / asmatrix dimensionality
# ---------------------------------------------------------------------------

def test_matrix_too_many_dims_raises():
    with pytest.raises(ValueError):
        cn.matrix(np.ones((2, 2, 2)))


def test_matrix_low_dim_expands_to_2d():
    m = cn.matrix(5)          # 0-D -> expanded to (1, 1)
    assert isinstance(m, Array)
    assert m.shape == (1, 1)
    m1 = cn.matrix([1, 2, 3])  # 1-D -> (1, 3)
    assert m1.shape == (1, 3)


def test_asmatrix_too_many_dims_raises():
    with pytest.raises(ValueError):
        cn.asmatrix(np.ones((2, 2, 2)))


def test_asmatrix_and_mat_expand():
    am = cn.asmatrix([1, 2, 3])
    assert am.shape == (1, 3)
    # mat is a thin alias for asmatrix
    mm = cn.mat([1, 2, 3])
    assert mm.shape == (1, 3)
