# -*- coding: utf-8 -*-
"""Regression tests for the 2026-06-19 ``math-compat`` audit (P3-* findings).

Covered findings (see ``docs/issues-found-20260619-math-compat.md``):

* P3-H1 (``activations.py``)      -- ``gelu`` must promote integer inputs to a
  floating dtype before computing; both the approximate and exact branches were
  silently wrong on integer input.
* P3-H2 (``compat_pytorch.py``)   -- ``unflatten`` must honour a negative ``dim``
  (PyTorch semantics) and reject out-of-range dims.
* P3-M1 (``compat_tensorflow.py``) -- ``segment_mean`` / ``unsorted_segment_mean``
  / ``unsorted_segment_sqrt_n`` must convert ``data`` to a jax array before
  ``jnp.ones_like`` (do not rely on the deprecated implicit ``__jax_array__``).
"""

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.math import (
    activations as act,
    compat_pytorch as cpt,
    compat_tensorflow as ctf,
)


def _j(x):
    return bm.as_jax(x)


def _finite(x):
    return bool(jnp.all(jnp.isfinite(_j(x))))


# ---------------------------------------------------------------------------
# P3-H1: gelu integer-input promotion
# ---------------------------------------------------------------------------

def test_gelu_integer_input_matches_float_approximate():
    """P3-H1: approximate gelu on int input must equal the float computation."""
    xi = jnp.array([1, 2, 3], dtype=jnp.int32)
    xf = jnp.array([1., 2., 3.])
    ri = np.asarray(_j(act.gelu(xi, approximate=True)))
    rf = np.asarray(_j(act.gelu(xf, approximate=True)))
    np.testing.assert_allclose(ri, rf, atol=1e-6)
    # and it must agree with jax.nn.gelu (the reference implementation)
    np.testing.assert_allclose(ri, np.asarray(jnn.gelu(xi, approximate=True)), atol=1e-6)
    # specifically: NOT the truncated x/2 result the bug produced
    assert not np.allclose(ri, np.asarray(xf) / 2.0)


def test_gelu_integer_input_matches_float_exact():
    """P3-H1: exact gelu on int input must not be truncated back to int."""
    xi = jnp.array([1, 2, 3], dtype=jnp.int32)
    xf = jnp.array([1., 2., 3.])
    ri = act.gelu(xi, approximate=False)
    assert jnp.issubdtype(_j(ri).dtype, jnp.floating)
    np.testing.assert_allclose(
        np.asarray(_j(ri)),
        np.asarray(_j(act.gelu(xf, approximate=False))),
        atol=1e-6,
    )
    # reference parity
    np.testing.assert_allclose(
        np.asarray(_j(ri)), np.asarray(jnn.gelu(xi, approximate=False)), atol=1e-5)


def test_gelu_float_unchanged():
    """The float path must be unchanged by the promotion fix."""
    x = bm.asarray([-1., 0., 1., 2.])
    for approx in (True, False):
        np.testing.assert_allclose(
            np.asarray(_j(act.gelu(x, approximate=approx))),
            np.asarray(jnn.gelu(_j(x), approximate=approx)),
            atol=1e-5,
        )


def test_gelu_accepts_brainpy_array_and_is_finite():
    x = bm.asarray([-3., -1., 0., 1., 3.])
    assert _finite(act.gelu(x, approximate=True))
    assert _finite(act.gelu(x, approximate=False))


# ---------------------------------------------------------------------------
# P3-H2: unflatten negative dim
# ---------------------------------------------------------------------------

def test_unflatten_negative_dim():
    """P3-H2: negative dim must be normalised like torch.unflatten."""
    x = bm.asarray(jnp.arange(6.))
    r = cpt.unflatten(x, -1, (2, 3))
    assert _j(r).shape == (2, 3)
    # equivalent to the positive-dim call
    np.testing.assert_allclose(
        np.asarray(_j(r)), np.asarray(_j(cpt.unflatten(x, 0, (2, 3)))))


def test_unflatten_negative_dim_higher_rank():
    x = bm.asarray(jnp.arange(24.).reshape(2, 12))
    r = cpt.unflatten(x, -1, (3, 4))
    assert _j(r).shape == (2, 3, 4)
    r2 = cpt.unflatten(x, -2, (1, 2))
    assert _j(r2).shape == (1, 2, 12)


def test_unflatten_positive_dim_still_works():
    x = bm.asarray(jnp.arange(6.))
    assert _j(cpt.unflatten(x, 0, (2, 3))).shape == (2, 3)
    assert _j(cpt.unflatten(x, 0, (-1, 3))).shape == (2, 3)


def test_unflatten_dim_out_of_range():
    x = bm.asarray(jnp.arange(6.))
    with pytest.raises((ValueError, AssertionError, IndexError)):
        cpt.unflatten(x, 5, (2, 3))
    with pytest.raises((ValueError, AssertionError, IndexError)):
        cpt.unflatten(x, -5, (2, 3))


# ---------------------------------------------------------------------------
# P3-M1: TF segment helpers must not lean on implicit __jax_array__
# ---------------------------------------------------------------------------

def test_segment_mean_array_input():
    data = bm.asarray([1., 2., 3., 4.])
    seg = bm.asarray([0, 0, 1, 1])
    np.testing.assert_allclose(np.asarray(_j(ctf.segment_mean(data, seg))), [1.5, 3.5])


def test_unsorted_segment_mean_array_input():
    data = bm.asarray([1., 2., 3., 4.])
    seg = bm.asarray([0, 0, 1, 1])
    np.testing.assert_allclose(
        np.asarray(_j(ctf.unsorted_segment_mean(data, seg, 2))), [1.5, 3.5])


def test_unsorted_segment_sqrt_n_array_input():
    data = bm.asarray([1., 1., 1., 1.])
    seg = bm.asarray([0, 0, 1, 1])
    # sum over 2-element segments divided by sqrt(2)
    np.testing.assert_allclose(
        np.asarray(_j(ctf.unsorted_segment_sqrt_n(data, seg, 2))),
        [2.0 / np.sqrt(2.0), 2.0 / np.sqrt(2.0)],
        atol=1e-6,
    )


def test_unsorted_segment_mean_under_jit():
    """The denominator (``jnp.ones_like``) must trace cleanly under jit.

    ``unsorted_segment_mean`` takes a static ``num_segments`` so it is
    jit-compatible (unlike ``segment_mean`` which infers it from the data).
    """
    data = jnp.array([1., 2., 3., 4.])
    seg = jnp.array([0, 0, 1, 1])
    f = jax.jit(lambda d: bm.as_jax(ctf.unsorted_segment_mean(bm.asarray(d), bm.asarray(seg), 2)))
    np.testing.assert_allclose(np.asarray(f(data)), [1.5, 3.5])
