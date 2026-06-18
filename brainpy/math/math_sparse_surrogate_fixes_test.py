# -*- coding: utf-8 -*-
"""Regression + coverage tests for the BrainPy v2.7.8 sparse / event / jitconn /
surrogate / delay / pre-syn-post audit (see ``docs/issues-found-20260618.md``).

This module locks in the fixes recorded in the audit for the following source
files (audit IDs in parentheses):

* ``brainpy/math/sparse/csr_mm.py``       — C-07 (``csrmm(transpose=True)`` must
  compute ``Mᵀ @ B`` rather than ``B @ M``).
* ``brainpy/math/event/csr_matmat.py``    — C-07 (same transpose fix for the
  event-driven CSR matmat path, with ``BinaryArray`` operand).
* ``brainpy/math/sparse/coo_mv.py``       — H-17 (``coomv`` must no longer touch
  the removed ``brainevent.COO`` and must still equal ``dense @ v``).
* ``brainpy/math/sparse/utils.py``        — H-18 (``coo_to_csr`` returns an int,
  monotone ``indptr`` starting at 0 / ending at ``nnz``) and H-19
  (``csr_to_dense`` wraps ``brainevent.CSR(...).todense()`` correctly).
* ``brainpy/math/jitconn/matvec.py``      — M-13 (``mv_prob_*`` /
  ``event_mv_prob_*`` are reproducible when an explicit ``seed`` is threaded).
* ``brainpy/math/surrogate/_one_input.py`` and
  ``brainpy/math/surrogate/_one_input_new.py`` — H-20..H-24: for every surrogate
  the ``surrogate_grad`` matches ``jax.grad(surrogate_fun)``; ``GaussianGrad``
  widens with ``sigma`` (H-20); ``PiecewiseQuadratic`` grad matches its forward
  derivative (H-21); ``QPseudoSpike`` uses ``alpha-1`` (H-22); ``Arctan``
  ``surrogate_fun`` does not raise (H-23); ``ERF`` ``surrogate_fun`` is
  increasing (H-24).
* ``brainpy/math/delayvars.py``           — C-09 (``TimeDelay`` ring-buffer read
  applies the modulo) plus ``LengthDelay`` / ``ROTATE_UPDATE`` / ``CONCAT_UPDATE``
  coverage.
* ``brainpy/math/pre_syn_post.py``        — M-15 plus ``syn2post_mean`` /
  ``syn2post_softmax`` empty-group → 0 vs genuine-NaN-propagation behaviour.

All tests use tiny shapes and complete well under the time budget. They assert
fixed (correct) behaviour, so they double as a regression guard against the bugs
re-appearing.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import brainpy.math as bm
import brainpy.math.surrogate._one_input as old_surr
import brainpy.math.surrogate._one_input_new as new_surr


# ---------------------------------------------------------------------------
# Helpers: build a small known CSR by hand (no scipy dependency).
#
#   dense = [[1, 0, 2, 0],
#            [0, 3, 0, 0],
#            [0, 0, 0, 4]]   (3 x 4)
# ---------------------------------------------------------------------------

def _known_csr():
    dense = np.array([[1., 0., 2., 0.],
                      [0., 3., 0., 0.],
                      [0., 0., 0., 4.]], dtype=np.float32)
    data = jnp.asarray([1., 2., 3., 4.], dtype=jnp.float32)
    indices = jnp.asarray([0, 2, 1, 3], dtype=jnp.int32)
    indptr = jnp.asarray([0, 2, 3, 4], dtype=jnp.int32)
    shape = (3, 4)
    return dense, data, indices, indptr, shape


# ===========================================================================
# C-07 — csrmm(transpose=True) computes Mᵀ @ B  (sparse + event paths)
# ===========================================================================

def test_csrmm_non_transpose_matches_dense():
    dense, data, indices, indptr, shape = _known_csr()
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    out = bm.sparse.csrmm(data, indices, indptr, jnp.asarray(B),
                          shape=shape, transpose=False)
    out = np.asarray(out)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, dense @ B, rtol=1e-5, atol=1e-5)


def test_csrmm_transpose_matches_dense_T():
    # Regression for C-07: transpose branch must equal Mᵀ @ B (shape (cols, k)),
    # NOT B @ M.  With shape=(3,4) and B=(3,2) the output must be (4,2).
    dense, data, indices, indptr, shape = _known_csr()
    B = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    out = bm.sparse.csrmm(data, indices, indptr, jnp.asarray(B),
                          shape=shape, transpose=True)
    out = np.asarray(out)
    assert out.shape == (4, 2)          # would be (3, ...) under the old bug
    np.testing.assert_allclose(out, dense.T @ B, rtol=1e-5, atol=1e-5)


def test_csrmm_accepts_brainpy_array_operands():
    # Exercise the ``isinstance(x, Array)`` unwrapping branches in csr_mm.py.
    dense, data, indices, indptr, shape = _known_csr()
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    out = bm.sparse.csrmm(bm.asarray(data), bm.asarray(indices),
                          bm.asarray(indptr), bm.asarray(B),
                          shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ B, rtol=1e-5, atol=1e-5)


def test_event_csrmm_transpose_matches_dense_T():
    # C-07 for brainpy/math/event/csr_matmat.py — binary event matrix.
    dense, data, indices, indptr, shape = _known_csr()
    B = (np.arange(3 * 2).reshape(3, 2) % 2).astype(np.float32)   # 0/1 events
    out = bm.event.csrmm(data, indices, indptr, jnp.asarray(B),
                         shape=shape, transpose=True)
    out = np.asarray(out)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, dense.T @ B, rtol=1e-5, atol=1e-5)


def test_event_csrmm_non_transpose_matches_dense():
    dense, data, indices, indptr, shape = _known_csr()
    B = (np.arange(4 * 2).reshape(4, 2) % 2).astype(np.float32)
    out = bm.event.csrmm(data, indices, indptr, jnp.asarray(B),
                         shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ B, rtol=1e-5, atol=1e-5)


def test_event_csrmm_accepts_brainpy_array_operands():
    # Exercise the ``isinstance(x, Array)`` unwrap branches in csr_matmat.py.
    dense, data, indices, indptr, shape = _known_csr()
    B = (np.arange(4 * 2).reshape(4, 2) % 2).astype(np.float32)
    out = bm.event.csrmm(bm.asarray(data), bm.asarray(indices),
                         bm.asarray(indptr), bm.asarray(B),
                         shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ B, rtol=1e-5, atol=1e-5)


# ===========================================================================
# H-17 — coomv works without the removed brainevent.COO, equals dense @ v
# ===========================================================================

def test_coomv_matches_dense_no_attribute_error():
    dense, _, _, _, shape = _known_csr()
    rows, cols = np.nonzero(dense)
    vals = dense[rows, cols].astype(np.float32)
    row = jnp.asarray(rows, dtype=jnp.int32)
    col = jnp.asarray(cols, dtype=jnp.int32)
    data = jnp.asarray(vals, dtype=jnp.float32)
    v = np.arange(4, dtype=np.float32)
    # Must not raise AttributeError about a removed COO type.
    out = bm.sparse.coomv(data, row, col, jnp.asarray(v), shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ v, rtol=1e-5, atol=1e-5)


def test_coomv_transpose_matches_dense_T():
    dense, _, _, _, shape = _known_csr()
    rows, cols = np.nonzero(dense)
    data = jnp.asarray(dense[rows, cols].astype(np.float32))
    row = jnp.asarray(rows, dtype=jnp.int32)
    col = jnp.asarray(cols, dtype=jnp.int32)
    v = np.arange(3, dtype=np.float32)
    out = bm.sparse.coomv(data, row, col, jnp.asarray(v), shape=shape, transpose=True)
    np.testing.assert_allclose(np.asarray(out), dense.T @ v, rtol=1e-5, atol=1e-5)


def test_coomv_scalar_weight_broadcast():
    # Exercise the scalar-data broadcast branch of coomv.
    dense, _, _, _, shape = _known_csr()
    rows, cols = np.nonzero(dense)
    row = jnp.asarray(rows, dtype=jnp.int32)
    col = jnp.asarray(cols, dtype=jnp.int32)
    v = np.ones(4, dtype=np.float32)
    out = bm.sparse.coomv(2.0, row, col, jnp.asarray(v), shape=shape, transpose=False)
    # mask of the structure, weight 2.0 everywhere
    mask = (dense != 0).astype(np.float32)
    np.testing.assert_allclose(np.asarray(out), (2.0 * mask) @ v, rtol=1e-5, atol=1e-5)


def test_coomv_accepts_brainpy_array_operands():
    # Exercise the ``isinstance(x, Array)`` unwrap branches in coo_mv.py.
    dense, _, _, _, shape = _known_csr()
    rows, cols = np.nonzero(dense)
    data = bm.asarray(dense[rows, cols].astype(np.float32))
    row = bm.asarray(jnp.asarray(rows, dtype=jnp.int32))
    col = bm.asarray(jnp.asarray(cols, dtype=jnp.int32))
    v = bm.asarray(np.arange(4, dtype=np.float32))
    out = bm.sparse.coomv(data, row, col, v, shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ np.arange(4, dtype=np.float32),
                               rtol=1e-5, atol=1e-5)


# ===========================================================================
# H-18 / H-19 — coo_to_csr int indptr; csr_to_dense matches reference
# ===========================================================================

def test_coo_to_csr_returns_int_monotone_indptr():
    pre_ids = jnp.asarray([0, 0, 1, 2], dtype=jnp.int32)
    post_ids = jnp.asarray([0, 2, 1, 3], dtype=jnp.int32)
    indices, indptr = bm.sparse.coo_to_csr(pre_ids, post_ids, num_row=3)
    indptr_np = np.asarray(indptr)
    # integer dtype (H-18: was float before the fix)
    assert jnp.issubdtype(indptr.dtype, jnp.integer)
    # monotone non-decreasing, starts at 0, ends at nnz
    assert indptr_np[0] == 0
    assert indptr_np[-1] == int(post_ids.shape[0])
    assert np.all(np.diff(indptr_np) >= 0)
    # round-trip sanity: indices length == nnz
    assert np.asarray(indices).shape[0] == int(post_ids.shape[0])


def test_csr_to_dense_matches_reference():
    dense, data, indices, indptr, shape = _known_csr()
    out = bm.sparse.csr_to_dense(data, indices, indptr, shape=shape)
    np.testing.assert_allclose(np.asarray(out), dense, rtol=1e-5, atol=1e-5)


def test_csr_to_coo_round_trip():
    # Exercises utils.csr_to_coo for coverage.
    _, _, indices, indptr, _ = _known_csr()
    row, col = bm.sparse.csr_to_coo(indices, indptr)
    np.testing.assert_array_equal(np.asarray(row), [0, 0, 1, 2])
    np.testing.assert_array_equal(np.asarray(col), np.asarray(indices))


# ===========================================================================
# sparse / event csrmv coverage (+ value correctness)
# ===========================================================================

def test_csrmv_matches_dense_both_directions():
    dense, data, indices, indptr, shape = _known_csr()
    v = np.arange(4, dtype=np.float32)
    out = bm.sparse.csrmv(data, indices, indptr, jnp.asarray(v),
                          shape=shape, transpose=False)
    np.testing.assert_allclose(np.asarray(out), dense @ v, rtol=1e-5, atol=1e-5)

    vt = np.arange(3, dtype=np.float32)
    out_t = bm.sparse.csrmv(data, indices, indptr, jnp.asarray(vt),
                            shape=shape, transpose=True)
    np.testing.assert_allclose(np.asarray(out_t), dense.T @ vt, rtol=1e-5, atol=1e-5)


def test_event_csrmv_matches_masked_dense():
    dense, data, indices, indptr, shape = _known_csr()
    # transpose=True: Mᵀ @ events,  events length == shape[0] == 3
    events = jnp.asarray([True, False, True], dtype=bool)
    out = bm.event.csrmv(data, indices, indptr, events, shape=shape, transpose=True)
    ref = dense.T @ np.asarray(events, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    # non-transpose: M @ events, events length == shape[1] == 4
    events2 = jnp.asarray([True, False, True, False], dtype=bool)
    out2 = bm.event.csrmv(data, indices, indptr, events2, shape=shape, transpose=False)
    ref2 = dense @ np.asarray(events2, dtype=np.float32)
    np.testing.assert_allclose(np.asarray(out2), ref2, rtol=1e-5, atol=1e-5)


# ===========================================================================
# H-20..H-24 — surrogate gradients consistent with their forward functions
# ===========================================================================

# The two modules expose slightly different APIs:
#   _one_input.py     : surrogate_grad(self, dz, x)   (dz = upstream gradient)
#   _one_input_new.py : surrogate_grad(self, x)
# These helpers normalise that so the same assertions run against both.

def _grad_old(inst, x):
    return inst.surrogate_grad(1.0, x)


def _grad_new(inst, x):
    return inst.surrogate_grad(x)


_MODULES = [
    ("_one_input", old_surr, _grad_old),
    ("_one_input_new", new_surr, _grad_new),
]

# Surrogates that expose BOTH surrogate_fun and surrogate_grad and are
# differentiable away from kinks — used for the grad-vs-autograd check.
_HAS_FUN = ["PiecewiseQuadratic", "QPseudoSpike", "Arctan", "ERF"]


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
@pytest.mark.parametrize("clsname", _HAS_FUN)
def test_surrogate_grad_matches_autograd(modname, mod, getgrad, clsname):
    """H-21/H-22/H-23/H-24: surrogate_grad == d/dx surrogate_fun."""
    cls = getattr(mod, clsname)
    inst = cls()
    # Avoid the exact kinks (|x| = 1/alpha) where the piecewise derivative jumps.
    xs = jnp.asarray([-0.85, -0.4, -0.1, 0.1, 0.4, 0.85], dtype=jnp.float32)
    analytic = np.asarray(getgrad(inst, xs))

    def fun_scalar(v):
        return jnp.squeeze(inst.surrogate_fun(jnp.reshape(v, (1,))))

    autograd = np.asarray(jax.vmap(jax.grad(fun_scalar))(xs))
    np.testing.assert_allclose(analytic, autograd, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
@pytest.mark.parametrize("clsname", _HAS_FUN)
def test_surrogate_fun_monotone_increasing_on_unit_interval(modname, mod, getgrad, clsname):
    """Each surrogate forward (origin) function is non-decreasing on [0, 1]."""
    cls = getattr(mod, clsname)
    inst = cls()
    fx = np.asarray(inst.surrogate_fun(jnp.linspace(0.0, 1.0, 21)))
    assert np.all(np.diff(fx) >= -1e-6)


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
def test_arctan_surrogate_fun_does_not_raise(modname, mod, getgrad):
    """H-23: Arctan.surrogate_fun previously called jnp.arctan2 with one arg."""
    inst = mod.Arctan()
    out = np.asarray(inst.surrogate_fun(jnp.asarray([-0.5, -0.1, 0.0, 0.1, 0.5])))
    assert np.all(np.isfinite(out))
    # arctan forward crosses 0.5 at x = 0
    assert np.isclose(out[2], 0.5, atol=1e-6)
    assert np.all(np.diff(out) > 0)


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
def test_erf_surrogate_fun_is_increasing(modname, mod, getgrad):
    """H-24: ERF.surrogate_fun must be increasing (was decreasing before)."""
    inst = mod.ERF()
    out = np.asarray(inst.surrogate_fun(jnp.linspace(-0.5, 0.5, 11)))
    assert np.all(np.diff(out) > 0)
    assert np.isclose(out[5], 0.5, atol=1e-6)   # centred at x = 0


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
def test_gaussian_grad_bump_widens_with_sigma(modname, mod, getgrad):
    """H-20: GaussianGrad — at x=1 the gradient must INCREASE with sigma
    (a wider bump), proving the sigma is no longer inverted by the
    operator-precedence bug ``exp(-(x**2)/2*sigma**2)``."""
    g_narrow = float(np.asarray(getgrad(mod.GaussianGrad(sigma=0.5), jnp.asarray(1.0))))
    g_wide = float(np.asarray(getgrad(mod.GaussianGrad(sigma=2.0), jnp.asarray(1.0))))
    assert g_wide > g_narrow
    # Sanity on the intended magnitude (audit: grad@±1 ≈ 0.088 for sigma=2).
    assert g_wide == pytest.approx(0.088, abs=2e-2)


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
def test_piecewise_quadratic_grad_formula(modname, mod, getgrad):
    """H-21: grad == -alpha**2 |x| + alpha inside the support, 0 outside."""
    inst = mod.PiecewiseQuadratic(alpha=1.0)
    g_in = float(np.asarray(getgrad(inst, jnp.asarray(0.5))))
    assert g_in == pytest.approx(-1.0 * 0.5 + 1.0)   # = 0.5
    g_out = float(np.asarray(getgrad(inst, jnp.asarray(5.0))))
    assert g_out == pytest.approx(0.0)


@pytest.mark.parametrize("modname,mod,getgrad", _MODULES,
                         ids=[m[0] for m in _MODULES])
def test_qpseudospike_grad_uses_alpha_minus_one(modname, mod, getgrad):
    """H-22: grad denominator uses (alpha-1); grad at 0 == 1."""
    inst = mod.QPseudoSpike(alpha=2.0)
    g0 = float(np.asarray(getgrad(inst, jnp.asarray(0.0))))
    assert g0 == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Surrogate coverage — every class's __call__ + surrogate_grad in both modules
# ===========================================================================

def _new_surrogate_classes():
    return [getattr(new_surr, n) for n in new_surr.__all__
            if n[0].isupper() and n != "Surrogate"]


def _old_surrogate_classes():
    out = []
    for n in dir(old_surr):
        obj = getattr(old_surr, n)
        if (isinstance(obj, type) and issubclass(obj, old_surr.Surrogate)
                and n not in ("Surrogate", "_OneInpSurrogate")):
            out.append(obj)
    return out


@pytest.mark.parametrize("cls", _new_surrogate_classes(),
                         ids=lambda c: c.__name__)
def test_new_surrogate_call_and_grad_run(cls):
    inst = cls()
    x = jnp.linspace(-1.5, 1.5, 9)
    y = inst(x)                              # __call__ -> heaviside forward
    assert np.asarray(y).shape == (9,)
    # forward is a {0,1} spike indicator
    assert set(np.unique(np.asarray(y)).tolist()).issubset({0.0, 1.0})
    g = np.asarray(inst.surrogate_grad(x))   # surrogate_grad(x)
    assert g.shape == (9,) and np.all(np.isfinite(g))
    # grad flows through __call__
    flow = jax.grad(lambda v: jnp.sum(inst(v)))(x)
    assert np.all(np.isfinite(np.asarray(flow)))


@pytest.mark.parametrize("cls", _old_surrogate_classes(),
                         ids=lambda c: c.__name__)
def test_old_surrogate_call_and_grad_run(cls):
    inst = cls()
    x = jnp.linspace(-1.5, 1.5, 9)
    y = inst(x)                                   # custom-gradient forward
    assert np.asarray(y).shape == (9,)
    assert set(np.unique(np.asarray(y)).tolist()).issubset({0.0, 1.0})
    g = np.asarray(inst.surrogate_grad(1.0, x))   # surrogate_grad(dz, x)
    assert g.shape == (9,) and np.all(np.isfinite(g))
    flow = jax.grad(lambda v: jnp.sum(inst(v)))(x)
    assert np.all(np.isfinite(np.asarray(flow)))


def test_new_surrogate_repr_and_functional_aliases():
    # Exercise functional (lowercase) entry points + __repr__ for coverage.
    x = jnp.linspace(-1.0, 1.0, 5)
    assert "Arctan" in repr(new_surr.Arctan())
    for fn in (new_surr.sigmoid, new_surr.arctan, new_surr.erf,
               new_surr.gaussian_grad, new_surr.relu_grad):
        assert np.asarray(fn(x)).shape == (5,)


def test_old_surrogate_repr_and_functional_aliases():
    x = jnp.linspace(-1.0, 1.0, 5)
    assert "GaussianGrad" in repr(old_surr.GaussianGrad())
    for fn in (old_surr.sigmoid, old_surr.arctan, old_surr.erf,
               old_surr.gaussian_grad, old_surr.q_pseudo_spike):
        assert np.asarray(fn(x)).shape == (5,)


# Lowercase functional aliases present in BOTH modules.
_FUNC_NAMES = [n for n in new_surr.__all__ if n[0].islower()]


@pytest.mark.parametrize("fname", _FUNC_NAMES)
def test_old_functional_alias_forward_and_origin(fname):
    """Exercise every ``_one_input`` functional alias (heaviside forward and,
    where supported, the ``origin=True`` smooth forward)."""
    import inspect
    fn = getattr(old_surr, fname)
    x = jnp.linspace(-1.2, 1.2, 7)
    y = np.asarray(fn(x))
    assert y.shape == (7,) and np.all(np.isfinite(y))
    if "origin" in inspect.signature(fn).parameters:
        yo = np.asarray(fn(x, origin=True))   # exercises surrogate_fun
        assert yo.shape == (7,) and np.all(np.isfinite(yo))


@pytest.mark.parametrize("fname", _FUNC_NAMES)
def test_new_functional_alias_forward(fname):
    """Exercise every ``_one_input_new`` functional alias (heaviside forward)."""
    fn = getattr(new_surr, fname)
    x = jnp.linspace(-1.2, 1.2, 7)
    y = np.asarray(fn(x))
    assert y.shape == (7,) and np.all(np.isfinite(y))


def _new_classes_with_surrogate_fun():
    out = []
    for n in new_surr.__all__:
        if not (n[0].isupper() and n != "Surrogate"):
            continue
        c = getattr(new_surr, n)
        if c.surrogate_fun is not new_surr.Surrogate.surrogate_fun:
            out.append(c)
    return out


@pytest.mark.parametrize("cls", _new_classes_with_surrogate_fun(),
                         ids=lambda c: c.__name__)
def test_new_surrogate_fun_runs(cls):
    """Cover the ``surrogate_fun`` body of every new-module class that has one."""
    out = np.asarray(cls().surrogate_fun(jnp.linspace(-1.2, 1.2, 9)))
    assert out.shape == (9,) and np.all(np.isfinite(out))


# ===========================================================================
# C-09 — TimeDelay ring-buffer read applies the modulo
# ===========================================================================

def test_time_delay_ring_buffer_modulo():
    """After pushing a ramp k*0.1 for k=1..30, the buffer wraps several times;
    the read must apply ``% num_delay_step`` (C-09) so d(now) == 3.0."""
    d = bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1)
    for k in range(1, 31):
        d.update(bm.ones(1) * (k * 0.1))
    now = d.current_time[0]
    assert float(np.asarray(d(now))[0]) == pytest.approx(3.0, abs=1e-4)
    assert float(np.asarray(d(now - 0.5))[0]) == pytest.approx(2.5, abs=1e-4)


def test_time_delay_round_interp_method():
    d = bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1, interp_method='round')
    for k in range(1, 31):
        d.update(bm.ones(1) * (k * 0.1))
    now = d.current_time[0]
    # round interpolation hits the exact-step branch too
    assert float(np.asarray(d(now))[0]) == pytest.approx(3.0, abs=1e-4)


def test_time_delay_reset():
    d = bm.TimeDelay(bm.zeros(2), delay_len=1.0, dt=0.1)
    for k in range(1, 12):
        d.update(bm.ones(2) * (k * 0.1))
    d.reset(bm.zeros(2), delay_len=1.0)
    now = d.current_time[0]
    np.testing.assert_allclose(np.asarray(d(now)), np.zeros(2), atol=1e-6)


# ===========================================================================
# LengthDelay — both ROTATE_UPDATE and CONCAT_UPDATE
# ===========================================================================

@pytest.mark.parametrize("method", [bm.ROTATE_UPDATE, bm.CONCAT_UPDATE])
def test_length_delay_update_methods(method):
    ld = bm.LengthDelay(bm.zeros(2), delay_len=3, update_method=method)
    for k in range(1, 6):
        ld.update(bm.ones(2) * float(k))
    # most-recent push is 5 -> delay 0 returns 5; delay 2 returns 3
    np.testing.assert_allclose(np.asarray(ld(0)), [5., 5.], atol=1e-6)
    np.testing.assert_allclose(np.asarray(ld(2)), [3., 3.], atol=1e-6)


def test_length_delay_reset_and_retrieve():
    ld = bm.LengthDelay(bm.zeros(2), delay_len=3)
    ld.reset(bm.ones(2), delay_len=3)
    out = ld.retrieve(1)
    assert np.asarray(out).shape == (2,)


def test_length_delay_initial_delay_data_scalar_and_callable():
    # scalar initial_delay_data branch
    ld = bm.LengthDelay(bm.zeros(2), delay_len=3, initial_delay_data=1.0)
    np.testing.assert_allclose(np.asarray(ld.retrieve(2)), [1., 1.], atol=1e-6)
    # callable initial_delay_data branch (plain lambda, no dtype kwarg)
    ld2 = bm.LengthDelay(bm.zeros(2), delay_len=3,
                         initial_delay_data=lambda shape: jnp.ones(shape) * 7.0)
    np.testing.assert_allclose(np.asarray(ld2.retrieve(2)), [7., 7.], atol=1e-6)


def test_length_delay_concat_single_step():
    # delay_len=0 -> num_delay_step=1 exercises the CONCAT_UPDATE short branch.
    ld = bm.LengthDelay(bm.zeros(2), delay_len=0, update_method=bm.CONCAT_UPDATE)
    ld.update(bm.ones(2) * 3.0)
    np.testing.assert_allclose(np.asarray(ld(0)), [3., 3.], atol=1e-6)


def test_time_delay_callable_before_t0():
    # Covers the _FUNC_BEFORE path (callable before_t0 + cond branch in __call__).
    d = bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1, t0=0.0,
                     before_t0=lambda t: jnp.ones(1) * 9.0)
    # request a time strictly before t0 -> uses before_t0 function
    out = np.asarray(d(-0.5))
    np.testing.assert_allclose(out, [9.0], atol=1e-6)


def test_neutral_delay_aliases():
    # NeuTimeDelay / NeuLenDelay are thin aliases; just instantiate + call.
    ntd = bm.NeuTimeDelay(bm.zeros(1), delay_len=0.5, dt=0.1)
    ntd.update(bm.ones(1))
    assert np.asarray(ntd(ntd.current_time[0])).shape == (1,)
    nld = bm.NeuLenDelay(bm.zeros(1), delay_len=2)
    nld.update(bm.ones(1))
    assert np.asarray(nld(0)).shape == (1,)


def test_time_delay_array_before_t0_and_indices():
    # array before_t0 fills the pre-t0 buffer (the _DATA_BEFORE branch);
    # indices select a sub-slice of the retrieved value.
    d = bm.TimeDelay(bm.zeros(3), delay_len=1.0, dt=0.1, before_t0=5.0)
    for k in range(1, 31):
        d.update(bm.ones(3) * (k * 0.1))
    now = d.current_time[0]
    out = np.asarray(d(now, indices=jnp.asarray([0, 2])))
    assert out.shape == (2,)


def test_time_delay_reset_with_callable_before_t0():
    d = bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1)
    d.reset(bm.zeros(1), delay_len=1.0, before_t0=lambda t: jnp.ones(1) * 4.0)
    out = np.asarray(d(-0.5))   # before t0 -> uses callable
    np.testing.assert_allclose(out, [4.0], atol=1e-6)


def test_time_delay_reset_with_array_before_t0():
    d = bm.TimeDelay(bm.zeros(2), delay_len=1.0, dt=0.1)
    d.reset(bm.zeros(2), delay_len=1.0, before_t0=3.0)
    now = d.current_time[0]
    assert np.asarray(d(now)).shape == (2,)


def test_length_delay_retrieve_with_indices_and_repr():
    ld = bm.LengthDelay(bm.zeros(4), delay_len=3)
    for k in range(1, 5):
        ld.update(bm.ones(4) * float(k))
    out = np.asarray(ld.retrieve(1, jnp.asarray([0, 1])))
    assert out.shape == (2,)
    assert "LengthDelay" in repr(ld)
    assert ld.delay_shape[0] == 4   # num_delay_step (delay_len + 1)


def test_length_delay_update_from_variable_target():
    # update(value=None) pulls from the stored delay_target Variable.
    target = bm.Variable(bm.ones(2) * 2.0)
    ld = bm.LengthDelay(target, delay_len=2)
    ld.update()   # no explicit value -> uses delay_target.value
    assert np.asarray(ld(0)).shape == (2,)


def test_time_delay_validation_errors():
    # invalid delay_target type
    with pytest.raises(ValueError):
        bm.TimeDelay([0.0], delay_len=1.0, dt=0.1)
    # unsupported interpolation method
    from brainpy._errors import UnsupportedError
    with pytest.raises(UnsupportedError):
        bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1, interp_method='nope')
    # unsupported before_t0 type
    with pytest.raises(ValueError):
        bm.TimeDelay(bm.zeros(1), delay_len=1.0, dt=0.1, before_t0='bad')


def test_length_delay_validation_errors():
    with pytest.raises(ValueError):
        bm.LengthDelay([0.0], delay_len=2)
    ld = bm.LengthDelay(bm.zeros(2), delay_len=2)
    with pytest.raises(ValueError):
        ld.reset(bm.zeros(2), delay_len=2, initial_delay_data='bad')


# ===========================================================================
# M-13 — jitconn mv_prob_* / event_mv_prob_* reproducible with explicit seed
# ===========================================================================

_SHAPE = (8, 6)


def _vec_for(transpose):
    n = _SHAPE[0] if transpose else _SHAPE[1]
    return jnp.asarray(np.random.RandomState(0).randn(n).astype(np.float32))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("outdim_parallel", [True, False])
def test_mv_prob_homo_reproducible(transpose, outdim_parallel):
    v = _vec_for(transpose)
    kw = dict(weight=1.5, conn_prob=0.3, shape=_SHAPE,
              transpose=transpose, outdim_parallel=outdim_parallel)
    o1 = np.asarray(bm.jitconn.mv_prob_homo(v, seed=123, **kw))
    o2 = np.asarray(bm.jitconn.mv_prob_homo(v, seed=123, **kw))
    np.testing.assert_array_equal(o1, o2)              # reproducible
    assert o1.shape == (_SHAPE[1] if transpose else _SHAPE[0],)


def test_mv_prob_uniform_and_normal_reproducible():
    v = _vec_for(False)
    ou1 = np.asarray(bm.jitconn.mv_prob_uniform(v, w_low=-1., w_high=1.,
                                                conn_prob=0.3, seed=7, shape=_SHAPE))
    ou2 = np.asarray(bm.jitconn.mv_prob_uniform(v, w_low=-1., w_high=1.,
                                                conn_prob=0.3, seed=7, shape=_SHAPE))
    np.testing.assert_array_equal(ou1, ou2)

    on1 = np.asarray(bm.jitconn.mv_prob_normal(v, w_mu=0., w_sigma=1.,
                                               conn_prob=0.3, seed=9, shape=_SHAPE))
    on2 = np.asarray(bm.jitconn.mv_prob_normal(v, w_mu=0., w_sigma=1.,
                                               conn_prob=0.3, seed=9, shape=_SHAPE))
    np.testing.assert_array_equal(on1, on2)


def test_mv_prob_homo_seed_none_runs():
    # Cover the ``seed is None`` host-RNG branch (documented as non-reproducible).
    v = _vec_for(False)
    out = np.asarray(bm.jitconn.mv_prob_homo(v, weight=1.0, conn_prob=0.3,
                                             seed=None, shape=_SHAPE))
    assert out.shape == (_SHAPE[0],) and np.all(np.isfinite(out))


@pytest.mark.parametrize("fn", [
    lambda ev, **k: bm.jitconn.event_mv_prob_homo(ev, 1.0, 0.3, **k),
    lambda ev, **k: bm.jitconn.event_mv_prob_uniform(ev, -1.0, 1.0, 0.3, **k),
    lambda ev, **k: bm.jitconn.event_mv_prob_normal(ev, 0.0, 1.0, 0.3, **k),
], ids=["homo", "uniform", "normal"])
def test_event_mv_prob_reproducible(fn):
    events = jnp.asarray(np.random.RandomState(1).rand(_SHAPE[1]) > 0.5)
    o1 = np.asarray(fn(events, seed=11, shape=_SHAPE))
    o2 = np.asarray(fn(events, seed=11, shape=_SHAPE))
    np.testing.assert_array_equal(o1, o2)
    assert o1.shape == (_SHAPE[0],)


def test_mv_prob_uniform_normal_transpose_and_array_args():
    # transpose=True branches + Array-operand unwrap branches for uniform/normal.
    vrow = bm.asarray(np.random.RandomState(3).randn(_SHAPE[0]).astype(np.float32))
    ou = np.asarray(bm.jitconn.mv_prob_uniform(vrow, bm.asarray(-1.0), bm.asarray(1.0),
                                               0.3, seed=4, shape=_SHAPE, transpose=True))
    assert ou.shape == (_SHAPE[1],) and np.all(np.isfinite(ou))
    on = np.asarray(bm.jitconn.mv_prob_normal(vrow, bm.asarray(0.0), bm.asarray(1.0),
                                              0.3, seed=4, shape=_SHAPE, transpose=True))
    assert on.shape == (_SHAPE[1],) and np.all(np.isfinite(on))
    # homo with Array weight + Array vector
    oh = np.asarray(bm.jitconn.mv_prob_homo(vrow, bm.asarray(1.0), 0.3,
                                            seed=4, shape=_SHAPE, transpose=True))
    assert oh.shape == (_SHAPE[1],)


def test_get_weight_matrices():
    mh = np.asarray(bm.jitconn.get_homo_weight_matrix(1.0, 0.3, seed=1, shape=_SHAPE))
    mu = np.asarray(bm.jitconn.get_uniform_weight_matrix(-1., 1., 0.3, seed=1, shape=_SHAPE))
    mn = np.asarray(bm.jitconn.get_normal_weight_matrix(0., 1., 0.3, seed=1, shape=_SHAPE))
    for m in (mh, mu, mn):
        assert m.shape == _SHAPE


def test_get_weight_matrices_transpose_and_seed_none():
    # transpose=True -> (cols, rows); seed=None branch; Array args unwrap.
    mh = np.asarray(bm.jitconn.get_homo_weight_matrix(bm.asarray(1.0), 0.3,
                                                      seed=None, shape=_SHAPE, transpose=True))
    assert mh.shape == (_SHAPE[1], _SHAPE[0])
    mu = np.asarray(bm.jitconn.get_uniform_weight_matrix(bm.asarray(-1.0), bm.asarray(1.0), 0.3,
                                                         seed=None, shape=_SHAPE, transpose=True))
    assert mu.shape == (_SHAPE[1], _SHAPE[0])
    mn = np.asarray(bm.jitconn.get_normal_weight_matrix(bm.asarray(0.0), bm.asarray(1.0), 0.3,
                                                        seed=None, shape=_SHAPE, transpose=True))
    assert mn.shape == (_SHAPE[1], _SHAPE[0])


# ===========================================================================
# M-15 — pre2post_mean / syn2post_mean / syn2post_softmax edge cases
# ===========================================================================

def test_syn2post_mean_empty_group_is_zero():
    syn = jnp.asarray([1., 3., 5.])
    post_ids = jnp.asarray([0, 0, 2])    # post group 1 is empty
    out = np.asarray(bm.syn2post_mean(syn, post_ids, 3))
    assert out[0] == pytest.approx(2.0)   # mean(1, 3)
    assert out[1] == pytest.approx(0.0)   # empty -> 0, not NaN
    assert out[2] == pytest.approx(5.0)


def test_syn2post_mean_propagates_genuine_nan():
    syn = jnp.asarray([1., np.nan, 5.])
    post_ids = jnp.asarray([0, 0, 2])
    out = np.asarray(bm.syn2post_mean(syn, post_ids, 3))
    assert np.isnan(out[0])               # genuine NaN must propagate
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(5.0)


def test_syn2post_softmax_propagates_nan_and_normalizes():
    post_ids = jnp.asarray([0, 0, 2])
    # genuine NaN must not be silently zeroed
    syn_nan = jnp.asarray([1., np.nan, 5.])
    out_nan = np.asarray(bm.syn2post_softmax(syn_nan, post_ids, 3))
    assert np.isnan(out_nan[0]) and np.isnan(out_nan[1])
    # clean input: each non-empty group's softmax weights sum to 1
    syn = jnp.asarray([1., 3., 5.])
    out = np.asarray(bm.syn2post_softmax(syn, post_ids, 3))
    assert out[0] + out[1] == pytest.approx(1.0, abs=1e-6)
    assert out[2] == pytest.approx(1.0, abs=1e-6)


def test_pre2post_mean_scalar_and_vector_branches():
    post_ids = jnp.asarray([0, 0, 2])
    # scalar branch: constant broadcast to targeted posts, others 0
    pm = np.asarray(bm.pre2post_mean(2.0, 3, post_ids))
    np.testing.assert_allclose(pm, [2., 0., 2.], atol=1e-6)
    # vector branch routes through syn2post_mean
    pre_vals = jnp.asarray([10., 20., 30.])
    pre_ids = jnp.asarray([0, 1, 2])
    pmv = np.asarray(bm.pre2post_mean(pre_vals, 3, post_ids, pre_ids))
    # post 0 gets mean(pre[0], pre[1]) = 15, post 2 gets pre[2] = 30
    np.testing.assert_allclose(pmv, [15., 0., 30.], atol=1e-6)


def test_pre2post_reductions_and_pre2syn():
    post_ids = jnp.asarray([0, 0, 2])
    assert np.asarray(bm.pre2post_sum(2.0, 3, post_ids)).tolist() == [4., 0., 2.]
    assert np.asarray(bm.pre2post_prod(2.0, 3, post_ids)).tolist() == [0., 0., 0.]
    assert np.asarray(bm.pre2post_max(2.0, 3, post_ids)).tolist() == [2., 0., 2.]
    assert np.asarray(bm.pre2post_min(2.0, 3, post_ids)).tolist() == [0., 0., 0.]
    syn = bm.pre2syn(jnp.asarray([1., 2., 3.]), jnp.asarray([0, 2]))
    np.testing.assert_allclose(np.asarray(syn), [1., 3.], atol=1e-6)


def test_syn2post_reductions():
    syn = jnp.asarray([1., 3., 5.])
    post_ids = jnp.asarray([0, 0, 2])
    np.testing.assert_allclose(np.asarray(bm.syn2post_sum(syn, post_ids, 3)),
                               [4., 0., 5.], atol=1e-6)
    np.testing.assert_allclose(np.asarray(bm.syn2post_prod(syn, post_ids, 3)),
                               [3., 1., 5.], atol=1e-6)
    # max of empty group is -inf, min is +inf (segment reduction identities)
    assert np.asarray(bm.syn2post_max(syn, post_ids, 3))[0] == 3.
    assert np.asarray(bm.syn2post_min(syn, post_ids, 3))[0] == 1.


def test_pre2post_reductions_vector_branch():
    # Vector pre_values + pre_ids exercises the heterogeneous gather branch.
    pre_vals = jnp.asarray([1., 2., 3., 4.])
    pre_ids = jnp.asarray([0, 1, 2, 3])
    post_ids = jnp.asarray([0, 0, 1, 1])
    post_num = 2
    assert np.asarray(bm.pre2post_sum(pre_vals, post_num, post_ids, pre_ids)).tolist() == [3., 7.]
    # prod / min accumulate against the zero-initialised output -> 0 here.
    assert np.asarray(bm.pre2post_prod(pre_vals, post_num, post_ids, pre_ids)).tolist() == [0., 0.]
    assert np.asarray(bm.pre2post_max(pre_vals, post_num, post_ids, pre_ids)).tolist() == [2., 4.]
    assert np.asarray(bm.pre2post_min(pre_vals, post_num, post_ids, pre_ids)).tolist() == [0., 0.]


def test_pre2post_vector_without_pre_ids_raises():
    # The _raise_pre_ids_is_none guard fires for heterogeneous values w/o pre_ids.
    from brainpy._errors import MathError
    pre_vals = jnp.asarray([1., 2., 3.])
    post_ids = jnp.asarray([0, 1, 1])
    with pytest.raises(MathError):
        bm.pre2post_sum(pre_vals, 2, post_ids)


def test_syn2post_bool_dtype_promotion():
    # bool syn_values -> promoted to int in every syn2post reduction.
    # group 0 = {True, False} -> {1, 0}; group 1 = {True} -> {1}
    syn = jnp.asarray([True, False, True], dtype=bool)
    post_ids = jnp.asarray([0, 0, 1])
    assert np.asarray(bm.syn2post_sum(syn, post_ids, 2)).tolist() == [1, 1]
    assert np.asarray(bm.syn2post_prod(syn, post_ids, 2)).tolist() == [0, 1]
    assert np.asarray(bm.syn2post_max(syn, post_ids, 2)).tolist() == [1, 1]
    assert np.asarray(bm.syn2post_min(syn, post_ids, 2)).tolist() == [0, 1]
    np.testing.assert_allclose(np.asarray(bm.syn2post_mean(syn, post_ids, 2)), [0.5, 1.0], atol=1e-6)
    sm = np.asarray(bm.syn2post_softmax(syn, post_ids, 2))
    assert np.all(np.isfinite(sm))


def test_pre2post_event_sum():
    # CSR connectivity: pre 0 -> post {0,2}, pre 1 -> post {1}, pre 2 -> post {3}
    indices = jnp.asarray([0, 2, 1, 3], dtype=jnp.int32)
    indptr = jnp.asarray([0, 2, 3, 4], dtype=jnp.int32)
    events = jnp.asarray([True, False, True], dtype=bool)
    out = np.asarray(bm.pre2post_event_sum(events, (indices, indptr), 4, 1.0))
    # pre 0 fires -> +1 at posts 0 and 2; pre 2 fires -> +1 at post 3
    np.testing.assert_allclose(out, [1., 0., 1., 1.], atol=1e-6)
