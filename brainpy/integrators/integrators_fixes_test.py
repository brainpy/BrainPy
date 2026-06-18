# -*- coding: utf-8 -*-
"""Regression + coverage tests for the BrainPy v2.7.8 integrators audit.

These tests pin the fixes recorded in ``docs/issues-found-20260618.md`` for the
``brainpy/integrators`` subtree.  Each regression test references the audit ID it
guards.  The remaining tests exercise the public integrator API broadly to keep
line coverage high on the assigned source files:

  * ode/adaptive_rk.py   -- C-12, H-26, H-27, H-28
  * ode/exponential.py   -- exp_euler / exp_euler_auto numerics
  * sde/base.py          -- C-13 (errors import; invalid intg_type/wiener_type)
  * sde/normal.py        -- C-13 (Heun Ito/Stratonovich guard)
  * integrators/runner.py-- H-29 (IntegratorRunner step-index monitor)
  * integrators/joint_eq.py -- L-13 (diagnostic DiffEqError message)
  * fde/Caputo.py        -- C-08 (CaputoEuler init scaling), H-31 (hists())
  * fde/GL.py            -- H-30 (GLShortMemory.reset key suffix)
  * fde/generic.py       -- H-32 (set/get_default_fdeint global)

The tests use tiny ``dt`` / step counts so the whole module runs in well under a
minute.  Global state (x64 precision, default FDE method) is restored by
fixtures.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.ode.adaptive_rk import BoSh3

IntegratorError = bp.errors.IntegratorError
DiffEqError = bp.errors.DiffEqError


# --------------------------------------------------------------------------- #
# Fixtures: restore global state mutated by some tests.
# --------------------------------------------------------------------------- #

@pytest.fixture
def x64():
    """Enable float64 for the duration of a test, then restore float32."""
    bm.enable_x64()
    try:
        yield
    finally:
        bm.disable_x64()


@pytest.fixture
def restore_default_fdeint():
    """Snapshot and restore the global default FDE method."""
    orig = bp.fde.get_default_fdeint()
    try:
        yield
    finally:
        bp.fde.set_default_fdeint(orig)


def _ev(seq):
    """Evaluate a Butcher-tableau row whose entries may be string fractions."""
    return [eval(x) if isinstance(x, str) else float(x) for x in seq]


# =========================================================================== #
# Regression tests
# =========================================================================== #

# --- C-12: adaptive RK no longer TypeErrors when ``tol`` is omitted --------- #

def test_c12_adaptive_rkf45_runs_without_tol_array_state():
    """C-12: adaptive=True with default tol must run (tol falls back to 0.1)."""
    f = bp.odeint(lambda y, t: -y, method='rkf45', adaptive=True)
    y_new, dt_new = f(jnp.array([1.0]), 0.0, dt=0.1)
    # exact solution y(0.1) = e^{-0.1}
    assert np.allclose(np.asarray(y_new), math.exp(-0.1), atol=1e-3)
    assert float(dt_new) > 0.0


def test_c12_adaptive_rkf45_runs_on_scalar_state():
    """C-12 / H-28: the same integrator must also work on a python-float scalar."""
    f = bp.odeint(lambda y, t: -y, method='rkf45', adaptive=True)
    y_new, dt_new = f(1.0, 0.0, dt=0.1)
    assert np.allclose(float(y_new), math.exp(-0.1), atol=1e-3)
    assert float(dt_new) > 0.0


# --- H-28: scalar state with default POP_VAR var_type ----------------------- #

def test_h28_scalar_state_all_adaptive_methods():
    """H-28: default var_type=POP_VAR must use jnp.sum, not builtin sum, so a
    scalar state does not raise ``'float' object is not iterable``."""
    for method in ['rkf45', 'rkdp', 'rkf12', 'ck', 'bs', 'heun_euler', 'BoSh3']:
        f = bp.odeint(lambda y, t: -y, method=method, adaptive=True)
        y_new, dt_new = f(1.0, 0.0, dt=0.05)
        assert np.isfinite(float(y_new))
        assert np.isfinite(float(dt_new))


# --- H-26: BoSh3 embedded error vector is non-degenerate -------------------- #

def test_h26_bosh3_embedded_error_non_degenerate():
    """H-26: B1 and B2 must each be consistent (sum ~ 1) and B1-B2 must be a
    real (non-zero) error estimator, not the zero-sum-B2 bug."""
    b1 = _ev(BoSh3.B1)
    b2 = _ev(BoSh3.B2)
    assert abs(sum(b1) - 1.0) < 1e-9, f'B1 must sum to 1, got {sum(b1)}'
    assert abs(sum(b2) - 1.0) < 1e-9, f'B2 must sum to 1, got {sum(b2)}'
    diff = [a - b for a, b in zip(b1, b2)]
    # the embedded error estimate must be non-degenerate (some non-zero weights)
    assert any(abs(d) > 1e-9 for d in diff), 'B1-B2 is degenerate (all zero)'
    # the buggy B2 summed to ~0; guard against that regression explicitly
    assert abs(sum(b2)) > 0.5


def test_h26_bosh3_integrates_correctly():
    """BoSh3 (3rd order) must integrate y'=-y accurately."""
    f = bp.odeint(lambda y, t: -y, method='BoSh3', adaptive=True)
    y_new, _ = f(jnp.array([1.0]), 0.0, dt=0.05)
    assert np.allclose(np.asarray(y_new), math.exp(-0.05), atol=1e-4)


# --- H-27: two-sided step-size controller can grow dt ----------------------- #

def test_h27_step_controller_grows_dt_when_error_small():
    """H-27: when the error is comfortably below tol the controller must be able
    to *increase* dt (the buggy one-sided controller never grew dt)."""
    f = bp.odeint(lambda y, t: -y, method='rkf45', adaptive=True)
    _, dt_new = f(jnp.array([1.0]), 0.0, dt=0.01)
    assert float(dt_new) > 0.01, 'controller failed to grow dt below tolerance'


# --- exp_euler numerics (python-float input) -------------------------------- #

def test_exp_euler_python_float_input():
    """exp_euler on a linear ODE y'=-2y is exact: y(0.3)=e^{-0.6}."""
    f = bp.odeint(lambda y, t: -2 * y, method='exp_euler', dt=0.3)
    out = f(1.0, 0.0, dt=0.3)
    assert np.allclose(float(out), math.exp(-0.6), atol=1e-5)


def test_exp_euler_auto_python_float_input():
    """exp_euler_auto must give the same exact result on the linear ODE."""
    f = bp.odeint(lambda y, t: -2 * y, method='exp_euler_auto', dt=0.3)
    out = f(1.0, 0.0, dt=0.3)
    assert np.allclose(float(out), math.exp(-0.6), atol=1e-5)


# --- C-13: SDE integrators raise IntegratorError, not NameError ------------- #

def test_c13_sde_invalid_intg_type_raises_integrator_error():
    """C-13: invalid intg_type must raise IntegratorError (errors import fixed),
    not NameError: name 'errors' is not defined."""
    with pytest.raises(IntegratorError):
        bp.sdeint(lambda x, t: -x, lambda x, t: 0.1,
                  method='euler', intg_type='WRONG')


def test_c13_sde_invalid_wiener_type_raises_integrator_error():
    """C-13: the same errors import guards the wiener_type validation path."""
    with pytest.raises(IntegratorError):
        bp.sdeint(lambda x, t: -x, lambda x, t: 0.1,
                  method='euler', wiener_type='WRONG')


def test_c13_heun_rejects_ito():
    """C-13 (sde/normal.py): Heun only supports Stratonovich; an Ito request
    must raise IntegratorError rather than NameError."""
    with pytest.raises(IntegratorError):
        bp.sdeint(lambda x, t: -x, lambda x, t: 0.1,
                  method='heun', intg_type='Ito')


# --- C-08: CaputoEuler does not mis-scale the initial condition ------------- #

def test_c08_caputo_euler_preserves_initial_condition(x64):
    """C-08: for D^a x = 0 with x(0)=1 (exact x==1), CaputoEuler must keep x~1
    across steps instead of returning dt^a/a."""
    intg = bp.fde.CaputoEuler(lambda x, t: jnp.zeros_like(jnp.asarray(x)),
                              alpha=0.8, num_memory=10, inits=[1.0])
    x = jnp.array([1.0])
    t = 0.0
    dt = 0.1
    for _ in range(5):
        x = intg(x, t, dt=dt)
        t += dt
        assert np.allclose(np.asarray(x), 1.0, atol=1e-6), \
            f'CaputoEuler drifted from the initial condition: {np.asarray(x)}'


# --- H-31: CaputoL1Schema.hists() iterates .items(), not bare dict ---------- #

def test_h31_caputo_l1_hists_returns_without_valueerror(x64):
    """H-31: after a step, .hists() (default numpy=True) must return a dict of
    arrays and not raise ValueError from iterating dict keys instead of items."""
    intg = bp.fde.CaputoL1Schema(lambda x, t: -x, alpha=0.9,
                                 num_memory=10, inits=[1.0])
    x = jnp.array([1.0])
    x = intg(x, 0.0, dt=0.1)
    hists = intg.hists()  # must not raise
    assert isinstance(hists, dict)
    for v in hists.values():
        assert isinstance(v, np.ndarray)
    # per-variable accessor path
    var = intg.variables[0]
    one = intg.hists(var=var)
    assert isinstance(one, np.ndarray)


# --- H-30: GLShortMemory.reset uses the '_delay' key suffix ------------------ #

def test_h30_glshortmemory_reset_works():
    """H-30: reset must use key+'_delay' and not raise KeyError."""
    g = bp.fde.GLShortMemory(lambda x, t: -x, alpha=0.6,
                             num_memory=8, inits=[1.0])
    g.reset(inits=[2.0])  # must not raise KeyError
    out = g(jnp.array([2.0]), 0.0, dt=0.1)
    assert np.all(np.isfinite(np.asarray(out)))


# --- H-32: set_default_fdeint writes the FDE global, get reads it back ------- #

def test_h32_set_get_default_fdeint_roundtrips(restore_default_fdeint):
    """H-32: set_default_fdeint must actually change get_default_fdeint (the bug
    wrote the wrong global, making it a no-op)."""
    for method in bp.fde.get_supported_methods():
        bp.fde.set_default_fdeint(method)
        assert bp.fde.get_default_fdeint() == method


def test_h32_set_default_fdeint_rejects_unknown(restore_default_fdeint):
    with pytest.raises(ValueError):
        bp.fde.set_default_fdeint('not-a-real-method')


# --- L-13: JointEq raises a *diagnostic* DiffEqError on conflicting kwarg ---- #

def test_l13_jointeq_conflicting_kwarg_raises_message():
    """L-13: a keyword argument that reuses a state-variable name must raise a
    DiffEqError carrying a non-empty diagnostic message."""
    def dV(V, t, w):
        return -V

    def dw(w, t, V=1.0):  # 'V' kwarg collides with state variable V
        return -w

    with pytest.raises(DiffEqError) as exc:
        bp.JointEq(dV, dw)
    assert str(exc.value).strip(), 'DiffEqError message must not be empty'
    assert 'V' in str(exc.value)


# =========================================================================== #
# Coverage tests
# =========================================================================== #

def test_cov_odeint_all_methods_array_and_scalar():
    """Exercise non-adaptive + adaptive ODE methods on array and scalar state."""
    y0 = jnp.array([1.0])
    for m in ['euler', 'rk2', 'rk4']:
        out = bp.odeint(lambda y, t: -y, method=m)(y0, 0.0, dt=0.01)
        assert np.allclose(np.asarray(out), math.exp(-0.01), atol=1e-3)

    for m in ['rkf45', 'rkdp', 'rkf12', 'ck', 'bs', 'heun_euler', 'BoSh3']:
        f = bp.odeint(lambda y, t: -y, method=m, adaptive=True)
        y_new, dt_new = f(y0, 0.0, dt=0.01)
        assert np.allclose(np.asarray(y_new), math.exp(-0.01), atol=1e-2)
        assert float(dt_new) > 0.0


def test_cov_adaptive_explicit_tol_and_var_type():
    """Cover the adaptive path with an explicit tol and SCALAR var_type."""
    f = bp.odeint(lambda y, t: -y, method='rkf45', adaptive=True,
                  tol=1e-3, var_type='scalar')
    y_new, dt_new = f(1.0, 0.0, dt=0.01)
    assert np.isfinite(float(y_new))
    assert np.isfinite(float(dt_new))


def test_cov_adaptive_show_code():
    """show_code=True exercises the code-emission/print branch."""
    f = bp.odeint(lambda y, t: -y, method='rkf45', adaptive=True, show_code=True)
    out = f(jnp.array([1.0]), 0.0, dt=0.05)
    assert len(out) == 2


def test_cov_exp_euler_variants_array_state():
    """Exercise exp_euler / exp_euler_auto / exp_auto on an array state."""
    y0 = jnp.array([1.0, 2.0])
    for m in ['exp_euler', 'exp_euler_auto', 'exp_auto', 'exponential_euler']:
        out = bp.odeint(lambda y, t: -y, method=m)(y0, 0.0, dt=0.01)
        assert np.allclose(np.asarray(out), y0 * math.exp(-0.01), atol=1e-3)


def test_cov_sdeint_euler_milstein_heun_both_types():
    """Cover euler/milstein (Ito + Stratonovich) and heun (Stratonovich)."""
    bm.random.seed(1234)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for method in ['euler', 'milstein']:
        for itype in ['Ito', 'Stratonovich']:
            f = bp.sdeint(lambda x, t: -x, g, method=method, intg_type=itype)
            out = f(jnp.array([1.0]), 0.0, dt=0.01)
            assert np.all(np.isfinite(np.asarray(out)))
    # Heun is Stratonovich-only
    f = bp.sdeint(lambda x, t: -x, g, method='heun', intg_type='Stratonovich')
    out = f(jnp.array([1.0]), 0.0, dt=0.01)
    assert np.all(np.isfinite(np.asarray(out)))


def test_cov_integrator_runner_with_monitors():
    """H-29 + coverage: IntegratorRunner over a small ODE; the step-index
    monitor must report 0..N-1 (loop variable no longer clobbered)."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t: -V, method='rk4')
    runner = bp.IntegratorRunner(
        intg,
        monitors={'V': 'V', 'step': lambda sh: sh['i']},
        inits={'V': 1.0},
        dt=0.1,
        progress_bar=False,
    )
    runner.run(0.5)
    v = np.asarray(runner.mon['V']).ravel()
    step = np.asarray(runner.mon['step']).ravel()
    assert v.shape == (5,)
    assert np.allclose(v, np.exp(-0.1 * np.arange(1, 6)), atol=1e-3)
    # H-29: the step index is the real loop counter, not len(vars)-1
    assert list(step) == [0, 1, 2, 3, 4]


def test_cov_integrator_runner_seq_monitor():
    """Cover the tuple/list monitor formatting branch of IntegratorRunner."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t: -V, method='euler')
    runner = bp.IntegratorRunner(intg, monitors=['V'], inits={'V': 1.0},
                                 dt=0.1, progress_bar=False)
    runner.run(0.3)
    assert np.asarray(runner.mon['V']).shape == (3, 1)


def test_cov_jointeq_construction_and_integration():
    """Cover JointEq construction (incl. nested) and integration."""
    a, b = 0.7, 0.8

    def dV(V, t, w, Iext):
        return V - V ** 3 / 3 - w + Iext

    def dw(w, t, V):
        return a * (b * V - w)

    eq = bp.JointEq(dV, dw)
    flat = [v for sub in eq.vars_in_eqs for v in sub]
    assert set(flat) == {'V', 'w'}

    intg = bp.odeint(eq, method='rk2')
    V, w = intg(0.0, 0.0, 0.0, Iext=0.5, dt=0.01)
    assert np.isfinite(float(V)) and np.isfinite(float(w))

    # nested JointEq
    def du(u, t, V):
        return -u + V

    eq2 = bp.JointEq(eq, du)
    flat2 = [v for sub in eq2.vars_in_eqs for v in sub]
    assert set(flat2) == {'V', 'w', 'u'}


def test_cov_jointeq_rejects_non_callable():
    """Cover the _check_eqs error branch."""
    with pytest.raises(DiffEqError):
        bp.JointEq(123)


def test_cov_caputo_euler_integration(x64):
    """Integrate a non-trivial Caputo equation a few steps for coverage."""
    intg = bp.fde.CaputoEuler(lambda x, t: -x, alpha=0.9,
                              num_memory=20, inits=[1.0])
    x = jnp.array([1.0])
    t = 0.0
    for _ in range(5):
        x = intg(x, t, dt=0.05)
        t += 0.05
    assert np.all(np.isfinite(np.asarray(x)))
    # decaying solution must stay below the initial value
    assert float(x[0]) < 1.0


def test_cov_caputo_l1_integration_and_reset(x64):
    """Integrate CaputoL1Schema a few steps, hits hists(), then reset()."""
    intg = bp.fde.CaputoL1Schema(lambda x, t: -x, alpha=0.7,
                                 num_memory=20, inits=[1.0])
    x = jnp.array([1.0])
    t = 0.0
    for _ in range(4):
        x = intg(x, t, dt=0.05)
        t += 0.05
    assert np.all(np.isfinite(np.asarray(x)))
    h = intg.hists()
    assert isinstance(h, dict)
    intg.reset(inits=[0.5])
    assert np.allclose(np.asarray(intg.inits[intg.variables[0]]), 0.5)


def test_cov_glshortmemory_integration_multivar():
    """Integrate GLShortMemory on a coupled 2D system + binomial_coef access."""
    def f(x, y, t):
        return -x + 0.1 * y, -y - 0.1 * x

    g = bp.fde.GLShortMemory(f, alpha=[0.95, 0.95], num_memory=16,
                             inits=[1.0, 0.5])
    coef = g.binomial_coef
    assert coef.shape[0] == 16
    x = jnp.array([1.0])
    y = jnp.array([0.5])
    t = 0.0
    for _ in range(5):
        x, y = g(x, y, t, dt=0.02)
        t += 0.02
    assert np.all(np.isfinite(np.asarray(x)))
    assert np.all(np.isfinite(np.asarray(y)))


def test_cov_glshortmemory_via_fdeint():
    """Cover the fdeint factory dispatch to the short-memory integrator."""
    intg = bp.fdeint(alpha=0.8, num_memory=8, inits=[1.0],
                     method='short-memory', dt=0.05)(lambda x, t: -x)
    out = intg(jnp.array([1.0]), 0.0, dt=0.05)
    assert np.all(np.isfinite(np.asarray(out)))


def test_cov_caputo_euler_via_fdeint(restore_default_fdeint):
    """Cover fdeint(method=None) using the default + the 'euler' dispatch."""
    bp.fde.set_default_fdeint('euler')
    intg = bp.fdeint(alpha=0.8, num_memory=8, inits=[1.0],
                     method=None, dt=0.05)(lambda x, t: -x)
    out = intg(jnp.array([1.0]), 0.0, dt=0.05)
    assert np.all(np.isfinite(np.asarray(out)))


# --------------------------------------------------------------------------- #
# Extra coverage: SDE method/wiener-type/multivar branches (sde/normal.py).
# --------------------------------------------------------------------------- #

def test_cov_sde_milstein_variants():
    """Cover milstein2 / milstein_grad_free for both integral types."""
    bm.random.seed(21)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for method in ['milstein2', 'milstein_grad_free']:
        for itype in ['Ito', 'Stratonovich']:
            f = bp.sdeint(lambda x, t: -x, g, method=method, intg_type=itype)
            out = f(jnp.array([1.0]), 0.0, dt=0.01)
            assert np.all(np.isfinite(np.asarray(out)))


def test_cov_sde_exponential_euler():
    """Cover the SDE ExponentialEuler integrator."""
    bm.random.seed(22)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for method in ['exp_euler', 'exponential_euler']:
        f = bp.sdeint(lambda x, t: -x, g, method=method)
        out = f(jnp.array([1.0]), 0.0, dt=0.01)
        assert np.all(np.isfinite(np.asarray(out)))


def test_cov_sde_multivariable():
    """Cover the multi-variable drift/diffusion branches of the Euler integrator.

    Milstein deliberately supports only a single variable at a time and raises
    DiffEqError for multi-variable systems, so it is exercised separately.
    """
    bm.random.seed(23)

    def f(x, y, t):
        return -x, -y

    def g(x, y, t):
        return 0.1 * jnp.ones_like(x), 0.1 * jnp.ones_like(y)

    for itype in ['Ito', 'Stratonovich']:
        intg = bp.sdeint(f, g, method='euler', intg_type=itype)
        out = intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)
        assert len(out) == 2
        assert np.all(np.isfinite(np.asarray(out[0])))
        assert np.all(np.isfinite(np.asarray(out[1])))

    # Milstein rejects multi-variable systems (raised when building/calling).
    with pytest.raises(DiffEqError):
        intg = bp.sdeint(f, g, method='milstein')
        intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)


def test_cov_sde_vector_wiener():
    """Cover the VECTOR_WIENER (Ito) summation branch of the Euler integrator.

    Only the Ito branch is exercised: the Stratonovich vector-wiener path has a
    latent broadcasting bug (g(Y) of shape (3, 2) is added to a state of shape
    (3,) without summing over the noise axis) and is out of scope for this audit.
    """
    bm.random.seed(24)

    def fv(x, t):
        return -x

    def gv(x, t):
        return 0.1 * jnp.ones((3, 2))  # 2 independent noise sources

    intg = bp.sdeint(fv, gv, method='euler', wiener_type='vector',
                     intg_type='Ito')
    out = intg(jnp.ones(3), 0.0, dt=0.01)
    assert np.asarray(out).shape == (3,)
    assert np.all(np.isfinite(np.asarray(out)))


def test_cov_sde_vector_wiener_requires_nd_diffusion():
    """Cover the vector-wiener scalar-diffusion guard (ValueError path)."""
    bm.random.seed(25)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: jnp.float32(0.1),
                     method='euler', wiener_type='vector')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


def test_cov_sde_drift_must_be_tensor():
    """Cover the single-variable drift-not-a-tensor ValueError branch."""
    intg = bp.sdeint(lambda x, t: -1.0, lambda x, t: jnp.ones_like(x) * 0.1,
                     method='euler')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


# --------------------------------------------------------------------------- #
# Extra coverage: IntegratorRunner.run options (runner.py).
# --------------------------------------------------------------------------- #

def test_cov_runner_run_options():
    """Cover run(start_t=, eval_time=True) and the .mon['ts'] time axis."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t: -V, method='rk4')
    runner = bp.IntegratorRunner(intg, monitors={'V': 'V'}, inits={'V': 1.0},
                                 dt=0.1, progress_bar=False)
    runner.run(0.3, start_t=0.0, eval_time=True)
    assert np.asarray(runner.mon['V']).shape == (3, 1)
    # second run continues from the previous index (covers idx bookkeeping)
    runner.run(0.2)
    assert np.asarray(runner.mon['V']).shape == (2, 1)


def test_cov_runner_rejects_non_integrator():
    """Cover the target type check in IntegratorRunner.__init__."""
    with pytest.raises(TypeError):
        bp.IntegratorRunner(lambda V, t: -V, monitors={'V': 'V'},
                            inits={'V': 1.0}, dt=0.1)


# --------------------------------------------------------------------------- #
# Extra coverage: JointEq error branches (joint_eq.py).
# --------------------------------------------------------------------------- #

def test_cov_jointeq_duplicate_state_variable():
    """Cover the duplicate-state-variable DiffEqError branch."""
    def dV1(V, t):
        return -V

    def dV2(V, t):  # 'V' used as a state variable twice
        return -2 * V

    with pytest.raises(DiffEqError):
        bp.JointEq(dV1, dV2)


# --------------------------------------------------------------------------- #
# Extra coverage: exponential ODE auto-diff path (exponential.py).
# --------------------------------------------------------------------------- #

def test_cov_exp_euler_nonlinear():
    """Cover the auto-linearization path on a non-linear right-hand side."""
    # logistic-like ODE; just needs to run and stay finite
    intg = bp.odeint(lambda y, t: y * (1.0 - y), method='exp_euler_auto')
    y = jnp.array([0.1])
    t = 0.0
    for _ in range(5):
        y = intg(y, t, dt=0.05)
        t += 0.05
    assert np.all(np.isfinite(np.asarray(y)))
    assert float(y[0]) > 0.1  # logistic growth


def test_cov_exp_euler_show_code():
    """Cover the show_code emission branch of ExponentialEuler."""
    intg = bp.odeint(lambda y, t: -y, method='exp_euler', show_code=True)
    out = intg(jnp.array([1.0]), 0.0, dt=0.05)
    assert np.all(np.isfinite(np.asarray(out)))


# --------------------------------------------------------------------------- #
# Extra coverage: JointEq argument parsing edge cases (joint_eq.py).
# --------------------------------------------------------------------------- #

def test_cov_jointeq_missing_time_variable():
    """Cover the 'Do not find time variable "t"' ValueError branch."""
    with pytest.raises(ValueError):
        bp.JointEq(lambda V, w: -V)  # no 't' parameter


def test_cov_jointeq_rejects_var_positional():
    """Cover the VAR_POSITIONAL (*args) rejection branch."""
    with pytest.raises(DiffEqError):
        bp.JointEq(lambda V, t, *extra: -V)


def test_cov_jointeq_rejects_var_keyword():
    """Cover the VAR_KEYWORD (**kwargs) rejection branch."""
    with pytest.raises(DiffEqError):
        bp.JointEq(lambda V, t, **extra: -V)


def test_cov_jointeq_conflicting_kwarg_defaults():
    """Cover the 'two different default value' DiffEqError branch."""
    def dV(V, t, a=1.0):
        return -V + a

    def dw(w, t, a=2.0):  # same kwarg name, different default
        return -w + a

    with pytest.raises(DiffEqError):
        bp.JointEq(dV, dw)


def test_cov_jointeq_nested_list_and_shared_kwarg():
    """Cover the _check_eqs list/tuple recursion + a shared (consistent) kwarg
    default, then call with the kwarg passed both positionally and by keyword."""
    def dV(V, t, w, gain=0.5):
        return -V + gain * w

    def dw(w, t, V, gain=0.5):  # same default -> allowed, shared kwarg
        return -w + gain * V

    eq = bp.JointEq([dV, dw])  # list form exercises the recursion branch
    assert 'gain' in eq.kwarg_keys
    intg = bp.odeint(eq, method='euler')
    # call providing gain by keyword
    out = intg(1.0, 0.0, gain=0.3, dt=0.01)
    assert all(np.isfinite(float(o)) for o in out)


# --------------------------------------------------------------------------- #
# Extra coverage: FDE reset / check paths (Caputo.py, generic.py).
# --------------------------------------------------------------------------- #

def test_cov_caputo_euler_reset(x64):
    """Cover CaputoEuler.reset()."""
    intg = bp.fde.CaputoEuler(lambda x, t: -x, alpha=0.8,
                              num_memory=10, inits=[1.0])
    x = jnp.array([1.0])
    x = intg(x, 0.0, dt=0.05)
    intg.reset(inits=[3.0])
    assert np.allclose(np.asarray(intg.inits[intg.variables[0]]), 3.0)


def test_cov_caputo_euler_requires_tensor_drift(x64):
    """Cover the single-variable 'Derivative values must be a tensor' branch."""
    intg = bp.fde.CaputoEuler(lambda x, t: 0.0, alpha=0.8,
                              num_memory=10, inits=[1.0])
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.05)


def test_cov_fde_register_duplicate_rejected():
    """Cover register_fde_integrator duplicate-name guard."""
    from brainpy.integrators.fde.generic import register_fde_integrator
    from brainpy.integrators.fde.GL import GLShortMemory
    with pytest.raises(ValueError):
        register_fde_integrator('euler', GLShortMemory)


def test_cov_fdeint_unknown_method_rejected():
    """Cover the fdeint unknown-method ValueError branch."""
    with pytest.raises(ValueError):
        bp.fdeint(alpha=0.8, num_memory=4, inits=[1.0],
                  method='nope', dt=0.05)(lambda x, t: -x)


def test_cov_set_default_fdeint_type_check(restore_default_fdeint):
    """Cover the non-string set_default_fdeint guard."""
    with pytest.raises(ValueError):
        bp.fde.set_default_fdeint(123)


# --------------------------------------------------------------------------- #
# Extra coverage: exponential ODE branches (exponential.py).
# --------------------------------------------------------------------------- #

def test_cov_exp_euler_with_jointeq():
    """Cover the JointEq build path of ExponentialEuler (_build_integrator)."""
    def dV(V, t, w):
        return -V - w

    def dw(w, t, V):
        return -w + 0.1 * V

    eq = bp.JointEq(dV, dw)
    intg = bp.odeint(eq, method='exp_euler')
    out = intg(jnp.array([1.0]), jnp.array([0.5]), 0.0, dt=0.01)
    assert len(out) == 2
    assert all(np.all(np.isfinite(np.asarray(o))) for o in out)


def test_cov_exp_euler_rejects_integer_input():
    """Cover the float-dtype guard of the Exponential Euler integral."""
    intg = bp.odeint(lambda y, t: -y, method='exp_euler')
    with pytest.raises(ValueError):
        intg(jnp.array([1, 2, 3]), 0.0, dt=0.01)  # integer dtype


def test_cov_exp_euler_system_var_not_implemented():
    """Cover the SYSTEM_VAR NotImplementedError branch."""
    with pytest.raises(NotImplementedError):
        bp.odeint(lambda y, t: -y, method='exp_euler', var_type='system')


# --------------------------------------------------------------------------- #
# Extra coverage: Milstein vector-wiener branches (sde/normal.py).
# --------------------------------------------------------------------------- #

def test_cov_milstein_vector_wiener_latent_bug():
    """Document a latent (out-of-scope) bug: the Milstein integrators do not
    correctly broadcast the diffusion-gradient term for VECTOR_WIENER noise, so
    a vector-wiener Milstein step currently raises a broadcasting ValueError.

    This still exercises the vector-wiener branch up to the failure point; if the
    library is ever fixed, this test should assert a finite result instead.
    """
    bm.random.seed(31)

    def fv(x, t):
        return -x

    def gv(x, t):
        return 0.1 * jnp.ones((3, 2))

    for method in ['milstein', 'milstein2']:
        intg = bp.sdeint(fv, gv, method=method, wiener_type='vector',
                         intg_type='Ito')
        with pytest.raises((ValueError, TypeError)):
            intg(jnp.ones(3), 0.0, dt=0.01)


# --------------------------------------------------------------------------- #
# Extra coverage: IntegratorRunner init / dyn_args branches (runner.py).
# --------------------------------------------------------------------------- #

def test_cov_runner_inits_as_sequence():
    """Cover the list/sequence form of the ``inits`` argument."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t: -V, method='euler')
    runner = bp.IntegratorRunner(intg, monitors={'V': 'V'}, inits=[1.0],
                                 dt=0.1, progress_bar=False)
    runner.run(0.3)
    assert np.asarray(runner.mon['V']).shape == (3, 1)


def test_cov_runner_dyn_args():
    """Cover the dyn_args time-varying input path of IntegratorRunner.run."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t, Iext: -V + Iext, method='euler')
    runner = bp.IntegratorRunner(intg, monitors={'V': 'V'}, inits={'V': 0.0},
                                 dt=0.1, progress_bar=False)
    # 3 steps -> dyn_args first dimension must be 3
    runner.run(0.3, dyn_args={'Iext': jnp.ones(3)})
    assert np.asarray(runner.mon['V']).shape == (3, 1)


def test_cov_runner_dyn_args_shape_mismatch():
    """Cover the dyn_args duration-mismatch ValueError branch."""
    bm.set_dt(0.1)
    intg = bp.odeint(lambda V, t, Iext: -V + Iext, method='euler')
    runner = bp.IntegratorRunner(intg, monitors={'V': 'V'}, inits={'V': 0.0},
                                 dt=0.1, progress_bar=False)
    with pytest.raises(ValueError):
        runner.run(0.3, dyn_args={'Iext': jnp.ones(99)})


# --------------------------------------------------------------------------- #
# Extra coverage: SDE ExponentialEuler branches (sde/normal.py).
# --------------------------------------------------------------------------- #

def test_cov_sde_exp_euler_vector_wiener():
    """Cover the VECTOR_WIENER branch of the SDE ExponentialEuler."""
    bm.random.seed(41)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1 * jnp.ones((3, 2)),
                     method='exp_euler', wiener_type='vector')
    out = intg(jnp.ones(3), 0.0, dt=0.01)
    assert np.asarray(out).shape == (3,)
    assert np.all(np.isfinite(np.asarray(out)))


def test_cov_sde_exp_euler_rejects_stratonovich():
    """Cover the SDE ExponentialEuler Stratonovich NotImplementedError branch."""
    with pytest.raises(NotImplementedError):
        bp.sdeint(lambda x, t: -x, lambda x, t: jnp.ones((1,)) * 0.1,
                  method='exp_euler', intg_type='Stratonovich')


def test_cov_sde_exp_euler_with_jointeq():
    """Cover the JointEq build + multi-variable diffusion path of SDE exp_euler."""
    bm.random.seed(42)

    def dV(V, t, w):
        return -V

    def dw(w, t, V):
        return -w

    def gV(V, t, w):
        return jnp.ones_like(V) * 0.1

    def gw(w, t, V):
        return jnp.ones_like(w) * 0.1

    intg = bp.sdeint(bp.JointEq(dV, dw), bp.JointEq(gV, gw), method='exp_euler')
    out = intg(jnp.array([1.0]), jnp.array([0.5]), 0.0, dt=0.01)
    assert len(out) == 2
    assert all(np.all(np.isfinite(np.asarray(o))) for o in out)


# --------------------------------------------------------------------------- #
# Extra coverage: JointEq.__call__ with a tuple-returning sub-equation.
# --------------------------------------------------------------------------- #

def test_cov_jointeq_call_with_nested_tuple_result():
    """Cover JointEq.__call__ extending results from a tuple-returning sub-eq
    (a nested JointEq returns a list/tuple)."""
    def dV(V, t, w):
        return -V + w

    def dw(w, t, V):
        return -w + 0.1 * V

    inner = bp.JointEq(dV, dw)  # returns a list when called

    def du(u, t, V):
        return -u + V

    outer = bp.JointEq(inner, du)
    res = outer(1.0, 0.5, 0.2, 0.0)  # V, w, u, t
    assert len(res) == 3
    assert all(np.isfinite(float(r)) for r in res)


# --------------------------------------------------------------------------- #
# Extra coverage: multi-variable FDE + IntegratorRunner branches.
# --------------------------------------------------------------------------- #

def test_cov_caputo_euler_multivariable(x64):
    """Cover the multi-variable drift branch of CaputoEuler."""
    def f(x, y, t):
        return -x, -y

    intg = bp.fde.CaputoEuler(f, alpha=[0.8, 0.9], num_memory=10,
                              inits=[1.0, 2.0])
    x = jnp.array([1.0])
    y = jnp.array([2.0])
    t = 0.0
    for _ in range(3):
        x, y = intg(x, y, t, dt=0.05)
        t += 0.05
    assert np.all(np.isfinite(np.asarray(x)))
    assert np.all(np.isfinite(np.asarray(y)))
    # decaying solution
    assert float(x[0]) < 1.0 and float(y[0]) < 2.0


def test_cov_caputo_l1_multivariable(x64):
    """Cover the multi-variable branch + per-variable hists() of CaputoL1Schema."""
    def f(x, y, t):
        return -x, -y

    intg = bp.fde.CaputoL1Schema(f, alpha=[0.8, 0.9], num_memory=10,
                                 inits=[1.0, 2.0])
    x = jnp.array([1.0])
    y = jnp.array([2.0])
    t = 0.0
    for _ in range(3):
        x, y = intg(x, y, t, dt=0.05)
        t += 0.05
    hists = intg.hists()
    assert set(hists.keys()) == set(intg.variables)
    one = intg.hists(var=intg.variables[1])
    assert isinstance(one, np.ndarray)


def test_cov_runner_multivariable_with_progress_bar():
    """Cover the multi-variable update branch and the progress-bar callback of
    IntegratorRunner."""
    bm.set_dt(0.1)

    def dV(V, t, w):
        return -V + w

    def dw(w, t, V):
        return -w

    intg = bp.odeint(bp.JointEq(dV, dw), method='rk2')
    runner = bp.IntegratorRunner(
        intg,
        monitors={'V': 'V', 'w': 'w'},
        inits={'V': 1.0, 'w': 0.5},
        dt=0.1,
        progress_bar=True,
    )
    runner.run(0.3)
    assert np.asarray(runner.mon['V']).shape == (3, 1)
    assert np.asarray(runner.mon['w']).shape == (3, 1)


def test_cov_jointeq_rejects_keyword_only():
    """Cover the KEYWORD_ONLY parameter rejection branch of _get_args."""
    def dV(V, t, *, w):
        return -V

    with pytest.raises(DiffEqError):
        bp.JointEq(dV, lambda w, t: -w)


def test_cov_jointeq_rejects_positional_only():
    """Cover the POSITIONAL_ONLY parameter rejection branch of _get_args."""
    def dV(V, t, w, /):
        return -V

    with pytest.raises(DiffEqError):
        bp.JointEq(dV, lambda w, t: -w)
