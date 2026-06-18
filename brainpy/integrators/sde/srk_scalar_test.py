# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.integrators.sde.srk_scalar``.

Drives every scalar-Wiener stochastic Runge-Kutta integrator
(``SRK1W1``, ``SRK2W1``, ``KlPl``) through a few steps of a simple scalar
linear SDE ``dx = -x dt + 0.1 x dW`` and checks the results are finite.
``KlPl`` in particular had no prior test coverage, so its ``build`` /
generated step function is exercised here.

Also asserts that each method rejects a ``VECTOR_WIENER`` process (the
``assert self.wiener_type == SCALAR_WIENER`` guard in each ``__init__``).
"""

import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators import sde


def f(x, t):
    return -x


def g(x, t):
    return 0.1 * x


def _run(method_cls, steps=20, **kwargs):
    bm.random.seed(123)
    intg = bm.jit(method_cls(f=f, g=g, dt=0.01, show_code=True, **kwargs))
    x = 1.0
    for i in range(steps):
        x = intg(x, i * 0.01)
    return np.asarray(bm.as_jax(x))


class TestSRK1W1:
    def test_runs_finite(self):
        out = _run(sde.SRK1W1)
        assert np.all(np.isfinite(out))

    def test_rejects_vector_wiener(self):
        with pytest.raises(AssertionError):
            sde.SRK1W1(f=f, g=g, dt=0.01, wiener_type=bp.integrators.VECTOR_WIENER)


class TestSRK2W1:
    def test_runs_finite(self):
        out = _run(sde.SRK2W1)
        assert np.all(np.isfinite(out))

    def test_rejects_vector_wiener(self):
        with pytest.raises(AssertionError):
            sde.SRK2W1(f=f, g=g, dt=0.01, wiener_type=bp.integrators.VECTOR_WIENER)


class TestKlPl:
    def test_runs_finite(self):
        # KlPl uses triple_integral=False noise terms and a 2-stage scheme.
        out = _run(sde.KlPl)
        assert np.all(np.isfinite(out))

    def test_rejects_vector_wiener(self):
        with pytest.raises(AssertionError):
            sde.KlPl(f=f, g=g, dt=0.01, wiener_type=bp.integrators.VECTOR_WIENER)


class TestViaSdeint:
    @pytest.mark.parametrize('method', ['srk1w1', 'srk2w1', 'klpl'])
    def test_sdeint_dispatch(self, method):
        bm.random.seed(7)
        intg = bp.sdeint(f=f, g=g, method=method,
                         intg_type=bp.integrators.ITO_SDE,
                         wiener_type=bp.integrators.SCALAR_WIENER,
                         dt=0.01)
        x = 1.0
        for i in range(10):
            x = intg(x, i * 0.01)
        assert np.all(np.isfinite(np.asarray(bm.as_jax(x))))


# ---------------------------------------------------------------------------
# Regression for P6-H1: the KlPl final-stage diffusion weights were wrong
# (g1 = -I1 + I11/dt_sqrt + I10/dt, g2 = I11/dt_sqrt), so g1 + g2 != I1 and
# the scheme did not converge. The corrected weights are
#   g1 = I1 - I11/dt_sqrt,   g2 = I11/dt_sqrt   (=> g1 + g2 = I1).
# ---------------------------------------------------------------------------

_A, _B, _X0 = -0.5, 0.3, 1.0  # dX = a X dt + b X dW (Ito)


def _gbm_exact(w_total, t):
    return _X0 * np.exp((_A - _B * _B / 2) * t + _B * w_total)


def _ref_klpl_terminal(n, dW, dI0, t_end=1.0):
    """NumPy reference of the *corrected* KlPl step, driven by a fixed path."""
    dt = t_end / n
    ds = np.sqrt(dt)
    x = _X0
    for i in range(n):
        I1 = dW[i]
        I11 = 0.5 * (I1 ** 2 - dt)
        f1 = _A * x
        g1s1 = _B * x
        h1s2 = x + dt * f1 + ds * g1s1
        g1s2 = _B * h1s2
        gg1 = I1 - I11 / ds
        gg2 = I11 / ds
        x = x + dt * f1 + gg1 * g1s1 + gg2 * g1s2
    return x


class TestKlPlConvergence:
    """P6-H1: KlPl must converge (strong order ~1.0) on a linear SDE."""

    def test_generated_weights_corrected(self):
        # The generated step code must encode g1 = I1 - I11/dt_sqrt and
        # g2 = I11/dt_sqrt (so g1 + g2 = I1). The buggy version had
        # g1 = -x_I1 + x_I11/dt_sqrt + x_I10/dt.
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            sde.KlPl(f=f, g=g, dt=0.01, show_code=True)
        code = buf.getvalue()
        assert 'x_g1 = x_I1 - x_I11/dt_sqrt' in code, code
        assert 'x_g2 = x_I11 / dt_sqrt' in code, code
        # the final-stage diffusion weight g1 must no longer contain the
        # spurious mixed-integral term I10/dt
        g1_line = [ln for ln in code.splitlines() if 'x_g1 =' in ln][0]
        assert 'I10' not in g1_line, g1_line

    def test_reference_strong_convergence(self):
        # The corrected scheme (NumPy reference) converges to the exact GBM
        # solution as dt -> 0; the buggy scheme had a flat ~0.3 error.
        errs = []
        for n in (50, 100, 200, 400):
            dt = 1.0 / n
            rng = np.random.default_rng(7)
            path_errs = []
            for _ in range(300):
                dW = rng.normal(0, np.sqrt(dt), n)
                dI0 = rng.normal(0, np.sqrt(dt), n)
                x = _ref_klpl_terminal(n, dW, dI0)
                path_errs.append(abs(x - _gbm_exact(dW.sum(), 1.0)))
            errs.append(float(np.mean(path_errs)))
        rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert np.mean(rates) > 0.7, f'KlPl not converging: errs={errs}, rates={rates}'
        assert errs[-1] < 1e-3, f'KlPl terminal error too large: {errs}'

    def test_integrator_matches_reference(self):
        # Run the *actual* compiled KlPl integrator and a NumPy reference on
        # the same Wiener path. The generated code draws, per step,
        #   I1 = dt_sqrt * randn(),  I0 = dt_sqrt * randn()
        # so seeding bm.random lets us mirror the exact draws.
        n, dt = 8, 0.01
        ds = np.sqrt(dt)
        bm.random.seed(2024)
        # mirror the generated draw order (I1 then I0 per step) with the SAME
        # RandomState by drawing from it directly first to fix the sequence.
        rng_state = bm.random.RandomState(2024)
        dW = np.empty(n)
        dI0 = np.empty(n)
        for i in range(n):
            dW[i] = ds * float(rng_state.randn())
            dI0[i] = ds * float(rng_state.randn())

        bm.random.seed(2024)
        intg = sde.KlPl(f=lambda x, t: _A * x, g=lambda x, t: _B * x, dt=dt)
        x = _X0
        for i in range(n):
            x = intg(x, i * dt)
        got = float(bm.as_jax(x))

        ref = _ref_klpl_terminal(n, dW, dI0, t_end=n * dt)
        assert np.isclose(got, ref, rtol=1e-4, atol=1e-5), f'got={got}, ref={ref}'
