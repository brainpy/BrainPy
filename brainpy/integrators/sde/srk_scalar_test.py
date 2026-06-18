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
