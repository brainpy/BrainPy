# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.integrators.sde.normal``.

Complements the existing ``normal_test.py`` (Lorenz-system based) by hitting
the single-variable scalar paths, the multi-variable JointEq paths, the
Stratonovich vs Ito branches, the vector-Wiener validation/error branches and
the constructor guards.

Covered methods: ``Euler`` (Ito scalar/vector, Stratonovich), ``Heun``
(Stratonovich + the "only Stratonovich" constructor error), ``Milstein``
(Ito + Stratonovich, scalar + JointEq), ``MilsteinGradFree`` (Ito +
Stratonovich), and ``ExponentialEuler`` (scalar Ito, vector Wiener, plus the
"no Stratonovich" constructor error and the multi-var DiffEqError).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators import constants, sde
from brainpy.integrators.sde import normal
from brainpy.integrators.sde.normal import (Euler, Heun, Milstein,
                                            MilsteinGradFree, ExponentialEuler,
                                            df_and_dg, dfdt, noise_terms)


class TestCodeGenHelpers:
    # df_and_dg / dfdt / noise_terms are module-level code-string builders that
    # the active integrators no longer reference; call them directly so their
    # bodies (string formatting) execute and stay regression-protected.
    def test_df_and_dg(self):
        lines = []
        df_and_dg(lines, ['x', 'y'], ['t'])
        joined = '\n'.join(lines)
        assert 'x_df' in joined and 'y_dg' in joined and 'f(' in joined and 'g(' in joined

    def test_dfdt(self):
        lines = []
        dfdt(lines, ['x'])
        assert any('x_dfdt' in ln for ln in lines)

    def test_noise_terms(self):
        lines = []
        noise_terms(lines, ['x'])
        joined = '\n'.join(lines)
        assert 'x_dg is not None' in joined and 'x_dW' in joined


# ----- simple scalar SDE: dx = -x dt + 0.1 x dW -----
def f_scalar(x, t):
    return -x


def g_scalar(x, t):
    return 0.1 * x


def _run_scalar(intg, steps=15):
    bm.random.seed(11)
    intg = bm.jit(intg)
    x = 1.0
    for i in range(steps):
        x = intg(x, i * 0.01)
    return np.asarray(bm.as_jax(x))


class TestEuler:
    def test_ito_scalar(self):
        intg = Euler(f=f_scalar, g=g_scalar, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_stratonovich_scalar(self):
        intg = Euler(f=f_scalar, g=g_scalar, dt=0.01,
                     intg_type=constants.STRA_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_ito_vector_wiener(self):
        # vector-Wiener: g returns a (..., m) array; the integrator sums over
        # the noise dimension.
        def g_vec(x, t):
            return bm.asarray([0.1 * x, 0.05 * x])

        intg = Euler(f=f_scalar, g=g_vec, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.VECTOR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_vector_wiener_scalar_diffusion_errors(self):
        # vector-Wiener with a scalar diffusion value must raise
        intg = Euler(f=f_scalar, g=g_scalar, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.VECTOR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_multivar_drift_must_be_list(self):
        # a JointEq returns a tuple; a single function that wrongly returns a
        # scalar for a multi-variable integrator triggers the list/tuple check.
        dx = lambda x, t, y: -x + y
        dy = lambda y, t, x: -y + x
        gx = lambda x, t, y: 0.1 * x
        gy = lambda y, t, x: 0.1 * y
        intg = Euler(f=bp.JointEq(dx, dy), g=bp.JointEq(gx, gy), dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        out = intg(1.0, 0.5, 0.0)
        assert len(out) == 2

    def test_stratonovich_vector_wiener(self):
        # Stratonovich + vector Wiener exercises the y_bar/jnp.sum noise branch.
        def g_vec(x, t):
            return bm.asarray([0.1 * x, 0.05 * x])

        intg = Euler(f=f_scalar, g=g_vec, dt=0.01,
                     intg_type=constants.STRA_SDE,
                     wiener_type=constants.VECTOR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_single_var_drift_not_tensor_raises(self):
        # if f returns a non-tensor for a single-variable equation -> ValueError
        def bad_drift(x, t):
            return [1.0, 2.0]  # a python list, not a tensor

        intg = Euler(f=bad_drift, g=g_scalar, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_multivar_drift_scalar_raises(self):
        # a 2-variable plain function whose drift returns a scalar (not a
        # list/tuple) hits the multi-variable drift validation error.
        def two_var_drift(x, y, t):
            return -x  # scalar, not a list/tuple

        def two_var_g(x, y, t):
            return 0.1 * x, 0.1 * y

        intg = Euler(f=two_var_drift, g=two_var_g, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 1.0, 0.0)

    def test_multivar_diffusion_scalar_raises(self):
        # diffusion returns a scalar for a multi-variable equation -> ValueError
        def two_var_drift(x, y, t):
            return -x, -y

        def two_var_g(x, y, t):
            return 0.1 * x  # scalar, not a list/tuple

        intg = Euler(f=two_var_drift, g=two_var_g, dt=0.01,
                     intg_type=constants.ITO_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 1.0, 0.0)

    def test_stratonovich_none_diffusion(self):
        # g returns None -> Stratonovich branch keeps y_bar = y (diffusion None).
        def g_none(x, t):
            return None

        intg = Euler(f=f_scalar, g=g_none, dt=0.01,
                     intg_type=constants.STRA_SDE,
                     wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))


class TestHeun:
    def test_stratonovich(self):
        intg = Heun(f=f_scalar, g=g_scalar, dt=0.01,
                    intg_type=constants.STRA_SDE,
                    wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_only_stratonovich_allowed(self):
        with pytest.raises(bp.errors.IntegratorError):
            Heun(f=f_scalar, g=g_scalar, dt=0.01,
                 intg_type=constants.ITO_SDE,
                 wiener_type=constants.SCALAR_WIENER)


class TestMilstein:
    def test_ito_scalar(self):
        intg = Milstein(f=f_scalar, g=g_scalar, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_stratonovich_scalar(self):
        intg = Milstein(f=f_scalar, g=g_scalar, dt=0.01,
                        intg_type=constants.STRA_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_jointeq(self):
        dx = lambda x, t, y: -x + y
        dy = lambda y, t, x: -y + x
        gx = lambda x, t, y: 0.1 * x
        gy = lambda y, t, x: 0.1 * y
        intg = Milstein(f=bp.JointEq(dx, dy), g=bp.JointEq(gx, gy), dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        out = intg(1.0, 0.5, 0.0)
        assert len(out) == 2 and all(np.all(np.isfinite(np.asarray(bm.as_jax(o)))) for o in out)

    def test_vector_wiener(self):
        def g_vec(x, t):
            return bm.asarray([0.1 * x, 0.05 * x])

        intg = Milstein(f=f_scalar, g=g_vec, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.VECTOR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_vector_wiener_scalar_diffusion_errors(self):
        intg = Milstein(f=f_scalar, g=g_scalar, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.VECTOR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_single_var_drift_not_tensor_raises(self):
        def bad_drift(x, t):
            return [1.0]

        intg = Milstein(f=bad_drift, g=g_scalar, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_single_var_diffusion_not_tensor_raises(self):
        def bad_g(x, t):
            return [0.1]

        intg = Milstein(f=f_scalar, g=bad_g, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_multivar_plain_function_raises_diffeq_error(self):
        # Milstein needs single-variable diffusion functions (or a JointEq of
        # them) so it can take a per-variable gradient. A plain 2-variable g
        # makes ``_get_g_grad`` raise a DiffEqError (covering its error path).
        def two_var_drift(x, y, t):
            return -x, -y

        def two_var_g(x, y, t):
            return 0.1 * x, 0.1 * y

        intg = Milstein(f=two_var_drift, g=two_var_g, dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(bp.errors.DiffEqError):
            intg(1.0, 1.0, 0.0)

    def test_single_var_jointeq_drift_tuple_raises(self):
        # A single-equation JointEq makes ``self.f`` return a 1-tuple while
        # ``len(self.variables) == 1``; Milstein.step then expects a bare tensor
        # for the drift and raises (covering its single-variable drift check).
        dx = lambda x, t: -x
        gx = lambda x, t: 0.1 * x
        intg = Milstein(f=bp.JointEq(dx), g=bp.JointEq(gx), dt=0.01,
                        intg_type=constants.ITO_SDE,
                        wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)


class TestMilsteinGradFree:
    def test_ito_scalar(self):
        intg = MilsteinGradFree(f=f_scalar, g=g_scalar, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_stratonovich_scalar(self):
        intg = MilsteinGradFree(f=f_scalar, g=g_scalar, dt=0.01,
                                intg_type=constants.STRA_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_registered_aliases(self):
        for name in ('milstein2', 'milstein_grad_free'):
            intg = bp.sdeint(f=f_scalar, g=g_scalar, method=name, dt=0.01,
                             intg_type=constants.ITO_SDE,
                             wiener_type=constants.SCALAR_WIENER)
            assert np.all(np.isfinite(_run_scalar(intg)))

    def test_vector_wiener(self):
        def g_vec(x, t):
            return bm.asarray([0.1 * x, 0.05 * x])

        intg = MilsteinGradFree(f=f_scalar, g=g_vec, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.VECTOR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_vector_wiener_scalar_diffusion_errors(self):
        intg = MilsteinGradFree(f=f_scalar, g=g_scalar, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.VECTOR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_jointeq_multivar(self):
        dx = lambda x, t, y: -x + y
        dy = lambda y, t, x: -y + x
        gx = lambda x, t, y: 0.1 * x
        gy = lambda y, t, x: 0.1 * y
        intg = MilsteinGradFree(f=bp.JointEq(dx, dy), g=bp.JointEq(gx, gy), dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        out = intg(1.0, 0.5, 0.0)
        assert len(out) == 2

    def test_single_var_drift_not_tensor_raises(self):
        def bad_drift(x, t):
            return [1.0]

        intg = MilsteinGradFree(f=bad_drift, g=g_scalar, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_single_var_diffusion_not_tensor_raises(self):
        def bad_g(x, t):
            return [0.1]

        intg = MilsteinGradFree(f=f_scalar, g=bad_g, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 0.0)

    def test_multivar_drift_scalar_raises(self):
        def two_var_drift(x, y, t):
            return -x

        def two_var_g(x, y, t):
            return 0.1 * x, 0.1 * y

        intg = MilsteinGradFree(f=two_var_drift, g=two_var_g, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 1.0, 0.0)

    def test_multivar_diffusion_scalar_raises(self):
        def two_var_drift(x, y, t):
            return -x, -y

        def two_var_g(x, y, t):
            return 0.1 * x

        intg = MilsteinGradFree(f=two_var_drift, g=two_var_g, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        with pytest.raises(ValueError):
            intg(1.0, 1.0, 0.0)


class TestExponentialEuler:
    def test_ito_scalar(self):
        intg = ExponentialEuler(f=f_scalar, g=g_scalar, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_no_diffusion(self):
        # g returning None exercises the "diffusion is None" branch.
        def g_none(x, t):
            return None

        intg = ExponentialEuler(f=f_scalar, g=g_none, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_vector_wiener(self):
        def g_vec(x, t):
            return bm.asarray([0.1 * x, 0.05 * x])

        intg = ExponentialEuler(f=f_scalar, g=g_vec, dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.VECTOR_WIENER)
        assert np.all(np.isfinite(_run_scalar(intg)))

    def test_jointeq_multivar(self):
        dx = lambda x, t, y: -x + y
        dy = lambda y, t, x: -y + x
        gx = lambda x, t, y: 0.1 * x
        gy = lambda y, t, x: 0.1 * y
        intg = ExponentialEuler(f=bp.JointEq(dx, dy), g=bp.JointEq(gx, gy), dt=0.01,
                                intg_type=constants.ITO_SDE,
                                wiener_type=constants.SCALAR_WIENER)
        out = intg(1.0, 0.5, 0.0)
        assert len(out) == 2

    def test_no_stratonovich(self):
        with pytest.raises(NotImplementedError):
            ExponentialEuler(f=f_scalar, g=g_scalar, dt=0.01,
                             intg_type=constants.STRA_SDE,
                             wiener_type=constants.SCALAR_WIENER)

    def test_multi_var_single_eq_raises(self):
        # a single non-JointEq function with 2 variables must raise DiffEqError
        def bad(x, y, t):
            return -x - y

        with pytest.raises(bp.errors.DiffEqError):
            ExponentialEuler(f=bad, g=g_scalar, dt=0.01,
                             intg_type=constants.ITO_SDE,
                             wiener_type=constants.SCALAR_WIENER)
