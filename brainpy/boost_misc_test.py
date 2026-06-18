# -*- coding: utf-8 -*-
"""Coverage-boost tests for the BrainPy v2.7.8 audit (mid-laggard files).

This module is part of the audit coverage-boost effort. The sibling
``tests/audit/test_*_fixes.py`` files pin the audited regressions; this file
targets the remaining *uncovered* branches of five mid-laggard source files so
their line coverage reaches ~90% wherever the live code allows:

  * ``brainpy/algorithms/offline.py``           (78% -> exercise every algorithm
    via closed-form and gradient-descent, registry helpers, ``__repr__``).
  * ``brainpy/integrators/sde/normal.py``       (81% -> Euler/Heun/Milstein/
    Milstein2 for Ito & Stratonovich, scalar & vector Wiener, JointEq diffusion).
  * ``brainpy/running/jax_multiprocessing.py``  (85% -> vectorize/parallelize map
    with list & dict inputs, clear_buffer on/off, TypeError guards).
  * ``brainpy/integrators/joint_eq.py``         (83% -> nested JointEq, derivative
    evaluation, the dead ``_std_func`` helper, conflicting-name DiffEqError).
  * ``brainpy/dyn/synapses/abstract_models.py`` (88% -> DualExponV2 forward pass,
    ``add_current``/``return_info`` of every synapse model).

Known dead / broken lines that cannot be driven to success are pinned with
``pytest.raises`` and documented inline (see NOTE comments):
  * ``LogisticRegression.call`` -> ``IndexError`` (flattens targets then indexes
    ``targets.shape[1]``); its body (offline.py 391-415) is unreachable.
  * Vector-Wiener Milstein/Milstein2 -> broadcasting ``ValueError`` (normal.py).

The tests use tiny inputs / few iterations so the whole module runs well under
the 4-minute budget. Mutated global state (dt) is restored by a fixture.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm

DiffEqError = bp.errors.DiffEqError


@pytest.fixture(autouse=True)
def _restore_dt():
    """Restore the global integration time step mutated by some tests."""
    old = bm.get_dt()
    yield
    bm.set_dt(old)


# =========================================================================== #
# offline.py -- regression algorithms (closed-form + gradient descent),
#               registry helpers, __repr__.
# =========================================================================== #

def _reg_xy(n=24, d=2, seed=0):
    rng = np.random.RandomState(seed)
    x = jnp.asarray(rng.randn(n, d).astype('float32'))
    w_true = jnp.asarray(rng.randn(d, 1).astype('float32'))
    y = jnp.asarray(np.asarray(x) @ np.asarray(w_true))
    return x, y


def test_offline_linear_regression_both_paths():
    from brainpy.algorithms.offline import LinearRegression

    bm.random.seed(0)
    x, y = _reg_xy()
    # closed-form lstsq path
    w = LinearRegression()(y, x)
    assert w.shape == (2, 1)
    assert not bool(jnp.isnan(w).any())
    # gradient-descent path
    w_gd = LinearRegression(gradient_descent=True, max_iter=30, learning_rate=1e-3)(y, x)
    assert w_gd.shape == (2, 1)
    assert not bool(jnp.isnan(w_gd).any())


def test_offline_ridge_regression_both_paths_and_repr():
    from brainpy.algorithms.offline import RidgeRegression

    bm.random.seed(1)
    x, y = _reg_xy()
    # closed-form ridge (alpha > 0 -> penalty branch)
    w = RidgeRegression(alpha=1e-3)(y, x)
    assert w.shape == (2, 1)
    # gradient-descent ridge
    w_gd = RidgeRegression(alpha=1e-4, gradient_descent=True, max_iter=30)(y, x)
    assert w_gd.shape == (2, 1)
    # __repr__ override (offline.py line 287)
    assert 'RidgeRegression' in repr(RidgeRegression(alpha=0.5))


def test_offline_ridge_beta_deprecation_warning():
    """Cover the deprecated ``beta=`` branch (offline.py lines 253-256)."""
    from brainpy.algorithms.offline import RidgeRegression

    with pytest.warns(UserWarning):
        model = RidgeRegression(beta=0.25)
    assert model.regularizer.alpha == 0.25


def test_offline_lasso_regression_and_predict():
    from brainpy.algorithms.offline import LassoRegression

    bm.random.seed(2)
    x, y = _reg_xy()
    model = LassoRegression(alpha=0.05, degree=2, max_iter=30)
    w = model(y, x)
    assert not bool(jnp.isnan(w).any())
    # predict() path applies polynomial_features + normalize (lines 342-344)
    pred = model.predict(w, x)
    assert pred.shape[0] == x.shape[0]


def test_offline_elastic_net_regression_fit_and_predict():
    """ElasticNet ``call`` (fit) and ``predict`` are now consistent.

    Fixed in audit 2026-06-19: ``call`` previously built features with the default
    ``add_bias=True`` while ``predict`` used ``add_bias=self.add_bias`` (default
    ``False``), so the feature counts differed and ``predict`` crashed. ``call``
    now passes ``add_bias=self.add_bias``, so fit and predict agree.
    """
    from brainpy.algorithms.offline import ElasticNetRegression

    bm.random.seed(3)
    x, y = _reg_xy()
    model = ElasticNetRegression(alpha=0.05, degree=2, l1_ratio=0.5, max_iter=30)
    w = model(y, x)
    assert not bool(jnp.isnan(w).any())
    pred = model.predict(w, x)
    assert pred.shape[0] == x.shape[0]


def test_offline_polynomial_regression_and_predict():
    from brainpy.algorithms.offline import PolynomialRegression

    bm.random.seed(4)
    x, y = _reg_xy()
    model = PolynomialRegression(degree=2, gradient_descent=True, max_iter=20)
    w = model(y, x)
    assert not bool(jnp.isnan(w).any())
    pred = model.predict(w, x)            # lines 450-452
    assert pred.shape[0] == x.shape[0]


def test_offline_polynomial_ridge_regression_and_predict():
    from brainpy.algorithms.offline import PolynomialRidgeRegression

    bm.random.seed(5)
    x, y = _reg_xy()
    # closed-form (gradient_descent=False) with add_bias -> exercises the
    # intercept-not-penalized branch in RidgeRegression.call.
    model = PolynomialRidgeRegression(alpha=1e-2, degree=2, add_bias=True,
                                      gradient_descent=False)
    w = model(y, x)
    assert not bool(jnp.isnan(w).any())
    pred = model.predict(w, x)            # lines 487-489
    assert pred.shape[0] == x.shape[0]


def test_offline_logistic_regression_fit():
    """``LogisticRegression.call`` now fits instead of crashing.

    Fixed in audit 2026-06-19: ``call`` previously flattened ``targets`` to 1-D
    and then indexed ``targets.shape[1]`` (``IndexError``); it now initialises a
    1-D parameter vector and runs the gradient-descent solver.
    """
    from brainpy.algorithms.offline import LogisticRegression

    bm.random.seed(6)
    rng = np.random.RandomState(6)
    x = jnp.asarray(rng.randn(20, 2).astype('float32'))
    y = jnp.asarray((np.asarray(x)[:, :1] > 0).astype('float32'))
    w = LogisticRegression(max_iter=50)(y, x)
    assert w is not None
    assert not bool(jnp.isnan(jnp.asarray(w)).any())


def test_offline_registry_helpers_and_base_repr():
    from brainpy.algorithms import offline
    from brainpy.algorithms.offline import OfflineAlgorithm, LinearRegression

    methods = offline.get_supported_offline_methods()
    for name in ('linear', 'lstsq', 'ridge', 'lasso', 'logistic',
                 'polynomial', 'polynomial_ridge', 'elastic_net'):
        assert name in methods

    # get() success + failure
    assert offline.get('ridge') is offline.RidgeRegression
    with pytest.raises(ValueError):
        offline.get('does_not_exist')

    # base OfflineAlgorithm.__repr__ (offline.py line 104)
    assert repr(LinearRegression()) == 'LinearRegression'

    # register_offline_method: success then duplicate + type guards
    inst = LinearRegression()
    unique = 'boost_misc_custom_method'
    if unique not in offline.name2func:
        offline.register_offline_method(unique, inst)
    assert unique in offline.get_supported_offline_methods()
    with pytest.raises(ValueError):           # duplicate name (line 570)
        offline.register_offline_method(unique, inst)
    with pytest.raises(ValueError):           # not an OfflineAlgorithm (line 572)
        offline.register_offline_method('boost_misc_bad', object())
    # restore global registry state
    offline.name2func.pop(unique, None)


def test_offline_base_call_not_implemented():
    """Cover OfflineAlgorithm.call NotImplementedError (offline.py line 101)."""
    from brainpy.algorithms.offline import OfflineAlgorithm

    base = OfflineAlgorithm()
    with pytest.raises(NotImplementedError):
        base(jnp.ones((2, 1)), jnp.ones((2, 1)))


def test_offline_regression_initialize_noop():
    """Cover the no-op ``RegressionAlgorithm.initialize`` (offline.py line 141),
    which the framework never calls but is part of the public API."""
    from brainpy.algorithms.offline import LinearRegression

    model = LinearRegression()
    assert model.initialize(1, 2, foo='bar') is None


def test_offline_check_data_flatten_3d():
    """Cover ``_check_data_2d_atls`` flatten branch (offline.py line 111) and the
    ndim<2 ValueError (line 109)."""
    from brainpy.algorithms.offline import _check_data_2d_atls

    flat = _check_data_2d_atls(bm.ones((2, 3, 4)))
    assert flat.ndim == 2
    with pytest.raises(ValueError):
        _check_data_2d_atls(bm.ones(5))


# =========================================================================== #
# sde/normal.py -- Euler / Heun / Milstein / Milstein2 integrators.
# =========================================================================== #

def test_sde_euler_scalar_wiener_ito_and_stratonovich():
    bm.random.seed(10)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for itype in ['Ito', 'Stratonovich']:
        intg = bp.sdeint(lambda x, t: -x, g, method='euler', intg_type=itype)
        x = jnp.array([1.0])
        for i in range(3):
            x = intg(x, i * 0.01, dt=0.01)
        assert np.all(np.isfinite(np.asarray(x)))


def test_sde_heun_stratonovich_runs_and_ito_rejected():
    bm.random.seed(11)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    intg = bp.sdeint(lambda x, t: -x, g, method='heun', intg_type='Stratonovich')
    out = intg(jnp.array([1.0]), 0.0, dt=0.01)
    assert np.all(np.isfinite(np.asarray(out)))
    # Heun only supports Stratonovich -> IntegratorError on Ito.
    with pytest.raises(bp.errors.IntegratorError):
        bp.sdeint(lambda x, t: -x, g, method='heun', intg_type='Ito')


def test_sde_milstein_scalar_wiener_ito_and_stratonovich():
    bm.random.seed(12)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for itype in ['Ito', 'Stratonovich']:
        intg = bp.sdeint(lambda x, t: -x, g, method='milstein', intg_type=itype)
        x = jnp.array([1.0])
        for i in range(3):
            x = intg(x, i * 0.01, dt=0.01)
        assert np.all(np.isfinite(np.asarray(x)))


def test_sde_milstein2_scalar_wiener_ito_and_stratonovich():
    bm.random.seed(13)
    g = lambda x, t: jnp.ones_like(x) * 0.1
    for method in ['milstein2', 'milstein_grad_free']:
        for itype in ['Ito', 'Stratonovich']:
            intg = bp.sdeint(lambda x, t: -x, g, method=method, intg_type=itype)
            out = intg(jnp.array([1.0]), 0.0, dt=0.01)
            assert np.all(np.isfinite(np.asarray(out)))


def test_sde_milstein_multivariable_jointeq_diffusion():
    """Milstein with a JointEq f/g (multi-variable) exercises ``_get_g_grad``
    recursion over JointEq sub-equations (normal.py lines 286-292)."""
    bm.random.seed(14)

    def dV(V, t, w):
        return -V + w

    def dw(w, t, V):
        return -w + V

    def gV(V, t, w):
        return jnp.ones_like(V) * 0.1

    def gw(w, t, V):
        return jnp.ones_like(w) * 0.1

    f = bp.JointEq(dV, dw)
    g = bp.JointEq(gV, gw)
    intg = bp.sdeint(f, g, method='milstein', intg_type='Ito')
    out = intg(jnp.array([1.0]), jnp.array([0.5]), 0.0, dt=0.01)
    assert len(out) == 2
    assert all(np.all(np.isfinite(np.asarray(o))) for o in out)


def test_sde_milstein_multivar_non_jointeq_raises():
    """A plain (non-JointEq) multi-variable f/g triggers the
    ``_get_g_grad`` failure path -> DiffEqError (normal.py 315-319 region)."""
    bm.random.seed(15)

    def f(x, y, t):
        return -x, -y

    def g(x, y, t):
        return 0.1 * jnp.ones_like(x), 0.1 * jnp.ones_like(y)

    with pytest.raises(DiffEqError):
        intg = bp.sdeint(f, g, method='milstein')
        intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)


def test_sde_euler_vector_wiener_ito_runs():
    """Cover the Euler VECTOR_WIENER Ito summation branch (normal.py ~155-156)."""
    bm.random.seed(16)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1 * jnp.ones((3, 2)),
                     method='euler', wiener_type='vector', intg_type='Ito')
    out = intg(jnp.ones(3), 0.0, dt=0.01)
    assert np.asarray(out).shape == (3,)
    assert np.all(np.isfinite(np.asarray(out)))


def test_sde_euler_vector_wiener_stratonovich_known_broadcast_error():
    """NOTE: broken path. The Euler (Euler-Heun) VECTOR_WIENER *Stratonovich*
    branch adds ``g(Y)`` of shape ``(3, 2)`` to a state of shape ``(3,)`` without
    summing over the noise axis, so it raises a broadcasting error. We still
    exercise the branch (normal.py ~168-178) up to the failure point and pin it.
    """
    bm.random.seed(17)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1 * jnp.ones((3, 2)),
                     method='euler', wiener_type='vector', intg_type='Stratonovich')
    with pytest.raises((ValueError, TypeError)):
        intg(jnp.ones(3), 0.0, dt=0.01)


def test_sde_euler_vector_wiener_scalar_diffusion_guard():
    """Cover the vector-wiener scalar-diffusion ValueError guard (normal.py 138-143)."""
    bm.random.seed(18)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: jnp.float32(0.1),
                     method='euler', wiener_type='vector')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


def test_sde_euler_single_var_drift_not_tensor_guard():
    """Cover the single-variable drift-not-a-tensor ValueError (normal.py ~117)."""
    bm.random.seed(19)
    intg = bp.sdeint(lambda x, t: -1.0, lambda x, t: jnp.ones_like(x) * 0.1,
                     method='euler')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


def test_sde_milstein_vector_wiener_known_broadcast_error():
    """NOTE: broken path. The Milstein / Milstein2 integrators do not correctly
    broadcast the diffusion-gradient term for VECTOR_WIENER noise, so a
    vector-wiener Milstein step raises a broadcasting ``ValueError`` (or
    ``TypeError``). We pin the failure while still exercising the branch up to
    the failure point (normal.py vector-wiener Milstein code).
    """
    bm.random.seed(20)

    def gv(x, t):
        return 0.1 * jnp.ones((3, 2))

    for method in ['milstein', 'milstein2']:
        intg = bp.sdeint(lambda x, t: -x, gv, method=method,
                         wiener_type='vector', intg_type='Ito')
        with pytest.raises((ValueError, TypeError)):
            intg(jnp.ones(3), 0.0, dt=0.01)


def test_sde_milstein2_vector_wiener_scalar_diffusion_guard():
    """Cover the Milstein2 vector-wiener scalar-diffusion guard (normal.py 469-474)."""
    bm.random.seed(21)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: jnp.float32(0.1),
                     method='milstein2', wiener_type='vector')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


def test_sde_euler_multivar_drift_not_list_guard():
    """Cover the multi-variable drift-not-list ValueError (normal.py ~121-124)."""
    bm.random.seed(26)

    def f(x, y, t):           # returns a single tensor, not a list/tuple
        return -x

    def g(x, y, t):
        return 0.1 * jnp.ones_like(x), 0.1 * jnp.ones_like(y)

    intg = bp.sdeint(f, g, method='euler')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)


def test_sde_euler_multivar_diffusion_not_list_guard():
    """Cover the multi-variable diffusion-not-list ValueError (normal.py 134-137)."""
    bm.random.seed(27)

    def f(x, y, t):
        return -x, -y

    def g(x, y, t):           # returns a single tensor, not a list/tuple
        return 0.1 * jnp.ones_like(x)

    intg = bp.sdeint(f, g, method='euler')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)


def test_sde_milstein_single_var_diffusion_not_tensor_guard():
    """Cover the single-variable Milstein diffusion-not-a-tensor ValueError
    (normal.py ~342)."""
    bm.random.seed(28)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1, method='milstein')
    with pytest.raises(ValueError):
        intg(jnp.array([1.0]), 0.0, dt=0.01)


def test_sde_milstein2_multivariable_runs():
    """Cover the Milstein2 multi-variable drift/diffusion branches (normal.py
    446-468 multi-var paths)."""
    bm.random.seed(22)

    def f(x, y, t):
        return -x, -y

    def g(x, y, t):
        return 0.1 * jnp.ones_like(x), 0.1 * jnp.ones_like(y)

    intg = bp.sdeint(f, g, method='milstein2', intg_type='Ito')
    out = intg(jnp.array([1.0]), jnp.array([2.0]), 0.0, dt=0.01)
    assert len(out) == 2
    assert all(np.all(np.isfinite(np.asarray(o))) for o in out)


def test_sde_milstein2_vector_wiener_known_broadcast_error():
    """NOTE: broken path. Like the gradient Milstein, the derivative-free
    Milstein2 VECTOR_WIENER branch (normal.py 495-509) mis-broadcasts the
    ``(diffusion_bar - diffusion)`` correction term ``(3, 2)`` against the
    summed-noise state ``(3,)`` and raises a broadcasting ``ValueError``. We
    exercise the branch up to the failure and pin it.
    """
    bm.random.seed(23)
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1 * jnp.ones((3, 2)),
                     method='milstein2', wiener_type='vector', intg_type='Ito')
    with pytest.raises((ValueError, TypeError)):
        intg(jnp.ones(3), 0.0, dt=0.01)


def test_sde_exponential_euler_scalar_and_vector():
    """Cover the SDE ExponentialEuler build/integral (normal.py 560-646)."""
    bm.random.seed(24)
    # scalar wiener
    for method in ['exp_euler', 'exponential_euler']:
        intg = bp.sdeint(lambda x, t: -x, lambda x, t: jnp.ones_like(x) * 0.1,
                         method=method)
        x = jnp.array([1.0])
        for i in range(3):
            x = intg(x, i * 0.01, dt=0.01)
        assert np.all(np.isfinite(np.asarray(x)))
    # vector wiener
    intg = bp.sdeint(lambda x, t: -x, lambda x, t: 0.1 * jnp.ones((3, 2)),
                     method='exp_euler', wiener_type='vector')
    out = intg(jnp.ones(3), 0.0, dt=0.01)
    assert np.asarray(out).shape == (3,)
    assert np.all(np.isfinite(np.asarray(out)))


def test_sde_exponential_euler_jointeq_multivariable():
    """Cover the ExponentialEuler JointEq build + multi-variable diffusion path
    (normal.py _build_integrator recursion, 624-646)."""
    bm.random.seed(25)

    def dV(V, t, w):
        return -V + w

    def dw(w, t, V):
        return -w + V

    def g(V, w, t):
        return jnp.ones_like(V) * 0.1, jnp.ones_like(w) * 0.1

    intg = bp.sdeint(bp.JointEq(dV, dw), g, method='exp_euler')
    out = intg(jnp.array([1.0]), jnp.array([0.5]), 0.0, dt=0.01)
    assert len(out) == 2
    assert all(np.all(np.isfinite(np.asarray(o))) for o in out)


def test_sde_exponential_euler_rejects_stratonovich():
    """Cover the ExponentialEuler Stratonovich NotImplementedError (normal.py 570-573)."""
    with pytest.raises(NotImplementedError):
        bp.sdeint(lambda x, t: -x, lambda x, t: jnp.ones((1,)) * 0.1,
                  method='exp_euler', intg_type='Stratonovich')


def test_sde_dead_codegen_helpers():
    """NOTE: dead code. ``df_and_dg``, ``dfdt`` and ``noise_terms`` are
    module-level code-generation helpers that are never called anywhere in the
    package (the live SDE codegen lives in ``srk_scalar.py``). We call them
    directly with throwaway lists to cover lines 37-60.
    """
    from brainpy.integrators.sde import normal

    lines = []
    normal.df_and_dg(lines, ['V', 'w'], ['t', 'Iext'])
    assert any('f(' in ln for ln in lines)
    lines2 = []
    normal.dfdt(lines2, ['V', 'w'])
    assert any('_dfdt' in ln for ln in lines2)
    lines3 = []
    normal.noise_terms(lines3, ['V', 'w'])
    assert any('_dW' in ln for ln in lines3)


# =========================================================================== #
# jax_multiprocessing.py -- vectorize / parallelize map.
# =========================================================================== #

def test_jax_vectorize_map_list_input():
    from brainpy.running.jax_multiprocessing import jax_vectorize_map

    def f(a, b):
        return a + b

    a = bm.arange(6.0)
    b = bm.arange(6.0) * 2
    out = jax_vectorize_map(f, [a, b], num_parallel=3, clear_buffer=False)
    assert np.allclose(np.asarray(out), np.asarray(a) + np.asarray(b))


def test_jax_vectorize_map_dict_input_and_clear_buffer():
    from brainpy.running.jax_multiprocessing import jax_vectorize_map

    def f(a, b):
        return a * b

    a = bm.arange(6.0)
    b = bm.arange(6.0) + 1
    expected = np.asarray(a) * np.asarray(b)
    # dict input + clear_buffer=True -> np.asarray / concatenate branch.
    # NOTE: clear_buffer=True normally calls ``bm.clear_buffer_memory()``, a
    # PROCESS-GLOBAL operation that deletes EVERY live device array (module-level
    # constants and persistent Variables in *other* test modules included),
    # poisoning the rest of the shared pytest session. We patch it to a no-op so
    # the clear_buffer code path is still exercised for coverage without nuking
    # the session.
    _orig_clear = bm.clear_buffer_memory
    bm.clear_buffer_memory = lambda *a, **k: None
    try:
        out = jax_vectorize_map(f, {'a': a, 'b': b}, num_parallel=2, clear_buffer=True)
    finally:
        bm.clear_buffer_memory = _orig_clear
    assert np.allclose(np.asarray(out), expected)


def test_jax_vectorize_map_type_error_and_length_mismatch():
    from brainpy.running.jax_multiprocessing import jax_vectorize_map

    # TypeError: arguments must be sequence or dict (line 60).
    with pytest.raises(TypeError):
        jax_vectorize_map(lambda a: a, 123, num_parallel=2)
    # ValueError: unequal element lengths (line 66-67).
    with pytest.raises(ValueError):
        jax_vectorize_map(lambda a, b: a + b,
                          [bm.arange(4.0), bm.arange(3.0)], num_parallel=2)


def test_jax_parallelize_map_list_and_dict():
    from brainpy.running.jax_multiprocessing import jax_parallelize_map

    # Default single-CPU pmap can map up to the local device count (1).
    def f(a, b):
        return a + b

    a = bm.arange(2.0)
    b = bm.arange(2.0) * 3
    expected = np.asarray(a) + np.asarray(b)
    out_list = jax_parallelize_map(f, [a, b], num_parallel=1, clear_buffer=False)
    assert np.allclose(np.asarray(out_list), expected)

    a = bm.arange(2.0)
    b = bm.arange(2.0) * 3
    expected = np.asarray(a) + np.asarray(b)
    # See note above: patch the process-global buffer wipe to a no-op so the
    # clear_buffer=True branch is covered without poisoning the shared session.
    _orig_clear = bm.clear_buffer_memory
    bm.clear_buffer_memory = lambda *a, **k: None
    try:
        out_dict = jax_parallelize_map(f, {'a': a, 'b': b}, num_parallel=1,
                                       clear_buffer=True)
    finally:
        bm.clear_buffer_memory = _orig_clear
    assert np.allclose(np.asarray(out_dict), expected)


def test_jax_parallelize_map_type_error():
    from brainpy.running.jax_multiprocessing import jax_parallelize_map

    with pytest.raises(TypeError):                 # line 125
        jax_parallelize_map(lambda a: a, 3.14, num_parallel=1)
    with pytest.raises(ValueError):                # length mismatch (line 131)
        jax_parallelize_map(lambda a, b: a + b,
                            [bm.arange(2.0), bm.arange(1.0)], num_parallel=1)


def test_jax_map_empty_input_returns_none():
    """Cover the ``res_tree is None -> return None`` branch of both maps (the loop
    body never executes for an empty task list): jax_multiprocessing 88-89, 155-156."""
    from brainpy.running.jax_multiprocessing import (jax_vectorize_map,
                                                     jax_parallelize_map)

    assert jax_vectorize_map(lambda a: a, [bm.zeros((0,))], num_parallel=1) is None
    assert jax_parallelize_map(lambda a: a, [bm.zeros((0,))], num_parallel=1) is None


# =========================================================================== #
# joint_eq.py -- nested JointEq, derivative evaluation, dead _std_func,
#                conflicting-name DiffEqError.
# =========================================================================== #

def test_jointeq_nested_derivative_evaluation():
    a, b = 0.02, 0.20
    dV = lambda V, t, u, Iext: 0.04 * V * V + 5 * V + 140 - u + Iext
    du = lambda u, t, V: a * (b * V - u)
    eq = bp.JointEq(dV, du)

    dw = lambda w, t, V: a * (b * V - w)
    eq2 = bp.JointEq(eq, dw)          # nested JointEq

    # arg_keys collected across nested equations: V, u, w are state variables.
    assert eq2.arg_keys[:3] == ['V', 'u', 'w']
    assert 'Iext' in eq2.arg_keys     # positional parameter propagated

    # derivative evaluation returns one value per state variable (3).
    res = eq2(-65.0, -14.0, -14.0, 0.0, Iext=10.0)
    assert len(res) == 3
    assert all(np.isfinite(float(r)) for r in res)


def test_jointeq_call_with_keyword_argument():
    def dV(V, t, w, gain=0.5):
        return -V + gain * w

    def dw(w, t, V, gain=0.5):
        return -w + gain * V

    eq = bp.JointEq([dV, dw])         # list-form exercises _check_eqs recursion
    assert 'gain' in eq.kwarg_keys
    res = eq(1.0, 2.0, 0.0, gain=0.3)
    assert len(res) == 2
    assert all(np.isfinite(float(r)) for r in res)


def test_jointeq_conflicting_kwarg_name_with_state_variable():
    """Cover the 'keyword argument conflicts with existing name' DiffEqError
    (joint_eq.py lines 189-194)."""
    def dV(V, t, w):
        return -V + w

    def dw(w, t, V=1.0):    # kwarg 'V' reuses the state variable name 'V'
        return -w + V

    with pytest.raises(DiffEqError):
        bp.JointEq(dV, dw)


def test_jointeq_conflicting_kwarg_defaults():
    def dV(V, t, a=1.0):
        return -V + a

    def dw(w, t, a=2.0):    # same kwarg name, different default
        return -w + a

    with pytest.raises(DiffEqError):
        bp.JointEq(dV, dw)


def test_jointeq_missing_time_variable_and_var_kinds():
    with pytest.raises(ValueError):                 # no 't' parameter (line 58)
        bp.JointEq(lambda V, w: -V)
    with pytest.raises(DiffEqError):                # *args (VAR_POSITIONAL)
        bp.JointEq(lambda V, t, *extra: -V)
    with pytest.raises(DiffEqError):                # **kwargs (VAR_KEYWORD)
        bp.JointEq(lambda V, t, **extra: -V)
    with pytest.raises(DiffEqError):                # non-callable element
        bp.JointEq(123)


def test_jointeq_rejects_keyword_only_and_positional_only():
    """Cover the KEYWORD_ONLY (line 40) and POSITIONAL_ONLY (line 43) rejection
    branches of ``_get_args``."""
    # KEYWORD_ONLY: parameter after a bare ``*``.
    def kw_only(V, t, *, x):
        return -V + x

    with pytest.raises(DiffEqError):
        bp.JointEq(kw_only)

    # POSITIONAL_ONLY: parameter before ``/`` (needs an exec'd def for the syntax).
    ns = {}
    exec("def pos_only(V, t, x, /):\n    return -V + x", ns)
    with pytest.raises(DiffEqError):
        bp.JointEq(ns['pos_only'])


def test_jointeq_call_with_kwarg_passed_positionally():
    """Cover the ``__call__`` branch where a trailing positional arg maps onto a
    keyword key (joint_eq.py line 235)."""
    def dV(V, t, w, gain=0.5):
        return -V + gain * w

    def dw(w, t, V, gain=0.5):
        return -w + gain * V

    eq = bp.JointEq(dV, dw)
    # arg_keys = [V, w, t]; passing a 4th positional arg maps onto kwarg 'gain'.
    res = eq(1.0, 2.0, 0.0, 0.3)
    assert len(res) == 2
    assert all(np.isfinite(float(r)) for r in res)


def test_jointeq_std_func_dependency_from_state_vars():
    """Cover the _std_func branch where a positional dependency is resolved from
    the state-variable tuple via ``all_vars.index`` (joint_eq.py line 76) and the
    branch where the dependency is supplied via keyword (line 72)."""
    from brainpy.integrators.joint_eq import _std_func

    def dV(V, t, w):    # 'w' is a dependency that lives in all_vars
        return -V + w

    wrapper = _std_func(dV, ['V', 'w'])
    out = wrapper(0.0, 1.0, 2.0)   # w resolved positionally (line 76)
    assert np.isfinite(float(out))
    # 'w' supplied as a keyword -> line 72 (par in args_and_kwargs).
    out2 = wrapper(0.0, 1.0, 2.0, w=3.0)
    assert np.isfinite(float(out2))


def test_jointeq_duplicate_state_variable_error():
    """Cover the duplicate-state-variable DiffEqError branch (joint_eq.py 157)."""
    def dV1(V, t):
        return -V

    def dV2(V, t):    # 'V' reused as a state variable
        return -2 * V

    with pytest.raises(DiffEqError):
        bp.JointEq(dV1, dV2)


def test_jointeq_dead_std_func_helper():
    """NOTE: ``_std_func`` is dead code (defined but never called anywhere in the
    package). We invoke it directly to exercise lines 64-82. It builds a wrapper
    that re-routes positional state vars / dependency lookups before calling the
    underlying derivative function.
    """
    from brainpy.integrators.joint_eq import _std_func

    def dV(V, t, w, gain=0.5):
        return -V + gain * w

    all_vars = ['V', 'w']
    wrapper = _std_func(dV, all_vars)
    # call(t, *vars, **args_and_kwargs): V and w are positional state vars;
    # 'w' is a dependency that is looked up from `vars` via all_vars.index.
    out = wrapper(0.0, 1.0, 2.0)          # V=1.0, w=2.0
    assert np.isfinite(float(out))
    # pass the kwarg explicitly to cover the kwargs branch (lines 77-79).
    out2 = wrapper(0.0, 1.0, 2.0, gain=0.9)
    assert np.isfinite(float(out2))


def test_jointeq_std_func_missing_dependency_raises():
    """Cover the 'Missing {par}' DiffEqError inside _std_func (line 75)."""
    from brainpy.integrators.joint_eq import _std_func

    def dV(V, t, missing_dep):
        return -V + missing_dep

    wrapper = _std_func(dV, ['V'])   # 'missing_dep' is neither passed nor a var
    with pytest.raises(DiffEqError):
        wrapper(0.0, 1.0)


# =========================================================================== #
# abstract_models.py -- synapse forward passes + add_current / return_info.
# =========================================================================== #

def _share(t=0.0, dt=0.1, i=0):
    bp.share.save(t=t, dt=dt, i=i)


def test_dualexponv2_forward_add_current_return_info():
    """DualExponV2 is not covered by the sibling synapse-forward sweep.
    Drive update(x) (add_current branch) and return_info (lines 404-437)."""
    bm.random.seed(50)
    syn = bp.dyn.DualExponV2(3, tau_decay=5.0, tau_rise=1.0)
    syn.reset_state()
    _share()
    spike = bm.asarray([1.0, 0.0, 1.0])
    out = syn.update(spike)               # x not None -> add_current branch
    assert jnp.asarray(out).shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(out)))
    # update with no current (x=None branch)
    out_none = syn.update()
    assert jnp.all(jnp.isfinite(jnp.asarray(out_none)))
    # return_info -> ReturnInfo with a callable (line 436)
    info = syn.return_info()
    assert info is not None


def test_expon_forward_add_current_return_info():
    bm.random.seed(51)
    syn = bp.dyn.Expon(3, tau=5.0)
    syn.reset_state()
    _share()
    out = syn.update(bm.ones(3))          # x not None -> add_current
    assert jnp.asarray(out).shape == (3,)
    out_none = syn.update()               # x None branch
    assert jnp.all(jnp.isfinite(jnp.asarray(out_none)))
    assert syn.return_info() is syn.g


def test_synapse_forward_and_return_info_methods():
    """Drive ``update()`` (covering each derivative + the discrete spike jump)
    and ``return_info`` for the JointEq-based synapses (abstract_models lines
    547/550/554-556, 727/730/737-741, 793-803, 867-888, plus return_info 292,
    559, 744, 806, 891)."""
    bm.random.seed(52)
    _share()
    spike_bool = bm.asarray([True, False, True], dtype=bool)
    spike_float = spike_bool.astype(float)
    for cls, kwargs, inp in [
        (bp.dyn.DualExpon, dict(tau_decay=5.0, tau_rise=1.0), spike_bool),
        (bp.dyn.Alpha, dict(tau_decay=5.0), spike_bool),
        (bp.dyn.NMDA, dict(tau_decay=10.0, tau_rise=2.0), spike_float),
        (bp.dyn.STD, dict(tau=20.0), spike_bool),
        (bp.dyn.STP, dict(), spike_bool),
    ]:
        syn = cls(3, **kwargs)
        syn.reset_state()
        out = syn.update(inp)
        assert jnp.asarray(out).shape == (3,)
        assert jnp.all(jnp.isfinite(jnp.asarray(out)))
        info = syn.return_info()
        assert info is not None


def test_dualexpon_forward_runs():
    """Exercise DualExpon.update (line 283-289) + return_info (line 292)."""
    bm.random.seed(53)
    syn = bp.dyn.DualExpon(3, tau_decay=5.0, tau_rise=1.0)
    syn.reset_state()
    _share()
    out = syn.update(bm.asarray([1.0, 0.0, 1.0]))
    assert jnp.asarray(out).shape == (3,)
    assert jnp.all(jnp.isfinite(jnp.asarray(out)))
    assert syn.return_info() is syn.g
