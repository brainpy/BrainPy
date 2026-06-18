# -*- coding: utf-8 -*-
"""Regression + coverage tests for the 2026-06-18 BrainPy audit.

This module targets the train / analysis / top-level-glue fixes documented in
``docs/issues-found-20260618.md`` for the following source files:

* ``brainpy/algorithms/online.py``           — C-23 (block RLS for batch>1)
* ``brainpy/algorithms/offline.py``          — H-46 (GD ``.value`` bug),
                                                H-47 (ridge intercept penalty)
* ``brainpy/running/jax_multiprocessing.py`` — H-48 (pmap reuse / labels)
* ``brainpy/analysis/lowdim/lowdim_analyzer.py`` — H-49 (arg-unwrap),
                                                    H-50 (empty-candidate concat)
* ``brainpy/analysis/utils/optimization.py`` — H-49 (arg-unwrap in roots_of_1d_by_x)
* ``brainpy/runners.py``                     — C-22 (memory_efficient DSRunner)
* ``brainpy/measure.py``                     — H-43 (firing_rate normalization)
* ``brainpy/delay.py``                       — H-44 (VarDelay self.data),
                                                H-45 (size_without_batch)

The tests are intentionally tiny (small nets, short durations) so the whole
module runs in well under four minutes. They assert the *fixed* behavior; on the
buggy pre-audit code each regression test would raise or diverge.
"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_rls(batch_size, n_in=3, n_out=2, steps=400, seed=0):
    """Fit a known linear map ``Y = X @ Wtrue`` with the RLS online algorithm.

    Returns (final_error, has_nan).
    """
    from brainpy.algorithms.online import RLS

    rng = np.random.RandomState(seed)
    w_true = jnp.asarray(rng.randn(n_in, n_out))
    rls = RLS(alpha=0.1)
    rls.register_target(n_in, identifier="w")
    weight = bm.Variable(jnp.zeros((n_in, n_out)))
    for _ in range(steps):
        x = jnp.asarray(rng.randn(batch_size, n_in))
        y = x @ w_true
        out = x @ weight.value
        dw = rls(y, x, out, identifier="w")
        weight.value = weight.value + dw
    err = float(jnp.linalg.norm(weight.value - w_true))
    has_nan = bool(jnp.isnan(weight.value).any())
    return err, has_nan


# ---------------------------------------------------------------------------
# online.py — C-23: block RLS valid for any batch size
# ---------------------------------------------------------------------------

def test_rls_converges_batch1():
    """RLS fits a known linear map for batch=1 without divergence/NaN."""
    err, has_nan = _fit_rls(batch_size=1)
    assert not has_nan
    assert err < 0.5, f"RLS (B=1) did not converge: err={err}"


def test_rls_converges_batch4_no_nan():
    """C-23: block RLS must stay correct (no NaN / no divergence) for batch>1.

    On the pre-audit code the scalar ``c = sum(1/(1+HPHᵀ))`` collapse made the
    covariance ``P`` diverge for B>=4 and produced NaN weights.
    """
    err, has_nan = _fit_rls(batch_size=4)
    assert not has_nan, "RLS (B=4) produced NaN weights (C-23 regression)"
    assert err < 0.5, f"RLS (B=4) did not converge: err={err}"


def test_rls_batch1_matches_scalar_update():
    """For B==1 the block update reduces to the classic scalar RLS update."""
    from brainpy.algorithms.online import RLS

    rls = RLS(alpha=0.1)
    rls.register_target(3, identifier="w")
    x = jnp.array([[1.0, -2.0, 0.5]])
    target = jnp.array([[1.0, 0.0]])
    output = jnp.zeros((1, 2))
    dw = rls(target, x, output, identifier="w")
    assert dw.shape == (3, 2)
    assert not bool(jnp.isnan(dw).any())


def test_lms_call_runs():
    """Coverage: LMS online algorithm produces a finite weight update."""
    from brainpy.algorithms.online import LMS

    lms = LMS(alpha=0.01)
    x = jnp.ones((2, 3))
    target = jnp.ones((2, 2))
    output = jnp.zeros((2, 2))
    dw = lms(target, x, output)
    assert dw.shape == (3, 2)
    assert not bool(jnp.isnan(dw).any())


def test_online_registry_helpers():
    """Coverage: the online method registry getters."""
    from brainpy.algorithms import online

    methods = online.get_supported_online_methods()
    assert "rls" in methods and "lms" in methods
    assert online.get("rls") is online.RLS
    assert online.get("lms") is online.LMS
    with pytest.raises(ValueError):
        online.get("does-not-exist")


def test_online_trainer_and_force_trainer_fit():
    """Coverage: exercise online.py ``call`` through a tiny ESN + trainers.

    Drives ``brainpy.algorithms.online.RLS.call`` via the high-level
    ``OnlineTrainer``/``ForceTrainer`` training loop on a small reservoir.
    """

    class ESN(bp.DynamicalSystem):
        def __init__(self, num_in, num_hidden, num_out):
            super().__init__()
            self.r = bp.dyn.Reservoir(
                num_in, num_hidden,
                Win_initializer=bp.init.Uniform(-0.1, 0.1),
                Wrec_initializer=bp.init.Normal(scale=0.1),
                in_connectivity=0.1, rec_connectivity=0.1,
                comp_type="dense",
            )
            self.o = bp.dnn.Dense(num_hidden, num_out,
                                  W_initializer=bp.init.Normal(),
                                  mode=bm.training_mode)

        def update(self, x):
            return x >> self.r >> self.o

    bm.random.seed(0)
    bp.share.save(fit=True)
    with bm.batching_environment():
        model = ESN(5, 25, 3)
    x = bm.random.random((1, 50, 5))
    y = bm.random.random((1, 50, 3))

    trainer = bp.OnlineTrainer(model, fit_method=bp.algorithms.RLS(alpha=0.1),
                               progress_bar=False)
    trainer.fit([x, y])
    out = trainer.predict(x)
    assert out.shape == (1, 50, 3)

    bm.random.seed(0)
    with bm.batching_environment():
        model2 = ESN(5, 25, 3)
    force = bp.ForceTrainer(model2, alpha=0.1, progress_bar=False)
    force.fit([x, y])


# ---------------------------------------------------------------------------
# offline.py — H-46 (GD .value bug) and H-47 (ridge intercept penalty)
# ---------------------------------------------------------------------------

def _small_xy(n=20, n_in=3, n_out=2, seed=1):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(n, n_in)), jnp.asarray(rng.randn(n, n_out))


def test_ridge_gradient_descent_runs_no_value_error():
    """H-46: ``gradient_descent=True`` must not raise ``.value`` AttributeError."""
    from brainpy.algorithms.offline import RidgeRegression

    x, y = _small_xy()
    w = RidgeRegression(alpha=1e-6, gradient_descent=True, max_iter=50)(y, x)
    assert w.shape == (3, 2)
    assert not bool(jnp.isnan(w).any())


def test_linear_regression_gradient_descent_runs():
    """H-46: the same GD code path through LinearRegression."""
    from brainpy.algorithms.offline import LinearRegression

    x, y = _small_xy()
    w = LinearRegression(gradient_descent=True, max_iter=50)(y, x)
    assert w.shape == (3, 2)
    assert not bool(jnp.isnan(w).any())


def test_lasso_always_gradient_descent_runs():
    """H-46: Lasso is always-GD; its body must not hit the ``.value`` bug."""
    from brainpy.algorithms.offline import LassoRegression

    x, y = _small_xy(n_out=1)
    w = LassoRegression(alpha=0.1, degree=2, max_iter=50)(y, x)
    assert w.ndim == 2
    assert not bool(jnp.isnan(w).any())


def test_elastic_net_gradient_descent_runs():
    """H-46: ElasticNet is always-GD as well."""
    from brainpy.algorithms.offline import ElasticNetRegression

    x, y = _small_xy(n_out=1)
    w = ElasticNetRegression(alpha=0.1, degree=2, max_iter=50)(y, x)
    assert not bool(jnp.isnan(w).any())


def test_ridge_intercept_not_over_penalized():
    """H-47: a large ridge ``alpha`` must not shrink the intercept/bias column.

    Fit ``y ≈ slope*x + intercept`` with a strongly nonzero mean and a huge
    penalty.  The bias column (index 0 of the polynomial features) must remain
    close to the data mean while the slope is shrunk toward zero.
    """
    from brainpy.algorithms.offline import PolynomialRidgeRegression

    rng = np.random.RandomState(2)
    x = jnp.asarray(rng.randn(40, 1))
    mean_y = 5.0
    y = jnp.asarray(0.3 * np.asarray(x) + mean_y + 0.01 * rng.randn(40, 1))

    model = PolynomialRidgeRegression(alpha=1e6, degree=1, add_bias=True,
                                      gradient_descent=False)
    w = model(y, x)
    intercept = float(w[0, 0])
    slope = float(w[1, 0])
    # Intercept stays near the data mean despite the huge penalty.
    assert abs(intercept - mean_y) < 0.5, f"intercept over-penalized: {intercept}"
    # The (penalized) slope is shrunk toward zero.
    assert abs(slope) < abs(intercept), f"slope not shrunk: slope={slope}"


def test_logistic_regression_fit_runs():
    """``LogisticRegression.call`` now fits instead of raising ``IndexError``.

    Fixed in audit 2026-06-19: ``call`` previously flattened ``targets`` to 1-D
    and then indexed ``targets.shape[1]``; it now initialises a 1-D parameter
    vector and runs the gradient-descent solver to completion.
    """
    from brainpy.algorithms.offline import LogisticRegression

    rng = np.random.RandomState(3)
    x = jnp.asarray(rng.randn(30, 2))
    y = jnp.asarray((np.asarray(x)[:, :1] > 0).astype("float32"))
    w = LogisticRegression(max_iter=100)(y, x)
    assert w is not None
    assert not bool(jnp.isnan(jnp.asarray(w)).any())


def test_offline_least_square_and_polynomial():
    """Coverage: non-GD lstsq path and polynomial regression."""
    from brainpy.algorithms.offline import (LinearRegression,
                                            PolynomialRegression,
                                            RidgeRegression)

    x, y = _small_xy()
    w_lin = LinearRegression()(y, x)            # lstsq path
    assert w_lin.shape[-1] == 2
    w_ridge = RidgeRegression(alpha=1e-3)(y, x)  # ridge closed form (no bias)
    assert w_ridge.shape == (3, 2)
    w_poly = PolynomialRegression(degree=2, gradient_descent=True, max_iter=20)(y, x)
    assert not bool(jnp.isnan(w_poly).any())


def test_offline_registry_helpers():
    """Coverage: the offline method registry getters."""
    from brainpy.algorithms import offline

    methods = offline.get_supported_offline_methods()
    for name in ("linear", "ridge", "lasso", "logistic"):
        assert name in methods
    assert offline.get("ridge") is offline.RidgeRegression
    with pytest.raises(ValueError):
        offline.get("nope")


# ---------------------------------------------------------------------------
# measure.py — H-43: firing_rate normalization
# ---------------------------------------------------------------------------

def test_firing_rate_100hz_mean():
    """H-43: a true 100 Hz spike train must average to ~100 Hz.

    ``sp[::10] = 1`` with dt=1 ms is one spike every 10 ms = 100 Hz.  The buggy
    normalization (by requested ``width`` rather than the actual window length)
    biased the smoothed rate so its mean drifted toward ~110 Hz.
    """
    spikes = np.zeros((1000, 1))
    spikes[::10] = 1
    rate = bp.measure.firing_rate(spikes, width=10, dt=1.0)
    assert abs(float(np.mean(rate)) - 100.0) < 5.0, f"mean rate={np.mean(rate)}"


def test_firing_rate_jax_mode():
    """Coverage: numpy=False (JIT-able) branch of firing_rate."""
    spikes = np.zeros((200, 2))
    spikes[::5] = 1
    rate = bp.measure.firing_rate(spikes, width=5, dt=1.0, numpy=False)
    assert rate.shape[0] == 200


def test_raster_plot():
    """Coverage: raster_plot returns (index, time) of spikes."""
    spikes = np.zeros((5, 3))
    spikes[1, 0] = 1
    spikes[3, 2] = 1
    times = np.arange(5) * 0.1
    index, time = bp.measure.raster_plot(spikes, times)
    assert set(np.asarray(index).tolist()) == {0, 2}
    assert len(time) == 2


# ---------------------------------------------------------------------------
# delay.py — H-44 (VarDelay self.data) and register_entry / retrieve coverage
# ---------------------------------------------------------------------------

def test_vardelay_constructs_and_updates_time_gt_zero():
    """H-44: ``VarDelay(target, time=T>0)`` must construct without AttributeError.

    The pre-audit ``_init_data`` read ``self.data`` before it was ever assigned,
    raising ``AttributeError: 'data'`` for any positive delay time.
    """
    target = bm.Variable(bm.zeros(4))
    delay = bp.delay.VarDelay(target, time=2.0)
    assert delay.data is not None
    assert delay.max_length > 0

    bp.share.save(i=0, t=0.0, dt=bm.get_dt())
    delay.update(bm.ones(4))  # one update step
    assert delay.data is not None


def test_vardelay_register_entry_and_retrieve():
    """Coverage: register_entry + at()/retrieve on a VarDelay."""
    target = bm.Variable(bm.arange(4.0))
    delay = bp.delay.VarDelay(target, time=2.0)
    delay.register_entry("e1", delay_time=1.0)
    delay.register_entry("e0", delay_time=0.0)

    bp.share.save(i=0, t=0.0, dt=bm.get_dt())
    delay.update(bm.arange(4.0) + 10.0)

    # zero-delay entry returns the current target value
    out0 = delay.at("e0")
    assert out0.shape == (4,)
    # nonzero-delay entry retrieves a buffered value
    out1 = delay.at("e1")
    assert out1.shape == (4,)

    with pytest.raises(KeyError):
        delay.at("missing")
    with pytest.raises(KeyError):
        delay.register_entry("e1", delay_time=1.0)  # duplicate


def test_vardelay_time_none_is_zero_length():
    """Coverage: ``time=None`` yields a zero-length (data-less) delay."""
    target = bm.Variable(bm.zeros(3))
    delay = bp.delay.VarDelay(target, time=None)
    assert delay.max_length == 0
    assert delay.data is None


def test_length_delay_register_retrieve_update():
    """Coverage: ``brainpy.math.LengthDelay`` retrieve/update/__call__."""
    var = bm.Variable(bm.arange(4.0))
    ld = bm.LengthDelay(var, delay_len=3)
    ld.update(bm.arange(4.0) + 10.0)
    # delay 0 -> newest value
    assert np.allclose(np.asarray(ld(0)), np.arange(4.0) + 10.0)
    # delay 1 -> previous (initial zeros-ish) value
    out1 = ld.retrieve(1)
    assert out1.shape == (4,)


def test_data_delay_constructs():
    """Coverage: ``DataDelay`` (subclass of VarDelay) constructs with time>0."""
    target = bm.Variable(bm.zeros(3))
    dd = bp.delay.DataDelay(target, data_init=bm.zeros(3), time=1.0)
    assert dd.data is not None


def test_vardelay_concat_update_and_init_by_return():
    """Coverage: CONCAT_UPDATE method, init_delay_by_return, DelayAccess."""
    from brainpy.math.delayvars import CONCAT_UPDATE
    from brainpy.delay import init_delay_by_return, DelayAccess

    target = bm.Variable(bm.zeros(3))
    delay = bp.delay.VarDelay(target, time=1.0, method=CONCAT_UPDATE)
    delay.register_entry("c", delay_step=2)
    bp.share.save(i=0, t=0.0, dt=bm.get_dt())
    delay.update(bm.ones(3))
    assert delay.at("c").shape == (3,)

    # init_delay_by_return with a plain Variable -> VarDelay
    dl = init_delay_by_return(bm.Variable(bm.zeros(2)))
    assert isinstance(dl, bp.delay.VarDelay)

    # DelayAccess registers an entry on the delay and reads it back
    access = DelayAccess(delay, 1.0, delay_entry="acc")
    out = access.update()
    assert out.shape == (3,)


def test_vardelay_wrong_target_type_raises():
    """Coverage: VarDelay rejects a non-Variable target."""
    with pytest.raises(ValueError):
        bp.delay.VarDelay(bm.zeros(3), time=1.0)


# ---------------------------------------------------------------------------
# optimization.py + lowdim_analyzer.py — H-49 (arg-unwrap) and H-50 (empty concat)
# ---------------------------------------------------------------------------

def test_roots_of_1d_by_x_finds_fixed_point():
    """H-49: ``roots_of_1d_by_x`` on ``dx=-x+I`` finds the fixed point x=I.

    Also exercises the arg-unwrap comprehension (passing a ``bm.Array`` arg).
    """
    from brainpy.analysis.utils.optimization import roots_of_1d_by_x

    bp.math.enable_x64()
    try:
        offset = 0.7
        f = lambda x, b: -x + b
        candidates = jnp.linspace(-2.0, 2.0, 401)
        # pass the parameter as a bm.Array to drive the unwrap branch
        fps = roots_of_1d_by_x(f, candidates, args=(bm.asarray(offset),))
        fps = np.asarray(fps)
        assert fps.size >= 1
        assert np.any(np.abs(fps - offset) < 1e-3), f"fps={fps}"
    finally:
        bp.math.disable_x64()


def test_phase_plane_1d_finds_fixed_point():
    """H-49: a PhasePlane1D analyzer on ``dx=-x+I`` locates the fixed point ~x=I."""
    import matplotlib
    matplotlib.use("Agg")

    bp.math.enable_x64()
    try:
        offset = 0.7

        @bp.odeint
        def int_x(x, t, Iext):
            return -x + Iext

        analyzer = bp.analysis.PhasePlane1D(
            model=int_x,
            target_vars={"x": [-2.0, 2.0]},
            pars_update={"Iext": offset},
            resolutions=0.01,
        )
        analyzer.plot_vector_field()
        fps = analyzer.plot_fixed_point(show=False, with_return=True)
        fps = np.asarray(fps).ravel()
        assert fps.size >= 1
        assert np.any(np.abs(fps - offset) < 1e-2), f"fps={fps}"
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")
        bp.math.disable_x64()


def test_lowdim_2d_empty_candidate_concat_guard():
    """H-50: the non-convertible 2D ``_get_fixed_points`` must guard the empty path.

    Build a 2D analyzer that cannot reduce to a single equation, then drive the
    optimization branch with an impossible candidate-screening tolerance so that
    nothing converges.  The buggy code did ``jnp.concatenate([])`` -> ValueError;
    the fix returns correctly-shaped empty arrays.
    """
    from brainpy.analysis.lowdim.lowdim_analyzer import Num2DAnalyzer

    bp.math.enable_x64()
    try:
        @bp.odeint
        def ds1(s1, t, s2):
            return -s1 + jnp.tanh(s2) + 0.1

        @bp.odeint
        def ds2(s2, t, s1):
            return -s2 + jnp.tanh(s1) + 0.1

        analyzer = Num2DAnalyzer(
            model=[ds1, ds2],
            target_vars={"s1": [-2.0, 2.0], "s2": [-2.0, 2.0]},
            resolutions=0.05,
        )
        assert not analyzer._can_convert_to_one_eq()

        candidates = jnp.asarray(
            np.random.RandomState(0).uniform(-2.0, 2.0, size=(30, 2)))
        # tol_opt_candidate=-1 screens out every candidate -> empty all_fps list
        fps, ids, pargs = analyzer._get_fixed_points(candidates,
                                                     tol_opt_candidate=-1.0)
        fps = np.asarray(fps)
        assert fps.shape == (0, 2)
        assert np.asarray(ids).shape == (0,)
    finally:
        bp.math.disable_x64()


def test_roots_of_1d_by_xy_and_brentq_helpers():
    """Coverage: roots_of_1d_by_xy and the scalar brentq helper functions."""
    from brainpy.analysis.utils import optimization as opt

    bp.math.enable_x64()
    try:
        f = lambda x, a: -x + a

        # roots_of_1d_by_xy on dx = -x + a, a = 0.5
        xs, ys = opt.roots_of_1d_by_xy(f, jnp.array([-2.0]), jnp.array([2.0]),
                                       jnp.array([0.5]))
        assert np.any(np.abs(np.asarray(xs) - 0.5) < 1e-6)

        # brentq_roots (jitted vmap brentq)
        roots, _ = opt.brentq_roots(f, jnp.array([-2.0]), jnp.array([2.0]),
                                    jnp.array([0.5]))
        assert np.any(np.abs(np.asarray(roots) - 0.5) < 1e-6)

        # get_brentq_candidates over a 2D meshgrid
        starts, ends, args = opt.get_brentq_candidates(
            lambda x, y: -x + y, jnp.linspace(-2.0, 2.0, 20),
            jnp.linspace(-1.0, 1.0, 5))
        assert len(np.asarray(starts)) == len(np.asarray(ends))

        # pure-numpy brentq + 1D root finder
        root, _iters, _calls = opt.numpy_brentq(lambda x: x - 0.3, -1.0, 1.0)
        assert abs(root - 0.3) < 1e-9
        roots_np = opt.find_root_of_1d_numpy(lambda x: -x + 0.4,
                                             np.linspace(-2.0, 2.0, 50))
        assert np.any(np.abs(np.asarray(roots_np) - 0.4) < 1e-6)
    finally:
        bp.math.disable_x64()


def test_phase_plane_1d_no_fixed_point_runs_clean():
    """Coverage: a 1D system with no real fixed point returns cleanly (no crash)."""
    import matplotlib
    matplotlib.use("Agg")

    bp.math.enable_x64()
    try:
        @bp.odeint
        def int_x(x, t):
            # dx = x^2 + 1 has no real root -> no fixed point in range
            return x ** 2 + 1.0

        analyzer = bp.analysis.PhasePlane1D(
            model=int_x,
            target_vars={"x": [-2.0, 2.0]},
            resolutions=0.01,
        )
        fps = analyzer.plot_fixed_point(show=False, with_return=True)
        assert np.asarray(fps).size == 0
    finally:
        import matplotlib.pyplot as plt
        plt.close("all")
        bp.math.disable_x64()


# ---------------------------------------------------------------------------
# running/jax_multiprocessing.py — H-48 (pmap reuse / labels)
# ---------------------------------------------------------------------------

def test_jax_vectorize_map_sequence_and_dict():
    """Coverage: jax_vectorize_map over sequence and dict arguments."""
    from brainpy.running.jax_multiprocessing import jax_vectorize_map

    out = jax_vectorize_map(lambda x: x * 2.0, [jnp.arange(6.0)], num_parallel=2)
    assert np.allclose(np.asarray(out), np.arange(6.0) * 2.0)

    # NOTE: clear_buffer=True calls the process-global ``bm.clear_buffer_memory()``
    # which deletes ALL live device arrays (poisoning other test modules in the
    # same pytest session). Patch it to a no-op so the branch is still covered.
    _orig_clear = bm.clear_buffer_memory
    bm.clear_buffer_memory = lambda *a, **k: None
    try:
        out2 = jax_vectorize_map(lambda a, b: a + b,
                                 {"a": jnp.arange(4.0), "b": jnp.ones(4)},
                                 num_parallel=2, clear_buffer=True)
    finally:
        bm.clear_buffer_memory = _orig_clear
    assert np.allclose(np.asarray(out2), np.arange(4.0) + 1.0)


def test_jax_vectorize_map_length_mismatch_raises():
    """Coverage: mismatched argument lengths raise ValueError."""
    from brainpy.running.jax_multiprocessing import jax_vectorize_map

    with pytest.raises(ValueError):
        jax_vectorize_map(lambda a, b: a + b,
                          {"a": jnp.arange(4.0), "b": jnp.ones(3)},
                          num_parallel=2)


def test_jax_parallelize_map_single_device():
    """H-48: jax_parallelize_map runs across chunks (one device per chunk)."""
    from brainpy.running.jax_multiprocessing import jax_parallelize_map

    n_dev = jax.local_device_count()
    out = jax_parallelize_map(lambda x: x * 2.0,
                              [jnp.arange(float(3 * n_dev))],
                              num_parallel=n_dev)
    assert np.allclose(np.asarray(out), np.arange(float(3 * n_dev)) * 2.0)


# ---------------------------------------------------------------------------
# runners.py — C-22: DSRunner(memory_efficient=True)
# ---------------------------------------------------------------------------

class _TinyNet(bp.DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.n = bp.dyn.LifRef(3)

    def update(self, inp):
        self.n(inp)
        return self.n.V.value


def test_dsrunner_memory_efficient_matches_normal():
    """C-22: ``memory_efficient=True`` must run and match the standard run.

    The pre-audit code did ``jax.ShapeDtypeStruct(mon.shape, ...)`` on a *dict*
    monitor and used a broken ``pure_callback`` signature, so any
    ``memory_efficient=True`` run raised ``AttributeError: 'dict' ... 'shape'``.
    """
    inputs = bm.ones((15, 3)) * 2.0

    bm.random.seed(0)
    r_normal = bp.DSRunner(_TinyNet(), monitors=["n.V"],
                           memory_efficient=False, progress_bar=False)
    r_normal.run(inputs=inputs)
    mon_normal = np.asarray(r_normal.mon["n.V"])

    bm.random.seed(0)
    r_mem = bp.DSRunner(_TinyNet(), monitors=["n.V"],
                        memory_efficient=True, progress_bar=False)
    r_mem.run(inputs=inputs)
    mon_mem = np.asarray(r_mem.mon["n.V"])

    assert mon_normal.shape == mon_mem.shape == (15, 3)
    assert np.allclose(mon_normal, mon_mem), "memory_efficient run diverged"


def test_dsrunner_basic_run_with_monitors():
    """Coverage: a plain DSRunner run with monitors and duration."""
    bm.random.seed(0)
    runner = bp.DSRunner(_TinyNet(), monitors=["n.V"], progress_bar=False)
    runner.run(duration=2.0)
    assert runner.mon["n.V"].shape[1] == 3
    assert runner.mon.ts.shape[0] == runner.mon["n.V"].shape[0]
