# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Coverage tests for :mod:`brainpy.analysis.highdim.slow_points`.

Complements ``slow_points_test.py`` (which only exercises the two
``find_fps_*`` entry points) by driving the full post-processing pipeline and
the many guard/error branches of :class:`SlowPointFinder`:

- constructor argument validation (args / target_vars / excluded_vars / f_type
  / f_cell type, and the unsupported callable-only kwargs);
- the read-only property setters;
- ``find_fps_with_gd_method`` (custom optimizer error, ``args`` + custom loss),
  ``find_fps_with_opt_solver`` (and the ndim guard);
- ``filter_loss`` / ``keep_unique`` / ``exclude_outliers`` (incl. early returns);
- ``compute_jacobians`` for 1D/2D points, dict points, ``plot=True`` and the
  multi-dim / 3D error guards;
- ``decompose_eigenvalues`` for both ``sort_by`` values and the error branch;
- the DynamicalSystem flow incl. ``_check_candidates`` key/shape errors.

A tiny linear system ``dx = -x`` (single fixed point at the origin) is used so
optimization converges in a handful of steps.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy import losses
from brainpy._errors import AnalyzerError, UnsupportedError

CONTINUOUS = bp.analysis.CONTINUOUS


def _linear_step(x):
    return -x


@pytest.fixture(autouse=True)
def _close_figs():
    try:
        yield
    finally:
        plt.close('all')


# --------------------------------------------------------------------------- #
# constructor validation
# --------------------------------------------------------------------------- #
def test_init_args_must_be_tuple():
    with pytest.raises(ValueError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, args=[1])


def test_init_target_vars_must_be_dict():
    with pytest.raises(TypeError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, target_vars=[1])


def test_init_excluded_vars_must_be_sequence():
    with pytest.raises(TypeError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, excluded_vars=5)


def test_init_excluded_vars_elements_must_be_variable():
    with pytest.raises(TypeError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, excluded_vars=[1])


def test_init_unknown_f_type():
    with pytest.raises(AnalyzerError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type='weird')


def test_init_non_callable_cell():
    with pytest.raises(ValueError):
        bp.analysis.SlowPointFinder(f_cell=123, f_type=CONTINUOUS)


@pytest.mark.parametrize('kwargs', [
    dict(inputs=[('x', 1)]),
    dict(t=1.),
    dict(dt=1.),
    dict(target_vars={'x': bm.Variable(bm.zeros(2))}),
])
def test_init_callable_only_kwargs_unsupported(kwargs):
    with pytest.raises(UnsupportedError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, **kwargs)


def test_init_excluded_vars_for_callable_unsupported():
    with pytest.raises(UnsupportedError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS,
                                    excluded_vars=[bm.Variable(bm.zeros(2))])


@pytest.mark.parametrize('kwargs', [
    dict(f_loss_batch=lambda *a: a),
    dict(fun_inputs=lambda *a: a),
])
def test_init_deprecated_kwargs(kwargs):
    with pytest.raises(UnsupportedError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, **kwargs)


def test_init_target_and_excluded_simultaneously():
    v = bm.Variable(bm.zeros(2))
    with pytest.raises(ValueError):
        bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS,
                                    target_vars={'x': v}, excluded_vars=[v])


# --------------------------------------------------------------------------- #
# read-only properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('attr', ['opt_losses', 'fixed_points', 'losses', 'selected_ids'])
def test_property_setters_raise(attr):
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=False)
    with pytest.raises(UnsupportedError):
        setattr(f, attr, 1)


# --------------------------------------------------------------------------- #
# gd method: optimizer error + args + custom loss
# --------------------------------------------------------------------------- #
def test_gd_bad_optimizer_type():
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=False)
    with pytest.raises(ValueError):
        f.find_fps_with_gd_method(np.random.rand(5, 2), optimizer='nope')


def test_gd_with_args_and_custom_loss_verbose():
    def step(x, scale):
        return -x * scale

    f = bp.analysis.SlowPointFinder(f_cell=step, f_type=CONTINUOUS, args=(2.0,),
                                    f_loss=losses.mean_square, verbose=True)
    rng = bm.random.RandomState(1)
    f.find_fps_with_gd_method(rng.random((5, 2)) * 0.3, num_opt=200, num_batch=50)
    assert f.num_fps == 5
    assert f.opt_losses.ndim == 1


# --------------------------------------------------------------------------- #
# full pipeline for a callable system
# --------------------------------------------------------------------------- #
def test_full_pipeline_callable():
    rng = bm.random.RandomState(42)
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=True)
    f.find_fps_with_gd_method(rng.random((20, 2)) * 0.5, num_opt=300, num_batch=100)
    assert f.num_fps == 20

    f.filter_loss(1e-2)
    f.keep_unique(tolerance=0.1)   # collapses near-identical points toward origin
    assert f.num_fps >= 1
    f.exclude_outliers(1e0)

    # properties are all materialised
    assert f.losses.ndim == 1
    assert f.selected_ids.ndim == 1

    # jacobians + decomposition
    J = f.compute_jacobians(f.fixed_points)
    J = np.asarray(J)
    assert J.shape[1:] == (2, 2)
    decomp = bp.analysis.SlowPointFinder.decompose_eigenvalues(
        J, sort_by='real', do_compute_lefts=True)
    assert decomp[0]['eig_values'].shape == (2,)
    assert decomp[0]['L'] is not None


def test_exclude_outliers_removes_with_multiple_fps():
    # dx = x - x^3 has fixed points at -1, 0, +1.  Seeding near -1 and +1 gives
    # two distinct surviving fixed points, so exclude_outliers runs its removal
    # body (rather than the num_fps<=1 early return).
    def step(x):
        return x - x ** 3

    rng = bm.random.RandomState(11)
    f = bp.analysis.SlowPointFinder(f_cell=step, f_type=CONTINUOUS, verbose=True)
    cands = np.concatenate([rng.normal(-1, 0.05, (8, 1)),
                            rng.normal(1, 0.05, (8, 1))], axis=0)
    f.find_fps_with_gd_method(cands, num_opt=400, num_batch=100)
    f.filter_loss(1e-3)
    f.keep_unique(0.1)
    assert f.num_fps == 2
    f.exclude_outliers(5.0)  # both neighbours within tol -> both kept
    assert f.num_fps == 2


def test_exclude_outliers_early_returns():
    rng = bm.random.RandomState(7)
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=False)
    f.find_fps_with_gd_method(rng.random((5, 2)) * 0.1, num_opt=200, num_batch=50)
    # inf tolerance -> immediate return
    n_before = f.num_fps
    f.exclude_outliers(np.inf)
    assert f.num_fps == n_before
    # collapse to a single fixed point, then exclude_outliers returns (num_fps<=1)
    f.keep_unique(tolerance=1.0)
    f.exclude_outliers(1e0)
    assert f.num_fps == 1


# --------------------------------------------------------------------------- #
# opt solver
# --------------------------------------------------------------------------- #
def test_opt_solver_and_jacobian_plot():
    rng = bm.random.RandomState(0)
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=True)
    f.find_fps_with_opt_solver(rng.random((10, 2)) * 0.3)
    assert f.num_fps >= 1

    # jacobians with plotting for a handful of explicit 2D points
    pts = np.array([[0., 0.], [0.01, 0.0], [0., 0.01], [0.02, 0.0], [0.0, 0.02]])
    J = f.compute_jacobians(pts, plot=True, num_col=2)
    assert np.asarray(J).shape == (5, 2, 2)

    # magnitude sort (default) path
    decomp = bp.analysis.SlowPointFinder.decompose_eigenvalues(J, sort_by='magnitude')
    assert len(decomp) == 5


def test_compute_jacobians_1d_point():
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=False)
    # a single 1D point (num_feature,) -> num_point == 1
    J = f.compute_jacobians(np.array([0., 0.]))
    assert np.asarray(J).shape == (1, 2, 2)


def test_compute_jacobians_3d_points_error():
    f = bp.analysis.SlowPointFinder(f_cell=_linear_step, f_type=CONTINUOUS, verbose=False)
    with pytest.raises(ValueError):
        f.compute_jacobians(np.zeros((2, 2, 2)))


def test_decompose_eigenvalues_bad_sort():
    J = np.stack([np.eye(2)])
    with pytest.raises(ValueError):
        bp.analysis.SlowPointFinder.decompose_eigenvalues(J, sort_by='nope')


# --------------------------------------------------------------------------- #
# DynamicalSystem flow
# --------------------------------------------------------------------------- #
def _linear_ds():
    class Lin(bp.dyn.NeuDyn):
        def __init__(self):
            super().__init__(size=1)
            self.x = bm.Variable(bm.zeros(1))
            self.intg = bp.odeint(lambda x, t: -x)

        def update(self):
            self.x.value = self.intg(self.x, bp.share['t'])

    return Lin()


def test_ds_check_candidates_errors():
    f = bp.analysis.SlowPointFinder(f_cell=_linear_ds(), verbose=False)
    # candidates must be a dict for a DynamicalSystem
    with pytest.raises(ValueError):
        f.find_fps_with_gd_method(np.zeros((5, 1)))
    # unknown variable key
    with pytest.raises(KeyError):
        f.find_fps_with_gd_method({'wrong': np.zeros((5, 1))})


def test_ds_full_pipeline_and_jacobian():
    f = bp.analysis.SlowPointFinder(f_cell=_linear_ds(), verbose=False)
    rng = bm.random.RandomState(3)
    f.find_fps_with_gd_method({'x': rng.random((6, 1)) * 0.4}, num_opt=300, num_batch=100)
    assert f.num_fps == 6
    f.filter_loss(1e2)
    f.keep_unique(tolerance=0.1)
    f.exclude_outliers(1e0)
    J = f.compute_jacobians({'x': f.fixed_points['x']})
    assert np.asarray(J).shape[1:] == (1, 1)


def test_ds_target_vars_subset_makes_rest_excluded():
    # Providing only a subset of variables as ``target_vars`` moves the rest
    # into ``excluded_vars`` automatically.
    class Two(bp.dyn.NeuDyn):
        def __init__(self):
            super().__init__(size=1)
            self.x = bm.Variable(bm.zeros(1))
            self.y = bm.Variable(bm.zeros(1))
            self.ix = bp.odeint(lambda x, t: -x)
            self.iy = bp.odeint(lambda y, t: -y)

        def update(self):
            t = bp.share['t']
            self.x.value = self.ix(self.x, t)
            self.y.value = self.iy(self.y, t)

    m = Two()
    f = bp.analysis.SlowPointFinder(f_cell=m, target_vars={'x': m.x}, verbose=False)
    assert list(f.target_vars.keys()) == ['x']
    assert 'y' in f.excluded_vars
    rng = bm.random.RandomState(2)
    f.find_fps_with_gd_method({'x': rng.random((4, 1))}, num_opt=150, num_batch=50)
    assert f.num_fps == 4


def _input_ds():
    class Inp(bp.dyn.NeuDyn):
        def __init__(self):
            super().__init__(size=1)
            self.x = bm.Variable(bm.zeros(1))
            self.inp = bm.Variable(bm.zeros(1))
            self.intg = bp.odeint(lambda x, t, I: -x + I)

        def update(self):
            self.x.value = self.intg(self.x, bp.share['t'], self.inp)

    return Inp()


def test_ds_with_fixed_inputs():
    # exercises the ``_step_func_input`` fixed-operation branch
    m = _input_ds()
    f = bp.analysis.SlowPointFinder(f_cell=m, excluded_vars=[m.inp],
                                    inputs=(m.inp, 0.5, 'fix', '='), verbose=False)
    rng = bm.random.RandomState(1)
    f.find_fps_with_gd_method({'x': rng.random((4, 1))}, num_opt=150, num_batch=50)
    assert f.num_fps == 4


def test_ds_with_callable_inputs():
    # exercises the callable-``inputs`` branch of ``_step_func_input``
    m = _input_ds()

    def inp_fn():
        m.inp.value = m.inp.value + 0.0

    f = bp.analysis.SlowPointFinder(f_cell=m, excluded_vars=[m.inp],
                                    inputs=inp_fn, verbose=False)
    rng = bm.random.RandomState(1)
    f.find_fps_with_gd_method({'x': rng.random((4, 1))}, num_opt=150, num_batch=50)
    assert f.num_fps == 4


def test_ds_opt_solver_ndim_guard():
    # When a target variable has ndim != 1, the opt solver must refuse
    # ("Cannot use opt solver.").
    class TwoD(bp.dyn.NeuDyn):
        def __init__(self):
            super().__init__(size=1)
            self.x = bm.Variable(bm.zeros((2, 2)))  # 2D target variable
            self.intg = bp.odeint(lambda x, t: -x)

        def update(self):
            self.x.value = self.intg(self.x, bp.share['t'])

    f = bp.analysis.SlowPointFinder(f_cell=TwoD(), verbose=False)
    rng = bm.random.RandomState(5)
    with pytest.raises(ValueError):
        f.find_fps_with_opt_solver({k: rng.random((4, 2, 2)) for k in f.target_vars})
