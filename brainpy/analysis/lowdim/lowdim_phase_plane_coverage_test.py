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
"""Coverage tests for :mod:`brainpy.analysis.lowdim.lowdim_phase_plane`.

Complements the existing ``phase_plane_test.py`` by covering the un-exercised
branches of ``PhasePlane1D`` / ``PhasePlane2D``:

- the ``target_pars`` rejection guards in both constructors;
- ``plot_vector_field`` with ``plot_method='quiver'`` and the unknown-method
  error;
- ``plot_fixed_point`` with ``select_candidates`` variants and the
  "no nullcline points" / unknown-selector errors;
- ``plot_trajectory`` for both ``axes='v-v'`` / ``axes='t-v'`` and the
  unknown-axes error;
- ``plot_limit_cycle_by_sim`` for both the cycle-found and no-cycle paths.

Resolutions are coarse; matplotlib runs headless via conftest.
"""

import matplotlib.pyplot as plt
import pytest

import brainpy as bp
import brainpy.math as bm


@pytest.fixture(autouse=True)
def _x64_and_close():
    bp.math.enable_x64()
    try:
        yield
    finally:
        plt.close('all')
        bp.math.disable_x64()


def _fhn_2d(Iext=0.5):
    @bp.odeint
    def dV(V, t, w, I=Iext):
        return V - V ** 3 / 3 - w + I

    @bp.odeint
    def dw(w, t, V, a=0.7, b=0.8):
        return (V + a - b * w) / 12.5

    return [dV, dw]


# --------------------------------------------------------------------------- #
# constructor guards
# --------------------------------------------------------------------------- #
def test_pp1d_rejects_target_pars():
    @bp.odeint
    def int_x(x, t, Iext):
        return x ** 3 - x + Iext

    with pytest.raises(bp.errors.AnalyzerError):
        bp.analysis.PhasePlane1D(model=int_x,
                                 target_vars={'x': [-2, 2]},
                                 target_pars={'Iext': [0., 1.]},
                                 resolutions=0.1)


def test_pp2d_rejects_target_pars():
    with pytest.raises(bp.errors.AnalyzerError):
        bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                 target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                 target_pars={'I': [0., 1.]},
                                 resolutions=0.5)


# --------------------------------------------------------------------------- #
# PhasePlane1D vector field + fixed point with return
# --------------------------------------------------------------------------- #
def test_pp1d_vector_field_and_fp_with_return():
    @bp.odeint
    def int_x(x, t, Iext=0.):
        return x ** 3 - x + Iext

    pp = bp.analysis.PhasePlane1D(model=int_x,
                                  target_vars={'x': [-2, 2]},
                                  pars_update={'Iext': 0.},
                                  resolutions=0.05)
    yval = pp.plot_vector_field(with_return=True)
    assert yval is not None
    fps = pp.plot_fixed_point(with_return=True)
    assert len(fps) > 0


# --------------------------------------------------------------------------- #
# PhasePlane2D vector field
# --------------------------------------------------------------------------- #
def test_pp2d_quiver_and_return():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    dx, dy = pp.plot_vector_field(plot_method='quiver', with_return=True)
    assert dx.shape == dy.shape


# NOTE (defect): passing ``linewidth`` inside ``plot_style`` to the
# ``streamplot`` branch crashes.  The code reads ``plot_style.get('linewidth')``
# but never pops it, so ``pyplot.streamplot(..., linewidth=linewidth,
# **plot_style)`` passes ``linewidth`` twice -> ``TypeError: got multiple values
# for keyword argument 'linewidth'`` (lowdim_phase_plane.py:235).
def test_pp2d_streamplot_custom_linewidth_is_buggy():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    with pytest.raises(TypeError):
        pp.plot_vector_field(plot_method='streamplot', plot_style=dict(linewidth=1.0))


def test_pp2d_unknown_plot_method_raises():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.5)
    with pytest.raises(bp.errors.AnalyzerError):
        pp.plot_vector_field(plot_method='nope')


# --------------------------------------------------------------------------- #
# PhasePlane2D fixed-point selectors / errors
# --------------------------------------------------------------------------- #
def test_pp2d_fixed_point_without_nullcline_raises():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    # fx-nullcline selector requires plot_nullcline() to have been called first
    with pytest.raises(bp.errors.AnalyzerError):
        pp.plot_fixed_point(select_candidates='fx-nullcline')


def test_pp2d_fixed_point_unknown_selector_raises():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    with pytest.raises(ValueError):
        pp.plot_fixed_point(select_candidates='unknown')


def test_pp2d_fixed_point_aux_rank():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    fps = pp.plot_fixed_point(select_candidates='aux_rank', num_rank=50,
                              with_return=True)
    assert fps is not None


def test_pp2d_fixed_point_after_nullcline():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.1)
    pp.plot_nullcline()
    fps = pp.plot_fixed_point(select_candidates='fx-nullcline', with_return=True)
    assert fps is not None


# --------------------------------------------------------------------------- #
# PhasePlane2D trajectory + limit cycle
# --------------------------------------------------------------------------- #
def test_pp2d_trajectory_vv_and_tv():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    pp.plot_trajectory(initials={'V': [-2.8], 'w': [-0.5]}, duration=30., axes='v-v')
    res = pp.plot_trajectory(initials={'V': [-2.8], 'w': [-0.5]}, duration=30.,
                             axes='t-v', with_return=True)
    assert res is not None


def test_pp2d_trajectory_unknown_axes_raises():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.5)
    with pytest.raises(bp.errors.AnalyzerError):
        pp.plot_trajectory(initials={'V': [0.], 'w': [0.]}, duration=5., axes='bad')


def test_pp2d_limit_cycle_found():
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(Iext=0.5),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    # a sustained oscillation produces a limit cycle
    pp.plot_limit_cycle_by_sim(initials={'V': [-2.8], 'w': [-0.5]}, duration=100.)


def test_pp2d_limit_cycle_not_found():
    # decaying system: no limit cycle -> exercises the "no limit cycle" branch
    pp = bp.analysis.PhasePlane2D(model=_fhn_2d(Iext=0.0),
                                  target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
                                  resolutions=0.2)
    pp.plot_limit_cycle_by_sim(initials={'V': [0.01], 'w': [0.0]}, duration=10.)
