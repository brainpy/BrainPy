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
"""Coverage tests for :mod:`brainpy.analysis.lowdim.lowdim_bifurcation`.

Exercises ``Bifurcation1D`` (co-dimension 1 and 2), ``Bifurcation2D``
(co-dimension 1 and 2 + limit-cycle plotting + candidate-selection branches),
and the ``FastSlow1D`` / ``FastSlow2D`` subclasses (bifurcation + trajectory
plotting).  Also drives error/early-return branches.  Resolutions are kept
coarse so the tests are fast; ``matplotlib`` runs headless via conftest.
"""

import matplotlib.pyplot as plt
import pytest

import brainpy as bp
import brainpy.math as bm


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fhn_system():
    class FHN(bp.DynamicalSystem):
        def __init__(self):
            super().__init__()
            self.a = 0.7
            self.b = 0.8
            self.tau = 12.5
            self.V = bm.Variable(bm.zeros(1))
            self.w = bm.Variable(bm.zeros(1))
            self.Iext = bm.Variable(bm.zeros(1))

            def dV(V, t, w, Iext=0.):
                return V - V * V * V / 3 - w + Iext

            def dw(w, t, V, a=0.7, b=0.8):
                return (V + a - b * w) / self.tau

            self.int_V = bp.odeint(dV)
            self.int_w = bp.odeint(dw)

        def update(self):
            t = bp.share['t']
            self.V.value = self.int_V(self.V, t, self.w, self.Iext)
            self.w.value = self.int_w(self.w, t, self.V, self.a, self.b)
            self.Iext[:] = 0.

    return FHN()


@pytest.fixture(autouse=True)
def _x64_and_close():
    bp.math.enable_x64()
    try:
        yield
    finally:
        plt.close('all')
        bp.math.disable_x64()


# --------------------------------------------------------------------------- #
# Bifurcation1D
# --------------------------------------------------------------------------- #
def test_bifurcation1d_codim1():
    @bp.odeint
    def int_x(x, t, a=1., b=1.):
        return bp.math.sin(a * x) + bp.math.cos(b * x)

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-bp.math.pi, bp.math.pi]},
        target_pars={'a': [0.5, 1.0]},
        resolutions={'a': 0.2},
    )
    fps, pars, dfxdx = bf.plot_bifurcation(show=False, with_return=True)
    assert len(fps) == len(dfxdx)


def test_bifurcation1d_codim2():
    @bp.odeint
    def int_x(x, t, a=1., b=1.):
        return bp.math.sin(a * x) + bp.math.cos(b * x)

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-bp.math.pi, bp.math.pi]},
        target_pars={'a': [0.5, 1.0], 'b': [0.5, 1.0]},
        resolutions={'a': 0.5, 'b': 0.5},
    )
    bf.plot_bifurcation(show=False)


def test_bifurcation1d_empty_target_pars_raises():
    @bp.odeint
    def int_x(x, t, a=1.):
        return -x + a

    with pytest.raises(ValueError):
        bp.analysis.Bifurcation1D(
            model=int_x,
            target_vars={'x': [-2., 2.]},
            target_pars={},
            resolutions={'x': 0.1},
        )


# --------------------------------------------------------------------------- #
# Bifurcation2D
# --------------------------------------------------------------------------- #
def test_bifurcation2d_codim1_and_limit_cycle():
    m = _fhn_system()
    bif = bp.analysis.Bifurcation2D(
        model=m,
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'Iext': [0., 1.]},
        resolutions={'Iext': 0.5},
    )
    fps, pars, jac = bif.plot_bifurcation(show=False, with_return=True)
    assert fps.shape[1] == 2
    assert jac.shape[1:] == (2, 2)
    # limit cycle plotting uses the cached fixed points
    res = bif.plot_limit_cycle_by_sim(duration=20., with_return=True)
    assert res is not None


def test_bifurcation2d_codim1_limit_cycle_detected():
    # FitzHugh-Nagumo in its oscillatory regime: a sustained limit cycle is
    # detected, exercising the codim-1 limit-cycle visualization branch.
    @bp.odeint
    def dV(V, t, w, Iext):
        return V - V ** 3 / 3 - w + Iext

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    bif = bp.analysis.Bifurcation2D(
        model=[dV, dw],
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'Iext': [0.3, 0.8]},
        resolutions={'Iext': 0.1},
    )
    bif.plot_bifurcation(show=False)
    vs, ps = bif.plot_limit_cycle_by_sim(duration=300., offset=1.0, with_return=True)
    assert any(len(p) > 0 for p in ps)


def test_bifurcation2d_codim2():
    @bp.odeint
    def dV(V, t, w, Iext, gNa):
        return V - V * V * V / 3 - w + Iext + 0. * gNa

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    bif = bp.analysis.Bifurcation2D(
        model=[dV, dw],
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'Iext': [0., 0.5], 'gNa': [0., 0.5]},
        resolutions={'Iext': 0.5, 'gNa': 0.5},
    )
    bif.plot_bifurcation(show=False)


def test_bifurcation2d_limit_cycle_early_return():
    @bp.odeint
    def dV(V, t, w, I):
        return V - V * V * V / 3 - w + I

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    bif = bp.analysis.Bifurcation2D(
        model=[dV, dw],
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'I': [0., 1.]},
        resolutions={'I': 0.5},
    )
    # plot_bifurcation has NOT been called -> _fixed_points is None -> early return
    assert bif.plot_limit_cycle_by_sim(duration=10.) is None


def _codim1_2d_model():
    @bp.odeint
    def dV(V, t, w, I):
        return V - V * V * V / 3 - w + I

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    return [dV, dw]


@pytest.mark.parametrize('selector', ['fx-nullcline', 'fy-nullcline', 'nullclines'])
def test_bifurcation2d_nullcline_selectors(selector):
    bif = bp.analysis.Bifurcation2D(
        model=_codim1_2d_model(),
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'I': [0., 1.]},
        resolutions={'I': 0.5},
    )
    bif.plot_bifurcation(show=False, select_candidates=selector)


def test_bifurcation2d_unknown_selector_raises():
    bif = bp.analysis.Bifurcation2D(
        model=_codim1_2d_model(),
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'I': [0., 1.]},
        resolutions={'I': 0.5},
    )
    with pytest.raises(ValueError):
        bif.plot_bifurcation(show=False, select_candidates='bogus')


def test_bifurcation2d_codim2_limit_cycle():
    @bp.odeint
    def dV(V, t, w, I, g):
        return V - V * V * V / 3 - w + I + 0. * g

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    bif = bp.analysis.Bifurcation2D(
        model=[dV, dw],
        target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        target_pars={'I': [0.4, 0.6], 'g': [0., 0.2]},
        resolutions={'I': 0.1, 'g': 0.1},
    )
    bif.plot_bifurcation(show=False)
    vs, ps = bif.plot_limit_cycle_by_sim(duration=300., offset=1.0, with_return=True)
    # oscillatory regime -> codim-2 limit-cycle visualization branch runs
    assert any(len(p) > 0 for p in ps)


def test_bifurcation2d_empty_target_pars_raises():
    @bp.odeint
    def dV(V, t, w):
        return V - w

    @bp.odeint
    def dw(w, t, V):
        return (V - 0.8 * w) / 12.5

    with pytest.raises(ValueError):
        bp.analysis.Bifurcation2D(
            model=[dV, dw],
            target_vars={'V': [-3., 3.], 'w': [-1., 3.]},
            target_pars={},
            resolutions={'V': 0.5, 'w': 0.5},
        )


# --------------------------------------------------------------------------- #
# FastSlow1D
# --------------------------------------------------------------------------- #
def test_fastslow1d():
    @bp.odeint
    def int_x(x, t, y):
        return x - x ** 3 / 3 - y

    fs = bp.analysis.FastSlow1D(
        model=int_x,
        fast_vars={'x': [-2., 2.]},
        slow_vars={'y': [-0.5, 0.5]},
        resolutions={'x': 0.1, 'y': 0.1},
    )
    fs.plot_bifurcation(show=False)
    res = fs.plot_trajectory(initials={'x': [0.5], 'y': [0.0]},
                             duration=20., with_return=True)
    assert res is not None


def test_fastslow1d_two_slow_vars_trajectory():
    # two slow vars -> 3D trajectory plotting branch
    @bp.odeint
    def int_x(x, t, y, z):
        return x - x ** 3 / 3 - y + 0. * z

    fs = bp.analysis.FastSlow1D(
        model=int_x,
        fast_vars={'x': [-2., 2.]},
        slow_vars={'y': [-0.5, 0.5], 'z': [-0.5, 0.5]},
        resolutions={'x': 0.2, 'y': 0.5, 'z': 0.5},
    )
    fs.plot_trajectory(initials={'x': [0.5], 'y': [0.0], 'z': [0.0]}, duration=20.)


# --------------------------------------------------------------------------- #
# FastSlow2D
# --------------------------------------------------------------------------- #
def test_fastslow2d():
    @bp.odeint
    def dV(V, t, w, I):
        return V - V ** 3 / 3 - w + I

    @bp.odeint
    def dw(w, t, V):
        return (V + 0.7 - 0.8 * w) / 12.5

    fs2 = bp.analysis.FastSlow2D(
        model=[dV, dw],
        fast_vars={'V': [-3., 3.], 'w': [-1., 3.]},
        slow_vars={'I': [0., 1.]},
        resolutions={'V': 0.5, 'w': 0.5, 'I': 0.5},
    )
    fs2.plot_bifurcation(show=False)
    res = fs2.plot_trajectory(initials={'V': [1.0], 'w': [0.0], 'I': [0.5]},
                              duration=20., with_return=True)
    assert res is not None
