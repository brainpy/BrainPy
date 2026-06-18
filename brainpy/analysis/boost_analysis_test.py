# -*- coding: utf-8 -*-
"""Audit coverage-boost tests for BrainPy v2.7.8 low-dimensional analysis.

This module is part of the test-coverage audit. It exercises:

- ``brainpy/analysis/lowdim/lowdim_analyzer.py`` (baseline ~39% line coverage)
  via the public ``PhasePlane1D`` / ``PhasePlane2D`` / ``Bifurcation1D`` /
  ``Bifurcation2D`` / ``FastSlow1D`` / ``FastSlow2D`` analyzers, including the
  numerical nullcline / fixed-point / Jacobian machinery in ``Num1DAnalyzer``
  and ``Num2DAnalyzer`` (both the optimization branch and the
  "convert-to-one-equation" brentq branch).
- ``brainpy/analysis/utils/optimization.py`` (baseline ~74% line coverage)
  via direct calls to the root-finding helpers (``jax_brentq``,
  ``get_brentq_candidates``, ``brentq_candidates``, ``brentq_roots``,
  ``brentq_roots2``, ``roots_of_1d_by_x``, ``roots_of_1d_by_xy``,
  ``numpy_brentq``, ``find_root_of_1d_numpy``) and ``scipy_minimize_with_jax``.

All tests run on tiny models with coarse resolutions, short durations and small
parameter ranges so the whole module stays well under the runtime budget.
The tests do NOT modify any source; behaviour quirks observed on valid input
(e.g. duplicate 1D fixed points) are pinned with explicit assertions/notes.
"""

import matplotlib

matplotlib.use('Agg')  # headless backend; must precede pyplot / analysis imports

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from matplotlib import pyplot as plt

import brainpy as bp
import brainpy.math as bm
from brainpy.analysis import constants as C
from brainpy.analysis.lowdim.lowdim_analyzer import Num3DAnalyzer
from brainpy.analysis.utils import optimization as opt


# ---------------------------------------------------------------------------
# Module-level setup: brentq optimization in BrainPy requires x64.
# ---------------------------------------------------------------------------

def setup_module(module):
    bm.enable_x64()


def teardown_module(module):
    plt.close('all')
    bm.disable_x64()


def _close():
    plt.close('all')


# ===========================================================================
# optimization.py  --  direct helper calls on tiny functions
# ===========================================================================

def test_jax_brentq_simple_root():
    """jax_brentq finds a bracketed root and reports convergence."""
    solver = opt.jax_brentq(lambda x: x - 2.0)
    res = solver(0.0, 5.0)
    assert res['status'] == opt.ECONVERGED
    assert abs(float(res['root']) - 2.0) < 1e-6
    assert int(res['funcalls']) >= 2


def test_jax_brentq_root_at_endpoint():
    """When an endpoint is already a root, brentq returns it immediately."""
    solver = opt.jax_brentq(lambda x: x)
    # a == 0 is exactly a root -> early ECONVERGED path
    res = solver(0.0, 3.0)
    assert res['status'] == opt.ECONVERGED
    assert abs(float(res['root'])) < 1e-12


def test_jax_brentq_with_args():
    """jax_brentq passes through extra args to the objective."""
    solver = opt.jax_brentq(lambda x, p: x - p)
    res = solver(0.0, 10.0, (4.0,))
    assert res['status'] == opt.ECONVERGED
    assert abs(float(res['root']) - 4.0) < 1e-6


def test_roots_of_1d_by_x_quadratic():
    """roots_of_1d_by_x recovers both roots of x**2 - 1 on [-2, 2]."""
    cands = jnp.linspace(-2.0, 2.0, 41)
    roots = opt.roots_of_1d_by_x(lambda x: x ** 2 - 1.0, cands)
    vals = sorted(round(float(r), 3) for r in roots)
    assert vals == [-1.0, 1.0]


def test_roots_of_1d_by_x_exact_zero_candidate():
    """Candidates that sit exactly on a root take the zero-sign branch."""
    # candidate grid includes the exact roots 0 and +/-1 of x**3 - x
    cands = jnp.linspace(-1.5, 1.5, 7)  # ... -1, -0.5, 0, 0.5, 1 ...
    roots = opt.roots_of_1d_by_x(lambda x: x ** 3 - x, cands)
    vals = sorted(round(float(r), 3) for r in np.unique(np.asarray(roots)))
    assert -1.0 in vals and 0.0 in vals and 1.0 in vals


def test_roots_of_1d_by_x_no_roots():
    """A strictly positive function yields no sign changes -> empty result."""
    cands = jnp.linspace(-2.0, 2.0, 21)
    roots = opt.roots_of_1d_by_x(lambda x: x ** 2 + 1.0, cands)
    assert len(np.asarray(roots)) == 0


def test_get_brentq_candidates_and_roots_of_1d_by_xy():
    """get_brentq_candidates + roots_of_1d_by_xy round-trip on f(x, y) = x - y."""
    xs = jnp.linspace(-2.0, 2.0, 21)
    ys = jnp.linspace(-1.0, 1.0, 11)
    starts, ends, args = opt.get_brentq_candidates(lambda x, y: x - y, xs, ys)
    assert starts.shape == ends.shape == args.shape
    assert starts.shape[0] > 0
    xs_root, ys_root = opt.roots_of_1d_by_xy(lambda x, y: x - y, starts, ends, args)
    # For f = x - y, the root x equals the parameter y.
    assert xs_root.shape == ys_root.shape
    np.testing.assert_allclose(np.asarray(xs_root), np.asarray(ys_root), atol=1e-6)


def test_brentq_candidates_and_roots():
    """brentq_candidates / brentq_roots / brentq_roots2 agree for f(x, p) = x - p."""
    f = lambda x, p: x - p
    vmap_f = jax.jit(vmap(f))
    xs = jnp.linspace(-2.0, 2.0, 21)
    ps = jnp.linspace(-1.0, 1.0, 11)
    starts, ends, others = opt.brentq_candidates(vmap_f, xs, ps)
    assert starts.shape[0] > 0
    assert len(others) == 1

    roots, vargs = opt.brentq_roots(f, starts, ends, others[0])
    assert roots.shape[0] > 0
    np.testing.assert_allclose(np.asarray(roots), np.asarray(vargs[0]), atol=1e-6)

    vmap_brentq = jax.jit(vmap(opt.jax_brentq(f)))
    roots2, vargs2 = opt.brentq_roots2(vmap_brentq, starts, ends, others[0])
    np.testing.assert_allclose(np.asarray(roots2), np.asarray(vargs2[0]), atol=1e-6)


def test_brentq_roots_no_extra_args_is_broken():
    """NOTE (source bug): brentq_roots with no vmap_args/args is broken.

    When called with neither ``vmap_args`` nor ``args``, the function builds an
    ``in_axes`` tuple of length 3 ``(0, 0, ())`` but then invokes the vmapped
    optimizer with only two positional arguments ``(starts, ends)``. jax.vmap
    rejects the mismatch (len(in_axes)=3 vs len(args)=2). This is a latent bug
    on the ``else`` branch of ``brentq_roots`` -- the codebase always reaches it
    via ``brentq_roots2`` instead. Pinned here so any future fix is noticed.
    """
    f = lambda x: x - 1.0
    starts = jnp.array([0.0, -5.0])
    ends = jnp.array([2.0, 5.0])
    with pytest.raises(ValueError):
        opt.brentq_roots(f, starts, ends)


def test_brentq_roots2_no_extra_args():
    """brentq_roots2 (the actually-used variant) handles the no-args case."""
    f = lambda x: x - 1.0
    vmap_brentq = jax.jit(vmap(opt.jax_brentq(f)))
    starts = jnp.array([0.0, -5.0])
    ends = jnp.array([2.0, 5.0])
    roots, vargs = opt.brentq_roots2(vmap_brentq, starts, ends)
    assert vargs == ()
    np.testing.assert_allclose(np.sort(np.asarray(roots)), [1.0, 1.0], atol=1e-6)


def test_scipy_minimize_with_jax():
    """scipy_minimize_with_jax minimizes a quadratic and unflattens the result."""
    res = opt.scipy_minimize_with_jax(
        lambda x: ((x - 3.0) ** 2).sum(),
        jnp.array([0.0]),
        method='BFGS',
    )
    assert bool(res['success'])
    assert abs(float(np.asarray(res['x'])[0]) - 3.0) < 1e-4


def test_scipy_minimize_with_jax_callback_and_bounds():
    """Exercise the callback + bounds branches of scipy_minimize_with_jax."""
    seen = []

    def cb(xk):
        seen.append(np.asarray(jax.tree_util.tree_leaves(xk)[0]))
        return False  # do not terminate early

    res = opt.scipy_minimize_with_jax(
        lambda x: ((x - 1.0) ** 2).sum(),
        jnp.array([5.0]),
        method='L-BFGS-B',
        bounds=[(0.0, 10.0)],
        callback=cb,
    )
    assert bool(res['success'])
    assert abs(float(np.asarray(res['x'])[0]) - 1.0) < 1e-3
    assert len(seen) >= 1


def test_numpy_brentq_root_and_errors():
    """numpy_brentq finds a root and raises on bad bracket / params."""
    root, funcalls, itr = opt.numpy_brentq(lambda x: x ** 2 - 4.0, 0.0, 5.0)
    assert abs(root - 2.0) < 1e-6
    assert funcalls >= 2

    # endpoint is a root -> early-return branch (status ECONVERGED, itr == 0)
    root2, _, itr2 = opt.numpy_brentq(lambda x: x, 0.0, 5.0)
    assert abs(root2) < 1e-12
    assert itr2 == 0

    with pytest.raises(ValueError):
        opt.numpy_brentq(lambda x: x + 1.0, 0.0, 5.0)  # f(a), f(b) same sign
    with pytest.raises(ValueError):
        opt.numpy_brentq(lambda x: x, 0.0, 5.0, xtol=-1.0)  # bad xtol
    with pytest.raises(ValueError):
        opt.numpy_brentq(lambda x: x, 0.0, 5.0, maxiter=0)  # bad maxiter


def test_numpy_brentq_endpoint_b_is_root():
    """numpy_brentq returns immediately when the upper endpoint is the root."""
    root, _, itr = opt.numpy_brentq(lambda x: x - 5.0, 0.0, 5.0)
    assert abs(root - 5.0) < 1e-12
    assert itr == 0  # early ECONVERGED via fcur == 0


def test_find_root_of_1d_numpy():
    """find_root_of_1d_numpy recovers the roots of x**3 - x including exact ones."""
    pts = np.linspace(-3.0, 3.0, 61)
    roots = opt.find_root_of_1d_numpy(lambda x: x ** 3 - x, pts)
    vals = sorted(round(float(r), 3) for r in np.unique(np.asarray(roots)))
    assert -1.0 in vals and 0.0 in vals and 1.0 in vals


def test_find_root_of_1d_numpy_leading_and_trailing_zeros():
    """find_root_of_1d_numpy handles exact roots at the first/last grid points."""
    # f = x*(x-2): roots 0 (first point, leading-zero branch) and 2 (last point)
    pts = np.linspace(0.0, 2.0, 11)
    roots = opt.find_root_of_1d_numpy(lambda x: x * (x - 2.0), pts)
    vals = sorted(round(float(r), 3) for r in np.unique(np.asarray(roots)))
    assert 0.0 in vals and 2.0 in vals


# ===========================================================================
# PhasePlane1D  --  lowdim_analyzer Num1DAnalyzer paths
# ===========================================================================

def test_phase_plane_1d_linear():
    """Linear 1D system dx = -x + I has a single stable fixed point at x = I."""
    Iext = 0.5

    @bp.odeint
    def int_x(x, t, I=0.5):
        return -x + I

    pp = bp.analysis.PhasePlane1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        pars_update={'I': Iext},
        resolutions=0.05,
    )
    # vector field
    yv = pp.plot_vector_field(with_return=True, show=False)
    assert yv.shape[0] > 0

    # fixed points
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    assert len(fps) >= 1
    # every returned fixed point should sit at x == I
    np.testing.assert_allclose(fps, Iext, atol=1e-4)
    _close()


def test_phase_plane_1d_cubic():
    """Cubic 1D system dx = x - x**3 has three fixed points: -1, 0, 1."""

    @bp.odeint
    def int_x(x, t):
        return x - x ** 3

    pp = bp.analysis.PhasePlane1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        resolutions=0.02,
    )
    pp.plot_vector_field(show=False)
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    uniq = sorted(round(float(v), 2) for v in np.unique(np.round(fps, 2)))
    for expected in (-1.0, 0.0, 1.0):
        assert any(abs(u - expected) < 1e-2 for u in uniq), f'missing fp {expected}: {uniq}'
    _close()


def test_phase_plane_1d_no_plot_paths():
    """with_plot=False / with_return=False branches return None without drawing."""

    @bp.odeint
    def int_x(x, t):
        return -x

    pp = bp.analysis.PhasePlane1D(
        model=int_x,
        target_vars={'x': [-1.0, 1.0]},
        resolutions=0.1,
    )
    assert pp.plot_vector_field(with_plot=False, with_return=False, show=False) is None
    assert pp.plot_fixed_point(with_plot=False, with_return=False, show=False) is None
    _close()


def test_phase_plane_1d_rejects_target_pars():
    """PhasePlane analyzers reject target_pars (only PP via pars_update allowed)."""
    @bp.odeint
    def int_x(x, t, a=1.):
        return -a * x

    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(
            model=int_x,
            target_vars={'x': [-1.0, 1.0]},
            target_pars={'a': [0.5, 1.5]},
            resolutions=0.1,
        )


# ===========================================================================
# PhasePlane2D  --  lowdim_analyzer Num2DAnalyzer paths (FitzHugh-Nagumo)
# ===========================================================================

_FHN_A, _FHN_B, _FHN_TAU = 0.7, 0.8, 12.5


def _fhn_integrals(I=0.5):
    @bp.odeint
    def int_V(V, t, w, I=I):
        return V - V ** 3 / 3.0 - w + I

    @bp.odeint
    def int_w(w, t, V):
        return (V + _FHN_A - _FHN_B * w) / _FHN_TAU

    return int_V, int_w


def test_phase_plane_2d_full_optimization_branch():
    """Full PP2D pipeline (vector field, nullclines, fixed point, trajectory)."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
    )

    # streamplot + quiver vector field branches
    dx, dy = pp.plot_vector_field(with_return=True, show=False)
    assert dx.shape == dy.shape and dx.ndim == 2
    pp.plot_vector_field(plot_method='quiver', show=False)

    # nullclines (optimization branch, x-y coords)
    nc = pp.plot_nullcline(with_return=True, show=False)
    assert set(nc.keys()) == {'V', 'w'}

    # fixed points from the fx-nullcline candidates (default)
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    assert fps.shape[1] == 2
    # FHN with I=0.5 has a single (unstable) fixed point in this window
    assert len(fps) >= 1

    # trajectory in both axis modes
    traj = pp.plot_trajectory(
        initials={'V': [-1.0], 'w': [0.0]},
        duration=10.0, show=False, with_return=True,
    )
    assert traj is not None
    pp.plot_trajectory(
        initials={'V': [-1.0], 'w': [0.0]},
        duration=5.0, axes='t-v', show=False,
    )
    _close()


def test_phase_plane_2d_nullcline_alternate_coords():
    """coords='w-V' exercises the y_var-x_var branch of the nullcline solver."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
    )
    nc = pp.plot_nullcline(
        with_return=True, show=False,
        coords={'V': 'V-w', 'w': 'w-V'},
    )
    assert set(nc.keys()) == {'V', 'w'}
    _close()


def test_phase_plane_2d_invalid_plot_method_and_axes():
    """Unknown plot_method / axes raise analyzer errors."""
    int_V, int_w = _fhn_integrals()
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.2,
    )
    with pytest.raises(Exception):
        pp.plot_vector_field(plot_method='nope', show=False)
    with pytest.raises(Exception):
        pp.plot_trajectory(initials={'V': [0.0], 'w': [0.0]},
                           duration=1.0, axes='bad', show=False)
    _close()


def test_phase_plane_2d_limit_cycle_by_sim():
    """plot_limit_cycle_by_sim runs (no cycle expected for short sim, but exercised)."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.2,
    )
    # short duration -> exercises the "no limit cycle found" branch safely
    pp.plot_limit_cycle_by_sim(
        initials={'V': [-1.0], 'w': [0.0]},
        duration=20.0, show=False,
    )
    _close()


def test_phase_plane_2d_aux_rank_candidates():
    """select_candidates='aux_rank' exercises _get_fp_candidates_by_aux_rank."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
    )
    fps = np.asarray(pp.plot_fixed_point(
        with_return=True, select_candidates='aux_rank', num_rank=50, show=False))
    assert fps.shape[1] == 2
    assert len(fps) >= 1
    _close()


def test_phase_plane_2d_fixed_point_requires_nullcline():
    """fx/fy-nullcline candidate selection errors if nullclines not yet computed."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.2,
    )
    with pytest.raises(Exception):
        pp.plot_fixed_point(select_candidates='fy-nullcline', show=False)
    _close()


def test_phase_plane_2d_convert_to_one_equation():
    """Providing y_by_x_in_fy enables the brentq 'convert to one equation' branch."""
    int_V, int_w = _fhn_integrals(I=0.5)

    # w-nullcline of FHN: (V + a - b*w)/tau = 0  ->  w = (V + a) / b
    def w_by_V(V):
        return (V + _FHN_A) / _FHN_B

    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
        options={C.y_by_x_in_fy: w_by_V},
    )
    assert pp._can_convert_to_one_eq()
    assert pp.convert_type() == C.y_by_x
    # nullcline now uses the analytic F_y_by_x_in_fy branch
    nc = pp.plot_nullcline(with_return=True, show=False)
    assert set(nc.keys()) == {'V', 'w'}
    # fixed point via brentq optimization branch
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    assert fps.shape[1] == 2 and len(fps) >= 1
    _close()


def test_phase_plane_2d_convert_x_by_y_in_fy():
    """x_by_y_in_fy option drives the analytic fy-nullcline + x_by_y convert type."""
    int_V, int_w = _fhn_integrals(I=0.5)

    # w-nullcline: (V + a - b*w)/tau = 0  ->  V = b*w - a
    def V_by_w(w):
        return _FHN_B * w - _FHN_A

    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
        options={C.x_by_y_in_fy: V_by_w},
    )
    assert pp.convert_type() == C.x_by_y
    nc = pp.plot_nullcline(with_return=True, show=False)
    assert set(nc.keys()) == {'V', 'w'}
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    assert fps.shape[1] == 2 and len(fps) >= 1
    _close()


def test_phase_plane_2d_convert_y_by_x_in_fx():
    """y_by_x_in_fx option drives the analytic fx-nullcline + y_by_x convert type."""
    int_V, int_w = _fhn_integrals(I=0.5)

    # V-nullcline: V - V**3/3 - w + I = 0  ->  w = V - V**3/3 + I
    def w_by_V(V):
        return V - V ** 3 / 3.0 + 0.5

    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
        options={C.y_by_x_in_fx: w_by_V},
    )
    assert pp.convert_type() == C.y_by_x
    nc = pp.plot_nullcline(with_return=True, show=False)
    assert set(nc.keys()) == {'V', 'w'}
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    assert fps.shape[1] == 2 and len(fps) >= 1
    _close()


def test_phase_plane_2d_convert_x_by_y_in_fx():
    """x_by_y_in_fx option (linear system) drives the analytic fx-nullcline branch."""

    @bp.odeint
    def int_x(x, t, y):
        return -x + 0.5 * y

    @bp.odeint
    def int_y(y, t, x):
        return x - 2.0 * y

    # fx-nullcline: -x + 0.5*y = 0  ->  x = 0.5*y
    def x_by_y(y):
        return 0.5 * y

    pp = bp.analysis.PhasePlane2D(
        model=[int_x, int_y],
        target_vars={'x': [-2.0, 2.0], 'y': [-2.0, 2.0]},
        resolutions=0.1,
        options={C.x_by_y_in_fx: x_by_y},
    )
    assert pp.convert_type() == C.x_by_y
    nc = pp.plot_nullcline(with_return=True, show=False)
    assert set(nc.keys()) == {'x', 'y'}
    fps = np.asarray(pp.plot_fixed_point(with_return=True, show=False))
    # the only fixed point of this linear system is the origin
    assert fps.shape[1] == 2 and len(fps) >= 1
    np.testing.assert_allclose(fps[0], [0.0, 0.0], atol=1e-4)
    _close()


def test_num2d_jacobian_and_derivatives():
    """Directly exercise the F_jacobian / derivative properties of Num2DAnalyzer."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.2,
    )
    J = np.asarray(pp.F_jacobian(0.0, 0.0))
    assert J.shape == (2, 2)
    # partials at the origin: dfx/dw = -1 ; dfy/dV = 1/tau ; dfy/dw = -b/tau
    assert abs(float(pp.F_dfxdy(0.0, 0.0)) + 1.0) < 1e-6
    assert abs(float(pp.F_dfydx(0.0, 0.0)) - 1.0 / _FHN_TAU) < 1e-6
    assert abs(float(pp.F_dfydy(0.0, 0.0)) + _FHN_B / _FHN_TAU) < 1e-6
    _close()


# ===========================================================================
# Bifurcation1D / Bifurcation2D
# ===========================================================================

def test_bifurcation_1d_codim1():
    """Bifurcation1D co-dimension-1: dx = -x + I, fixed point tracks I."""

    @bp.odeint
    def int_x(x, t, I=0.0):
        return -x + I

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        target_pars={'I': [-1.0, 1.0]},
        resolutions={'I': 0.1},
    )
    fps, pars, dfdx = bf.plot_bifurcation(with_return=True, show=False)
    fps = np.asarray(fps)
    p = np.asarray(pars[0])
    assert fps.shape[0] > 0 and fps.shape == p.shape
    # for dx = -x + I, fixed point x == I and df/dx == -1 (stable)
    np.testing.assert_allclose(fps, p, atol=1e-4)
    assert np.all(np.asarray(dfdx) < 0)
    _close()


def test_bifurcation_1d_codim2():
    """Bifurcation1D co-dimension-2 (3D scatter): dx = -a*x + b."""

    @bp.odeint
    def int_x(x, t, a=1.0, b=0.0):
        return -a * x + b

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        target_pars={'a': [0.5, 1.5], 'b': [-1.0, 1.0]},
        resolutions={'a': 0.3, 'b': 0.3},
    )
    fps, pars, dfdx = bf.plot_bifurcation(with_return=True, show=False)
    assert np.asarray(fps).shape[0] > 0
    assert len(pars) == 2
    _close()


def test_bifurcation_1d_float_resolution_warns():
    """A single float resolution with target_pars warns and uses jnp.arange grids."""

    @bp.odeint
    def int_x(x, t, I=0.0):
        return -x + I

    with pytest.warns(UserWarning):
        bf = bp.analysis.Bifurcation1D(
            model=int_x,
            target_vars={'x': [-2.0, 2.0]},
            target_pars={'I': [-1.0, 1.0]},
            resolutions=0.2,
        )
    fps, pars, dfdx = bf.plot_bifurcation(with_return=True, show=False)
    assert np.asarray(fps).shape[0] > 0
    _close()


def test_bifurcation_2d_segmented():
    """num_par_segments>1 and num_fp_segment>1 exercise the segment-loop branches."""
    int_V, int_w = _fhn_integrals()

    @bp.odeint
    def int_V2(V, t, w, Iext=0.0):
        return V - V ** 3 / 3.0 - w + Iext

    bif = bp.analysis.Bifurcation2D(
        model=[int_V2, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        target_pars={'Iext': [0.0, 1.0]},
        resolutions={'Iext': 0.3},
    )
    fps, pars, jac = bif.plot_bifurcation(
        with_return=True, show=False,
        select_candidates='fx-nullcline',
        num_par_segments=2, num_fp_segment=2, nullcline_aux_filter=1.0,
    )
    assert np.asarray(fps).shape[1] == 2
    _close()


def test_bifurcation_2d_codim1_and_limit_cycle():
    """Bifurcation2D co-dimension-1 on FHN over Iext, plus limit-cycle sim."""
    int_V, int_w = _fhn_integrals()

    @bp.odeint
    def int_V2(V, t, w, Iext=0.0):
        return V - V ** 3 / 3.0 - w + Iext

    bif = bp.analysis.Bifurcation2D(
        model=[int_V2, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        target_pars={'Iext': [0.0, 1.0]},
        resolutions={'Iext': 0.2},
    )
    fps, pars, jac = bif.plot_bifurcation(with_return=True, show=False)
    fps = np.asarray(fps)
    assert fps.shape[1] == 2
    assert np.asarray(jac).shape[1:] == (2, 2)
    assert np.asarray(pars).shape[1] == 1

    # limit cycle by simulation off the recorded fixed points
    lc = bif.plot_limit_cycle_by_sim(duration=20.0, with_return=True, show=False)
    assert lc is not None
    _close()


def test_bifurcation_2d_nullcline_candidate_selection():
    """Bifurcation2D with fx/fy/nullclines candidate selection + aux filtering.

    This drives the analytic-free nullcline solvers plus the ``_fp_filter``
    auxiliary-loss filtering branch (nullcline_aux_filter > 0).
    """
    int_V, int_w = _fhn_integrals()

    @bp.odeint
    def int_V2(V, t, w, Iext=0.0):
        return V - V ** 3 / 3.0 - w + Iext

    for select in ('fx-nullcline', 'fy-nullcline', 'nullclines'):
        bif = bp.analysis.Bifurcation2D(
            model=[int_V2, int_w],
            target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
            target_pars={'Iext': [0.0, 1.0]},
            resolutions={'Iext': 0.3},
        )
        fps, pars, jac = bif.plot_bifurcation(
            with_return=True, show=False,
            select_candidates=select, nullcline_aux_filter=1.0,
        )
        assert np.asarray(fps).shape[1] == 2
        assert np.asarray(fps).shape[0] >= 1
        _close()


def test_bifurcation_2d_limit_cycle_without_bifurcation():
    """plot_limit_cycle_by_sim returns early when no fixed points recorded yet."""
    int_V, int_w = _fhn_integrals()

    @bp.odeint
    def int_V2(V, t, w, Iext=0.0):
        return V - V ** 3 / 3.0 - w + Iext

    bif = bp.analysis.Bifurcation2D(
        model=[int_V2, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        target_pars={'Iext': [0.0, 1.0]},
        resolutions={'Iext': 0.3},
    )
    # _fixed_points is None -> early return (None)
    assert bif.plot_limit_cycle_by_sim(duration=5.0, show=False) is None
    _close()


# ===========================================================================
# FastSlow1D / FastSlow2D
# ===========================================================================

def test_fast_slow_1d():
    """FastSlow1D: fast var x, slow var u as the bifurcation parameter."""

    @bp.odeint
    def int_x(x, t, u=0.0):
        return -x + u

    @bp.odeint
    def int_u(u, t, x):
        return 0.01 * (x - u)

    fs = bp.analysis.FastSlow1D(
        model=[int_x, int_u],
        fast_vars={'x': [-2.0, 2.0]},
        slow_vars={'u': [-1.0, 1.0]},
        resolutions={'u': 0.1},
    )
    fps, pars, dfdx = fs.plot_bifurcation(with_return=True, show=False)
    fps = np.asarray(fps)
    assert fps.shape[0] > 0
    # x* == u and stable (df/dx == -1)
    np.testing.assert_allclose(fps, np.asarray(pars[0]), atol=1e-4)

    traj = fs.plot_trajectory(
        initials={'x': [0.5], 'u': [0.5]},
        duration=10.0, show=False, with_return=True,
    )
    assert traj is not None
    _close()


def test_fast_slow_2d():
    """FastSlow2D: 2 fast vars (V, w), 1 slow var u acting like an input current."""

    @bp.odeint
    def int_V(V, t, w, u=0.0):
        return V - V ** 3 / 3.0 - w + u

    @bp.odeint
    def int_w(w, t, V):
        return (V + _FHN_A - _FHN_B * w) / _FHN_TAU

    @bp.odeint
    def int_u(u, t, V):
        return 0.01 * (V - u)

    fs = bp.analysis.FastSlow2D(
        model=[int_V, int_w, int_u],
        fast_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        slow_vars={'u': [0.0, 1.0]},
        resolutions={'u': 0.2},
    )
    fps, pars, jac = fs.plot_bifurcation(with_return=True, show=False)
    assert np.asarray(fps).shape[1] == 2

    traj = fs.plot_trajectory(
        initials={'V': [-1.0], 'w': [0.0], 'u': [0.5]},
        duration=10.0, show=False, with_return=True,
    )
    assert traj is not None
    _close()


# ===========================================================================
# LowDimAnalyzer construction / validation paths
# ===========================================================================

def test_analyzer_resolution_dict_and_array():
    """resolutions can be a dict mixing floats and explicit 1D arrays."""

    @bp.odeint
    def int_x(x, t, I=0.0):
        return -x + I

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        target_pars={'I': [-1.0, 1.0]},
        resolutions={'x': np.linspace(-2.0, 2.0, 25), 'I': 0.2},
    )
    assert bf.resolutions['x'].shape[0] == 25
    assert bf.resolutions['I'].shape[0] > 0
    _close()


def test_analyzer_resolution_none_default():
    """resolutions=None builds the default 20-point linspace for vars and pars."""

    @bp.odeint
    def int_x(x, t, I=0.0):
        return -x + I

    bf = bp.analysis.Bifurcation1D(
        model=int_x,
        target_vars={'x': [-2.0, 2.0]},
        target_pars={'I': [-1.0, 1.0]},
        resolutions=None,
    )
    assert bf.resolutions['x'].shape[0] == 20
    assert bf.resolutions['I'].shape[0] == 20
    _close()


def test_analyzer_validation_errors():
    """Constructor validates target_vars / fixed_vars / reversed ranges."""

    @bp.odeint
    def int_x(x, t, I=0.0):
        return -x + I

    # target_vars must be a dict
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars=['x'], resolutions=0.1)

    # reversed variable range
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [2.0, -2.0]}, resolutions=0.1)

    # unknown target variable
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'z': [-1.0, 1.0]}, resolutions=0.1)

    # unknown resolution target key
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 resolutions={'zzz': 0.1})

    # fixed_vars must be a dict
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 fixed_vars=['x'], resolutions=0.1)

    # pars_update must reference a real parameter
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 pars_update={'nope': 1.0}, resolutions=0.1)

    # reversed parameter range
    with pytest.raises(Exception):
        bp.analysis.Bifurcation1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                  target_pars={'I': [1.0, -1.0]}, resolutions={'I': 0.1})

    # unknown target parameter
    with pytest.raises(Exception):
        bp.analysis.Bifurcation1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                  target_pars={'nope': [-1.0, 1.0]}, resolutions={'nope': 0.1})

    # resolution value must be a 1D array, not 2D
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 resolutions={'x': np.ones((2, 2))})

    # unknown resolution value type (e.g. a string)
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 resolutions={'x': 'bad'})

    # unknown resolution container type
    with pytest.raises(Exception):
        bp.analysis.PhasePlane1D(model=int_x, target_vars={'x': [-1.0, 1.0]},
                                 resolutions='all')


def test_phase_plane_2d_fx_nullcline_alternate_coords_and_tol_opt_screen():
    """fx-nullcline 'w-V' coords + tol_opt_screen candidate screening path."""
    int_V, int_w = _fhn_integrals(I=0.5)
    pp = bp.analysis.PhasePlane2D(
        model=[int_V, int_w],
        target_vars={'V': [-2.5, 2.5], 'w': [-1.0, 2.5]},
        resolutions=0.1,
    )
    # 'w-V' triggers the y_var-x_var optimisation branch for the fx nullcline
    pp.plot_nullcline(show=False, coords={'V': 'w-V'})
    # tol_opt_screen exercises the tol_opt_candidate screening in _get_fixed_points
    fps = np.asarray(pp.plot_fixed_point(with_return=True, tol_opt_screen=1e-2, show=False))
    assert fps.shape[1] == 2 and len(fps) >= 1
    _close()


def test_num3d_analyzer():
    """Num3DAnalyzer instantiates and evaluates the third derivative F_fz."""

    @bp.odeint
    def int_x(x, t, y, z):
        return -x + y

    @bp.odeint
    def int_y(y, t, x, z):
        return -y + z

    @bp.odeint
    def int_z(z, t, x, y):
        return -z + x

    ana = Num3DAnalyzer(
        model=[int_x, int_y, int_z],
        target_vars={'x': [-1.0, 1.0], 'y': [-1.0, 1.0], 'z': [-1.0, 1.0]},
        resolutions=0.2,
    )
    assert ana.z_var == 'z'
    # dz = -z + x  ->  at (x=0.3, y=0.2, z=0.3): -0.3 + 0.3 == 0 ... use distinct vals
    val = float(ana.F_fz(0.1, 0.2, 0.3))
    assert abs(val - (-0.3 + 0.1)) < 1e-6


def test_num3d_analyzer_requires_three_vars():
    """Num3DAnalyzer rejects models with fewer than three target variables."""

    @bp.odeint
    def int_x(x, t, y):
        return -x + y

    @bp.odeint
    def int_y(y, t, x):
        return -y + x

    with pytest.raises(Exception):
        Num3DAnalyzer(
            model=[int_x, int_y],
            target_vars={'x': [-1.0, 1.0], 'y': [-1.0, 1.0]},
            resolutions=0.2,
        )
