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
"""Coverage tests for :mod:`brainpy.analysis.stability`.

Exercises every branch of ``stability_analysis`` (1D / 2D / 3D classification)
plus the small ``get_*_stability_types`` enumeration helpers and the
unknown-shape error branch.  These are pure functions on eigenvalues /
Jacobian matrices, so we drive each return path directly with hand-built
matrices.
"""

import numpy as np
import pytest

from brainpy.analysis import stability as st
from brainpy.analysis.stability import stability_analysis


# --------------------------------------------------------------------------- #
# enumeration helpers
# --------------------------------------------------------------------------- #
def test_stability_type_lists():
    t1 = st.get_1d_stability_types()
    t2 = st.get_2d_stability_types()
    t3 = st.get_3d_stability_types()
    assert st.SADDLE_NODE in t1 and st.STABLE_POINT_1D in t1 and st.UNSTABLE_POINT_1D in t1
    assert st.CENTER_2D in t2 and st.STABLE_NODE_2D in t2 and st.UNSTABLE_LINE_2D in t2
    # lists are non-empty and contain only strings
    for lst in (t1, t2, t3):
        assert len(lst) > 0
        assert all(isinstance(x, str) for x in lst)


# --------------------------------------------------------------------------- #
# 1D classification
# --------------------------------------------------------------------------- #
def test_1d_branches():
    assert stability_analysis(0.) == st.SADDLE_NODE
    assert stability_analysis(2.5) == st.UNSTABLE_POINT_1D
    assert stability_analysis(-2.5) == st.STABLE_POINT_1D


# --------------------------------------------------------------------------- #
# 2D classification
# --------------------------------------------------------------------------- #
def test_2d_saddle_node():
    # q = det < 0  ->  saddle node
    J = [[1., 0.], [0., -1.]]  # det = -1
    assert stability_analysis(J) == st.SADDLE_NODE


def test_2d_center_manifold_and_unstable_line():
    # q == 0 (det zero). p = trace.
    # p <= 0 -> center manifold
    J = [[-1., 0.], [0., 0.]]  # det 0, trace -1
    assert stability_analysis(J) == st.CENTER_MANIFOLD
    # p > 0 -> unstable line
    J = [[1., 0.], [0., 0.]]  # det 0, trace 1
    assert stability_analysis(J) == st.UNSTABLE_LINE_2D


def test_2d_center():
    # q > 0 and p == 0 -> center
    J = [[0., -1.], [1., 0.]]  # trace 0, det 1
    assert stability_analysis(J) == st.CENTER_2D


def test_2d_unstable_focus_and_node():
    # p > 0, e = p*p - 4q < 0 -> unstable focus
    J = [[1., -1.], [1., 1.]]  # trace 2, det 2, e = 4 - 8 < 0
    assert stability_analysis(J) == st.UNSTABLE_FOCUS_2D
    # p > 0, e > 0 -> unstable node
    J = [[3., 0.], [0., 1.]]  # trace 4, det 3, e = 16 - 12 > 0
    assert stability_analysis(J) == st.UNSTABLE_NODE_2D


def test_2d_unstable_degenerate_and_star():
    # p > 0, e == 0. Distinct construction so eigenvalues equal -> degenerate.
    J = [[2., 0.], [0., 2.]]  # trace 4, det 4, e = 16 - 16 = 0; eigvals both 2
    assert stability_analysis(J) == st.UNSTABLE_DEGENERATE_2D
    # p > 0, e == 0 but eigvals not literally equal (numerical) -> star branch.
    # Use a Jordan-like block so np.linalg.eigvals returns slightly different.
    J = [[2., 1.], [0., 2.]]  # trace 4, det 4, e = 0; defective matrix
    res = stability_analysis(J)
    assert res in (st.UNSTABLE_DEGENERATE_2D, st.UNSTABLE_STAR_2D)


def test_2d_stable_focus_and_node():
    # p < 0, e < 0 -> stable focus
    J = [[-1., -1.], [1., -1.]]  # trace -2, det 2, e = 4 - 8 < 0
    assert stability_analysis(J) == st.STABLE_FOCUS_2D
    # p < 0, e > 0 -> stable node
    J = [[-3., 0.], [0., -1.]]  # trace -4, det 3, e = 16 - 12 > 0
    assert stability_analysis(J) == st.STABLE_NODE_2D


def test_2d_stable_degenerate_and_star():
    # p < 0, e == 0 -> stable degenerate (eigvals equal)
    J = [[-2., 0.], [0., -2.]]  # trace -4, det 4, e = 0
    assert stability_analysis(J) == st.STABLE_DEGENERATE_2D
    # defective -> degenerate or star
    J = [[-2., 1.], [0., -2.]]
    res = stability_analysis(J)
    assert res in (st.STABLE_DEGENERATE_2D, st.STABLE_STAR_2D)


# --------------------------------------------------------------------------- #
# 3D classification (all-real eigenvalues)
# --------------------------------------------------------------------------- #
def _diag3(a, b, c):
    return [[a, 0., 0.], [0., b, 0.], [0., 0., c]]


def test_3d_real_stable_node():
    # all eigenvalues < 0 -> stable node (sorted[2] < 0)
    assert stability_analysis(_diag3(-1., -2., -3.)) == st.STABLE_NODE_3D


def test_3d_real_largest_zero_unknown():
    # sorted[2] == 0 -> unknown
    assert stability_analysis(_diag3(-1., -2., 0.)) == st.UNKNOWN_3D


def test_3d_real_unstable_node():
    # sorted[2] > 0 and sorted[0] > 0 -> unstable node
    assert stability_analysis(_diag3(1., 2., 3.)) == st.UNSTABLE_NODE_3D


def test_3d_real_smallest_zero_unknown():
    # sorted[2] > 0, sorted[0] == 0 -> unknown
    assert stability_analysis(_diag3(0., 1., 2.)) == st.UNKNOWN_3D


def test_3d_real_saddle_node():
    # sorted[2] > 0, sorted[0] < 0, sorted[1] < 0 -> saddle node
    assert stability_analysis(_diag3(-2., -1., 3.)) == st.SADDLE_NODE


def test_3d_real_middle_zero_unknown():
    # sorted[2] > 0, sorted[0] < 0, sorted[1] == 0 -> unknown
    assert stability_analysis(_diag3(-1., 0., 2.)) == st.UNKNOWN_3D


def test_3d_real_unstable_saddle():
    # sorted[2] > 0, sorted[0] < 0, sorted[1] > 0 -> unstable saddle
    assert stability_analysis(_diag3(-1., 1., 2.)) == st.UNSTABLE_SADDLE_3D


# --------------------------------------------------------------------------- #
# 3D classification (one real + complex conjugate pair)
# --------------------------------------------------------------------------- #
def _block3(real_eig, alpha, omega):
    """3x3 matrix with one real eigenvalue ``real_eig`` and a complex pair
    ``alpha +/- i*omega``."""
    return [[real_eig, 0., 0.],
            [0., alpha, -omega],
            [0., omega, alpha]]


def test_3d_complex_stable_focus():
    # v0 < 0, complex real part < 0 -> stable focus
    assert stability_analysis(_block3(-1., -0.5, 2.)) == st.STABLE_FOCUS_3D


def test_3d_complex_v0neg_zero_real_unknown():
    # v0 < 0, complex real part == 0 -> unknown
    assert stability_analysis(_block3(-1., 0., 2.)) == st.UNKNOWN_3D


def test_3d_complex_v0neg_unstable_focus():
    # v0 < 0, complex real part > 0 -> unstable focus
    assert stability_analysis(_block3(-1., 0.5, 2.)) == st.UNSTABLE_FOCUS_3D


def test_3d_complex_v0zero_unknown():
    # v0 == 0, complex real part <= 0 -> unknown
    assert stability_analysis(_block3(0., -0.5, 2.)) == st.UNKNOWN_3D


def test_3d_complex_v0zero_unstable_point():
    # v0 == 0, complex real part > 0 -> unstable point
    assert stability_analysis(_block3(0., 0.5, 2.)) == st.UNSTABLE_POINT_3D


def test_3d_complex_v0pos_unstable_focus():
    # v0 > 0, complex real part < 0 -> unstable focus
    assert stability_analysis(_block3(1., -0.5, 2.)) == st.UNSTABLE_FOCUS_3D


def test_3d_complex_v0pos_unstable_center():
    # v0 > 0, complex real part == 0 -> unstable center
    assert stability_analysis(_block3(1., 0., 2.)) == st.UNSTABLE_CENTER_3D


def test_3d_complex_v0pos_unstable_point():
    # v0 > 0, complex real part > 0 -> unstable point
    assert stability_analysis(_block3(1., 0.5, 2.)) == st.UNSTABLE_POINT_3D


# --------------------------------------------------------------------------- #
# error branch
# --------------------------------------------------------------------------- #
def test_unknown_derivative_shape_raises():
    with pytest.raises(ValueError):
        stability_analysis(np.ones((2, 3)))  # size 6, unsupported
