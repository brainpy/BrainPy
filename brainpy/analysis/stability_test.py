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
import numpy as np

from brainpy.analysis.stability import *


def test_d1():
    assert stability_analysis(1.) == UNSTABLE_POINT_1D
    assert stability_analysis(-1.) == STABLE_POINT_1D
    assert stability_analysis(0.) == SADDLE_NODE


# --------------------------------------------------------------------------- #
# 2D star (proper) vs degenerate (improper) node — regression for P13-C1.
#
# A *star* node has a full 2-D eigenspace, i.e. the Jacobian is a scalar
# multiple of the identity (b == c == 0 and a == d). A *degenerate*
# (improper) node has a repeated eigenvalue but is defective (single
# eigenvector), e.g. a Jordan block.
# --------------------------------------------------------------------------- #
def test_2d_stable_star_proper_node():
    # -I : repeated eigenvalue -1, full eigenspace -> stable STAR.
    J = np.array([[-1., 0.], [0., -1.]])
    assert stability_analysis(J) == STABLE_STAR_2D


def test_2d_stable_degenerate_defective():
    # Jordan block with repeated eigenvalue -1, defective -> stable DEGENERATE.
    J = np.array([[-1., 1.], [0., -1.]])
    assert stability_analysis(J) == STABLE_DEGENERATE_2D


def test_2d_unstable_star_proper_node():
    # 2*I : repeated eigenvalue +2, full eigenspace -> unstable STAR.
    J = np.array([[2., 0.], [0., 2.]])
    assert stability_analysis(J) == UNSTABLE_STAR_2D


def test_2d_unstable_degenerate_defective():
    # Jordan block with repeated eigenvalue +2, defective -> unstable DEGENERATE.
    J = np.array([[2., 1.], [0., 2.]])
    assert stability_analysis(J) == UNSTABLE_DEGENERATE_2D
