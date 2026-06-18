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
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

import brainpy as bp
from brainpy.integrators import sde

block = False
sigma = 10
beta = 8 / 3
rho = 28
p = 0.1


def lorenz_f(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def lorenz_g(x, y, z, t):
    return p * x, p * y, p * z


def lorenz_system(method, **kwargs):
    bp.math.random.seed()
    integral = bp.math.jit(method(f=lorenz_f,
                                  g=lorenz_g,
                                  show_code=True,
                                  dt=0.005,
                                  **kwargs))

    times = np.arange(0, 10, 0.01)
    mon1 = []
    mon2 = []
    mon3 = []
    x, y, z = 1, 1, 1
    for t in times:
        x, y, z = integral(x, y, z, t)
        mon1.append(x)
        mon2.append(y)
        mon3.append(z)
    mon1 = bp.math.array(mon1).to_numpy()
    mon2 = bp.math.array(mon2).to_numpy()
    mon3 = bp.math.array(mon3).to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(mon1, mon2, mon3)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')
    plt.show(block=block)
    plt.close(fig)


class TestScalarWienerIntegral(unittest.TestCase):
    def test_srk1w1_try1(self):
        lorenz_system(sde.SRK1W1)

    def test_srk1w1_try2(self):
        with pytest.raises(AssertionError):
            lorenz_system(sde.SRK1W1, wiener_type=bp.integrators.VECTOR_WIENER)

    def test_srk2w1(self):
        lorenz_system(sde.SRK2W1)

    def test_euler(self):
        lorenz_system(sde.Euler, intg_type=bp.integrators.ITO_SDE)
        lorenz_system(sde.Euler, intg_type=bp.integrators.STRA_SDE)

    def test_milstein(self):
        lorenz_system(sde.MilsteinGradFree, intg_type=bp.integrators.ITO_SDE)
        lorenz_system(sde.MilsteinGradFree, intg_type=bp.integrators.STRA_SDE)
