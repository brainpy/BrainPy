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

import brainpy as bp

block = False


class TestGLShortMemory(unittest.TestCase):
    def test_lorenz(self):
        a, b, c = 10, 28, 8 / 3

        def lorenz(x, y, z, t):
            dx = a * (y - x)
            dy = x * (b - z) - y
            dz = x * y - c * z
            return dx, dy, dz

        bp.math.random.seed()
        integral = bp.fde.GLShortMemory(lorenz,
                                        alpha=0.99,
                                        num_memory=500,
                                        inits=[1., 0., 1.])
        runner = bp.IntegratorRunner(integral,
                                     monitors=list('xyz'),
                                     inits=[1., 0., 1.],
                                     dt=0.005)
        runner.run(100.)

        plt.plot(runner.mon.x.flatten(), runner.mon.z.flatten())
        plt.show(block=block)
