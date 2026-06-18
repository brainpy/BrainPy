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
from unittest import TestCase

import numpy as np

import brainpy as bp

plt = None
block = False


def show(current, duration, title=''):
    global plt
    if plt is None:
        import matplotlib.pyplot as plt
    ts = np.arange(0, duration, bp.math.get_dt())
    plt.plot(ts, current)
    plt.title(title)
    plt.xlabel('Time [ms]')
    plt.ylabel('Current Value')
    plt.show(block=block)


class TestCurrents(TestCase):

    def test_section_input(self):
        current1, duration = bp.inputs.section_input(values=[0, 1., 0.],
                                                     durations=[100, 300, 100],
                                                     return_length=True,
                                                     dt=0.1)
        show(current1, duration, 'values=[0, 1, 0], durations=[100, 300, 100]')

    def test_constant_input(self):
        current2, duration = bp.inputs.constant_input([(0, 100), (1, 300), (0, 100)])
        show(current2, duration, '[(0, 100), (1, 300), (0, 100)]')

    def test_spike_input(self):
        current3 = bp.inputs.spike_input(
            sp_times=[10, 20, 30, 200, 300],
            sp_lens=1.,  # can be a list to specify the spike length at each point
            sp_sizes=0.5,  # can be a list to specify the spike current size at each point
            duration=400.)

        show(current3, 400, 'Spike Input Example')

    def test_ramp_input(self):
        duration = 500
        current4 = bp.inputs.ramp_input(0, 1, duration)

        show(current4, duration, r'$c_{start}$=0, $c_{end}$=%d, duration, '
                                 r'$t_{start}$=0, $t_{end}$=None' % (duration))

    def test_ramp_input2(self):
        duration, t_start, t_end = 500, 100, 400
        current5 = bp.inputs.ramp_input(0, 1, duration, t_start, t_end)

        show(current5, duration, r'$c_{start}$=0, $c_{end}$=1, duration=%d, '
                                 r'$t_{start}$=%d, $t_{end}$=%d' % (duration, t_start, t_end))

    def test_wiener_process(self):
        duration = 200
        current6 = bp.inputs.wiener_process(duration, n=2, t_start=10., t_end=180.)
        show(current6, duration, 'Wiener Process')

    def test_ou_process(self):
        duration = 200
        current7 = bp.inputs.ou_process(mean=1., sigma=0.1, tau=10., duration=duration, n=2, t_start=10., t_end=180.)
        show(current7, duration, 'Ornstein-Uhlenbeck Process')

    # def test_sinusoidal_input(self):
    #     duration = 2000 * u.ms
    #     current8 = bp.inputs.sinusoidal_input(amplitude=1., frequency=2.0 * u.Hz,
    #                                           duration=duration, t_start=100. * u.ms, dt=0.1 * u.ms)
    #     show(current8, duration, 'Sinusoidal Input')
    #
    # def test_square_input(self):
    #     duration = 2000 * u.ms
    #     current9 = bp.inputs.square_input(amplitude=1., frequency=2.0 * u.Hz,
    #                                       duration=duration, t_start=100 * u.ms, dt=0.1 * u.ms)
    #     show(current9, duration, 'Square Input')

    def test_general1(self):
        I1 = bp.inputs.section_input(values=[0, 1, 2], durations=[10, 20, 30], dt=0.1)
        I2 = bp.inputs.section_input(values=[0, 1, 2], durations=[10, 20, 30], dt=0.01)
        self.assertTrue(I1.shape[0] == 600)
        self.assertTrue(I2.shape[0] == 6000)

    def test_general2(self):
        bp.math.random.seed(123)
        current = bp.inputs.section_input(values=[0, bp.math.ones(10),
                                                  bp.math.random.random((3, 10))],
                                          durations=[100, 300, 100])
        self.assertTrue(current.shape == (5000, 3, 10))
