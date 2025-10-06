# -*- coding: utf-8 -*-
# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import brainpy.version2 as bp


class TestFiringRate(unittest.TestCase):
    def test_fr1(self):
        spikes = bp.math.ones((1000, 10))
        print(bp.measure.firing_rate(spikes, 1.))

    def test_fr2(self):
        bp.math.random.seed()
        spikes = bp.math.random.random((1000, 10)) < 0.2
        print(bp.measure.firing_rate(spikes, 1.))
        print(bp.measure.firing_rate(spikes, 10.))

    def test_fr3(self):
        bp.math.random.seed()
        spikes = bp.math.random.random((1000, 10)) < 0.02
        print(bp.measure.firing_rate(spikes, 1.))
        print(bp.measure.firing_rate(spikes, 5.))
