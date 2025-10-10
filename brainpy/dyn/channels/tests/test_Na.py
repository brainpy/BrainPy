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
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class Test_Na(parameterized.TestCase):
    bm.random.seed(1234)

    def test_Na(self):
        class Neuron(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
                self.INa_1 = bp.dyn.INa_HH1952(size, E=50., g_max=120.)
                self.INa_2 = bp.dyn.INa_TM1991(size)
                self.INa_3 = bp.dyn.INa_Ba2002(size)

        model = Neuron(1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'INa_1.p', 'INa_1.q', 'INa_2.p', 'INa_2.q', 'INa_3.p', 'INa_3.q'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_1.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_1.q'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_2.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_2.q'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_3.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_3.q'].shape, (100, 1))
