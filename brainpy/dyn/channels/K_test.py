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


class Test_K(parameterized.TestCase):
    bm.random.seed(1234)

    def test_K(self):
        class Neuron(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
                self.IK_1 = bp.dyn.IKDR_Ba2002(size)
                self.IK_2 = bp.dyn.IK_TM1991(size)
                self.IK_3 = bp.dyn.IK_HH1952(size)
                self.IK_4 = bp.dyn.IKA1_HM1992(size)
                self.IK_5 = bp.dyn.IKA2_HM1992(size)
                self.IK_6 = bp.dyn.IKK2A_HM1992(size)
                self.IK_7 = bp.dyn.IKK2B_HM1992(size)
                self.IK_8 = bp.dyn.IKNI_Ya1989(size)

        model = Neuron(1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'IK_1.p', 'IK_2.p', 'IK_3.p', 'IK_4.p', 'IK_5.p', 'IK_6.p', 'IK_7.p',
                                       'IK_8.p'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_1.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_2.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_3.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_4.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_5.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_6.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_7.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IK_8.p'].shape, (100, 1))
