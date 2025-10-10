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


class Test_IH(parameterized.TestCase):
    bm.random.seed(1234)

    def test_IH(self):
        class Neuron(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super(Neuron, self).__init__(size)
                self.IH = bp.dyn.Ih_HM1992(size)
                self.Ca = bp.dyn.CalciumDetailed(size, IH=bp.dyn.Ih_De1996(size))

        model = Neuron(1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'IH.p', 'Ca.IH.O'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['IH.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['Ca.IH.O'].shape, (100, 1))
