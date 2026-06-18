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


class Test_delay_couplings(parameterized.TestCase):
    def test_DiffusiveCoupling(self):
        bm.random.seed()
        bm.set_dt(0.1)

        areas = bp.rates.FHN(80, x_ou_sigma=0.01, y_ou_sigma=0.01, name='fhn1')
        conn = bp.synapses.DiffusiveCoupling(areas.x, areas.x, areas.input,
                                             conn_mat=bp.conn.All2All(pre=areas.num, post=areas.num).require(
                                                 'conn_mat'),
                                             initial_delay_data=bp.init.Uniform(0, 0.05))
        net = bp.Network(areas, conn)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['fhn1.x'],
                             inputs=('fhn1.input', 35.))
        runner(10.)
        self.assertTupleEqual(runner.mon['fhn1.x'].shape, (100, 80))

    def test_AdditiveCoupling(self):
        bm.random.seed()
        bm.set_dt(0.1)

        areas = bp.rates.FHN(80, x_ou_sigma=0.01, y_ou_sigma=0.01, name='fhn2')
        conn = bp.synapses.AdditiveCoupling(areas.x, areas.input,
                                            conn_mat=bp.conn.All2All(pre=areas.num, post=areas.num).require('conn_mat'),
                                            initial_delay_data=bp.init.Uniform(0, 0.05))
        net = bp.Network(areas, conn)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['fhn2.x'],
                             inputs=('fhn2.input', 35.))
        runner(10.)
        self.assertTupleEqual(runner.mon['fhn2.x'].shape, (100, 80))
