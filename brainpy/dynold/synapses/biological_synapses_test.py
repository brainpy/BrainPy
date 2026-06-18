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

biological_models = [
    bp.synapses.AMPA,
    bp.synapses.GABAa,
    bp.synapses.BioNMDA,
]


class Test_Biological_Synapse(parameterized.TestCase):
    @parameterized.product(
        synapse=biological_models,
        delay_step=[None, 5, 1],
        mode=[bm.NonBatchingMode(), bm.BatchingMode(5)],
        stp=[None, bp.synplast.STP(), bp.synplast.STD()]
    )
    def test_all2all_synapse(self, synapse, delay_step, mode, stp):
        bm.random.seed()
        with bm.environment(mode=mode):
            pre_neu = bp.neurons.LIF(5)
            post_neu = bp.neurons.LIF(5)
            syn = synapse(pre_neu, post_neu, conn=bp.conn.All2All(), delay_step=delay_step, stp=stp)
            net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['pre.V', 'syn.g', 'post.V'],
                             inputs=('pre.input', 35.))
        runner(10.)

        expected_shape = (100, 5)
        if isinstance(mode, bm.BatchingMode):
            expected_shape = (mode.batch_size,) + expected_shape

        self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)

    @parameterized.product(
        synapse=biological_models,
        delay_step=[None, 10, 1],
        mode=[bm.NonBatchingMode(), bm.BatchingMode(5), ],
        stp=[None, bp.synplast.STP(), bp.synplast.STD()]
    )
    def test_one2one_synapse(self, synapse, delay_step, mode, stp):
        bm.random.seed()
        with bm.environment(mode=mode):
            pre_neu = bp.neurons.LIF(5)
            post_neu = bp.neurons.LIF(5)
            syn = synapse(pre_neu, post_neu, conn=bp.conn.One2One(), delay_step=delay_step, stp=stp)
            net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['pre.V', 'syn.g', 'post.V'],
                             inputs=('pre.input', 35.))
        runner(10.)

        expected_shape = (100, 5)
        if isinstance(mode, bm.BatchingMode):
            expected_shape = (mode.batch_size,) + expected_shape
        self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)

    @parameterized.product(
        synapse=biological_models,
        comp_method=['sparse', 'dense'],
        delay_step=[None, 10, 1],
        mode=[bm.NonBatchingMode(), bm.BatchingMode(5)],
        stp=[None, bp.synplast.STP(), bp.synplast.STD()]
    )
    def test_sparse_synapse(self, synapse, comp_method, delay_step, mode, stp):
        bm.random.seed()
        with bm.environment(mode=mode):
            pre_neu = bp.neurons.LIF(10)
            post_neu = bp.neurons.LIF(10)
            syn = synapse(pre_neu, post_neu, conn=bp.conn.FixedProb(0.5),
                          comp_method=comp_method, delay_step=delay_step,
                          stp=stp)
            net = bp.Network(pre=pre_neu, syn=syn, post=post_neu)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['pre.V', 'syn.g', 'post.V'],
                             inputs=('pre.input', 35.))
        runner(10.)

        expected_shape = (100, 10)
        if isinstance(mode, bm.BatchingMode):
            expected_shape = (mode.batch_size,) + expected_shape
        self.assertTupleEqual(runner.mon['pre.V'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['syn.g'].shape, expected_shape)
        self.assertTupleEqual(runner.mon['post.V'].shape, expected_shape)
