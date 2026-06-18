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
from brainpy.dyn.neurons import hh


class Test_HH(parameterized.TestCase):
    def test_HH(self):
        model = hh.HH(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HH_batching_mode(self):
        model = hh.HH(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_HHLTC(self):
        model = hh.HHLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HHLTC_batching_mode(self):
        model = hh.HHLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_MorrisLecar(self):
        model = hh.MorrisLecar(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecar_batching_mode(self):
        model = hh.MorrisLecar(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_MorrisLecarLTC(self):
        model = hh.MorrisLecarLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecarLTC_batching_mode(self):
        model = hh.MorrisLecarLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_WangBuzsakiModel(self):
        model = hh.WangBuzsakiHH(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModel_batching_mode(self):
        model = hh.WangBuzsakiHH(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_WangBuzsakiModelLTC(self):
        model = hh.WangBuzsakiHHLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModelLTC_batching_mode(self):
        model = hh.WangBuzsakiHHLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))
