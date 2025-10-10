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

import jax.numpy
import matplotlib.pyplot as plt
import pytest
from absl.testing import parameterized

import brainpy.math as bm
from brainpy.optim import scheduler

show = False

pytest.skip('Skip the test for now', allow_module_level=True)


class TestMultiStepLR(parameterized.TestCase):
    @parameterized.product(
        last_epoch=[-1, 0, 5, 10]
    )
    def test2(self, last_epoch):
        bm.random.seed()
        scheduler1 = scheduler.MultiStepLR(0.1, [10, 20], gamma=0.1, last_epoch=last_epoch)
        scheduler2 = scheduler.MultiStepLR(0.1, [10, 20], gamma=0.1, last_epoch=last_epoch)

        for i in range(1, 25):
            lr1 = scheduler1(i + last_epoch)
            lr2 = scheduler2()
            scheduler2.step_epoch()
            print(f'{scheduler2.last_epoch}, {lr1:.4f}, {lr2:.4f}')
            self.assertTrue(lr1 == lr2)


class TestStepLR(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': f'last_epoch={last_epoch}',
         'last_epoch': last_epoch}
        for last_epoch in [-1, 0, 5, 10]
    )
    def test1(self, last_epoch):
        bm.random.seed()
        scheduler1 = scheduler.StepLR(0.1, 10, gamma=0.1, last_epoch=last_epoch)
        scheduler2 = scheduler.StepLR(0.1, 10, gamma=0.1, last_epoch=last_epoch)
        for i in range(1, 25):
            lr1 = scheduler1(i + last_epoch)
            lr2 = scheduler2()
            scheduler2.step_epoch()
            print(f'{scheduler2.last_epoch}, {lr1:.4f}, {lr2:.4f}')
            self.assertTrue(lr1 == lr2)


class TestCosineAnnealingLR(unittest.TestCase):
    def test1(self):
        bm.random.seed()
        max_epoch = 50
        iters = 200
        sch = scheduler.CosineAnnealingLR(0.1, T_max=5, eta_min=0, last_epoch=-1)
        all_lr1 = [[], []]
        all_lr2 = [[], []]
        for epoch in range(max_epoch):
            for batch in range(iters):
                all_lr1[0].append(epoch + batch / iters)
                all_lr1[1].append(sch())
                sch.step_epoch()
            all_lr2[0].append(epoch)
            all_lr2[1].append(sch())
            sch.step_epoch()

        if show:
            plt.subplot(211)
            plt.plot(jax.numpy.asarray(all_lr1[0]), jax.numpy.asarray(all_lr1[1]))
            plt.subplot(212)
            plt.plot(jax.numpy.asarray(all_lr2[0]), jax.numpy.asarray(all_lr2[1]))
            plt.show()
            plt.close()


class TestCosineAnnealingWarmRestarts(unittest.TestCase):
    def test1(self):
        bm.random.seed()
        max_epoch = 50
        iters = 200
        sch = scheduler.CosineAnnealingWarmRestarts(0.1,
                                                    iters,
                                                    T_0=5,
                                                    T_mult=1,
                                                    last_call=-1)
        all_lr1 = []
        all_lr2 = []
        for epoch in range(max_epoch):
            for batch in range(iters):
                all_lr1.append(sch())
                sch.step_call()
            all_lr2.append(sch())
            sch.step_epoch()

        if show:
            plt.subplot(211)
            plt.plot(jax.numpy.asarray(all_lr1))
            plt.subplot(212)
            plt.plot(jax.numpy.asarray(all_lr2))
            plt.show()
            plt.close()
