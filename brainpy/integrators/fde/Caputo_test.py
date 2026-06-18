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

import numpy as np

import brainpy as bp
import brainpy.math as bm


def _scalar(x):
    return float(np.asarray(bm.as_numpy(x)).reshape(-1)[0])


class TestCaputoEulerReset(unittest.TestCase):
    def test_reset_preserves_variable(self):
        """``reset`` must keep the memory buffer a ``bm.Variable`` (P7-H1)."""
        intg = bp.fde.CaputoEuler(lambda y, t: -y, alpha=0.8, num_memory=20, inits=[1.])
        self.assertIsInstance(intg.f_states['y'], bm.Variable)
        intg.reset([2.])
        self.assertIsInstance(intg.f_states['y'], bm.Variable)

    def test_reset_then_run_matches_fresh(self):
        """A reset+run must reproduce a fresh-integrator run (P7-H1).

        Before the fix, ``reset`` orphaned the registered ``Variable`` so the
        ``IntegratorRunner`` snapshotted a stale buffer and produced wrong values.
        """
        bm.enable_x64()
        try:
            def f(y, t):
                return -y

            # fresh reference run
            fresh = bp.fde.CaputoEuler(f, alpha=0.8, num_memory=100, inits=[1.])
            runner_fresh = bp.IntegratorRunner(fresh, monitors=['y'], dt=0.05, inits=[1.])
            runner_fresh.run(1.0)
            ref = _scalar(runner_fresh.mon.y[-1])

            # reset then run
            intg = bp.fde.CaputoEuler(f, alpha=0.8, num_memory=100, inits=[1.])
            intg.reset([1.])
            runner = bp.IntegratorRunner(intg, monitors=['y'], dt=0.05, inits=[1.])
            runner.run(1.0)
            got = _scalar(runner.mon.y[-1])

            self.assertTrue(np.allclose(ref, got, atol=1e-10),
                            msg=f'reset run {got} != fresh run {ref}')
        finally:
            bm.disable_x64()


class TestFdeintDefaultMethod(unittest.TestCase):
    def test_set_default_fdeint_respected(self):
        """``fdeint`` must honor ``set_default_fdeint`` when method is omitted (P7-M1)."""
        from brainpy.integrators.fde.generic import (
            fdeint, set_default_fdeint, get_default_fdeint
        )
        original = get_default_fdeint()
        try:
            set_default_fdeint('euler')
            intg = fdeint(alpha=0.8, num_memory=20, inits=[1.], f=lambda y, t: -y)
            self.assertIsInstance(intg, bp.fde.CaputoEuler)

            set_default_fdeint('l1')
            intg2 = fdeint(alpha=0.8, num_memory=20, inits=[1.], f=lambda y, t: -y)
            self.assertIsInstance(intg2, bp.fde.CaputoL1Schema)
        finally:
            set_default_fdeint(original)


class TestCaputoL1(unittest.TestCase):
    def test1(self):
        bp.math.random.seed()
        bp.math.enable_x64()
        alpha = 0.9
        intg = bp.fde.CaputoL1Schema(lambda a, t: a,
                                     alpha=alpha,
                                     num_memory=10,
                                     inits=[1., ])
        for N in [2, 3, 4, 5, 6, 7, 8]:
            diff = np.random.rand(N - 1, 1)
            memory_trace = 0
            for i in range(N - 1):
                c = (N - i) ** (1 - alpha) - (N - i - 1) ** (1 - alpha)
                memory_trace += c * diff[i]

            intg.idx[0] = N - 1
            intg.diff_states['a_diff'][:N - 1] = bp.math.asarray(diff)
            idx = ((intg.num_memory - intg.idx) + np.arange(intg.num_memory)) % intg.num_memory
            memory_trace2 = intg.coef[idx, 0] @ intg.diff_states['a_diff']

            print()
            print(memory_trace[0], )
            print(memory_trace2[0], bp.math.array_equal(memory_trace[0], memory_trace2[0]))

        bp.math.disable_x64()
