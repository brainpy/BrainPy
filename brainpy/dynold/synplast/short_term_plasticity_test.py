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
"""Tests for ``brainpy.dynold.synplast.short_term_plasticity`` (STD / STP).

These STP components are attached as the ``stp=`` slot of a ``TwoEndConn``
synapse; ``register_master`` allocates their state from the master's
pre-synaptic group. The regressions here pin P11-M1: the discrete
Tsodyks-Markram jumps must act on the value *at spike arrival* (the decayed
local), not the pre-decay state held over from the previous step.
"""

import unittest

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share


def _make_std(num=4, tau=200., U=0.07):
    """Build an STD bound to a real master synapse."""
    pre = bp.neurons.LIF(num)
    post = bp.neurons.LIF(num)
    syn = bp.synapses.Exponential(pre, post, bp.connect.One2One(),
                                  stp=bp.synplast.STD(tau=tau, U=U),
                                  comp_method='dense')
    return syn.stp


def _make_stp(num=4, U=0.15, tau_f=1500., tau_d=200.):
    pre = bp.neurons.LIF(num)
    post = bp.neurons.LIF(num)
    syn = bp.synapses.Exponential(pre, post, bp.connect.One2One(),
                                  stp=bp.synplast.STP(U=U, tau_f=tau_f, tau_d=tau_d),
                                  comp_method='dense')
    return syn.stp


class TestSTD(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)
        bm.set_dt(0.1)

    def test_first_spike_from_rest(self):
        std = _make_std(3, tau=200., U=0.07)
        share.save(t=0.0, dt=bm.dt)
        std.update(bm.ones(3, dtype=bool))
        # from rest x=1 -> x^+ = 1 - U = 0.93 (decay over one dt is negligible)
        np.testing.assert_allclose(bm.as_jax(std.x.value), np.full(3, 1 - 0.07), atol=2e-3)

    def test_jump_uses_decayed_state(self):
        # P11-M1: depress, let x recover for one step (no spike), then spike. The
        # depression must scale with the *decayed* x (= x^- at spike arrival),
        # i.e. x^+ = x_dec - U*x_dec, NOT x_dec - U*x_prev.
        U, tau, dt = 0.5, 50., bm.dt
        std = _make_std(1, tau=tau, U=U)
        share.save(t=0.0, dt=dt)
        std.update(bm.ones(1, dtype=bool))           # x drops to ~1-U
        x_prev = float(bm.as_jax(std.x.value)[0])
        share.save(t=float(dt), dt=dt)
        std.update(bm.ones(1, dtype=bool))           # recover one dt, then spike
        x_after = float(bm.as_jax(std.x.value)[0])

        # decayed value at spike arrival
        x_dec = x_prev + (1 - x_prev) / tau * float(dt)
        expected_correct = x_dec - U * x_dec
        expected_buggy = x_dec - U * x_prev
        self.assertAlmostEqual(x_after, expected_correct, places=5)
        # the two differ enough (recovery over dt) that the buggy form is rejected
        self.assertNotAlmostEqual(expected_correct, expected_buggy, places=7)


class TestSTP(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)
        bm.set_dt(0.1)

    def test_jump_uses_decayed_state(self):
        # P11-M1: u^+ = u^- + U(1-u^-) and x^+ = x^- - u^+ x^- must use the
        # decayed (current-time) locals, not the previous-step Variables.
        U, tau_f, tau_d, dt = 0.5, 100., 50., bm.dt
        stp = _make_stp(1, U=U, tau_f=tau_f, tau_d=tau_d)
        share.save(t=0.0, dt=dt)
        stp.update(bm.ones(1, dtype=bool))
        u_prev = float(bm.as_jax(stp.u.value)[0])
        x_prev = float(bm.as_jax(stp.x.value)[0])
        share.save(t=float(dt), dt=dt)
        stp.update(bm.ones(1, dtype=bool))
        u_after = float(bm.as_jax(stp.u.value)[0])
        x_after = float(bm.as_jax(stp.x.value)[0])

        # decayed locals at spike arrival (exp_auto integrates exactly here)
        u_dec = u_prev + (U - u_prev / tau_f) * float(dt)
        x_dec = x_prev + (1 - x_prev) / tau_d * float(dt)
        u_correct = u_dec + U * (1 - u_dec)
        x_correct = x_dec - u_correct * x_dec
        self.assertAlmostEqual(u_after, u_correct, places=4)
        self.assertAlmostEqual(x_after, x_correct, places=4)
        # buggy variants (using the pre-decay Variables) must be distinguishable
        u_buggy = u_dec + U * (1 - u_prev)
        self.assertNotAlmostEqual(u_correct, u_buggy, places=6)


if __name__ == '__main__':
    unittest.main()
