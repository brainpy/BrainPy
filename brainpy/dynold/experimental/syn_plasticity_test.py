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
"""Coverage tests for ``brainpy.dynold.experimental.syn_plasticity``.

Exercises the experimental short-term plasticity components ``STD`` (fully
functional) and ``STP``.

.. note::

    ``STP`` is currently **unconstructable**: its ``reset_state`` calls
    ``variable_(jnp.ones, batch_size, self.num)`` with the ``batch_or_mode``
    and ``sizes`` arguments swapped relative to ``STD``. When ``__init__``
    calls ``reset_state(self.mode)`` the ``Mode`` object lands in the
    ``sizes`` slot and ``to_size`` raises ``ValueError: Cannot make a size
    for NonBatchingMode``. The DEFECT is pinned in
    ``TestSTP.test_stp_construction_is_broken`` below; the rest of STP's
    behaviour (the ``du``/``dx`` ODE RHS and ``update``) is exercised through
    a manually corrected instance.
"""

import unittest

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy import tools
from brainpy.context import share
from brainpy.dynold.experimental import syn_plasticity as sp
from brainpy.dynold.experimental.base import SynSTPNS
from brainpy.initialize import variable_, OneInit, parameter
from brainpy.integrators import odeint, JointEq


class TestSTD(unittest.TestCase):
    def setUp(self):
        bm.random.seed(123)
        bm.set_dt(0.1)

    def test_construction_and_reset(self):
        std = sp.STD(4, tau=200., U=0.07)
        self.assertEqual(std.num, 4)
        self.assertEqual(std.pre_size, (4,))
        # x is initialized to ones
        np.testing.assert_allclose(bm.as_jax(std.x.value), np.ones(4))

    def test_update_no_spike_recovers_toward_one(self):
        # With x initialised at 1 and no spike, dx = (1-x)/tau = 0, so x stays 1.
        std = sp.STD(3, tau=200., U=0.07)
        share.save(t=0.0, dt=bm.dt)
        r = std.update(bm.zeros(3, dtype=bool))
        np.testing.assert_allclose(bm.as_jax(r), np.ones(3), rtol=1e-5)

    def test_update_spike_depresses(self):
        # On a spike, x <- x - U * x, so the resource fraction drops by U.
        std = sp.STD(3, tau=200., U=0.07)
        share.save(t=0.0, dt=bm.dt)
        r = bm.as_jax(std.update(bm.ones(3, dtype=bool)))
        # x integrated forward (still ~1) then reduced by U*x_old (=0.07)
        self.assertTrue(np.all(r < 1.0))
        np.testing.assert_allclose(r, np.full(3, 1.0 - 0.07), atol=2e-3)

    def test_scalar_pre_size(self):
        std = sp.STD(5)
        self.assertEqual(std.num, 5)

    def test_named(self):
        std = sp.STD(2, name='my_std')
        self.assertEqual(std.name, 'my_std')


class TestSTP(unittest.TestCase):
    def setUp(self):
        bm.random.seed(123)
        bm.set_dt(0.1)

    def test_stp_construction_is_broken(self):
        # NOTE: DEFECT -- STP.reset_state has swapped (batch_or_mode, sizes)
        # arguments to variable_, so constructing STP with the default
        # NonBatchingMode raises ValueError ("Cannot make a size for ...Mode").
        # STD.reset_state uses the correct order; STP should mirror it.
        with self.assertRaises(ValueError):
            sp.STP(4)

    def _make_stp(self, num=4, U=0.15, tau_f=1500., tau_d=200.):
        """Build a working STP instance, working around the reset_state defect."""
        stp = sp.STP.__new__(sp.STP)
        SynSTPNS.__init__(stp)
        stp.pre_size = tools.to_size(num)
        stp.num = tools.size2num(stp.pre_size)
        stp.tau_f = parameter(tau_f, stp.num)
        stp.tau_d = parameter(tau_d, stp.num)
        stp.U = parameter(U, stp.num)
        stp.method = 'exp_auto'
        stp.integral = odeint(JointEq([stp.du, stp.dx]), method=stp.method)
        # correct argument order (mirrors STD.reset_state)
        stp.x = variable_(jnp.ones, stp.num, None)
        stp.u = variable_(OneInit(stp.U), stp.num, None)
        return stp

    def test_du_dx_rhs(self):
        stp = self._make_stp()
        # du = U - u/tau_f ; dx = (1-x)/tau_d
        u = bm.ones(4)
        du = bm.as_jax(stp.du(u, 0.))
        np.testing.assert_allclose(du, 0.15 - bm.as_jax(u) / 1500.)
        x = bm.full(4, 0.5)
        dx = bm.as_jax(stp.dx(x, 0.))
        np.testing.assert_allclose(dx, (1 - bm.as_jax(x)) / 200.)

    def test_update_no_spike(self):
        stp = self._make_stp()
        share.save(t=0.0, dt=bm.dt)
        r = bm.as_jax(stp.update(bm.zeros(4, dtype=bool)))
        # returns x * u ; with x=1, u=U initially and no spike, ~ U
        self.assertEqual(r.shape, (4,))
        self.assertTrue(np.all(r > 0))

    def test_update_with_spike_changes_state(self):
        stp = self._make_stp()
        share.save(t=0.0, dt=bm.dt)
        x_before = bm.as_jax(stp.x.value).copy()
        u_before = bm.as_jax(stp.u.value).copy()
        stp.update(bm.ones(4, dtype=bool))
        # a spike facilitates u (u increases) and depresses x (x decreases)
        self.assertTrue(np.all(bm.as_jax(stp.u.value) >= u_before - 1e-6))
        self.assertTrue(np.all(bm.as_jax(stp.x.value) <= x_before + 1e-6))


if __name__ == '__main__':
    unittest.main()
