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
"""Coverage tests for ``brainpy.dynold.experimental.syn_outs``.

Exercises the three experimental synaptic-output components (``COBA``,
``CUBA`` and ``MgBlock``): their construction, the numerical value produced
by ``update`` (the core math of each model), and ``reset_state`` (a no-op
inherited from ``SynOutNS``).
"""

import unittest

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.dynold.experimental import syn_outs


class TestCUBA(unittest.TestCase):
    def test_identity(self):
        out = syn_outs.CUBA()
        g = bm.asarray([1., 2., 3.])
        # CUBA is the identity on the conductance, ignoring the potential.
        r = out.update(g, potential=bm.asarray([-65., -65., -65.]))
        np.testing.assert_allclose(bm.as_jax(r), bm.as_jax(g))
        # potential defaults to None
        r2 = out.update(g)
        np.testing.assert_allclose(bm.as_jax(r2), bm.as_jax(g))

    def test_reset_state_noop(self):
        out = syn_outs.CUBA()
        # reset_state is inherited from SynOutNS and must be a no-op.
        self.assertIsNone(out.reset_state())
        self.assertIsNone(out.reset_state(batch_size=5))

    def test_named(self):
        out = syn_outs.CUBA(name='my_cuba')
        self.assertEqual(out.name, 'my_cuba')


class TestCOBA(unittest.TestCase):
    def test_conductance_based_current(self):
        E = 0.
        out = syn_outs.COBA(E=E)
        g = bm.asarray([1., 2.])
        v = bm.asarray([-65., -60.])
        r = bm.as_jax(out.update(g, v))
        expected = bm.as_jax(g) * (E - bm.as_jax(v))
        np.testing.assert_allclose(r, expected)

    def test_array_reversal_potential(self):
        E = bm.asarray([0., -80.])
        out = syn_outs.COBA(E=E)
        g = bm.asarray([1., 1.])
        v = bm.asarray([-50., -50.])
        r = bm.as_jax(out.update(g, v))
        expected = bm.as_jax(g) * (bm.as_jax(E) - bm.as_jax(v))
        np.testing.assert_allclose(r, expected)


class TestMgBlock(unittest.TestCase):
    def test_default_params(self):
        out = syn_outs.MgBlock()
        self.assertAlmostEqual(float(out.E), 0.)
        self.assertAlmostEqual(float(out.cc_Mg), 1.2)
        self.assertAlmostEqual(float(out.alpha), 0.062)
        self.assertAlmostEqual(float(out.beta), 3.57)

    def test_update_matches_formula(self):
        E, cc_Mg, alpha, beta = 0., 1.2, 0.062, 3.57
        out = syn_outs.MgBlock(E=E, cc_Mg=cc_Mg, alpha=alpha, beta=beta)
        g = bm.asarray([0.5, 1.0])
        v = bm.asarray([-65., -20.])
        r = bm.as_jax(out.update(g, v))
        vj = bm.as_jax(v)
        expected = bm.as_jax(g) * (E - vj) / (1 + cc_Mg / beta * jnp.exp(-alpha * vj))
        np.testing.assert_allclose(r, expected, rtol=1e-6)

    def test_depolarization_relieves_block(self):
        # The magnesium block is relieved as the membrane depolarizes, so the
        # transmitted current (E - V > 0 for V < E = 0) grows with V.
        out = syn_outs.MgBlock(E=0.)
        g = bm.asarray([1.])
        current_at_rest = bm.as_jax(out.update(g, bm.asarray([-70.])))[0]
        current_when_depol = bm.as_jax(out.update(g, bm.asarray([-10.])))[0]
        self.assertGreater(current_when_depol, current_at_rest)


if __name__ == '__main__':
    unittest.main()
