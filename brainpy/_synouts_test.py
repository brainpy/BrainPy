# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainstate


class TestSynOutModels(unittest.TestCase):
    def setUp(self):
        self.conductance = jnp.array([0.5, 1.0, 1.5])
        self.potential = jnp.array([-70.0, -65.0, -60.0])
        self.E = jnp.array([-70.0])
        self.alpha = jnp.array([0.062])
        self.beta = jnp.array([3.57])
        self.cc_Mg = jnp.array([1.2])
        self.V_offset = jnp.array([0.0])

    def test_COBA(self):
        model = brainstate.nn.COBA(E=self.E)
        output = model.update(self.conductance, self.potential)
        expected_output = self.conductance * (self.E - self.potential)
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_CUBA(self):
        model = brainstate.nn.CUBA()
        output = model.update(self.conductance)
        expected_output = self.conductance * model.scale
        self.assertTrue(u.math.allclose(output, expected_output))

    def test_MgBlock(self):
        model = brainstate.nn.MgBlock(E=self.E, cc_Mg=self.cc_Mg, alpha=self.alpha, beta=self.beta,
                                      V_offset=self.V_offset)
        output = model.update(self.conductance, self.potential)
        norm = (1 + self.cc_Mg / self.beta * jnp.exp(self.alpha * (self.V_offset - self.potential)))
        expected_output = self.conductance * (self.E - self.potential) / norm
        np.testing.assert_array_almost_equal(output, expected_output)


if __name__ == '__main__':
    unittest.main()
