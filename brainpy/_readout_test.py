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

import jax.numpy as jnp

import brainstate
import braintools
import brainpy


class TestReadoutModels(unittest.TestCase):
    def setUp(self):
        self.in_size = 3
        self.out_size = 3
        self.batch_size = 4
        self.tau = 5.0
        self.V_th = 1.0
        self.x = jnp.ones((self.batch_size, self.in_size))

    def test_LeakyRateReadout(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.LeakyRateReadout(in_size=self.in_size, out_size=self.out_size, tau=self.tau)
            model.init_state(batch_size=self.batch_size)
            output = model.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.out_size))

    def test_LeakySpikeReadout(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.LeakySpikeReadout(
                in_size=self.in_size, tau=self.tau, V_th=self.V_th,
                V_initializer=braintools.init.Constant(0. * u.mV),
                w_init=braintools.init.KaimingNormal()
            )
            model.init_state(batch_size=self.batch_size)
            with brainstate.environ.context(t=0.):
                output = model.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.out_size))


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1):
        unittest.main()
