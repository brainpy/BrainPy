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

import jax.numpy as jnp

import brainpy as bp


class TestVarDelay(unittest.TestCase):
    def test_delay1(self):
        bp.math.random.seed()
        a = bp.math.Variable((10, 20))
        delay = bp.VarDelay(a, )
        delay.register_entry('a', 1.)
        delay.register_entry('b', 2.)
        delay.register_entry('c', None)
        with self.assertRaises(KeyError):
            delay.register_entry('c', 10.)

    def test_rotation_delay(self):
        a = bp.math.Variable((1,))
        rotation_delay = bp.VarDelay(a)
        t0 = 0.
        t1, n1 = 1., 10
        t2, n2 = 2., 20

        rotation_delay.register_entry('a', t0)
        rotation_delay.register_entry('b', t1)
        rotation_delay.register_entry('c', t2)

        print()
        for i in range(100):
            bp.share.save(i=i)
            a.value = jnp.ones((1,)) * i
            rotation_delay()
            self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
            self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1 + 1, 0.)))
            self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2 + 1, 0.)))

    def test_concat_delay(self):
        a = bp.math.Variable((1,))
        rotation_delay = bp.VarDelay(a, method='concat')
        t0 = 0.
        t1, n1 = 1., 10
        t2, n2 = 2., 20

        rotation_delay.register_entry('a', t0)
        rotation_delay.register_entry('b', t1)
        rotation_delay.register_entry('c', t2)

        print()
        for i in range(100):
            bp.share.save(i=i)
            a.value = jnp.ones((1,)) * i
            rotation_delay()
            self.assertTrue(jnp.allclose(rotation_delay.at('a'), jnp.ones((1,)) * i))
            self.assertTrue(jnp.allclose(rotation_delay.at('b'), jnp.maximum(jnp.ones((1,)) * i - n1 + 1, 0.)))
            self.assertTrue(jnp.allclose(rotation_delay.at('c'), jnp.maximum(jnp.ones((1,)) * i - n2 + 1, 0.)))
