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

import brainpy as bp
import brainpy.math as bm


class TestMixIons(unittest.TestCase):
    def test_init(self):
        class HH(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super().__init__(size)

                self.k = bp.dyn.PotassiumFixed(size)
                self.ca = bp.dyn.CalciumFirstOrder(size)

                self.kca = bp.dyn.mix_ions(self.k, self.ca)
                self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))

        bm.random.seed()
        HH(10)

    def test_init2(self):
        class HH(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super().__init__(size)

                self.k = bp.dyn.PotassiumFixed(size)
                self.ca = bp.dyn.CalciumFirstOrder(size)

                self.kca = bp.dyn.mix_ions(self.k, self.ca)
                self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))
                self.kca.add_elem(na=bp.dyn.INa_Ba2002(size))

        bm.random.seed()
        with self.assertRaises(TypeError):
            HH(10)

    def test_init3(self):
        class HH(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super().__init__(size)

                self.na = bp.dyn.SodiumFixed(size)
                self.ca = bp.dyn.CalciumFirstOrder(size)

                self.kca = bp.dyn.mix_ions(self.na, self.ca)
                self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))
                self.kca.add_elem(na=bp.dyn.INa_Ba2002(size))

        bm.random.seed()
        with self.assertRaises(TypeError):
            HH(10)

    def test_init4(self):
        class HH(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super().__init__(size)

                self.na = bp.dyn.SodiumFixed(size)
                self.k = bp.dyn.PotassiumFixed(size)
                self.ca = bp.dyn.CalciumFirstOrder(size)

                self.kca = bp.dyn.mix_ions(self.na, self.k, self.ca)
                self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))

        bm.random.seed()
        HH(10)


class TestMixIons2(unittest.TestCase):
    def test_current1(self):
        class HH(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super().__init__(size)

                self.k = bp.dyn.PotassiumFixed(size)
                self.na = bp.dyn.SodiumFixed(size)
                self.ca = bp.dyn.CalciumFirstOrder(size)
                self.kca = bp.dyn.MixIons(self.na, self.k, self.ca)

                self.kca.add_elem(ahp=bp.dyn.IAHP_De1994v2(size))

        bm.random.seed()
        hh = HH(10)

        hh.reset_state()

        ICa = hh.ca.current(hh.V, external=True)
        INa = hh.na.current(hh.V, external=True)
        IK = hh.k.current(hh.V, external=True)
        print(ICa, INa, IK)

        self.assertTrue(bm.allclose(INa, 0.))
        self.assertTrue(bm.allclose(ICa, IK))
