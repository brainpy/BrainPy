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


class TestParamDesc(unittest.TestCase):
    def test1(self):
        a = bp.dyn.Expon(1)
        self.assertTrue(not isinstance(a, bp.mixin.ParamDescriber[bp.dyn.Expon]))
        self.assertTrue(not isinstance(a, bp.mixin.ParamDescriber[bp.DynamicalSystem]))

    def test2(self):
        a = bp.dyn.Expon.desc(1)
        self.assertTrue(isinstance(a, bp.mixin.ParamDescriber[bp.dyn.Expon]))
        self.assertTrue(isinstance(a, bp.mixin.ParamDescriber[bp.DynamicalSystem]))


class TestJointType(unittest.TestCase):
    def test1(self):
        T = bp.mixin.JointType[bp.DynamicalSystem]
        self.assertTrue(isinstance(bp.dnn.Layer(), T))

        T = bp.mixin.JointType[bp.DynamicalSystem, bp.mixin.ParamDesc]
        self.assertTrue(isinstance(bp.dyn.Expon(1), T))

    def test2(self):
        T = bp.mixin.JointType[bp.DynamicalSystem, bp.mixin.ParamDesc]
        self.assertTrue(not isinstance(bp.dyn.Expon(1), bp.mixin.ParamDescriber[T]))
        self.assertTrue(isinstance(bp.dyn.Expon.desc(1), bp.mixin.ParamDescriber[T]))


class TestDelayRegister(unittest.TestCase):
    # def test11(self):
    #   lif = bp.dyn.Lif(10)
    #   with self.assertWarns(UserWarning):
    #     lif.register_delay('pre.spike', 10, lif.spike)
    #
    #   with self.assertWarns(UserWarning):
    #     lif.get_delay_data('pre.spike', 10)

    def test2(self):
        bp.share.save(i=0)
        lif = bp.dyn.Lif(10)
        lif.register_local_delay('spike', 'a', delay_time=10.)
        data = lif.get_local_delay('spike', 'a')
        self.assertTrue(bm.allclose(data, bm.zeros(10)))

        with self.assertRaises(AttributeError):
            lif.register_local_delay('a', 'a', 10.)
