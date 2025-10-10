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

import brainpy.math as bm


class TestDefaults(unittest.TestCase):
    def test_dt(self):
        with bm.environment(dt=1.0):
            self.assertEqual(bm.dt, 1.0)
            self.assertEqual(bm.get_dt(), 1.0)

    def test_bool(self):
        with bm.environment(bool_=bm.int32):
            self.assertTrue(bm.bool_ == bm.int32)
            self.assertTrue(bm.get_bool() == bm.int32)

    def test_int(self):
        with bm.environment(int_=bm.int32):
            self.assertTrue(bm.int == bm.int32)
            self.assertTrue(bm.get_int() == bm.int32)

    def test_float(self):
        with bm.environment(float_=bm.float32):
            self.assertTrue(bm.float_ == bm.float32)
            self.assertTrue(bm.get_float() == bm.float32)

    def test_complex(self):
        with bm.environment(complex_=bm.complex64):
            self.assertTrue(bm.complex_ == bm.complex64)
            self.assertTrue(bm.get_complex() == bm.complex64)

    def test_mode(self):
        mode = bm.TrainingMode()
        with bm.environment(mode=mode):
            self.assertTrue(bm.mode == mode)
            self.assertTrue(bm.get_mode() == mode)
