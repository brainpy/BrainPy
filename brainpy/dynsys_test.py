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


def test1():
    class A(bp.DynamicalSystem):
        def update(self, x=None):
            # print(tdi)
            print(x)

    A()({}, 10.)


def test2():
    class B(bp.DynamicalSystem):
        def update(self, tdi, x=None):
            print(tdi)
            print(x)

    B()({}, 10.)
    B()(10.)


def test3():
    class A(bp.DynamicalSystem):
        def update(self, x=None):
            # print(tdi)
            print('A:', x)

    class B(A):
        def update(self, tdi, x=None):
            print('B:', tdi, x)
            super().update(x)

    B()(dict(), 1.)
    B()(1.)


class TestResetLevelDecorator(unittest.TestCase):
    _max_level = 10  # Define the maximum level for testing purposes

    @bp.reset_level(5)
    def test_function_with_reset_level_5(self):
        self.assertEqual(self.test_function_with_reset_level_5.reset_level, 5)

    def test1(self):
        with self.assertRaises(ValueError):
            @bp.reset_level(12)  # This should raise a ValueError
            def test_function_with_invalid_reset_level(self):
                pass  # Call the function here to trigger the ValueError

    @bp.reset_level(-3)
    def test_function_with_negative_reset_level(self):
        self.assertEqual(self.test_function_with_negative_reset_level.reset_level, self._max_level - 3)
