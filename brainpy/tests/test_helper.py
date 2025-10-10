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


class TestResetLevel(unittest.TestCase):

    def test1(self):
        class Level0(bp.DynamicalSystem):
            @bp.reset_level(0)
            def reset_state(self, *args, **kwargs):
                print('Level 0')

        class Level1(bp.DynamicalSystem):
            @bp.reset_level(1)
            def reset_state(self, *args, **kwargs):
                print('Level 1')

        class Net(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.l0 = Level0()
                self.l1 = Level1()
                self.l0_2 = Level0()
                self.l1_2 = Level1()

        net = Net()
        net.reset()
