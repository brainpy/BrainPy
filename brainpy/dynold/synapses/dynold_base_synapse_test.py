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


class Test_TwoEndConnAlignPre(unittest.TestCase):
    def test1(self):
        E = bp.neurons.HH(size=4)
        syn = bp.synapses.AMPA(E, E, bp.conn.All2All(include_self=False))
        self.assertTrue(syn.conn.include_self == syn.comm.include_self)
