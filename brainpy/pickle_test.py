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
import pickle
import unittest

import brainpy as bp


class TestPickle(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPickle, self).__init__(*args, **kwargs)

        self.pre = bp.neurons.LIF(10)
        self.post = bp.neurons.LIF(20)
        self.syn = bp.synapses.TwoEndConn(self.pre, self.post, bp.conn.FixedProb(0.2))
        self.net = bp.DynSysGroup(self.pre, self.post, self.syn)

    def test_net(self):
        self.skipTest('Currently do not support')
        with open('data/net.pickle', 'wb') as f:
            pickle.dump(self.net, f)
