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
import unittest

import brainpy as bp


class TestDynamicalSystem(unittest.TestCase):
    def test_delay(self):
        A = bp.neurons.LIF(1)
        B = bp.neurons.LIF(1)
        C = bp.neurons.LIF(1)
        A2B = bp.synapses.Exponential(A, B, bp.conn.All2All(), delay_step=1)
        A2C = bp.synapses.Exponential(A, C, bp.conn.All2All(), delay_step=None)
        net = bp.Network(A, B, C, A2B, A2C)

        runner = bp.DSRunner(net, )
        runner.run(10.)

    def test_receive_update_output(self):
        def aft_update(inp):
            assert inp is not None

        hh = bp.dyn.HH(1)
        hh.add_aft_update('aft_update', aft_update)
        bp.share.save(i=0, t=0.)
        hh(1.)

    def test_do_not_receive_update_output(self):
        def aft_update():
            pass

        hh = bp.dyn.HH(1)
        hh.add_aft_update('aft_update', bp.not_receive_update_output(aft_update))
        bp.share.save(i=0, t=0.)
        hh(1.)

    def test_not_receive_update_input(self):
        def bef_update():
            pass

        hh = bp.dyn.HH(1)
        hh.add_bef_update('bef_update', bef_update)
        bp.share.save(i=0, t=0.)
        hh(1.)

    def test_receive_update_input(self):
        def bef_update(inp):
            assert inp is not None

        hh = bp.dyn.HH(1)
        hh.add_bef_update('bef_update', bp.receive_update_input(bef_update))
        bp.share.save(i=0, t=0.)
        hh(1.)
