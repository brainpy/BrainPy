# -*- coding: utf-8 -*-

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

    runner = bp.DSRunner(net,)
    runner.run(10.)


