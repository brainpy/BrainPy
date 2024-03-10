# -*- coding: utf-8 -*-

import unittest

import brainpy as bp
import brainpy.math as bm


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

    bm.clear_buffer_memory()

  def test_receive_update_output(self):
    def aft_update(inp):
      assert inp is not None

    hh = bp.dyn.HH(1)
    hh.add_aft_update('aft_update', aft_update)
    bp.share.save(i=0, t=0.)
    hh(1.)

    bm.clear_buffer_memory()

  def test_do_not_receive_update_output(self):
    def aft_update():
      pass

    hh = bp.dyn.HH(1)
    hh.add_aft_update('aft_update', bp.not_receive_update_output(aft_update))
    bp.share.save(i=0, t=0.)
    hh(1.)

    bm.clear_buffer_memory()

  def test_not_receive_update_input(self):
    def bef_update():
      pass

    hh = bp.dyn.HH(1)
    hh.add_bef_update('bef_update', bef_update)
    bp.share.save(i=0, t=0.)
    hh(1.)

    bm.clear_buffer_memory()

  def test_receive_update_input(self):
    def bef_update(inp):
      assert inp is not None

    hh = bp.dyn.HH(1)
    hh.add_bef_update('bef_update', bp.receive_update_input(bef_update))
    bp.share.save(i=0, t=0.)
    hh(1.)

    bm.clear_buffer_memory()





