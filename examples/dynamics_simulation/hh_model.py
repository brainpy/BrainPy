# -*- coding: utf-8 -*-

import numpy as np

import brainpy as bp
from jax import pmap
import brainpy.math as bm

bm.set_host_device_count(20)


class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size):
    super().__init__(size)

    self.INa = bp.channels.INa_HH1952(size)
    self.IK = bp.channels.IK_HH1952(size)
    self.IL = bp.channels.IL(size, E=-54.387, g_max=0.03)


class HHLTC(bp.dyn.CondNeuGroupLTC):
  def __init__(self, size):
    super().__init__(size)

    self.INa = bp.channels.INa_HH1952(size)
    self.IK = bp.channels.IK_HH1952(size)
    self.IL = bp.channels.IL(size, E=-54.387, g_max=0.03)


class HHv2(bp.dyn.CondNeuGroupLTC):
  def __init__(self, size):
    super().__init__(size)

    self.Na = bp.dyn.SodiumFixed(size, E=50.)
    self.Na.add_elem(ina=bp.dyn.INa_HH1952v2(size))

    self.K = bp.dyn.PotassiumFixed(size, E=50.)
    self.K.add_elem(ik=bp.dyn.IK_HH1952v2(size))

    self.IL = bp.dyn.IL(size, E=-54.387, g_max=0.03)

    self.KNa = bp.dyn.MixIons(self.Na, self.K)
    self.KNa.add_elem()


# hh = HH(1)
# I, length = bp.inputs.section_input(values=[0, 5, 0],
#                                     durations=[100, 500, 100],
#                                     return_length=True)
# runner = bp.DSRunner(
#   hh,
#   monitors=['V', 'INa.p', 'INa.q', 'IK.p'],
#   inputs=[hh.input, I, 'iter'],
# )
# runner.run(length)
#
# bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)

