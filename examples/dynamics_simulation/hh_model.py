# -*- coding: utf-8 -*-
import numpy as np

import brainpy as bp
from jax import pmap
import brainpy.math as bm

bm.set_host_device_count(20)


class HH(bp.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size, keep_size=True)

    self.INa = bp.channels.INa_HH1952(size, keep_size=True)
    self.IK = bp.channels.IK_HH1952(size, keep_size=True)
    self.IL = bp.channels.IL(size, E=-54.387, g_max=0.03, keep_size=True)


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


hh = HH((20, 10000))
variables = hh.vars().unique()


iis = np.arange(1000000000)

def f(i):
  bp.share.save(i=i, t=i * bm.get_dt(), dt=bm.get_dt())
  hh(5.)


@pmap
def run(vars):
  for v, d in vars.items():
    variables[v]._value = d
  bm.for_loop(f, bm.arange(1000000000))
  print('Compiling End')
  return hh.spike


r = run(variables.dict())
print(r.shape)
