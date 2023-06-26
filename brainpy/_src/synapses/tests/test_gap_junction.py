# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from brainpy import rates
from absl.testing import parameterized
from brainpy._src.synapses import gap_junction


class Test_gap_junction(parameterized.TestCase):
  def test_gap_junction(self):
    neu = bp.neurons.HH(2, V_initializer=bp.init.Constant(-70.68))
    syn = gap_junction.GapJunction(neu, neu, conn=bp.connect.All2All(include_self=False))
    net = bp.Network(syn=syn, neu=neu)

    # 运行模拟
    runner = bp.DSRunner(net,
                         monitors=['neu.V'],
                         inputs=('neu.input', 35.))
    runner(10.)
    self.assertTupleEqual(runner.mon['neu.V'].shape, (100, 2))
