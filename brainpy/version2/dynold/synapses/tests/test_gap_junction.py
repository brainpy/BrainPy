# -*- coding: utf-8 -*-


from absl.testing import parameterized

import brainpy.version2 as bp
import brainpy.version2.math as bm
from brainpy.version2.dynold.synapses import gap_junction


class Test_gap_junction(parameterized.TestCase):
    def test_gap_junction(self):
        bm.random.seed()
        neu = bp.neurons.HH(2, V_initializer=bp.init.Constant(-70.68))
        syn = gap_junction.GapJunction(neu, neu, conn=bp.connect.All2All(include_self=False))
        net = bp.Network(syn=syn, neu=neu)

        # 运行模拟
        runner = bp.DSRunner(net,
                             monitors=['neu.V'],
                             inputs=('neu.input', 35.))
        runner(10.)
        self.assertTupleEqual(runner.mon['neu.V'].shape, (100, 2))
