# -*- coding: utf-8 -*-


from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class Test_Na(parameterized.TestCase):
    bm.random.seed(1234)

    def test_Na(self):
        class Neuron(bp.dyn.CondNeuGroup):
            def __init__(self, size):
                super(Neuron, self).__init__(size, V_initializer=bp.init.Uniform(-70, -50.))
                self.INa_1 = bp.dyn.INa_HH1952(size, E=50., g_max=120.)
                self.INa_2 = bp.dyn.INa_TM1991(size)
                self.INa_3 = bp.dyn.INa_Ba2002(size)

        model = Neuron(1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'INa_1.p', 'INa_1.q', 'INa_2.p', 'INa_2.q', 'INa_3.p', 'INa_3.q'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_1.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_1.q'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_2.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_2.q'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_3.p'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['INa_3.q'].shape, (100, 1))
