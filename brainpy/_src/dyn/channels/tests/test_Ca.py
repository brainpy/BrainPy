# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
from absl.testing import parameterized


class Test_Ca(parameterized.TestCase):
  def test_Ca(self):
    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca1 = bp.dyn.CalciumFixed(size)
        self.Ca2 = bp.dyn.CalciumDetailed(size)
        self.Ca3 = bp.dyn.CalciumFirstOrder(size)

    bm.random.seed(1234)
    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V', 'Ca2.C', 'Ca3.C'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca2.C'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca3.C'].shape, (100, 1))

  def test_ICaN_IS2008(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaN_IS2008(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V', 'Ca.ICa.p'],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))

  def test_ICaT_HM1992(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaT_HM1992(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V',
                                   'Ca.ICa.p',
                                   ],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))

  def test_ICaT_HP1992(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaT_HP1992(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V',
                                   'Ca.ICa.p',
                                   ],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))

  def test_ICaHT_HM1992(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaHT_HM1992(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V',
                                   'Ca.ICa.p',
                                   ],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))

  def test_ICaHT_Re1993(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaHT_Re1993(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V',
                                   'Ca.ICa.p',
                                   ],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))

  def test_ICaL_IS2008(self):
    bm.random.seed(1234)

    class Neuron(bp.dyn.CondNeuGroup):
      def __init__(self, size):
        super(Neuron, self).__init__(size)
        self.Ca = bp.dyn.CalciumDetailed(size,
                                         ICa=bp.dyn.ICaL_IS2008(size),
                                         )

    model = Neuron(1)
    runner = bp.DSRunner(model,
                         monitors=['V',
                                   'Ca.ICa.p',
                                   ],
                         progress_bar=False)
    runner.run(10.)
    self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
    self.assertTupleEqual(runner.mon['Ca.ICa.p'].shape, (100, 1))
