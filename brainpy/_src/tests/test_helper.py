import brainpy as bp

import unittest


class TestResetLevel(unittest.TestCase):

  def test1(self):
    class Level0(bp.DynamicalSystem):
      @bp.reset_level(0)
      def reset_state(self, *args, **kwargs):
        print('Level 0')

    class Level1(bp.DynamicalSystem):
      @bp.reset_level(1)
      def reset_state(self, *args, **kwargs):
        print('Level 1')

    class Net(bp.DynamicalSystem):
      def __init__(self):
        super().__init__()
        self.l0 = Level0()
        self.l1 = Level1()
        self.l0_2 = Level0()
        self.l1_2 = Level1()

    net = Net()
    net.reset()


