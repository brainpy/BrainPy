# -*- coding: utf-8 -*-

from unittest import TestCase

import brainpy as bp


class TestFF(TestCase):
  def test_one2one(self):
    i = bp.nn.Input(1)
    r = bp.nn.Reservoir(10)
    model = i >> r
    print(model.lnodes)
    self.assertTrue(model.ff_senders[r][0] == i)
    self.assertTrue(model.ff_receivers[i][0] == r)

  def test_many2one1(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)
    r = bp.nn.Reservoir(10)
    model = [i1, i2, i3] >> r
    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))

  def test_many2one2(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)
    r = bp.nn.Reservoir(10)
    model = (i1, i2, i3) >> r
    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))

  def test_many2one3(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)
    r = bp.nn.Reservoir(10)
    model = {i1, i2, i3} >> r
    self.assertTrue(model.ff_receivers[i1][0] == r)
    self.assertTrue(model.ff_receivers[i2][0] == r)
    self.assertTrue(model.ff_receivers[i3][0] == r)

  def test_one2many1(self):
    i = bp.nn.Input(1)
    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)
    with self.assertRaises(TypeError):
      model = i >> [o1, o2, o3]

  def test_one2many2(self):
    i = bp.nn.Input(1)
    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)
    with self.assertRaises(TypeError):
      model = i >> (o1, o2, o3)

  def test_one2many3(self):
    i = bp.nn.Input(1)
    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)
    model = i >> {o1, o2, o3}
    # model.plot_node_graph()
    self.assertTrue(model.ff_senders[o1][0] == i)
    self.assertTrue(model.ff_senders[o2][0] == i)
    self.assertTrue(model.ff_senders[o3][0] == i)

  def test_many2many1(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)

    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)

    model = bp.nn.ff_connect([i1, i2, i3], {o1, o2, o3})

    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))

    self.assertTrue(isinstance(model.ff_senders[o1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_senders[o2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_senders[o3][0], bp.nn.Concat))

  def test_many2many2(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)

    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)

    model = bp.nn.ff_connect((i1, i2, i3), {o1, o2, o3})

    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))

    self.assertTrue(isinstance(model.ff_senders[o1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_senders[o2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_senders[o3][0], bp.nn.Concat))

  def test_many2many3(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)

    o1 = bp.nn.Dense(3)
    o2 = bp.nn.Dense(4)
    o3 = bp.nn.Dense(5)

    model = bp.nn.ff_connect({i1, i2, i3}, {o1, o2, o3})
    model.plot_node_graph()

    self.assertTrue(len(model.ff_receivers[i1]) == 3)
    self.assertTrue(len(model.ff_receivers[i2]) == 3)
    self.assertTrue(len(model.ff_receivers[i3]) == 3)

    self.assertTrue(len(model.ff_senders[o1]) == 3)
    self.assertTrue(len(model.ff_senders[o2]) == 3)
    self.assertTrue(len(model.ff_senders[o3]) == 3)

  def test_many2one4(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)

    ii = bp.nn.Input(3)

    model = {i1, i2, i3} >> ii
    model.plot_node_graph()

    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))

  def test_many2one5(self):
    i1 = bp.nn.Input(1)
    i2 = bp.nn.Input(2)
    i3 = bp.nn.Input(3)
    ii = bp.nn.Input(3)

    model = (i1 >> ii) & (i2 >> ii)
    # model.plot_node_graph()
    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(len(model.ff_senders[ii]) == 1)
    self.assertTrue(isinstance(model.ff_senders[ii][0], bp.nn.Concat))

    model = model & (i3 >> ii)
    # model.plot_node_graph()
    self.assertTrue(isinstance(model.ff_receivers[i1][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i2][0], bp.nn.Concat))
    self.assertTrue(isinstance(model.ff_receivers[i3][0], bp.nn.Concat))
    self.assertTrue(len(model.ff_senders[ii]) == 1)
    self.assertTrue(isinstance(model.ff_senders[ii][0], bp.nn.Concat))


class TestFB(TestCase):
  def test_many2one(self):
    class FBNode(bp.nn.Node):
      def init_fb_conn(self):
        pass

    i1 = FBNode()
    i2 = FBNode()
    i3 = FBNode()
    i4 = FBNode()

    model = (i1 >> i2 >> i3) & (i1 << i2) & (i1 << i3)
    model.plot_node_graph()

    model = model & (i3 >> i4) & (i1 << i4)
    model.plot_node_graph()



