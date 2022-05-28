# -*- coding: utf-8 -*-


import unittest
from brainpy.compat.nn.graph_flow import find_entries_and_exits
from brainpy.compat.nn.graph_flow import detect_cycle


class TestGraphFlow(unittest.TestCase):
  def test_ff1(self):
    nodes = (1, 2, 3, 4, 5)
    ff_edges = ((1, 2), (2, 3), (3, 4), (4, 5))
    inputs, outputs = find_entries_and_exits(nodes, ff_edges)
    print()
    print(inputs, outputs)

    ff_edges = ((1, 2), (2, 3), (3, 4))
    inputs, outputs = find_entries_and_exits(nodes, ff_edges)
    print(inputs, outputs)

  def test_fb1(self):
    nodes = (1, 2, 3, 4, 5)
    ff_edges = ((1, 2), (2, 3), (3, 4), (4, 5))
    fb_edges = ((5, 2), (4, 2))
    inputs, outputs = find_entries_and_exits(nodes, ff_edges, fb_edges)
    print()
    print(inputs, outputs)

  def test_fb2(self):
    nodes = (1, 2, 3, 4, 5)
    ff_edges = ((1, 2), (2, 3), (3, 4))
    fb_edges = ((3, 2), (4, 5))
    # with self.assertRaises(ValueError):
    find_entries_and_exits(nodes, ff_edges, fb_edges)

  def test_fb3(self):
    nodes = (1, 2, 3, 4, 5)
    ff_edges = ((1, 2), (2, 3), (3, 4))
    fb_edges = ((5, 2), )
    inputs, outputs = find_entries_and_exits(nodes, ff_edges, fb_edges)
    print()
    print(inputs, outputs)

  def test_fb4(self):
    # 1 -> 2 -> 3 -> 4 -> 5 -> 6
    #      ^             |^    |
    #      ∟------------- ∟----
    nodes = (1, 2, 3, 4, 5, 6)
    ff_edges = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
    fb_edges = ((5, 2), (6, 5))
    inputs, outputs = find_entries_and_exits(nodes, ff_edges, fb_edges)
    print()
    print(inputs, outputs)

  def test_fb5(self):
    # 1 -> 2 -> 3 -> 4 -> 5 -> 6
    # ^                   |^    |
    # ∟------------------- ∟----
    nodes = (1, 2, 3, 4, 5, 6)
    ff_edges = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6))
    fb_edges = ((5, 1), (6, 5))
    inputs, outputs = find_entries_and_exits(nodes, ff_edges, fb_edges)
    print()
    print(inputs, outputs)


class TestDetectCycle(unittest.TestCase):
  def test1(self):
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 3)]
    print(detect_cycle(nodes, edges))
