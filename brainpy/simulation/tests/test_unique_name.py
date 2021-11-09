# -*- coding: utf-8 -*-


import brainpy as bp

def test1():
  assert bp.Network().name.startswith('Network')
  assert bp.Container().name.startswith('Container')
  assert bp.NeuGroup(10).name.startswith('NeuGroup')


