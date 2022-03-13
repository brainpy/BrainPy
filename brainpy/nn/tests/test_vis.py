# -*- coding: utf-8 -*-


import unittest
import brainpy as bp
import brainpy.math as bm


class TestVisualize(unittest.TestCase):
  def test(self):
    model = (
        bp.nn.Input(3)
        >>
        bp.nn.Reservoir(100, name='I')
        >>
        bp.nn.Reservoir(100)
        >>
        bp.nn.Reservoir(100, name='l1')
        >>
        bp.nn.LinearReadout(3, weight_initializer=bp.init.Normal())
        >>
        bp.nn.Reservoir(100)
        >>
        bp.nn.Reservoir(100)
        >>
        bp.nn.LinearReadout(3, weight_initializer=bp.init.Normal(), name='output')
    )
    model &= (model['l1'] << model['output'])
    model &= (model['I'] << model['output'])

    # model =
    # print(model.trainable)
    print()

    model.plot_node_graph('spring_layout', fig_size=(15, 5), node_size=1000, seed=2022)
