# -*- coding: utf-8 -*-

import pickle
import unittest

import brainpy.version2 as bp


class TestPickle(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPickle, self).__init__(*args, **kwargs)

        self.pre = bp.neurons.LIF(10)
        self.post = bp.neurons.LIF(20)
        self.syn = bp.synapses.TwoEndConn(self.pre, self.post, bp.conn.FixedProb(0.2))
        self.net = bp.DynSysGroup(self.pre, self.post, self.syn)

    def test_net(self):
        self.skipTest('Currently do not support')
        with open('data/net.pickle', 'wb') as f:
            pickle.dump(self.net, f)
