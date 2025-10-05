# -*- coding: utf-8 -*-


import unittest

import brainpy.version2 as bp


class TestFiringRate(unittest.TestCase):
    def test_fr1(self):
        spikes = bp.math.ones((1000, 10))
        print(bp.measure.firing_rate(spikes, 1.))

    def test_fr2(self):
        bp.math.random.seed()
        spikes = bp.math.random.random((1000, 10)) < 0.2
        print(bp.measure.firing_rate(spikes, 1.))
        print(bp.measure.firing_rate(spikes, 10.))

    def test_fr3(self):
        bp.math.random.seed()
        spikes = bp.math.random.random((1000, 10)) < 0.02
        print(bp.measure.firing_rate(spikes, 1.))
        print(bp.measure.firing_rate(spikes, 5.))
