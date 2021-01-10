# -*- coding: utf-8 -*-

import numpy as np
import brainpy
from brainpy import inputs


def test_SpikeTimeInput1():
    brainpy.profile.set(dt=0.1)
    group = inputs.SpikeTimeInput(30, [1., 1.2, 1.31], monitors=['spike'])
    group.run(2.)

    spikes = group.mon.spike

    assert spikes.sum() == 90
    assert np.all(spikes[int(np.ceil(1. / 0.1))] == 1.)
    assert np.all(spikes[int(np.ceil(1.2 / 0.1))] == 1.)
    assert np.all(spikes[int(np.ceil(1.31 / 0.1))] == 1.)


def test_SpikeTimeInput2():
    brainpy.profile.set(dt=0.1)

    group = inputs.SpikeTimeInput(4,
                                  times=[1., 1.2, 1.31, 1.45],
                                  indices=[0, 1, 3, 1],
                                  monitors=['spike'])
    group.run(2.)

    spikes = group.mon.spike

    assert spikes.sum() == 4
    assert np.all(spikes[int(np.ceil(1. / 0.1)), 0] == 1.)
    assert np.all(spikes[int(np.ceil(1.2 / 0.1)), 1] == 1.)
    assert np.all(spikes[int(np.ceil(1.31 / 0.1)), 3] == 1.)
    assert np.all(spikes[int(np.ceil(1.45 / 0.1)), 1] == 1.)


if __name__ == '__main__':
    test_SpikeTimeInput1()
    test_SpikeTimeInput2()
