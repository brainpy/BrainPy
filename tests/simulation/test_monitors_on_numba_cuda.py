# -*- coding: utf-8 -*-

import pytest
from numba import cuda

if not cuda.is_available():
    pytest.skip("cuda is not available", allow_module_level=True)

import numpy as np

from brainpy.backend import set
from brainpy.simulation.brainobjects import NeuGroup
from brainpy.simulation.monitors import Monitor


class TryGroup(NeuGroup):
    target_backend = 'general'

    def __init__(self, **kwargs):
        self.a = np.ones((2, 2))
        super(TryGroup, self).__init__(size=1, **kwargs)

    def update(self, _t):
        self.a += 1


def test_non_array():
    set('numba-cuda', dt=0.1)
    try1 = TryGroup(monitors=['a'])
    try1.a = 1.
    try1.run(100.)

    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 1
    assert np.allclose(np.arange(2, 1002).reshape((-1, 1)), try1.mon.a)


def test_1d_array():
    set('numba-cuda', dt=0.1)
    try1 = TryGroup(monitors=['a'])
    try1.a = np.ones(1)
    try1.run(100.)

    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 1
    assert np.allclose(np.arange(2, 1002).reshape((-1, 1)), try1.mon.a)


def test_2d_array():
    set('numba-cuda', dt=0.1)
    try1 = TryGroup(monitors=['a'])
    try1.a = np.ones((2, 2))
    try1.run(100.)

    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
    series = np.arange(2, 1002).reshape((-1, 1))
    series = np.repeat(series, 4, axis=1)
    assert np.allclose(series, try1.mon.a)


def test_monitor_with_every():
    set('numba-cuda', dt=0.1)

    # try1: 2d array
    try1 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try1.run(100.)
    assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    series = np.repeat(series, 4, axis=1)
    assert np.allclose(series, try1.mon.a)

    # try2: 1d array
    try2 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try2.a = np.array([1., 1.])
    try2.run(100.)
    assert np.ndim(try2.mon.a) == 2 and np.shape(try2.mon.a)[1] == 2
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    series = np.repeat(series, 2, axis=1)
    assert np.allclose(series, try2.mon.a)

    # try2: scalar
    try3 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
    try3.a = 1.
    try3.run(100.)
    assert np.ndim(try3.mon.a) == 2 and np.shape(try3.mon.a)[1] == 1
    series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
    assert np.allclose(series, try3.mon.a)


# test_non_array()
# test_1d_array()
# test_2d_array()
# test_monitor_with_every()
