# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm


# visualization
def mat_visualize(matrix, cmap=None):
    if cmap is None:
        cmap = plt.colormaps.get_cmap('coolwarm')
    plt.colormaps.get_cmap('coolwarm')
    im = plt.matshow(matrix, cmap=cmap)
    plt.colorbar(mappable=im, shrink=0.8, aspect=15)
    plt.show()


def _size2len(size):
    if isinstance(size, int):
        return size
    elif isinstance(size, (tuple, list)):
        length = 1
        for e in size:
            length *= e
        return length
    else:
        raise ValueError(f'Must be a list/tuple of int, but got {size}')


class TestGaussianDecayInit(unittest.TestCase):
    def test_gaussian_decay_init1(self):
        init = bp.init.GaussianDecay(sigma=4, max_w=1.)
        for size in [10, (10, 20), (10, 20, 30)]:
            weights = init(size)
            shape = _size2len(size)
            assert weights.shape == (shape, shape)
            assert isinstance(weights, bp.math.ndarray)

    def test_gaussian_decay_init2(self):
        init = bp.init.GaussianDecay(sigma=4, max_w=1., min_w=0.1, periodic_boundary=True,
                                     encoding_values=((-bm.pi, bm.pi), (10, 20), (0, 2 * bm.pi)),
                                     include_self=False, normalize=True)
        size = (10, 20, 30)
        weights = init(size)
        shape = _size2len(size)
        assert weights.shape == (shape, shape)
        assert isinstance(weights, bp.math.ndarray)


class TestDOGDecayInit(unittest.TestCase):
    def test_dog_decay_init1(self):
        init = bp.init.DOGDecay(sigmas=(1., 2.5), max_ws=(1.0, 0.7))
        for size in [10, (10, 20), (10, 20, 30)]:
            weights = init(size)
            shape = _size2len(size)
            assert weights.shape == (shape, shape)
            assert isinstance(weights, bp.math.ndarray)

    def test_dog_decay_init2(self):
        init = bp.init.DOGDecay(sigmas=(1., 2.5),
                                max_ws=(1.0, 0.7), min_w=0.1,
                                periodic_boundary=True,
                                encoding_values=((-bm.pi, bm.pi), (10, 20), (0, 2 * bm.pi)),
                                include_self=False,
                                normalize=True)
        size = (10, 20, 30)
        weights = init(size)
        shape = _size2len(size)
        assert weights.shape == (shape, shape)
        assert isinstance(weights, bp.math.ndarray)

    def test_dog_decay3(self):
        size = (10, 12)
        dog_init = bp.init.DOGDecay(sigmas=(1., 3.),
                                    max_ws=(10., 5.),
                                    min_w=0.1,
                                    include_self=True)
        weights = dog_init(size)
        print('shape of weights: {}'.format(weights.shape))
        # out: shape of weights: (120, 120)
        self.assertTrue(weights.shape == (np.prod(size), np.prod(size)))

        # visualize neuron(3, 4)
        mat_visualize(weights[:, 3 * 12 + 4].reshape((10, 12)), cmap=matplotlib.colormaps['Reds'])
        plt.close()
