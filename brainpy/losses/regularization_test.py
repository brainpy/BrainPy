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
"""Regression tests for :mod:`brainpy.losses.regularization`."""

import unittest

import numpy as np

import braintools.metric as M
import brainpy.math as bm
from brainpy import losses as L


def _close(a, b, atol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol)


class TestDelegatedEquivalence(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(11)
        self.x = bm.asarray(rng.randn(4, 5))
        self.y = bm.asarray(rng.randn(4, 5))
        oh = np.eye(5)[rng.randint(0, 5, size=(4,))]
        self.onehot = bm.asarray(oh)

    def test_log_cosh(self):
        self.assertTrue(_close(L.log_cosh(self.x), M.log_cosh(self.x)))

    def test_smooth_labels(self):
        for alpha in (0.0, 0.1, 0.5):
            self.assertTrue(_close(L.smooth_labels(self.onehot, alpha),
                                   M.smooth_labels(self.onehot, alpha)))

    def test_mean_square(self):
        self.assertTrue(_close(L.mean_square(self.x),
                               M.squared_error(self.x, None, reduction='mean')))

    def test_mean_square_axis(self):
        self.assertTrue(_close(L.mean_square(self.x, axis=-1),
                               M.squared_error(self.x, None, axis=-1, reduction='mean')))

    def test_mean_absolute(self):
        self.assertTrue(_close(L.mean_absolute(self.x),
                               M.absolute_error(self.x, None, reduction='mean')))

    def test_mean_absolute_axis(self):
        self.assertTrue(_close(L.mean_absolute(self.x, axis=0),
                               M.absolute_error(self.x, None, axis=0, reduction='mean')))


class TestPytreeEnvelope(unittest.TestCase):
    def test_smooth_labels_pytree(self):
        rng = np.random.RandomState(3)
        oh = bm.asarray(np.eye(4)[rng.randint(0, 4, size=(5,))])
        flat = L.smooth_labels(oh, 0.2)
        tree = L.smooth_labels({'a': oh}, 0.2)
        self.assertTrue(_close(flat, tree))

    def test_mean_square_pytree_sums_leaves(self):
        rng = np.random.RandomState(4)
        a = bm.asarray(rng.randn(3, 3))
        single = L.mean_square(a)
        multi = L.mean_square({'x': a, 'y': a})
        self.assertTrue(_close(2.0 * np.asarray(single), multi))


class TestL2NormUnchanged(unittest.TestCase):
    """l2_norm has a different definition upstream and stays as brainpy's."""

    def test_l2_norm_is_sqrt_sum_squares(self):
        x = bm.asarray(np.array([3.0, 4.0]))
        # brainpy l2_norm == Euclidean norm == 5.0 for [3, 4]
        self.assertTrue(_close(L.l2_norm(x), 5.0))


if __name__ == '__main__':
    unittest.main()
