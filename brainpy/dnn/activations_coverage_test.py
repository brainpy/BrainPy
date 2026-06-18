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
"""Coverage tests for ``brainpy/dnn/activations.py``.

Target: the ``extra_repr`` methods of every activation layer, which were not
exercised by ``activation_test.py`` (that file only constructs layers and runs
forward passes). Each ``extra_repr`` is called directly with both the inplace
and non-inplace formatting branches where applicable.
"""

from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp


class TestActivationExtraRepr(parameterized.TestCase):
    def test_threshold_repr(self):
        self.assertIn('inplace=True', bp.dnn.Threshold(5, 20, inplace=True).extra_repr())
        self.assertNotIn('inplace', bp.dnn.Threshold(5, 20, inplace=False).extra_repr())

    def test_relu_repr(self):
        self.assertEqual(bp.dnn.ReLU(inplace=True).extra_repr(), 'inplace=True')
        self.assertEqual(bp.dnn.ReLU(inplace=False).extra_repr(), '')

    def test_rrelu_repr(self):
        self.assertIn('inplace=True', bp.dnn.RReLU(inplace=True).extra_repr())
        self.assertIn('lower=', bp.dnn.RReLU(inplace=False).extra_repr())

    def test_hardtanh_repr(self):
        self.assertIn('inplace=True', bp.dnn.Hardtanh(inplace=True).extra_repr())
        self.assertIn('min_val=', bp.dnn.Hardtanh(inplace=False).extra_repr())

    def test_relu6_repr(self):
        self.assertEqual(bp.dnn.ReLU6(inplace=True).extra_repr(), 'inplace=True')
        self.assertEqual(bp.dnn.ReLU6(inplace=False).extra_repr(), '')

    def test_silu_repr(self):
        self.assertEqual(bp.dnn.SiLU(inplace=True).extra_repr(), 'inplace=True')
        self.assertEqual(bp.dnn.SiLU(inplace=False).extra_repr(), '')

    def test_mish_repr(self):
        self.assertEqual(bp.dnn.Mish(inplace=True).extra_repr(), 'inplace=True')
        self.assertEqual(bp.dnn.Mish(inplace=False).extra_repr(), '')

    def test_elu_repr(self):
        self.assertIn('inplace=True', bp.dnn.ELU(inplace=True).extra_repr())
        self.assertIn('alpha=', bp.dnn.ELU(inplace=False).extra_repr())

    def test_celu_repr(self):
        self.assertIn('inplace=True', bp.dnn.CELU(inplace=True).extra_repr())
        self.assertIn('alpha=', bp.dnn.CELU(inplace=False).extra_repr())

    def test_selu_repr(self):
        self.assertEqual(bp.dnn.SELU(inplace=True).extra_repr(), 'inplace=True')
        self.assertEqual(bp.dnn.SELU(inplace=False).extra_repr(), '')

    def test_glu_repr(self):
        self.assertEqual(bp.dnn.GLU(dim=-1).extra_repr(), 'dim=-1')

    def test_gelu_repr(self):
        self.assertIn('approximate=', bp.dnn.GELU().extra_repr())

    def test_hardshrink_repr(self):
        self.assertEqual(bp.dnn.Hardshrink(lambd=0.5).extra_repr(), '0.5')

    def test_leaky_relu_repr(self):
        self.assertIn('inplace=True', bp.dnn.LeakyReLU(inplace=True).extra_repr())
        self.assertIn('negative_slope=', bp.dnn.LeakyReLU(inplace=False).extra_repr())

    def test_softplus_repr(self):
        self.assertIn('beta=', bp.dnn.Softplus().extra_repr())

    def test_softshrink_repr(self):
        self.assertEqual(bp.dnn.Softshrink(lambd=0.5).extra_repr(), '0.5')

    def test_prelu_repr(self):
        self.assertIn('num_parameters=', bp.dnn.PReLU().extra_repr())

    def test_softmin_repr(self):
        self.assertEqual(bp.dnn.Softmin(dim=1).extra_repr(), 'dim=1')

    def test_softmax_repr(self):
        self.assertEqual(bp.dnn.Softmax(dim=1).extra_repr(), 'dim=1')

    def test_logsoftmax_repr(self):
        self.assertEqual(bp.dnn.LogSoftmax(dim=1).extra_repr(), 'dim=1')


if __name__ == '__main__':
    absltest.main()
