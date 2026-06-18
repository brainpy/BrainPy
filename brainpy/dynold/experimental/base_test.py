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
"""Coverage tests for ``brainpy.dynold.experimental.base``.

Exercises ``SynConnNS`` (connection initialisation, weight initialisation
across One2One / All2All / sparse(csr,ij,coo) / dense layouts and the
training-mode branch, plus the three ``_syn2post_with_*`` aggregation
helpers for both scalar and matrix weights) and the abstract bases
``SynOutNS`` and ``SynSTPNS``.

``SynConnNS`` is abstract (no public constructor of its own), so a concrete
``Exponential`` instance is used as the vehicle to drive the inherited
helper methods directly.
"""

import unittest

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.connect import MatConn, IJConn
from brainpy.dynold.experimental import abstract_synapses as asyn
from brainpy.dynold.experimental.base import SynConnNS, SynOutNS, SynSTPNS


def _make_syn(pre=5, post=4, prob=0.5):
    return asyn.Exponential(bp.conn.FixedProb(prob)(pre_size=pre, post_size=post))


class TestSynConnNSInitConn(unittest.TestCase):
    def setUp(self):
        bm.random.seed(7)
        bm.set_dt(0.1)
        self.s = _make_syn()
        # the helper uses pre_num/post_num
        self.s.pre_num, self.s.post_num = 5, 4

    def test_two_end_connector_passthrough(self):
        conn = bp.conn.All2All()(pre_size=5, post_size=4)
        out = self.s._init_conn(conn)
        self.assertIs(out, conn)

    def test_matrix_conn(self):
        mat = np.zeros((5, 4))
        mat[0, 0] = 1
        out = self.s._init_conn(mat)
        self.assertIsInstance(out, MatConn)

    def test_matrix_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self.s._init_conn(np.zeros((2, 2)))

    def test_dict_conn(self):
        out = self.s._init_conn({'i': np.array([0, 1]), 'j': np.array([1, 2])})
        self.assertIsInstance(out, IJConn)

    def test_dict_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            self.s._init_conn({'a': 1})

    def test_none_conn(self):
        self.assertIsNone(self.s._init_conn(None))

    def test_unknown_conn_type_raises(self):
        with self.assertRaises(ValueError):
            self.s._init_conn(12345)


class TestSynConnNSInitWeights(unittest.TestCase):
    def setUp(self):
        bm.random.seed(7)
        bm.set_dt(0.1)

    def test_bad_comp_method_raises(self):
        s = _make_syn()
        with self.assertRaises(ValueError):
            s._init_weights(1., 'banana')

    def test_bad_sparse_data_raises(self):
        s = _make_syn()
        with self.assertRaises(ValueError):
            s._init_weights(1., 'sparse', data_if_sparse='banana')

    def test_one2one_weights_scalar(self):
        # a scalar initializer yields a scalar weight (broadcast at runtime)
        s = asyn.Exponential(bp.conn.One2One()(pre_size=4, post_size=4))
        weight, mask = s._init_weights(1., 'sparse')
        self.assertEqual(float(weight), 1.0)
        self.assertIsNone(mask)

    def test_one2one_weights_array(self):
        s = asyn.Exponential(bp.conn.One2One()(pre_size=4, post_size=4))
        weight, mask = s._init_weights(bm.ones(4), 'sparse')
        self.assertEqual(bm.as_jax(weight).shape, (4,))
        self.assertIsNone(mask)

    def test_all2all_weights_array(self):
        s = asyn.Exponential(bp.conn.All2All()(pre_size=5, post_size=4))
        weight, mask = s._init_weights(bm.ones((5, 4)), 'sparse')
        self.assertEqual(bm.as_jax(weight).shape, (5, 4))
        self.assertIsNone(mask)

    def test_sparse_csr(self):
        s = _make_syn()
        weight, mask = s._init_weights(1., 'sparse', data_if_sparse='csr')
        self.assertEqual(len(mask), 2)

    def test_sparse_ij(self):
        s = _make_syn()
        weight, mask = s._init_weights(1., 'sparse', data_if_sparse='ij')
        self.assertEqual(len(mask), 2)

    def test_sparse_coo(self):
        s = _make_syn()
        weight, mask = s._init_weights(1., 'sparse', data_if_sparse='coo')
        self.assertEqual(len(mask), 2)

    def test_dense(self):
        s = _make_syn()
        weight, mask = s._init_weights(bm.ones((5, 4)), 'dense')
        self.assertEqual(bm.as_jax(weight).shape, (5, 4))
        self.assertIsNotNone(mask)

    def test_training_mode_makes_trainvar(self):
        with bm.environment(mode=bm.TrainingMode(3)):
            s = _make_syn()
        self.assertIsInstance(s.g_max, bm.TrainVar)


class TestSyn2Post(unittest.TestCase):
    def setUp(self):
        bm.random.seed(7)
        bm.set_dt(0.1)

    def test_all2all_scalar_include_self(self):
        s = asyn.Exponential(bp.conn.All2All()(pre_size=4, post_size=4))
        syn_value = bm.ones(4)
        r = bm.as_jax(s._syn2post_with_all2all(syn_value, 2.0, include_self=True))
        # scalar weight: 2 * sum(ones(4)) = 8 for every post
        np.testing.assert_allclose(r, 8.0)

    def test_all2all_scalar_exclude_self(self):
        s = asyn.Exponential(bp.conn.All2All()(pre_size=4, post_size=4))
        syn_value = bm.ones(4)
        r = bm.as_jax(s._syn2post_with_all2all(syn_value, 2.0, include_self=False))
        # 2 * (sum - self) = 2 * (4 - 1) = 6
        np.testing.assert_allclose(r, 6.0)

    def test_all2all_matrix_weight(self):
        s = asyn.Exponential(bp.conn.All2All()(pre_size=4, post_size=3))
        syn_value = bm.ones(4)
        weight = bm.ones((4, 3))
        r = bm.as_jax(s._syn2post_with_all2all(syn_value, weight, include_self=True))
        np.testing.assert_allclose(r, np.full(3, 4.0))

    def test_all2all_batching_mode(self):
        with bm.environment(mode=bm.BatchingMode(2)):
            s = asyn.Exponential(bp.conn.All2All()(pre_size=4, post_size=4))
            syn_value = bm.ones((2, 4))
            r = bm.as_jax(s._syn2post_with_all2all(syn_value, 2.0, include_self=True))
            # keepdims sum over last axis -> (2, 1) then *2 -> 8
            np.testing.assert_allclose(r, np.full((2, 1), 8.0))

    def test_one2one(self):
        s = _make_syn()
        r = bm.as_jax(s._syn2post_with_one2one(bm.ones(4), bm.full(4, 3.0)))
        np.testing.assert_allclose(r, np.full(4, 3.0))

    def test_dense_scalar_weight(self):
        s = _make_syn(pre=4, post=3, prob=1.0)
        conn_mat = bm.ones((4, 3))
        r = bm.as_jax(s._syn2post_with_dense(bm.ones(4), 2.0, conn_mat))
        np.testing.assert_allclose(r, np.full(3, 8.0))

    def test_dense_matrix_weight(self):
        s = _make_syn(pre=4, post=3, prob=1.0)
        conn_mat = bm.ones((4, 3))
        weight = bm.ones((4, 3))
        r = bm.as_jax(s._syn2post_with_dense(bm.ones(4), weight, conn_mat))
        np.testing.assert_allclose(r, np.full(3, 4.0))


class TestAbstractBases(unittest.TestCase):
    def test_synoutns_update_not_implemented(self):
        out = SynOutNS()
        with self.assertRaises(NotImplementedError):
            out.update(None, None)

    def test_synoutns_reset_state_noop(self):
        out = SynOutNS()
        self.assertIsNone(out.reset_state())
        self.assertIsNone(out.reset_state(batch_size=3))

    def test_synstpns_update_not_implemented(self):
        stp = SynSTPNS()
        with self.assertRaises(NotImplementedError):
            stp.update(None)


if __name__ == '__main__':
    unittest.main()
