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
"""Coverage tests for ``brainpy.dynold.synapses.base``.

Targets the error/edge branches of the dynold synapse base classes that the
happy-path simulation tests in ``abstract_synapses_test.py`` /
``biological_synapses_test.py`` do not reach:

* ``_SynapseComponent`` abstract methods (``clone``/``filter``), ``__repr__``,
  the ``isregistered`` setter validation, and ``register_master`` type/double
  registration guards.
* ``_SynOut`` ``target_var`` handling (string lookup success + KeyError, bad
  type, and the identity ``filter`` when no target var).
* ``TwoEndConn._init_weights`` validation errors and the matrix-weight
  branches of the three ``_syn2post_with_*`` aggregation helpers.
* ``_init_stp`` type guard and the ``_TwoEndConnAlignPre`` deprecated
  ``g_max`` property + unsupported ``comp_method``.
"""

import unittest
import warnings

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import UnsupportedError
from brainpy.dynold.synapses.base import (_SynapseComponent, _SynOut, _SynSTP,
                                          _NullSynOut, _init_stp)
from brainpy.dynold.synouts import CUBA


class TestSynapseComponent(unittest.TestCase):
    def setUp(self):
        bm.random.seed(3)
        bm.set_dt(0.1)

    def test_repr(self):
        self.assertEqual(repr(_SynapseComponent()), '_SynapseComponent')

    def test_clone_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _SynapseComponent().clone()

    def test_filter_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _SynapseComponent().filter(1.0)

    def test_isregistered_setter_type_guard(self):
        c = _SynapseComponent()
        with self.assertRaises(ValueError):
            c.isregistered = 'not-a-bool'
        c.isregistered = True
        self.assertTrue(c.isregistered)

    def test_register_master_wrong_type(self):
        with self.assertRaises(TypeError):
            _SynapseComponent().register_master(123)

    def test_register_master_double(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        a = bp.synapses.Exponential(pre, post, bp.conn.All2All())
        b = bp.synapses.Exponential(pre, post, bp.conn.All2All())
        out = CUBA()
        out.register_master(a)
        with self.assertRaises(ValueError):
            out.register_master(b)


class TestSynOut(unittest.TestCase):
    def setUp(self):
        bm.random.seed(3)
        bm.set_dt(0.1)

    def test_target_var_bad_type(self):
        with self.assertRaises(TypeError):
            _SynOut(target_var=123)

    def test_filter_identity_when_no_target(self):
        out = _SynOut()
        self.assertEqual(out.filter(5.0), 5.0)
        self.assertIsNone(out.update())

    def test_target_var_string_resolved(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Delta(pre, post, bp.conn.All2All(),
                                output=CUBA(target_var='V'))
        # the string 'V' is resolved to the post-group Variable on registration
        self.assertIsInstance(syn.output.target_var, bm.Variable)

    def test_target_var_string_missing_raises(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        with self.assertRaises(KeyError):
            bp.synapses.Delta(pre, post, bp.conn.All2All(),
                              output=CUBA(target_var='does_not_exist'))


class TestSynSTPBase(unittest.TestCase):
    def setUp(self):
        bm.random.seed(3)
        bm.set_dt(0.1)

    def test_base_update_is_noop(self):
        # the abstract _SynSTP.update is a no-op (pass)
        self.assertIsNone(_SynSTP().update(None))

    def test_return_info(self):
        # return_info requires a registered master with a `.pre` group
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        stp = bp.synplast.STD()
        bp.synapses.Exponential(pre, post, bp.conn.All2All(), stp=stp)
        info = stp.return_info()
        self.assertEqual(tuple(info.size), tuple(pre.varshape))

    def test_null_syn_out_clone(self):
        out = _NullSynOut()
        self.assertIsInstance(out.clone(), _NullSynOut)


class TestTwoEndConnWeights(unittest.TestCase):
    def setUp(self):
        bm.random.seed(3)
        bm.set_dt(0.1)
        self.pre = bp.neurons.LIF(4)
        self.post = bp.neurons.LIF(4)
        self.syn = bp.synapses.Delta(self.pre, self.post, bp.conn.FixedProb(0.5))

    def test_init_weights_bad_comp_method(self):
        with self.assertRaises(ValueError):
            self.syn._init_weights(1., 'banana')

    def test_init_weights_bad_sparse_data(self):
        with self.assertRaises(ValueError):
            self.syn._init_weights(1., 'sparse', sparse_data='banana')

    def test_init_weights_sparse_ij(self):
        weight, mask = self.syn._init_weights(1., 'sparse', sparse_data='ij')
        self.assertEqual(len(mask), 2)

    def test_init_weights_conn_none_raises(self):
        # a fresh synapse stub with conn=None should raise on weight init
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Delta(pre, post, bp.conn.FixedProb(0.5))
        syn.conn = None
        with self.assertRaises(ValueError):
            syn._init_weights(1., 'sparse')

    def test_syn2post_all2all_scalar_exclude_self(self):
        # scalar weight + exclude-self subtracts the diagonal contribution
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Delta(pre, post, bp.conn.All2All(include_self=False))
        r = bm.as_jax(syn._syn2post_with_all2all(bm.ones(4), 2.0))
        # 2 * (sum(ones(4)) - self) = 2 * (4 - 1) = 6
        np.testing.assert_allclose(r, np.full(4, 6.0))

    def test_syn2post_all2all_matrix_weight(self):
        r = bm.as_jax(self.syn._syn2post_with_all2all(bm.ones(4), bm.ones((4, 4))))
        np.testing.assert_allclose(r, np.full(4, 4.0))

    def test_syn2post_dense_matrix_weight(self):
        r = bm.as_jax(self.syn._syn2post_with_dense(bm.ones(4), bm.ones((4, 4)), bm.ones((4, 4))))
        np.testing.assert_allclose(r, np.full(4, 4.0))

    def test_syn2post_one2one(self):
        r = bm.as_jax(self.syn._syn2post_with_one2one(bm.ones(4), bm.full(4, 2.0)))
        np.testing.assert_allclose(r, np.full(4, 2.0))


class TestInitStpAndAlignPre(unittest.TestCase):
    def setUp(self):
        bm.random.seed(3)
        bm.set_dt(0.1)

    def test_init_stp_type_guard(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Exponential(pre, post, bp.conn.All2All())

        class _Stub:
            isregistered = False

        with self.assertRaises(TypeError):
            _init_stp(_Stub(), syn)

    def test_stp_clone_on_reuse(self):
        # reusing a registered stp clones it instead of erroring
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        stp = bp.synplast.STD()
        a = bp.synapses.Exponential(pre, post, bp.conn.All2All(), stp=stp)
        b = bp.synapses.Exponential(pre, post, bp.conn.All2All(), stp=stp)
        self.assertIsNotNone(a.stp)
        self.assertIsNotNone(b.stp)

    def test_output_clone_on_reuse(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        out = CUBA()
        a = bp.synapses.Exponential(pre, post, bp.conn.All2All(), output=out)
        b = bp.synapses.Exponential(pre, post, bp.conn.All2All(), output=out)
        self.assertIsNot(a.output, b.output)

    def test_alignpre_unsupported_comp_method(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        with self.assertRaises(UnsupportedError):
            bp.synapses.DualExponential(pre, post, bp.conn.FixedProb(0.5), comp_method='banana')

    def test_alignpre_g_max_deprecation(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.DualExponential(pre, post, bp.conn.All2All())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            gm = syn.g_max
            self.assertTrue(any(issubclass(x.category, UserWarning) for x in w))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            syn.g_max = gm
            self.assertTrue(any(issubclass(x.category, UserWarning) for x in w))


if __name__ == '__main__':
    unittest.main()
