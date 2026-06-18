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
"""Coverage tests for ``brainpy.dynold.experimental.abstract_synapses``.

Exercises the experimental ``SynConnNS``-style synapse models ``Exponential``,
``DualExponential`` and ``Alpha`` across the supported connectivity layouts
(All2All, One2One, dense, sparse), with and without an output component
(``CUBA``/``COBA``/``MgBlock``) and short-term depression (``STD``), in both
non-batching and batching modes.

These models take a *connector* (not pre/post groups) as their first
argument and are driven manually by saving ``t``/``dt`` into the shared
context and calling ``update(pre_spike, post_v)``.

.. note::

    Two DEFECTS in the sparse computation paths are pinned below
    (``test_*_sparse_*_defect``): ``Exponential.update`` (sparse + stp) and
    ``DualExponential.update`` (sparse, always) call
    ``bm.sparse.csrmv(..., method='cusparse')`` but the current ``csrmv``
    no longer accepts a ``method`` keyword, raising ``TypeError``.
"""

import unittest

from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
from brainpy.dynold.experimental import abstract_synapses as asyn
from brainpy.dynold.experimental import syn_outs, syn_plasticity


def _step(syn, pre_spike, post_v=None, n=3):
    """Drive ``syn`` for ``n`` manual steps, returning the last output."""
    out = None
    for i in range(n):
        share.save(t=i * bm.get_dt(), dt=bm.get_dt())
        out = syn.update(pre_spike, post_v)
    return bm.as_jax(out)


class TestExponential(parameterized.TestCase):
    def setUp(self):
        bm.random.seed(11)
        bm.set_dt(0.1)

    def test_all2all_no_out(self):
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn)
        r = _step(syn, bm.ones(4))
        self.assertEqual(r.shape, (4,))

    def test_all2all_cuba(self):
        conn = bp.conn.All2All(include_self=False)(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn, out=syn_outs.CUBA())
        r = _step(syn, bm.ones(4), post_v=bm.zeros(4))
        self.assertEqual(r.shape, (4,))

    def test_all2all_with_std_stp(self):
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn, stp=syn_plasticity.STD(4))
        r = _step(syn, bm.ones(4, dtype=bool))
        self.assertEqual(r.shape, (4,))

    def test_one2one(self):
        conn = bp.conn.One2One()(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn, out=syn_outs.COBA(E=0.))
        r = _step(syn, bm.ones(4), post_v=bm.zeros(4))
        self.assertEqual(r.shape, (4,))

    def test_one2one_with_stp(self):
        conn = bp.conn.One2One()(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn, stp=syn_plasticity.STD(4))
        r = _step(syn, bm.ones(4, dtype=bool))
        self.assertEqual(r.shape, (4,))

    def test_dense(self):
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.Exponential(conn, comp_method='dense')
        r = _step(syn, bm.ones(5))
        self.assertEqual(r.shape, (4,))

    def test_dense_with_stp(self):
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.Exponential(conn, comp_method='dense', stp=syn_plasticity.STD(5))
        r = _step(syn, bm.ones(5, dtype=bool))
        self.assertEqual(r.shape, (4,))

    def test_sparse_no_stp(self):
        # sparse + no stp uses bm.event.csrmv, which works.
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.Exponential(conn, comp_method='sparse')
        r = _step(syn, bm.ones(5))
        self.assertEqual(r.shape, (4,))

    def test_sparse_no_stp_batching(self):
        with bm.environment(mode=bm.BatchingMode(2)):
            conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
            syn = asyn.Exponential(conn, comp_method='sparse')
            share.save(t=0.0, dt=bm.get_dt())
            r = bm.as_jax(syn.update(bm.ones((2, 5))))
        self.assertEqual(r.shape, (2, 4))

    def test_sparse_with_stp_defect(self):
        # NOTE: DEFECT -- the sparse + stp path calls
        # bm.sparse.csrmv(..., method='cusparse'); csrmv no longer accepts
        # a `method` kwarg, so this raises TypeError.
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.Exponential(conn, comp_method='sparse', stp=syn_plasticity.STD(5))
        share.save(t=0.0, dt=bm.get_dt())
        with self.assertRaises(TypeError):
            syn.update(bm.ones(5, dtype=bool))

    def test_reset_state(self):
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.Exponential(conn, out=syn_outs.CUBA(), stp=syn_plasticity.STD(4))
        _step(syn, bm.ones(4, dtype=bool))
        syn.reset_state()
        import numpy as np
        np.testing.assert_allclose(bm.as_jax(syn.g.value), np.zeros(4))


class TestDualExponential(parameterized.TestCase):
    def setUp(self):
        bm.random.seed(11)
        bm.set_dt(0.1)

    def test_all2all_no_out_scalar_weight(self):
        # scalar g_max -> All2All aggregation collapses to a scalar conductance
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.DualExponential(conn)
        r = _step(syn, bm.ones(4))
        self.assertEqual(r.shape, ())

    def test_all2all_matrix_weight(self):
        # matrix g_max -> output has post_num shape
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.DualExponential(conn, g_max=bm.ones((4, 4)))
        r = _step(syn, bm.ones(4))
        self.assertEqual(r.shape, (4,))

    def test_all2all_coba_stp(self):
        conn = bp.conn.All2All(include_self=False)(pre_size=4, post_size=4)
        syn = asyn.DualExponential(conn, g_max=bm.ones((4, 4)),
                                   out=syn_outs.COBA(E=0.), stp=syn_plasticity.STD(4))
        r = _step(syn, bm.ones(4), post_v=bm.zeros(4))
        self.assertEqual(r.shape, (4,))

    def test_one2one(self):
        conn = bp.conn.One2One()(pre_size=4, post_size=4)
        syn = asyn.DualExponential(conn, out=syn_outs.CUBA())
        r = _step(syn, bm.ones(4), post_v=bm.zeros(4))
        self.assertEqual(r.shape, (4,))

    def test_dense(self):
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.DualExponential(conn, comp_method='dense')
        r = _step(syn, bm.ones(5))
        self.assertEqual(r.shape, (4,))

    def test_dense_stp(self):
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.DualExponential(conn, comp_method='dense', stp=syn_plasticity.STD(5))
        r = _step(syn, bm.ones(5, dtype=bool))
        self.assertEqual(r.shape, (4,))

    def test_dh_dg_rhs(self):
        conn = bp.conn.All2All()(pre_size=3, post_size=3)
        syn = asyn.DualExponential(conn, tau_rise=2., tau_decay=10.)
        import numpy as np
        h = bm.ones(3)
        np.testing.assert_allclose(bm.as_jax(syn.dh(h, 0.)), -bm.as_jax(h) / 2.)
        g = bm.ones(3)
        np.testing.assert_allclose(bm.as_jax(syn.dg(g, 0., h)), -bm.as_jax(g) / 10. + bm.as_jax(h))

    def test_sparse_defect(self):
        # NOTE: DEFECT -- DualExponential's sparse path always calls
        # bm.sparse.csrmv(..., method='cusparse'); csrmv no longer accepts a
        # `method` kwarg, so this raises TypeError.
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.DualExponential(conn, comp_method='sparse')
        share.save(t=0.0, dt=bm.get_dt())
        with self.assertRaises(TypeError):
            syn.update(bm.ones(5))

    def test_reset_state(self):
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.DualExponential(conn, out=syn_outs.CUBA(), stp=syn_plasticity.STD(4))
        _step(syn, bm.ones(4, dtype=bool))
        syn.reset_state()
        import numpy as np
        np.testing.assert_allclose(bm.as_jax(syn.g.value), np.zeros(4))
        np.testing.assert_allclose(bm.as_jax(syn.h.value), np.zeros(4))


class TestAlpha(parameterized.TestCase):
    def setUp(self):
        bm.random.seed(11)
        bm.set_dt(0.1)

    def test_all2all(self):
        conn = bp.conn.All2All()(pre_size=4, post_size=4)
        syn = asyn.Alpha(conn, out=syn_outs.MgBlock())
        r = _step(syn, bm.ones(4), post_v=bm.zeros(4))
        self.assertEqual(r.shape, (4,))

    def test_dense(self):
        conn = bp.conn.FixedProb(0.5)(pre_size=5, post_size=4)
        syn = asyn.Alpha(conn, comp_method='dense')
        r = _step(syn, bm.ones(5))
        self.assertEqual(r.shape, (4,))

    def test_tau_rise_equals_tau_decay(self):
        # Alpha forces tau_rise == tau_decay.
        conn = bp.conn.All2All()(pre_size=3, post_size=3)
        syn = asyn.Alpha(conn, tau_decay=7.)
        self.assertEqual(syn.tau_rise, syn.tau_decay)
        self.assertEqual(syn.tau_decay, 7.)


if __name__ == '__main__':
    unittest.main()
