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
"""Coverage tests for ``brainpy.dynold.synapses.abstract_models``.

Complements ``abstract_synapses_test.py`` (which exercises the All2All happy
paths) by covering the branches it misses:

* ``Delta.update`` across One2One, sparse (with and without STP, hitting both
  the ``event.csrmv`` and ``sparse.csrmv`` paths) and dense layouts, plus the
  ``stop_spike_gradient`` and ``post_ref_key`` branches.
* ``Exponential`` ``stop_spike_gradient``, the ``g`` getter/setter, and the
  unsupported ``comp_method`` guard.
* The scalar-size validation guards of ``DualExponential`` (``tau_rise`` /
  ``tau_decay``), ``Alpha`` (``tau_decay``) and ``NMDA`` (``a`` / ``tau_decay``
  / ``tau_rise``).
"""

import unittest

import numpy as np

import brainpy as bp
import brainpy.math as bm


def _simulate(syn, pre, post, n=5):
    net = bp.Network(pre=pre, syn=syn, post=post)
    runner = bp.DSRunner(net, monitors=['post.V'], inputs=('pre.input', 35.),
                         progress_bar=False)
    runner(n * bm.get_dt())
    return runner


class TestDelta(unittest.TestCase):
    def setUp(self):
        bm.random.seed(42)
        bm.set_dt(0.1)

    def test_one2one(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Delta(pre, post, bp.conn.One2One())
        _simulate(syn, pre, post)

    def test_one2one_with_stp(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Delta(pre, post, bp.conn.One2One(), stp=bp.synplast.STD())
        _simulate(syn, pre, post)

    def test_sparse_no_stp(self):
        # sparse + no stp -> bm.event.csrmv path
        pre = bp.neurons.LIF(6)
        post = bp.neurons.LIF(6)
        syn = bp.synapses.Delta(pre, post, bp.conn.FixedProb(0.5), comp_method='sparse')
        _simulate(syn, pre, post)

    def test_sparse_with_stp(self):
        # sparse + stp -> bm.sparse.csrmv path
        pre = bp.neurons.LIF(6)
        post = bp.neurons.LIF(6)
        syn = bp.synapses.Delta(pre, post, bp.conn.FixedProb(0.5),
                                comp_method='sparse', stp=bp.synplast.STD())
        _simulate(syn, pre, post)

    def test_dense(self):
        pre = bp.neurons.LIF(6)
        post = bp.neurons.LIF(6)
        syn = bp.synapses.Delta(pre, post, bp.conn.FixedProb(0.5), comp_method='dense')
        _simulate(syn, pre, post)

    def test_dense_with_stp(self):
        pre = bp.neurons.LIF(6)
        post = bp.neurons.LIF(6)
        syn = bp.synapses.Delta(pre, post, bp.conn.FixedProb(0.5),
                                comp_method='dense', stp=bp.synplast.STD())
        _simulate(syn, pre, post)

    def test_stop_spike_gradient_and_post_ref_key(self):
        pre = bp.neurons.LIF(4, ref_var=True)
        post = bp.neurons.LIF(4, ref_var=True)
        syn = bp.synapses.Delta(pre, post, bp.conn.All2All(),
                                post_ref_key='refractory',
                                stop_spike_gradient=True)
        _simulate(syn, pre, post)


class TestExponential(unittest.TestCase):
    def setUp(self):
        bm.random.seed(42)
        bm.set_dt(0.1)

    def test_g_getter_setter(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Exponential(pre, post, bp.conn.All2All())
        g0 = bm.as_jax(syn.g).copy()
        syn.g = syn.syn.g + 1.0
        np.testing.assert_allclose(bm.as_jax(syn.g), g0 + 1.0)

    def test_stop_spike_gradient(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        syn = bp.synapses.Exponential(pre, post, bp.conn.All2All(),
                                      stop_spike_gradient=True)
        _simulate(syn, pre, post)

    def test_bad_comp_method(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        with self.assertRaises(ValueError):
            bp.synapses.Exponential(pre, post, bp.conn.FixedProb(0.5), comp_method='banana')


class TestAlignPreStopGradient(unittest.TestCase):
    """AlignPre-based synapses (DualExponential/Alpha/NMDA) with stop gradient."""

    def setUp(self):
        bm.random.seed(42)
        bm.set_dt(0.1)

    def test_dual_exponential_stop_spike_gradient(self):
        pre = bp.neurons.LIF(4)
        post = bp.neurons.LIF(4)
        # stop_spike_gradient=True exercises the jax.lax.stop_gradient branch
        # inside _TwoEndConnAlignPre.update
        syn = bp.synapses.DualExponential(pre, post, bp.conn.All2All(),
                                          stop_spike_gradient=True)
        _simulate(syn, pre, post)

    def test_nmda_stop_spike_gradient(self):
        pre = bp.neurons.HH(4)
        post = bp.neurons.HH(4)
        syn = bp.synapses.NMDA(pre, post, bp.conn.All2All(),
                               stop_spike_gradient=True)
        _simulate(syn, pre, post)


class TestScalarSizeGuards(unittest.TestCase):
    def setUp(self):
        bm.random.seed(42)
        bm.set_dt(0.1)
        self.pre = bp.neurons.LIF(4)
        self.post = bp.neurons.LIF(4)

    def test_dual_tau_rise_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.DualExponential(self.pre, self.post, bp.conn.All2All(),
                                        tau_rise=bm.ones(2))

    def test_dual_tau_decay_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.DualExponential(self.pre, self.post, bp.conn.All2All(),
                                        tau_decay=bm.ones(2))

    def test_alpha_tau_decay_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.Alpha(self.pre, self.post, bp.conn.All2All(),
                              tau_decay=bm.ones(2))

    def test_nmda_a_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.NMDA(self.pre, self.post, bp.conn.All2All(), a=bm.ones(2))

    def test_nmda_tau_decay_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.NMDA(self.pre, self.post, bp.conn.All2All(), tau_decay=bm.ones(2))

    def test_nmda_tau_rise_must_be_scalar(self):
        with self.assertRaises(ValueError):
            bp.synapses.NMDA(self.pre, self.post, bp.conn.All2All(), tau_rise=bm.ones(2))


if __name__ == '__main__':
    unittest.main()
