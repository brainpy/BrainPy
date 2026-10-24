# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import unittest

import braintools
import brainstate
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import brainunit as u

from brainpy.state.delay import InputDelay
from brainpy.state import (AlignPostProj, CurrentProj, LIF, Expon, CUBA)


def _impulse(n_steps, n_pre, active):
    """A (n_steps, n_pre) drive: the given pre-neurons spike once at step 0."""
    seq = np.zeros((n_steps, n_pre))
    seq[0, active] = 1.0
    return jnp.asarray(seq)


class _AlignPostNet(brainstate.nn.Module):
    def __init__(self, n, delay, comm_w):
        super().__init__()
        self.pop = LIF(n, tau=1e6 * u.ms, V_rest=0. * u.mV, V_th=1e9 * u.mV, V_reset=0. * u.mV)
        self.proj = AlignPostProj(
            comm=brainstate.nn.Linear(n, n, w_init=comm_w, b_init=None),
            syn=Expon.desc(n, tau=100. * u.ms),
            out=CUBA.desc(scale=u.volt),
            post=self.pop,
            delay=delay,
        )

    def update(self, spk):
        with brainstate.environ.context(t=0. * u.ms):
            self.proj(spk)
            self.pop(0. * u.mA)
        return self.proj.syn.g.value


class _CurrentNet(brainstate.nn.Module):
    def __init__(self, n, delay, comm_w):
        super().__init__()
        self.pop = LIF(n, tau=1e6 * u.ms, V_rest=0. * u.mV, V_th=1e9 * u.mV, V_reset=0. * u.mV)
        self.proj = CurrentProj(
            comm=brainstate.nn.Linear(n, n, w_init=comm_w, b_init=None),
            out=CUBA(scale=u.volt),
            post=self.pop,
            delay=delay,
        )

    def update(self, spk):
        with brainstate.environ.context(t=0. * u.ms):
            self.proj(spk)
            self.pop(0. * u.mA)
        return self.pop.V.value


def _run(net, drive):
    return np.asarray([np.asarray(u.get_mantissa(net.update(drive[k]))) for k in range(drive.shape[0])])


class TestAlignPostProjDelay(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=1.0 * u.ms)

    def test_homogeneous_delay_shifts_onset(self):
        n, T = 4, 6
        drive = _impulse(T, n, active=0)
        net0 = _AlignPostNet(n, delay=None, comm_w=braintools.init.Constant(1.0 * u.mS))
        net2 = _AlignPostNet(n, delay=2. * u.ms, comm_w=braintools.init.Constant(1.0 * u.mS))
        brainstate.nn.init_all_states(net0)
        brainstate.nn.init_all_states(net2)
        g0 = _run(net0, drive).sum(axis=1)
        g2 = _run(net2, drive).sum(axis=1)
        # No delay: synapse charges at step 0.
        self.assertGreater(g0[0], 1.0)
        # delay = 2 steps: silent at steps 0, 1; charges at step 2.
        npt.assert_allclose(g2[0], 0., atol=1e-9)
        npt.assert_allclose(g2[1], 0., atol=1e-9)
        self.assertGreater(g2[2], 1.0)
        # The delayed response is the undelayed response shifted by exactly 2 steps.
        npt.assert_allclose(g2[2:], g0[:T - 2], rtol=1e-5, atol=1e-9)

    def test_delay_none_matches_baseline(self):
        # delay=None must take the original code path -> immediate onset.
        n, T = 4, 5
        drive = _impulse(T, n, active=0)
        net = _AlignPostNet(n, delay=None, comm_w=braintools.init.Constant(1.0 * u.mS))
        brainstate.nn.init_all_states(net)
        g = _run(net, drive).sum(axis=1)
        self.assertGreater(g[0], 1.0)

    def test_axonal_delay_per_post_onset(self):
        # identity comm so pre-neuron j feeds post-neuron j only.
        n, T = 4, 7
        delays = jnp.array([1., 2., 3., 4.]) * u.ms
        net = _AlignPostNet(n, delay=delays, comm_w=jnp.eye(n) * u.mS)
        brainstate.nn.init_all_states(net)
        drive = _impulse(T, n, active=slice(None))  # every pre-neuron spikes at step 0
        g = _run(net, drive)  # (T, n)
        for j in range(n):
            onset = int(np.argmax(g[:, j] > 1e-9))
            self.assertEqual(onset, j + 1, msg=f'post {j}: onset {onset} != {j + 1}')


class TestCurrentProjDelay(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=1.0 * u.ms)

    def test_homogeneous_delay_shifts_onset(self):
        n, T = 4, 6
        drive = _impulse(T, n, active=0)
        net0 = _CurrentNet(n, delay=None, comm_w=jnp.eye(n) * u.mS)
        net3 = _CurrentNet(n, delay=3. * u.ms, comm_w=jnp.eye(n) * u.mS)
        brainstate.nn.init_all_states(net0)
        brainstate.nn.init_all_states(net3)
        v0 = _run(net0, drive)[:, 0]
        v3 = _run(net3, drive)[:, 0]
        # No delay: post neuron 0 receives current at step 0.
        self.assertGreater(abs(v0[0]), 0.)
        # delay = 3 steps: silent until step 3.
        for k in range(3):
            npt.assert_allclose(v3[k], 0., atol=1e-12)
        self.assertGreater(abs(v3[3]), 0.)


class TestInputDelaySeam(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=1.0 * u.ms)

    def test_none_is_identity(self):
        seam = InputDelay(3, delay=None)
        brainstate.nn.init_all_states(seam)
        x = jnp.array([1., 2., 3.])
        out = seam.update(x)
        npt.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_global_delay_shifts_whole_frame(self):
        # delay = 2 steps (dt = 1 ms); output at step t is the input at step t-2.
        seam = InputDelay(2, delay=2.0 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([1., 10.]), jnp.array([2., 20.]),
               jnp.array([3., 30.]), jnp.array([4., 40.])]
        outs = [np.asarray(seam.update(x)) for x in seq]
        npt.assert_allclose(outs[0], [0., 0.], atol=1e-6)
        npt.assert_allclose(outs[1], [0., 0.], atol=1e-6)
        npt.assert_allclose(outs[2], [1., 10.], atol=1e-6)
        npt.assert_allclose(outs[3], [2., 20.], atol=1e-6)

    def test_axonal_delay_per_element(self):
        # element 0 delayed 1 step, element 1 delayed 3 steps.
        seam = InputDelay(2, delay=jnp.array([1.0, 3.0]) * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([1., 10.]), jnp.array([2., 20.]), jnp.array([3., 30.]),
               jnp.array([4., 40.]), jnp.array([5., 50.])]
        outs = [np.asarray(seam.update(x)) for x in seq]
        npt.assert_allclose(outs[1][0], 1., atol=1e-6)
        npt.assert_allclose(outs[2][0], 2., atol=1e-6)
        npt.assert_allclose(outs[3], [3., 10.], atol=1e-6)
        npt.assert_allclose(outs[4], [4., 20.], atol=1e-6)

    def test_fractional_delay_interpolates(self):
        # delay = 1.5 steps -> out[t] = 0.5*in[t-1] + 0.5*in[t-2].
        seam = InputDelay(1, delay=1.5 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = [jnp.array([10.]), jnp.array([20.]), jnp.array([30.]), jnp.array([40.])]
        outs = [float(np.asarray(seam.update(x))[0]) for x in seq]
        npt.assert_allclose(outs[0], 0., atol=1e-6)
        npt.assert_allclose(outs[1], 5., atol=1e-6)    # 0.5*10 + 0.5*0
        npt.assert_allclose(outs[2], 15., atol=1e-6)   # 0.5*20 + 0.5*10
        npt.assert_allclose(outs[3], 25., atol=1e-6)   # 0.5*30 + 0.5*20

    def test_runs_under_jitted_for_loop(self):
        seam = InputDelay(3, delay=2.0 * u.ms)
        brainstate.nn.init_all_states(seam)
        seq = jnp.arange(5 * 3, dtype=float).reshape(5, 3)
        outs = brainstate.transform.for_loop(lambda x: seam.update(x), seq)
        npt.assert_allclose(np.asarray(outs[2]), np.asarray(seq[0]), atol=1e-6)
        npt.assert_allclose(np.asarray(outs[4]), np.asarray(seq[2]), atol=1e-6)

    def test_heterogeneous_per_connection_diagonal_gather(self):
        # 3 connections from 3 distinct pre-neurons, delays [1, 5, 10] steps.
        # A single impulse delivers three pulses offset by 1, 5, 10 steps,
        # directly exercising retrieve_at_step(steps, pre_ids).
        n_pre, T = 3, 12
        pre_ids = jnp.array([0, 1, 2])
        delays = jnp.array([1., 5., 10.]) * u.ms
        seam = InputDelay(n_pre, delay=delays, indices=pre_ids)
        brainstate.nn.init_all_states(seam)
        seq = np.zeros((T, n_pre)); seq[0, :] = 1.0
        out = np.asarray([np.asarray(seam.update(jnp.asarray(seq[k]))) for k in range(T)])  # (T, 3)
        for k, d in enumerate([1, 5, 10]):
            self.assertEqual(int(np.argmax(out[:, k])), d)
            npt.assert_allclose(out[d, k], 1.0, atol=1e-6)
            npt.assert_allclose(out[:, k].sum(), 1.0, atol=1e-6)  # exactly one pulse

    def test_per_connection_length_mismatch_raises(self):
        seam = InputDelay(3, delay=jnp.array([1., 5.]) * u.ms, indices=jnp.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            brainstate.nn.init_all_states(seam)


if __name__ == '__main__':
    unittest.main()
