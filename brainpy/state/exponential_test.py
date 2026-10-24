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


import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp

from brainpy.state import Expon, DualExpon


class TestExponentialSynapse(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.in_size = 10
        self.batch_size = 5
        self.time_steps = 100

    def generate_input(self):
        return brainstate.random.randn(self.time_steps, self.batch_size, self.in_size) * u.mS

    def test_expon_synapse(self):
        tau = 20.0 * u.ms
        synapse = Expon(self.in_size, tau=tau)
        inputs = self.generate_input()

        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.tau, tau)

        synapse.init_state(self.batch_size)
        call = brainstate.transform.jit(synapse)
        with brainstate.environ.context(dt=0.1 * u.ms):
            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

        constant_input = jnp.ones((self.batch_size, self.in_size)) * u.mS
        out1 = call(constant_input)
        out2 = call(constant_input)
        self.assertTrue(jnp.all(out2 > out1))

    def test_dualexpon_synapse(self):
        tau_rise = 1.0 * u.ms
        tau_decay = 10.0 * u.ms
        synapse = DualExpon(self.in_size, tau_rise=tau_rise, tau_decay=tau_decay)

        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.tau_rise, tau_rise)
        self.assertEqual(synapse.tau_decay, tau_decay)
        self.assertTrue(synapse.normalize)

        synapse.init_state(self.batch_size)
        call = brainstate.transform.jit(synapse)
        inputs = self.generate_input()

        with brainstate.environ.context(dt=0.1 * u.ms):
            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

    def test_dualexpon_single_spike_response(self):
        synapse = DualExpon(self.in_size, tau_rise=1.0 * u.ms, tau_decay=10.0 * u.ms)
        synapse.init_state(self.batch_size)
        call = brainstate.transform.jit(synapse)

        with brainstate.environ.context(dt=0.1 * u.ms):
            spike_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            spike_input = spike_input.at[0, 0].set(1.0 * u.mS)

            out0 = call(spike_input)
            self.assertEqual(out0.shape, (self.batch_size, self.in_size))

            outputs = [out0]
            zero_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            for _ in range(20):
                outputs.append(call(zero_input))

            self.assertTrue(jnp.allclose(out0[0, 0].to_decimal(u.mS), 0.0))
            self.assertTrue(jnp.any(outputs[1][0, 0] > 0.0 * u.mS))
            self.assertTrue(jnp.any(outputs[-1][0, 0] >= 0.0 * u.mS))

    def test_dualexpon_without_normalization(self):
        syn_norm = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=True,
            amplitude=1.0,
        )
        syn_raw = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=False,
            amplitude=1.0,
        )

        syn_norm.init_state(self.batch_size)
        syn_raw.init_state(self.batch_size)

        call_norm = brainstate.transform.jit(syn_norm)
        call_raw = brainstate.transform.jit(syn_raw)

        with brainstate.environ.context(dt=0.1 * u.ms):
            spike_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            spike_input = spike_input.at[0, 0].set(1.0 * u.mS)
            zero_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS

            outs_norm = [call_norm(spike_input)]
            outs_raw = [call_raw(spike_input)]
            for _ in range(40):
                outs_norm.append(call_norm(zero_input))
                outs_raw.append(call_raw(zero_input))

            peak_norm = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs_norm]))
            peak_raw = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs_raw]))

            self.assertTrue(peak_norm > peak_raw)

    def test_dualexpon_amplitude_with_normalization(self):
        syn1 = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=True,
            amplitude=1.0,
        )
        syn2 = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=True,
            amplitude=2.0,
        )

        syn1.init_state(self.batch_size)
        syn2.init_state(self.batch_size)

        call1 = brainstate.transform.jit(syn1)
        call2 = brainstate.transform.jit(syn2)

        with brainstate.environ.context(dt=0.1 * u.ms):
            spike_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            spike_input = spike_input.at[0, 0].set(1.0 * u.mS)
            zero_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS

            outs1 = [call1(spike_input)]
            outs2 = [call2(spike_input)]
            for _ in range(40):
                outs1.append(call1(zero_input))
                outs2.append(call2(zero_input))

            peak1 = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs1]))
            peak2 = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs2]))

            self.assertTrue(peak2 > peak1)

    def test_dualexpon_amplitude_without_normalization(self):
        syn1 = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=False,
            amplitude=1.0,
        )
        syn2 = DualExpon(
            self.in_size,
            tau_rise=1.0 * u.ms,
            tau_decay=10.0 * u.ms,
            normalize=False,
            amplitude=2.0,
        )

        syn1.init_state(self.batch_size)
        syn2.init_state(self.batch_size)

        call1 = brainstate.transform.jit(syn1)
        call2 = brainstate.transform.jit(syn2)

        with brainstate.environ.context(dt=0.1 * u.ms):
            spike_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            spike_input = spike_input.at[0, 0].set(1.0 * u.mS)
            zero_input = jnp.zeros((self.batch_size, self.in_size)) * u.mS

            outs1 = [call1(spike_input)]
            outs2 = [call2(spike_input)]
            for _ in range(40):
                outs1.append(call1(zero_input))
                outs2.append(call2(zero_input))

            peak1 = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs1]))
            peak2 = jnp.max(jnp.asarray([x[0, 0].to_decimal(u.mS) for x in outs2]))

            self.assertTrue(peak2 > peak1)

    def test_dualexpon_non_callable_delta_input(self):
        synapse = DualExpon(self.in_size, tau_rise=1.0 * u.ms, tau_decay=10.0 * u.ms)
        synapse.init_state(self.batch_size)
        call = brainstate.transform.jit(synapse)

        with brainstate.environ.context(dt=0.1 * u.ms):
            delta = jnp.zeros((self.batch_size, self.in_size)) * u.mS
            delta = delta.at[0, 0].set(1.0 * u.mS)

            synapse.add_delta_input('test_delta', delta)

            out0 = call()
            out1 = call()

            self.assertTrue(jnp.allclose(out0[0, 0].to_decimal(u.mS), 0.0))
            self.assertTrue(out1[0, 0] > 0.0 * u.mS)

    def test_keep_size(self):
        in_size = (2, 3)
        for SynapseClass in [Expon, DualExpon]:
            synapse = SynapseClass(in_size)
            self.assertEqual(synapse.in_size, in_size)
            self.assertEqual(synapse.out_size, in_size)

            inputs = brainstate.random.randn(self.time_steps, self.batch_size, *in_size) * u.mS
            synapse.init_state(self.batch_size)
            call = brainstate.transform.jit(synapse)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertEqual(out.shape, (self.batch_size, *in_size))


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1 * u.ms):
        unittest.main()