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


"""
Reproduce the results of the``spytorch`` tutorial 1:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial1.ipynb

"""

import time

import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainpy
import brainstate
import braintools


class SNN(brainstate.nn.Module):
    def __init__(self, num_in, num_rec, num_out):
        super(SNN, self).__init__()

        # parameters
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out

        # synapse: i->r
        scale = 7 * (1 - (u.math.exp(-brainstate.environ.get_dt() / (1 * u.ms))))
        self.i2r = brainstate.nn.Sequential(
            brainstate.nn.Linear(
                num_in, num_rec,
                w_init=braintools.init.KaimingNormal(scale=scale, unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA)
            ),
            brainpy.Expon(num_rec, tau=5. * u.ms, g_initializer=braintools.init.Constant(0. * u.mA))
        )
        # recurrent: r
        self.r = brainpy.LIF(
            num_rec, tau=20 * u.ms, V_reset=0 * u.mV,
            V_rest=0 * u.mV, V_th=1. * u.mV,
            spk_fun=braintools.surrogate.ReluGrad()
        )
        # synapse: r->o
        self.r2o = brainstate.nn.Linear(num_rec, num_out, w_init=braintools.init.KaimingNormal())
        # # output: o
        self.o = brainpy.Expon(num_out, tau=10. * u.ms, g_initializer=braintools.init.Constant(0.))

    def update(self, spike):
        return self.o(self.r2o(self.r(self.i2r(spike))))

    def predict(self, spike):
        rec_spikes = self.r(self.i2r(spike))
        out = self.o(self.r2o(rec_spikes))
        return self.r.V.value, rec_spikes, out


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5, show=True):
    fig, gs = braintools.visualize.get_figure(*dim, 3, 3)
    if spk is not None:
        mem[spk > 0.0] = spike_height
    if isinstance(mem, u.Quantity):
        mem = mem.to_decimal(u.mV)
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(mem[:, i])
    if show:
        plt.show()


def print_classification_accuracy(output, target):
    """ Dirty little helper function to compute classification accuracy. """
    m = u.math.max(output, axis=0)  # max over time
    am = u.math.argmax(m, axis=1)  # argmax over output units
    acc = u.math.mean(target == am)  # compare to labels
    print("Accuracy %.3f" % acc)


def predict_and_visualize_net_activity(net):
    brainstate.nn.init_all_states(net, batch_size=num_sample)
    vs, spikes, outs = brainstate.transform.for_loop(net.predict, x_data, pbar=brainstate.transform.ProgressBar(10))
    plot_voltage_traces(vs, spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs)
    print_classification_accuracy(outs, y_data)


with brainstate.environ.context(dt=1.0 * u.ms):
    # network
    net = SNN(100, 4, 2)

    # dataset
    num_step = 200
    num_sample = 256
    freq = 5 * u.Hz
    x_data = brainstate.random.rand(num_step, num_sample, net.num_in) < freq * brainstate.environ.get_dt()
    y_data = u.math.asarray(brainstate.random.rand(num_sample) < 0.5, dtype=int)

    # Before training
    predict_and_visualize_net_activity(net)

    # brainstate optimizer
    optimizer = braintools.optim.Adam(lr=3e-3)
    optimizer.register_trainable_weights(net.states(brainstate.ParamState))

    def loss_fn():
        predictions = brainstate.compile.for_loop(net.update, x_data)
        predictions = u.math.mean(predictions, axis=0)  # [T, B, C] -> [B, C]
        return braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()


    @brainstate.compile.jit
    def train_fn():
        brainstate.nn.init_all_states(net, batch_size=num_sample)
        grads, l = brainstate.transform.grad(loss_fn, net.states(brainstate.ParamState), return_value=True)()
        optimizer.update(grads)
        return l


    # train the network
    train_losses = []
    t0 = time.time()
    for i in range(1, 3001):
        loss = train_fn()
        train_losses.append(loss)
        if i % 100 == 0:
            print(f'Train {i} epoch, loss = {loss:.4f}, used time {time.time() - t0:.4f} s')
            t0 = time.time()

    # visualize the training losses
    plt.plot(np.asarray(jnp.asarray(train_losses)))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")

    # predict the output according to the input data
    predict_and_visualize_net_activity(net)
