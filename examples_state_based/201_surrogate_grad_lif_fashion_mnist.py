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
Reproduce the results of the``spytorch`` tutorial 2 & 3:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial2.ipynb
- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial3.ipynb

"""

import time

import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

import brainpy.state_based as brainpy
import brainstate
import braintools

dataset = load_dataset("zalando-datasets/fashion_mnist")

# images
X_train = np.array(np.stack(dataset['train']['image']), dtype=np.uint8)
X_test = np.array(np.stack(dataset['test']['image']), dtype=np.uint8)
X_train = (X_train / 255).reshape(-1, 28 * 28).astype(jnp.float32)
X_test = (X_test / 255).reshape(-1, 28 * 28).astype(jnp.float32)
print(f'Training image shape: {X_train.shape}, testing image shape: {X_test.shape}')
# labels
Y_train = np.array(dataset['train']['label'], dtype=np.int32)
Y_test = np.array(dataset['test']['label'], dtype=np.int32)


class SNN(brainstate.nn.DynamicsGroup):
    """
    This class implements a spiking neural network model with three layers:

       i >> r >> o

    Each two layers are connected through the exponential synapse model.
    """

    def __init__(self, num_in, num_rec, num_out):
        super().__init__()

        # parameters
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out

        # synapse: i->r
        self.i2r = brainstate.nn.Sequential(
            brainstate.nn.Linear(num_in, num_rec, w_init=braintools.init.KaimingNormal(scale=40.)),
            brainpy.Expon(num_rec, tau=10. * u.ms, g_initializer=braintools.init.ZeroInit())
        )
        # recurrent: r
        self.r = brainpy.LIF(num_rec, tau=10 * u.ms, V_reset=0 * u.mV, V_rest=0 * u.mV, V_th=1. * u.mV)
        # synapse: r->o
        self.r2o = brainstate.nn.Sequential(
            brainstate.nn.Linear(num_rec, num_out, w_init=braintools.init.KaimingNormal(scale=2.)),
            brainpy.Expon(num_out, tau=10. * u.ms, g_initializer=braintools.init.ZeroInit())
        )

    def update(self, spikes):
        r_spikes = self.r(self.i2r(spikes) * u.mA)
        out = self.r2o(r_spikes)
        return out, r_spikes

    def predict(self, spikes):
        r_spikes = self.r(self.i2r(spikes) * u.mA)
        out = self.r2o(r_spikes)
        return out, r_spikes, self.r.V.value


with brainstate.environ.context(dt=1.0 * u.ms):
    # inputs
    batch_size = 256

    # spiking neural networks
    net = SNN(num_in=X_train.shape[-1], num_rec=100, num_out=10)

    # encoding inputs as spikes
    encoder = braintools.LatencyEncoder(tau=100 * u.ms)


    @brainstate.transform.jit
    def predict(xs):
        brainstate.nn.init_all_states(net, xs.shape[0])
        xs = encoder(xs)
        outs, spikes, vs = brainstate.transform.for_loop(net.predict, xs)
        return outs, spikes, vs


    def visualize(xs):
        # visualization function
        outs, spikes, vs = predict(xs)
        xs = np.asarray(encoder(xs))
        vs = np.asarray(vs.to_decimal(u.mV))
        # vs = np.where(spikes, vs, 5.0)
        fig, gs = braintools.visualize.get_figure(4, 4, 3., 4.)
        for i in range(4):
            ax = fig.add_subplot(gs[i, 0])
            i_indice, n_indices = np.where(xs[:, i])
            ax.plot(i_indice, n_indices, 'r.', markersize=1)
            plt.title('Input spikes')
            ax = fig.add_subplot(gs[i, 1])
            i_indice, n_indices = np.where(spikes[:, i])
            ax.plot(i_indice, n_indices, 'r.', markersize=1)
            plt.title('Recurrent spikes')
            ax = fig.add_subplot(gs[i, 2])
            ax.plot(vs[:, i])
            plt.title('Membrane potential')
            ax = fig.add_subplot(gs[i, 3])
            ax.plot(outs[:, i])
            plt.title('Output')
        plt.show()


    # visualization of the spiking activity
    visualize(X_test[:4])

    # optimizer
    optimizer = braintools.optim.Adam(lr=1e-3)
    optimizer.register_trainable_weights(net.states(brainstate.ParamState))


    def loss_fun(xs, ys):
        # initialize states
        brainstate.nn.init_all_states(net, xs.shape[0])

        # encode inputs
        xs = encoder(xs)

        # predictions
        outs, r_spikes = brainstate.transform.for_loop(net.update, xs)

        # Here we set up our regularize loss
        # The strength parameters here are merely a guess and there should be ample
        # room for improvement by tuning these parameters.
        l1_loss = 1e-5 * u.math.sum(r_spikes)  # L1 loss on total number of spikes
        l2_loss = 1e-5 * u.math.mean(
            u.math.sum(u.math.sum(r_spikes, axis=0), axis=0) ** 2)  # L2 loss on spikes per neuron

        # predictions
        predicts = u.math.max(outs, axis=0)  # max over time, [T, B, C] -> [B, C]
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(predicts, ys).mean()
        correct_n = u.math.sum(ys == u.math.argmax(predicts, axis=1))  # compare to labels
        return loss + l2_loss + l1_loss, correct_n


    @brainstate.transform.jit
    def train_fn(xs, ys):
        grads, loss, correct_n = brainstate.transform.grad(
            loss_fun, net.states(brainstate.ParamState), has_aux=True, return_value=True)(xs, ys)
        optimizer.update(grads)
        return loss, correct_n


    n_epoch = 20
    train_losses, train_accs = [], []
    indices = np.arange(X_train.shape[0])

    for epoch_i in range(n_epoch):
        indices = brainstate.random.shuffle(indices)

        # training phase
        t0 = time.time()
        loss, train_acc = [], 0.
        for i in range(0, X_train.shape[0], batch_size):
            X = X_train[indices[i: i + batch_size]]
            Y = Y_train[indices[i: i + batch_size]]
            l, correct_num = train_fn(X, Y)
            loss.append(l)
            train_acc += correct_num
        train_acc /= X_train.shape[0]
        train_loss = jnp.mean(jnp.asarray(loss))
        optimizer.lr.step_epoch()

        # testing phase
        loss, test_acc = [], 0.
        for i in range(0, X_test.shape[0], batch_size):
            X = X_test[i: i + batch_size]
            Y = Y_test[i: i + batch_size]
            l, correct_num = loss_fun(X, Y)
            loss.append(l)
            test_acc += correct_num
        test_acc /= X_test.shape[0]
        test_loss = jnp.mean(jnp.asarray(loss))

        t = (time.time() - t0) / 60
        print(f"Epoch {epoch_i}: train loss={train_loss:.3f}, acc={train_acc:.3f}, "
              f"test loss={test_loss:.3f}, acc={test_acc:.3f}, time={t:.2f} min")
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    fig, gs = braintools.visualize.get_figure(1, 2, 3, 4)
    fig.add_subplot(gs[0])
    plt.plot(np.asarray(train_losses))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.add_subplot(gs[1])
    plt.plot(np.asarray(train_accs))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    visualize(X_test[:4])
