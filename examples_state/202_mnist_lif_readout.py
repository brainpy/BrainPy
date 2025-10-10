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


import argparse
import time

import brainpy
import braintools
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

import brainstate

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-platform', default='cpu', help='device')
parser.add_argument('-batch', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
args = parser.parse_args()
print(args)


class SNN(brainstate.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.l1 = brainstate.nn.Linear(
            28 * 28, 10, b_init=None, w_init=braintools.init.LecunNormal(scale=10., unit=u.mA))
        self.l2 = brainpy.state.LIF(10, V_rest=0. * u.mV, V_reset=0. * u.mV, V_th=1. * u.mV, tau=tau * u.ms)

    def update(self, x):
        return self.l2(self.l1(x))

    def predict(self, x):
        spikes = self.l2(self.l1(x))
        return self.l2.V.value, spikes


with brainstate.environ.context(dt=1.0 * u.ms):
    net = SNN(args.tau)

    dataset = load_dataset('mnist')
    # images
    X_train = np.array(np.stack(dataset['train']['image']), dtype=np.uint8)
    X_test = np.array(np.stack(dataset['test']['image']), dtype=np.uint8)
    X_train = (X_train / 255).reshape(-1, 28 * 28).astype(jnp.float32)
    X_test = (X_test / 255).reshape(-1, 28 * 28).astype(jnp.float32)
    # labels
    Y_train = np.array(dataset['train']['label'], dtype=np.int32)
    Y_test = np.array(dataset['test']['label'], dtype=np.int32)


    @brainstate.transform.jit
    def predict(xs):
        brainstate.nn.init_all_states(net, xs.shape[0])
        xs = (xs + 0.02)
        xs = brainstate.random.rand(args.T, *xs.shape) < xs
        vs, outs = brainstate.transform.for_loop(net.predict, xs)
        return vs, outs


    def visualize(xs):
        vs, outs = predict(xs)
        vs = np.asarray(vs.to_decimal(u.mV))
        fig, gs = braintools.visualize.get_figure(4, 2, 3., 6.)
        for i in range(4):
            ax = fig.add_subplot(gs[i, 0])
            i_indice, n_indices = np.where(outs[:, i])
            ax.plot(i_indice, n_indices, 'r.', markersize=1)
            ax.set_xlim([0, args.T])
            ax.set_ylim([0, net.l2.varshape[0]])
            ax = fig.add_subplot(gs[i, 1])
            ax.plot(vs[:, i])
            ax.set_xlim([0, args.T])
        plt.show()


    # visualization of the spiking activity
    visualize(X_test[:4])


    @brainstate.transform.jit
    def loss_fun(xs, ys):
        # initialize states
        brainstate.nn.init_all_states(net, xs.shape[0])

        # encoding inputs as spikes
        xs = brainstate.random.rand(args.T, *xs.shape) < xs

        # shared arguments for looping over time
        outs = brainstate.transform.for_loop(net.update, xs)
        out_fr = u.math.mean(outs, axis=0)  # [T, B, C] -> [B, C]
        ys_onehot = brainstate.nn.one_hot(ys, 10, dtype=float)
        l = braintools.metric.squared_error(out_fr, ys_onehot).mean()
        n = u.math.sum(out_fr.argmax(1) == ys)
        return l, n


    # gradient function
    grad_fun = brainstate.transform.grad(loss_fun, net.states(brainstate.ParamState), has_aux=True, return_value=True)

    # optimizer
    optimizer = braintools.optim.Adam(lr=args.lr)
    optimizer.register_trainable_weights(net.states(brainstate.ParamState))


    # train
    @brainstate.transform.jit
    def train(xs, ys):
        print('compiling...')

        grads, l, n = grad_fun(xs, ys)
        optimizer.update(grads)
        return l, n


    # training loop
    for epoch_i in range(args.epochs):
        key = brainstate.random.split_key()
        X_train = brainstate.random.shuffle(X_train, key=key)
        Y_train = brainstate.random.shuffle(Y_train, key=key)

        # training phase
        t0 = time.time()
        loss, train_acc = [], 0.
        for i in range(0, X_train.shape[0], args.batch):
            X = X_train[i: i + args.batch]
            Y = Y_train[i: i + args.batch]
            l, correct_num = train(X, Y)
            loss.append(l)
            train_acc += correct_num
        train_acc /= X_train.shape[0]
        train_loss = jnp.mean(jnp.asarray(loss))
        optimizer.lr.step_epoch()

        # testing phase
        loss, test_acc = [], 0.
        for i in range(0, X_test.shape[0], args.batch):
            X = X_test[i: i + args.batch]
            Y = Y_test[i: i + args.batch]
            l, correct_num = loss_fun(X, Y)
            loss.append(l)
            test_acc += correct_num
        test_acc /= X_test.shape[0]
        test_loss = jnp.mean(jnp.asarray(loss))

        t = (time.time() - t0) / 60
        print(f'epoch {epoch_i}, used {t:.3f} min, '
              f'train loss = {train_loss:.4f}, acc = {train_acc:.4f}, '
              f'test loss = {test_loss:.4f}, acc = {test_acc:.4f}')

    # inference
    correct_num = 0.
    for i in range(0, X_test.shape[0], 512):
        X = X_test[i: i + 512]
        Y = Y_test[i: i + 512]
        correct_num += loss_fun(X, Y)[1]
    print('Max test accuracy: ', correct_num / X_test.shape[0])
