# -*- coding: utf-8 -*-

import argparse
import os.path
import sys
import time

import brainpy_datasets as bd
from tensorboardX import SummaryWriter

import brainpy as bp
import brainpy.math as bm

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-platform', default='cpu', help='device')
parser.add_argument('-b', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args()
print(args)

out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
  args_txt.write(str(args))
  args_txt.write('\n')
  args_txt.write(' '.join(sys.argv))

bm.set_platform(args.platform)


class SNN(bp.DynamicalSystem):
  def __init__(self, tau):
    super().__init__()

    self.layer = bp.Sequential(
      bp.layers.Flatten(),
      bp.layers.Dense(28 * 28, 10, b_initializer=None),
      bp.neurons.LIF(10, V_rest=0., V_reset=0., V_th=1., tau=tau, spike_fun=bm.surrogate.arctan),
    )

  def update(self, p, x):
    self.layer(p, x)
    return self.layer[-1].spike


net = SNN(args.tau)

# data
train_data = bd.vision.MNIST(r'D:/data', split='train', download=True)
test_data = bd.vision.MNIST(r'D:/data', split='test', download=True)
x_train = bm.asarray(train_data.data / 255, dtype=bm.float_)
y_train = bm.asarray(train_data.targets, dtype=bm.int_)
x_test = bm.asarray(test_data.data / 255, dtype=bm.float_)
y_test = bm.asarray(test_data.targets, dtype=bm.int_)


def get_train_data():
  key = bm.random.DEFAULT.split_key()
  X = bm.random.permutation(x_train, key=key)
  Y = bm.random.permutation(y_train, key=key)
  for i in range(0, X.shape[0], args.b):
    yield X[i:i + args.b], Y[i: i + args.b]


def get_test_data():
  key = bm.random.DEFAULT.split_key()
  X = bm.random.permutation(x_test, key=key)
  Y = bm.random.permutation(y_test, key=key)
  for i in range(0, X.shape[0], args.b):
    yield X[i:i + args.b], Y[i: i + args.b]


# optimizer
optimizer = bp.optim.Adam(lr=args.lr, train_vars=net.train_vars())

# loss
encoder = bp.encoding.PoissonEncoder(min_val=0., max_val=1.)


@bm.to_object(child_objs=(net, encoder))
def loss_fun(xs, ys):
  xs = encoder(xs, num_step=args.T)
  shared = bm.form_shared_args(num_step=xs.shape[0], dt=1.)
  outs = bm.for_loop(net, (shared, xs))
  out_fr = bm.mean(outs, axis=0)
  ys_onehot = bm.one_hot(ys, 10, dtype=bm.float_)
  loss = bp.losses.mean_squared_error(out_fr, ys_onehot)
  acc = bm.mean(out_fr.argmax(1) == ys)
  return loss, {'acc': acc}


# trainer
trainer = bp.train.BPTT(net,
                        loss_fun=loss_fun,
                        loss_has_aux=True,
                        optimizer=optimizer)

trainer.fit(get_test_data,
            get_test_data,
            num_epoch=args.epochs)

writer = SummaryWriter(out_dir, purge_step=0)

