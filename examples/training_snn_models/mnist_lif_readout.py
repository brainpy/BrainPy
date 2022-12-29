# -*- coding: utf-8 -*-

import time
import argparse
import os.path
import sys

import brainpy_datasets as bd

import brainpy as bp
import brainpy.math as bm

parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
parser.add_argument('-platform', default='cpu', help='device')
parser.add_argument('-batch', default=64, type=int, help='batch size')
parser.add_argument('-epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

args = parser.parse_args()
print(args)

out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.batch}_lr{args.lr}')
if not os.path.exists(out_dir):
  os.makedirs(out_dir)
with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
  args_txt.write(str(args))
  args_txt.write('\n')
  args_txt.write(' '.join(sys.argv))

bm.set_platform(args.platform)
bm.set_environment(mode=bm.training_mode, dt=1.)


class SNN(bp.DynamicalSystem):
  def __init__(self, tau):
    super().__init__()

    self.layer = bp.Sequential(
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
x_train = bm.asarray(train_data.data / 255, dtype=bm.float_).reshape(-1, 28 * 28)
y_train = bm.asarray(train_data.targets, dtype=bm.int_)
x_test = bm.asarray(test_data.data / 255, dtype=bm.float_).reshape(-1, 28 * 28)
y_test = bm.asarray(test_data.targets, dtype=bm.int_)

# loss
encoder = bp.encoding.PoissonEncoder(min_val=0., max_val=1.)


@bm.to_object(child_objs=(net, encoder))
def loss_fun(xs, ys):
  net.reset_state(batch_size=xs.shape[0])
  xs = encoder(xs, num_step=args.T)
  # shared arguments for looping over time
  shared = bm.shared_args_over_time(num_step=args.T)
  outs = bm.for_loop(net, (shared, xs))
  out_fr = bm.mean(outs, axis=0)
  ys_onehot = bm.one_hot(ys, 10, dtype=bm.float_)
  l = bp.losses.mean_squared_error(out_fr, ys_onehot)
  n = bm.sum(out_fr.argmax(1) == ys)
  return l, n


# gradient
grad_fun = bm.grad(loss_fun, grad_vars=net.train_vars().unique(), has_aux=True, return_value=True)

# optimizer
optimizer = bp.optim.Adam(lr=args.lr, train_vars=net.train_vars().unique())


# train
@bm.jit
@bm.to_object(child_objs=(grad_fun, optimizer))
def train(xs, ys):
  grads, l, n = grad_fun(xs, ys)
  optimizer.update(grads)
  return l, n


max_test_acc = 0.

# computing
for epoch_i in range(args.epochs):
  bm.random.shuffle(x_train, key=123)
  bm.random.shuffle(y_train, key=123)

  t0 = time.time()
  loss, train_acc = [], 0.
  for i in range(0, x_train.shape[0], args.batch):
    X = x_train[i: i + args.batch]
    Y = y_train[i: i + args.batch]
    l, correct_num = train(X, Y)
    loss.append(l)
    train_acc += correct_num
  train_acc /= x_train.shape[0]
  train_loss = bm.mean(bm.asarray(loss))

  loss, test_acc = [], 0.
  for i in range(0, x_test.shape[0], args.batch):
    X = x_test[i: i + args.batch]
    Y = y_test[i: i + args.batch]
    l, correct_num = loss_fun(X, Y)
    loss.append(l)
    test_acc += correct_num
  test_acc /= x_test.shape[0]
  test_loss = bm.mean(bm.asarray(loss))

  t = (time.time() - t0) / 60
  print(f'epoch {epoch_i}, used {t:.3f} min, '
        f'train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, '
        f'test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}')

  if max_test_acc < test_acc:
    max_test_acc = test_acc
    states = {
      'net': net.state_dict(),
      'optimizer': optimizer.state_dict(),
      'epoch_i': epoch_i,
      'train_acc': train_acc,
      'test_acc': test_acc,
    }
    bp.checkpoints.save(out_dir, states, epoch_i)

# inference
state_dict = bp.checkpoints.load(out_dir)
net.load_state_dict(state_dict['net'])

runner = bp.DSRunner(net, data_first_axis='T')
correct_num = 0
for i in range(0, x_test.shape[0], 512):
  X = encoder(x_test[i: i + 512], num_step=args.T)
  Y = y_test[i: i + 512]
  out_fr = bm.mean(runner.predict(inputs=X, reset_state=True), axis=0)
  correct_num += bm.sum(out_fr.argmax(1) == Y)

print('Max test accuracy: ', correct_num / x_test.shape[0])
