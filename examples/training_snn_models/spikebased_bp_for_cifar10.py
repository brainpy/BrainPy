# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Lee, Chankyu et al. “Enabling Spike-Based Backpropagation for
  Training Deep Neural Network Architectures.” Frontiers in
  Neuroscience 14 (2019): n. pag.

"""

import sys

import tqdm
import jax
import numpy as np

sys.path.append('../../')
import argparse
import time

import jax.numpy as jnp
from jax import custom_gradient, custom_vjp
from jax.lax import stop_gradient
import brainpy_datasets as bd

import brainpy as bp
import brainpy.math as bm


# bm.disable_gpu_memory_preallocation()

bm.set_environment(bm.TrainingMode())
bm.set_platform('gpu')

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('-data', default='/mnt/d/data', type=str, help='path to dataset')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('-T', '--timesteps', default=100, type=int, help='Simulation timesteps')
parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                    help='initial learning rate', dest='lr')


class LIFNode(bp.DynamicalSystem):
  def __init__(self, size, tau=100.0, v_threshold=1.0, v_reset=0.0,
               fire: bool = True):
    super().__init__()
    bp.check.is_subclass(self.mode, [bp.math.TrainingMode, bp.math.BatchingMode])

    self.size = size
    self.tau = tau
    self.v_reset = v_reset
    self.fire = fire
    self.v_threshold = v_threshold
    self.reset_state(batch_size=1)

    self.relu_grad = custom_vjp(self.f)
    self.relu_grad.defvjp(self.f_fwd, self.f_bwd)

  def reset_state(self, batch_size=None):
    self.v = bp.init.variable_(jnp.zeros, self.size, batch_size)
    self.v[:] = self.v_reset
    # Accumulated voltage (Assuming NO fire for this neuron)
    self.v_acc = bp.init.variable_(jnp.zeros, self.size, batch_size)
    # Accumulated voltage with leaky (Assuming NO fire for this neuron)
    self.v_acc_l = bp.init.variable_(jnp.zeros, self.size, batch_size)
    if self.fire:
      # accumulated gradient
      self.grad_acc = bp.init.variable_(jnp.zeros, self.size, batch_size)

  def f(self, x):
    comp = x <= 0.0
    return comp.astype(bm.float_)

  def f_fwd(self, x):
    comp = x <= 0.0
    return comp.astype(bm.float_), (comp,)

  def f_bwd(self, res, g):
    comp, = res
    g = jnp.where(comp, 0., g)
    return (self.grad_acc.value * g,)

  def update(self, s, dv):
    self.v += dv
    if self.fire:
      spike = self.relu_grad(self.v.value - self.v_threshold)
      s = stop_gradient(spike)
      self.v = s * self.v_reset + (1 - s) * self.v
      self.v_acc += spike
      self.v_acc_l = self.v - ((self.v - self.v_reset) / self.tau) + spike
    self.v -= stop_gradient(((self.v - self.v_reset) / self.tau))
    return self.v.value


class IFNode(bp.DynamicalSystem):
  def __init__(self, size, v_threshold=0.75, v_reset=0.0):
    super().__init__()
    bp.check.is_subclass(self.mode, [bm.TrainingMode, bm.BatchingMode])

    self.size = size
    self.v_threshold = v_threshold
    self.v_reset = v_reset
    self.reset_state(1)
    self.relu_grad = custom_gradient(self.relu)

  def reset_state(self, batch_size=None):
    self.v = bp.init.variable_(jnp.zeros, self.size, batch_size)
    self.v[:] = self.v_reset

  def relu(self, x):
    comp = x <= 0.0

    def grad(dz):
      return jnp.where(comp, 0., dz)

    return comp.astype(bp.math.float_), grad

  def update(self, s, dv):
    self.v += dv
    spike = self.relu_grad(self.v - self.v_threshold)
    s = stop_gradient(spike)
    self.v = s * self.v_reset + (1 - s) * self.v
    return spike


def conv_init(shape):
  n = shape[0] * shape[1] * shape[2]
  variance1 = jnp.sqrt(1.0 / n)
  return bm.random.normal(0., variance1, size=shape)


class ResNet11(bp.DynamicalSystem):
  def __init__(self):
    super().__init__()

    linear_init = bp.init.VarianceScaling(1., 'fan_out', 'normal', seed=bm.random.split_key())

    self.cnn11 = bp.layers.Conv2d(in_channels=3, out_channels=64, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.lif11 = bp.Sequential(LIFNode((32, 32, 64)),
                               bp.layers.Dropout(0.75))
    self.avgpool1 = bp.layers.AvgPool(2, 2, channel_axis=-1)
    self.if1 = IFNode((16, 16, 64))

    self.cnn21 = bp.layers.Conv2d(in_channels=64, out_channels=128, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.lif21 = bp.Sequential(LIFNode((16, 16, 128)),
                               bp.layers.Dropout(0.75))
    self.cnn22 = bp.layers.Conv2d(in_channels=128, out_channels=128, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.shortcut1 = bp.layers.Conv2d(64, 128, kernel_size=1, strides=1,
                                      b_initializer=None, w_initializer=conv_init)
    self.lif2 = bp.Sequential(LIFNode((16, 16, 128)),
                              bp.layers.Dropout(0.75))

    self.cnn31 = bp.layers.Conv2d(in_channels=128, out_channels=256, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.lif31 = bp.Sequential(LIFNode((16, 16, 256)),
                               bp.layers.Dropout(0.75))
    self.cnn32 = bp.layers.Conv2d(in_channels=256, out_channels=256, kernel_size=3, strides=2,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.shortcut2 = bp.layers.Conv2d(128, 256, kernel_size=1, strides=2,
                                      b_initializer=None, w_initializer=conv_init)
    self.lif3 = bp.Sequential(LIFNode((8, 8, 256)),
                              bp.layers.Dropout(0.75))

    self.cnn41 = bp.layers.Conv2d(in_channels=256, out_channels=512, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.lif41 = bp.Sequential(LIFNode((8, 8, 512)),
                               bp.layers.Dropout(0.75))
    self.cnn42 = bp.layers.Conv2d(in_channels=512, out_channels=512, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.shortcut3 = bp.layers.Conv2d(256, 512, kernel_size=1, strides=1,
                                      b_initializer=None, w_initializer=conv_init)
    self.lif4 = bp.Sequential(LIFNode((8, 8, 512)),
                              bp.layers.Dropout(0.75))

    self.cnn51 = bp.layers.Conv2d(in_channels=512, out_channels=512, kernel_size=3, strides=1,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.lif51 = bp.Sequential(LIFNode((8, 8, 512)),
                               bp.layers.Dropout(0.75))

    self.cnn52 = bp.layers.Conv2d(in_channels=512, out_channels=512, kernel_size=3, strides=2,
                                  padding=(1, 1), b_initializer=None, w_initializer=conv_init)
    self.shortcut4 = bp.layers.AvgPool((1, 1), stride=(2, 2), channel_axis=-1)
    self.lif5 = bp.Sequential(LIFNode((4, 4, 512)),
                              bp.layers.Dropout(0.75))

    self.fc0 = bp.layers.Dense(512 * 4 * 4, 1024, b_initializer=None, W_initializer=linear_init)
    self.lif6 = bp.Sequential(LIFNode((1024,)),
                              bp.layers.Dropout(0.75))
    self.fc1 = bp.layers.Dense(1024, 10, b_initializer=None, W_initializer=linear_init)
    self.lif_out = LIFNode((10,), fire=False)

  def update(self, s, x):
    x = self.if1(s, self.avgpool1(s, self.lif11(s, self.cnn11(s, x))))
    x = self.lif2(s, self.cnn22(s, self.lif21(s, self.cnn21(s, x))) + self.shortcut1(s, x))
    x = self.lif3(s, self.cnn32(s, self.lif31(s, self.cnn31(s, x))) + self.shortcut2(s, x))
    x = self.lif4(s, self.cnn42(s, self.lif41(s, self.cnn41(s, x))) + self.shortcut3(s, x))
    x = self.lif5(s, self.cnn52(s, self.lif51(s, self.cnn51(s, x))) + self.shortcut4(s, x))
    out = x.reshape(x.shape[0], -1)
    out = self.lif_out(s, self.fc1(s, self.lif6(s, self.fc0(s, out))))
    return out


def normalize(data):
  data = jnp.asarray(data, dtype=bm.float_)
  data = data - jnp.asarray([0.4914, 0.4822, 0.4465], dtype=bm.float_)
  return data / jnp.asarray([0.557, 0.549, 0.5534], dtype=bm.float_)


def main():
  args = parser.parse_args()
  learning_rate = args.lr
  batch_size = args.batch_size
  num_time = args.timesteps
  dataset_root_dir = args.data

  # Prepare model
  bm.random.seed(1234)
  net = ResNet11()

  @bm.jit
  @bm.to_object(child_objs=net, dyn_vars=bm.random.DEFAULT)
  def loss_fun(x, y, fit=True):
    net.reset_state(x.shape[0])
    yy = bm.one_hot(y, 10, dtype=bm.float_)
    # poisson encoding
    x = jnp.asarray((bm.random.rand(num_time, *x.shape) < jnp.abs(x)) * jnp.sign(x), bm.float_)
    # loop over time
    for i in range(num_time):
      o = net({'fit': fit}, x[i])
    for m in net.nodes():
      if isinstance(m, LIFNode) and m.fire:
        m.v_acc += (m.v_acc < 1e-3).astype(bm.float_)
        m.grad_acc.value = ((m.v_acc_l > 1e-3).astype(bm.float_) +
                            jnp.log(1 - 1 / m.tau) * m.v_acc_l / m.v_acc)
    l = bp.losses.mean_squared_error(o / num_time, yy, reduction='sum')
    n = jnp.sum(jnp.argmax(o, axis=1), y)
    return l, n

  inference_fun = bm.jit(bm.Partial(loss_fun.target, fit=False))
  optimizer = bp.optim.SGD(bp.optim.MultiStepLR(learning_rate, milestones=[70, 100, 125], gamma=0.2),
                           net.train_vars().unique(),
                           weight_decay=5e-4)
  grad_fun = bm.grad(loss_fun, grad_vars=net.train_vars().unique(), return_value=True)

  @bm.jit
  def train_fun(x, y):
    grads, l, n = grad_fun(x, y)
    optimizer.update(grads)
    return l, n

  train_set = bd.vision.CIFAR10(root=dataset_root_dir, split='train', download=True)
  test_set = bd.vision.CIFAR10(root=dataset_root_dir, split='test', download=True)
  x_train = jax.device_put(normalize(train_set.data))
  y_train = jax.device_put(jnp.asarray(train_set.targets, dtype=bm.int_))
  x_test = jax.device_put(normalize(test_set.data))
  y_test = jax.device_put(np.asarray(test_set.targets, dtype=bm.int_))

  for epoch_i in range(200):
    x_train = bm.random.permutation(x_train, key=123)
    y_train = bm.random.permutation(y_train, key=123)

    start_time = time.time()
    train_loss = []
    for i in tqdm.tqdm(range(0, x_train.shape[0], batch_size), desc=f'Train {epoch_i}'):
      img = x_train[i:i + batch_size]
      label = y_train[i:i + batch_size]
      loss = train_fun(img, label)
      train_loss.append(loss)
    train_loss = jnp.asarray(train_loss).mean()
    end_time = time.time()
    print(f'Epoch {epoch_i}, train time {end_time - start_time} s, train loss {train_loss}')
    optimizer.lr.update_epoch()

    start_time = time.time()
    test_loss, test_acc = [], []
    for i in tqdm.tqdm(range(0, x_test.shape[0], batch_size), desc=f'Test {epoch_i}'):
      img = x_test[i:i + batch_size]
      label = y_test[i:i + batch_size]
      loss, acc = inference_fun(img, label)
      test_loss.append(loss)
      test_acc.append(acc)
    test_loss = jnp.asarray(test_loss).mean()
    test_acc = jnp.asarray(test_acc).mean()
    end_time = time.time()
    print(f'Epoch {epoch_i}, test time {end_time - start_time} s, test loss {test_loss}, test acc {test_acc}')


if __name__ == '__main__':
  main()
