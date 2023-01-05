# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from functools import partial

import brainpy_datasets as bd

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp

bm.set_environment(mode=bm.training_mode, dt=1.)


class BasicBlock(bp.DynamicalSystem):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, is_last=False):
    super(BasicBlock, self).__init__()
    self.is_last = is_last
    self.conv1 = bp.layers.Conv2D(in_planes, planes, kernel_size=(3, 3), strides=stride, padding=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn1 = bp.layers.BatchNorm2D(planes)
    self.conv2 = bp.layers.Conv2D(planes, planes, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn2 = bp.layers.BatchNorm2D(planes)

    # self.shortcut = bp.layers.Identity()
    self.shortcut = bp.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = bp.Sequential(
        bp.layers.Conv2D(in_planes, self.expansion * planes, kernel_size=1, strides=stride,
                         w_initializer=bp.init.KaimingNormal(mode='fan_out')),
        bp.layers.BatchNorm2D(self.expansion * planes)
      )

  def update(self, s, x):
    out = bm.relu(self.bn1(s, self.conv1(s, x)))
    out = self.bn2(s, self.conv2(s, out))
    out += self.shortcut(s, x)
    preact = out
    out = bm.relu(out)
    if self.is_last:
      return out, preact
    else:
      return out


class Bottleneck(bp.DynamicalSystem):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, is_last=False):
    super(Bottleneck, self).__init__()
    self.is_last = is_last
    self.conv1 = bp.layers.Conv2D(in_planes, planes, kernel_size=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn1 = bp.layers.BatchNorm2D(planes)
    self.conv2 = bp.layers.Conv2D(planes, planes, kernel_size=(3, 3), strides=stride, padding=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn2 = bp.layers.BatchNorm2D(planes)
    self.conv3 = bp.layers.Conv2D(planes, self.expansion * planes, kernel_size=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn3 = bp.layers.BatchNorm2D(self.expansion * planes)

    # self.shortcut = bp.layers.Identity()
    self.shortcut = bp.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = bp.Sequential(
        bp.layers.Conv2D(in_planes, self.expansion * planes, kernel_size=1, strides=stride,
                         w_initializer=bp.init.KaimingNormal(mode='fan_out')),
        bp.layers.BatchNorm2D(self.expansion * planes)
      )

  def update(self, s, x):
    out = bm.relu(self.bn1(s, self.conv1(s, x)))
    out = bm.relu(self.bn2(s, self.conv2(s, out)))
    out = self.bn3(s, self.conv3(s, out))
    out += self.shortcut(s, x)
    preact = out
    out = bm.relu(out)
    if self.is_last:
      return out, preact
    else:
      return out


class ResNet(bp.DynamicalSystem):
  def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = bp.layers.Conv2D(3, 64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                                  w_initializer=bp.init.KaimingNormal(mode='fan_out'))
    self.bn1 = bp.layers.BatchNorm2D(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.linear = bp.layers.Dense(512 * block.expansion, num_classes)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.nodes():
        if isinstance(m, Bottleneck):
          # nn.init.constant_(m.bn3.weight, 0)
          m.bn3.scale[:] = 0
        elif isinstance(m, BasicBlock):
          m.bn2.scale[:] = 0

  def get_bn_before_relu(self):
    if isinstance(self.layer1[0], Bottleneck):
      bn1 = self.layer1[-1].bn3
      bn2 = self.layer2[-1].bn3
      bn3 = self.layer3[-1].bn3
      bn4 = self.layer4[-1].bn3
    elif isinstance(self.layer1[0], BasicBlock):
      bn1 = self.layer1[-1].bn2
      bn2 = self.layer2[-1].bn2
      bn3 = self.layer3[-1].bn2
      bn4 = self.layer4[-1].bn2
    else:
      raise NotImplementedError('ResNet unknown block error !!!')

    return [bn1, bn2, bn3, bn4]

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for i in range(num_blocks):
      stride = strides[i]
      layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
      self.in_planes = planes * block.expansion
    return bp.Sequential(*layers)

  def update(self, s, x, is_feat=False, preact=False):
    out = bm.relu(self.bn1(s, self.conv1(s, x)))
    f0 = out
    out, f1_pre = self.layer1(s, out)
    f1 = out
    out, f2_pre = self.layer2(s, out)
    f2 = out
    out, f3_pre = self.layer3(s, out)
    f3 = out
    out, f4_pre = self.layer4(s, out)
    f4 = out
    # out = self.avgpool(s, out)
    # out = out.reshape(128, -1)
    out = bm.mean(out, axis=(1, 2))
    f5 = out
    out = self.linear(s, out)
    if is_feat:
      if preact:
        return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
      else:
        return [f0, f1, f2, f3, f4, f5], out
    else:
      return out


def ResNet18(**kwargs):
  return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
  return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
  return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
  return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
  return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def main():
  parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
  parser.add_argument('-platform', default='cpu', help='platform')
  parser.add_argument('-batch', default=128, type=int, help='batch size')
  parser.add_argument('-n_epoch', default=64, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('-data-dir', default='./data', type=str, help='root dir of Fashion-MNIST dataset')
  parser.add_argument('-out-dir', default='./logs', type=str, help='root dir for saving logs and checkpoint')
  parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
  args = parser.parse_args()
  print(args)

  bm.set_platform(args.platform)

  out_dir = os.path.join(args.out_dir, f'b{args.batch}_lr{args.lr}_epoch{args.n_epoch}')

  # dataset
  train_set = bd.vision.MNIST(root=args.data_dir, split='train', download=True)
  test_set = bd.vision.MNIST(root=args.data_dir, split='test', download=True)
  x_train = bm.asarray(train_set.data / 255, dtype=bm.float_).reshape((-1, 28, 28, 1))
  y_train = bm.asarray(train_set.targets, dtype=bm.int_)
  x_test = bm.asarray(test_set.data / 255, dtype=bm.float_).reshape((-1, 28, 28, 1))
  y_test = bm.asarray(test_set.targets, dtype=bm.int_)

  with bm.training_environment():
    net = ResNet18(num_classes=10)

  # loss function
  @bm.to_object(child_objs=net)
  def loss_fun(X, Y, fit=True):
    s = {'fit': fit}
    predictions = net(s, X)
    l = bp.losses.cross_entropy_loss(predictions, Y)
    n = bm.sum(predictions.argmax(1) == Y)
    return l, n

  grad_fun = bm.grad(loss_fun, grad_vars=net.train_vars().unique(), has_aux=True, return_value=True)

  # optimizer
  optimizer = bp.optim.Adam(bp.optim.ExponentialDecay(args.lr, 1, 0.9999),
                            train_vars=net.train_vars().unique())

  @bm.jit
  @bm.to_object(child_objs=(grad_fun, optimizer))
  def train_fun(X, Y):
    grads, l, n = grad_fun(X, Y)
    optimizer.update(grads)
    return l, n

  predict_loss_fun = bm.jit(partial(loss_fun, fit=False), child_objs=loss_fun)

  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    args_txt.write(str(args))
    args_txt.write('\n')
    args_txt.write(' '.join(sys.argv))

  max_test_acc = -1
  for epoch_i in range(0, args.n_epoch):
    start_time = time.time()
    loss, train_acc = [], 0.
    for i in range(0, x_train.shape[0], args.batch):
      xs = x_train[i: i + args.batch]
      ys = y_train[i: i + args.batch]
      l, n = train_fun(xs, ys)
      if (i / args.batch) % 100 == 0:
        print(f'Epoch {epoch_i}: Train {i} batch, loss = {bm.mean(l):.4f}')
      loss.append(l)
      train_acc += n
    train_acc /= x_train.shape[0]
    train_loss = bm.mean(bm.asarray(loss))

    loss, test_acc = [], 0.
    for i in range(0, x_test.shape[0], args.batch):
      xs = x_test[i: i + args.batch]
      ys = y_test[i: i + args.batch]
      l, n = predict_loss_fun(xs, ys)
      loss.append(l)
      test_acc += n
    test_acc /= x_test.shape[0]
    test_loss = bm.mean(bm.asarray(loss))

    t = time.time() - start_time
    print(f'epoch {epoch_i}, used {t:.3f} seconds, '
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
  correct_num = 0
  for i in range(0, x_test.shape[0], 512):
    xs = x_test[i: i + 512]
    ys = y_test[i: i + 512]
    correct_num += predict_loss_fun(xs, ys)[1]
  print('Max test accuracy: ', correct_num / x_test.shape[0])


if __name__ == '__main__':
  main()
  # import time
  #
  # with bm.training_environment():
  #   net = ResNet34()
  #   x = bm.random.randn(2, 32, 32, 1)
  #   start = time.time()
  #   feats, logit = net({'fit': False}, x, is_feat=True, preact=True)
  #   end = time.time()
  #   print(f'time: {end - start}')
  #
  #   start = time.time()
  #   feats, logit = net({'fit': False}, x, is_feat=True, preact=True)
  #   end = time.time()
  #   print(f'time: {end - start}')
  #
  #   for f in feats:
  #     print(f.shape, f.min().item(), f.max().item())
  #   print(logit.shape)
  #
  #   for m in net.get_bn_before_relu():
  #     if isinstance(m, bp.layers.BatchNorm2D):
  #       print('pass')
  #     else:
  #       print('warning')
