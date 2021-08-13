# -*- coding: utf-8 -*-

import numpy as np

from brainpy.dnn.base import Module
from brainpy.dnn.imports import jmath, jax
from brainpy.math.jax.ndarray import TrainVar
from brainpy.primary import collector

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'NesterovMomentum',
  'Adam',
]


class Optimizer(Module):
  def __init__(self, train_vars, lr, name):
    super(Optimizer, self).__init__(name=name)
    if isinstance(train_vars, collector.ArrayCollector):
      train_vars = list(train_vars.subset(TrainVar).values())
      # train_vars = list(train_vars.values())
    elif isinstance(train_vars, (list, tuple)):
      train_vars = list(train_vars)
    else:
      raise ValueError
    self.lr = lr
    self._train_vars = train_vars
    self._dynamic_vars = []  # dynamic variables

  def register_dynamic_vars(self, variables):
    self._dynamic_vars.extend(variables)

  def vars(self, method='absolute'):
    gather = collector.ArrayCollector()
    for i, v in enumerate(self._dynamic_vars):
      gather[f'_v{i}'] = v
    gather.update(super(Optimizer, self).vars(method=method))
    return gather


class SGD(Optimizer):
  def __init__(self, lr, train_vars, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def __call__(self, grads):
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    for g, p in zip(grads, self._train_vars):
      p.value -= self.lr * g


class Momentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)
    self.momentum = momentum
    self.ms = [TrainVar(jmath.zeros_like(x)) for x in self._train_vars]
    self.register_dynamic_vars(self.ms)

  def __call__(self, grads):
    for g, p, m in zip(grads, self._train_vars, self.ms):
      m.value = g + self.momentum * m.value
      p.value -= self.lr * m.value


class NesterovMomentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(NesterovMomentum, self).__init__(lr=lr, train_vars=train_vars, name=name)
    self.momentum = momentum
    self.ms = [TrainVar(jmath.zeros_like(x)) for x in self._train_vars]
    self.register_dynamic_vars(self.ms)

  def __call__(self, grads):
    for g, p, m in zip(grads, self._train_vars, self.ms):
      m.value = g + self.momentum * m.value
      p.value -= self.lr * (g + self.momentum * m.value)


class Adam(Optimizer):
  def __init__(self, lr, train_vars, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr=lr, train_vars=train_vars, name=name)
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.step = jmath.array([0])
    self.ms = [TrainVar(jmath.zeros_like(x)) for x in self._train_vars]
    self.vs = [TrainVar(jmath.zeros_like(x)) for x in self._train_vars]
    self.register_dynamic_vars(self.ms)
    self.register_dynamic_vars(self.vs)

  def __call__(self, grads):
    """Updates variables and other state based on Adam algorithm.
    """
    assert len(grads) == len(self._train_vars), 'Expecting as many gradients as trainable variables'
    self.step.value += 1
    lr = self.lr * np.sqrt(1 - self.beta2 ** self.step.value) / (1 - self.beta1 ** self.step.value)
    for g, p, m, v in zip(grads, self._train_vars, self.ms, self.vs):
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      p.value -= lr * m.value * jax.lax.rsqrt(v.value + self.eps)
