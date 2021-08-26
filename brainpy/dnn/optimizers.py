# -*- coding: utf-8 -*-

import numpy as np

from brainpy.base.collector import ArrayCollector
from brainpy.dnn.base import Module
from brainpy.dnn.imports import jmath, jax

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

    if isinstance(train_vars, ArrayCollector):
      train_vars = train_vars.subset(jmath.TrainVar).unique()
    elif isinstance(train_vars, (list, tuple)):
      train_vars = ArrayCollector((f'_unknown{i}', var) for i, var in enumerate(train_vars))
      train_vars = train_vars.unique()
    else:
      raise ValueError
    self.lr = lr
    self._train_vars = train_vars
    self.dynamic_vars = ArrayCollector(train_vars)  # dynamic variables

  def register_dynamical_vars(self, vars: dict):
    for key, var in vars.items():
      if key in self.dynamic_vars:
        if id(self.dynamic_vars[key]) != id(var):
          raise ValueError(f'Name "{key}" conflicts: same name for '
                           f'{var} and {self.dynamic_vars[key]}.')
      self.dynamic_vars[key] = var

  def vars(self, method='absolute'):
    gather = ArrayCollector(self.dynamic_vars)
    gather.update(super(Optimizer, self).vars(method=method))
    return gather


class SGD(Optimizer):
  def __init__(self, lr, train_vars, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def __call__(self, grads: dict):
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    for key, p in self._train_vars.items():
      p.value -= self.lr * grads[key]


class Momentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', jmath.Variable(jmath.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)

  def __call__(self, grads: dict):
    if not (len(grads) == len(self._train_vars)):
      raise ValueError('Expecting as many gradients as trainable variables')
    for key, p in self._train_vars.items():
      m = self.dynamic_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= self.lr * m.value


class NesterovMomentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(NesterovMomentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', jmath.Variable(jmath.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)

  def __call__(self, grads: dict):
    if not (len(grads) == len(self._train_vars)):
      raise ValueError('Expecting as many gradients as trainable variables')
    for key, p in self._train_vars.items():
      m = self.dynamic_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= self.lr * (g + self.momentum * m.value)


class Adam(Optimizer):
  def __init__(self, lr, train_vars, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.step = jmath.Variable(jmath.array([0]))
    ms = dict((key + '_m', jmath.Variable(jmath.zeros_like(x))) for key, x in self._train_vars.items())
    vs = dict((key + '_v', jmath.Variable(jmath.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)
    self.register_dynamical_vars(vs)

  def __call__(self, grads: dict):
    """Updates variables and other state based on Adam algorithm.
    """
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    self.step.value += 1
    lr = self.lr * np.sqrt(1 - self.beta2 ** self.step.value) / (1 - self.beta1 ** self.step.value)
    for key, p in self._train_vars.items():
      m = self.dynamic_vars[key + '_m']
      v = self.dynamic_vars[key + '_v']
      g = grads[key]
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      p.value -= lr * m.value * jax.lax.rsqrt(v.value + self.eps)
