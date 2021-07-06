# -*- coding: utf-8 -*-

import numpy as np

from brainpy.simulation import collector
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.dnn.imports import jax_math, jax
from brainpy.simulation.dnn.layers import Module

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'NesterovMomentum',
  'Adam',
]


class Optimizer(Module):
  def __init__(self, lr, target, name):
    super(Optimizer, self).__init__(name=name)
    if not isinstance(target, Module):
      raise ValueError
    self.target = target
    self.lr = lr
    self.train_vars = list(target.vars().unique_values())
    self._dynamic_vars = []  # dynamic variables

  def register_dynamic_vars(self, variables):
    self._dynamic_vars.extend(variables)

  def vars(self, prefix=''):
    gather = collector.ArrayCollector()
    for i, v in enumerate(self._dynamic_vars):
      gather[f'_v{i}'] = v
    for k, v in self.__dict__.items():
      if isinstance(v, jax_math.ndarray):
        gather[prefix + k] = v
        gather[f'{self.name}.{k}'] = v
      elif isinstance(v, DynamicSystem):
        gather.update(v.vars(prefix=f'{prefix}{k}.'))
    return gather


class SGD(Optimizer):
  def __init__(self, target, lr, name=None):
    super(SGD, self).__init__(lr=lr, target=target, name=name)

  def __call__(self, grads):
    if len(grads) != len(self.train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    for g, p in zip(grads, self.train_vars):
      p.value -= self.lr * g


class Momentum(Optimizer):
  def __init__(self, target, lr, momentum, name=None):
    super(Momentum, self).__init__(lr, target=target, name=name)
    self.momentum = momentum
    self.ms = [jax_math.zeros_like(x) for x in self.train_vars]
    self.register_dynamic_vars(self.ms)

  def __call__(self, grads):
    for g, p, m in zip(grads, self.train_vars, self.ms):
      m.value = g + self.momentum * m.value
      p.value -= self.lr * m.value


class NesterovMomentum(Optimizer):
  def __init__(self, target, lr, momentum, name=None):
    super(NesterovMomentum, self).__init__(lr, target=target, name=name)
    self.momentum = momentum
    self.ms = [jax_math.zeros_like(x) for x in self.train_vars]
    self.register_dynamic_vars(self.ms)

  def __call__(self, grads):
    for g, p, m in zip(grads, self.train_vars, self.ms):
      m.value = g + self.momentum * m.value
      p.value -= self.lr * (g + self.momentum * m.value)


class Adam(Optimizer):
  def __init__(self, target, lr, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr, target=target, name=name)
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.step = jax_math.array([0])
    self.ms = [jax_math.zeros_like(x) for x in self.train_vars]
    self.vs = [jax_math.zeros_like(x) for x in self.train_vars]
    self.register_dynamic_vars(self.ms)
    self.register_dynamic_vars(self.vs)

  def __call__(self, grads):
    """Updates variables and other state based on Adam algorithm.
    """
    assert len(grads) == len(self.train_vars), 'Expecting as many gradients as trainable variables'
    self.step.value += 1
    lr = self.lr * np.sqrt(1 - self.beta2 ** self.step.value) / (1 - self.beta1 ** self.step.value)
    for g, p, m, v in zip(grads, self.train_vars, self.ms, self.vs):
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      p.value -= lr * m.value * jax.lax.rsqrt(v.value + self.eps)
