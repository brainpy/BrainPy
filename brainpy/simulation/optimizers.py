# -*- coding: utf-8 -*-

from brainpy.tools.collector import ArrayCollector
from brainpy.simulation._imports import mjax, jax
from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'NesterovMomentum',
  'Adam',
]


class Optimizer(DynamicalSystem):
  target_backend = 'jax'

  def __init__(self, train_vars, lr, name):
    super(Optimizer, self).__init__(name=name)

    if isinstance(train_vars, ArrayCollector):
      train_vars = train_vars.subset(mjax.TrainVar).unique()
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

  def update(self, grads: dict, **kwargs):
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    for key, p in self._train_vars.items():
      p.value -= self.lr * grads[key]


class Momentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', mjax.Variable(mjax.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)

  def update(self, grads: dict, **kwargs):
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
    ms = dict((key + '_m', mjax.Variable(mjax.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)

  def update(self, grads: dict, **kwargs):
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
    self.step = mjax.Variable(mjax.array([0]))
    ms = dict((key + '_m', mjax.Variable(mjax.zeros_like(x))) for key, x in self._train_vars.items())
    vs = dict((key + '_v', mjax.Variable(mjax.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_dynamical_vars(ms)
    self.register_dynamical_vars(vs)

  def update(self, grads: dict, **kwargs):
    """Updates variables and other state based on Adam algorithm.
    """
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    self.step += 1
    step = self.step.value[0]
    lr = self.lr * mjax.sqrt(1 - self.beta2 ** step) / (1 - self.beta1 ** step)
    for key, p in self._train_vars.items():
      m = self.dynamic_vars[key + '_m']
      v = self.dynamic_vars[key + '_v']
      g = grads[key]
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      p.value -= lr * m.value * jax.lax.rsqrt(v.value + self.eps)
