# -*- coding: utf-8 -*-

import jax.numpy as jn

import brainpy.math.jax as bm
from brainpy.base.base import Base
from brainpy.base.collector import TensorCollector

__all__ = [
  # optimizers
  'Optimizer',
  'SGD',
  'Momentum',
  'NesterovMomentum',
  'Adam',

  # schedules
  'constant',
  'exponential_decay',
  'inverse_time_decay',
  'polynomial_decay',
  'piecewise_constant',
]


class Optimizer(Base):
  """Base Optimizer Class.
  """
  target_backend = 'jax'

  def __init__(self, train_vars: dict, lr, name):
    super(Optimizer, self).__init__(name=name)

    assert isinstance(train_vars, dict), '"train_vars" must be a dict of JaxArray.'
    self.lr = _make_schedule(lr)
    self.step = bm.Variable(bm.array([0]))
    self._train_vars = train_vars
    self.implicit_vars = TensorCollector()
    self.implicit_vars.update(train_vars)  # dynamic variables

  def register_variables(self, vars: dict):
    if self.implicit_vars is None:
      raise ValueError('Please super initialize the Optimizer first, then call "register_variables()".')
    for key, var in vars.items():
      if key in self.implicit_vars:
        if id(self.implicit_vars[key]) != id(var):
          raise ValueError(f'Name "{key}" conflicts: same name for {var} and {self.implicit_vars[key]}.')
      self.implicit_vars[key] = var

  def update(self, grads):
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    self.step += 1
    lr = self.lr(self.step[0])
    return lr


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.
  """

  def __init__(self, lr, train_vars, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def update(self, grads: dict, **kwargs):
    lr = super(SGD, self).update(grads)
    for key, p in self._train_vars.items():
      p.value -= lr * grads[key]


class Momentum(Optimizer):
  """Momentum optimizer.

  """

  def __init__(self, lr, train_vars, momentum, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_variables(ms)

  def update(self, grads: dict, **kwargs):
    lr = super(Momentum, self).update(grads)
    for key, p in self._train_vars.items():
      m = self.implicit_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= lr * m.value


class NesterovMomentum(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(NesterovMomentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_variables(ms)

  def update(self, grads: dict, **kwargs):
    lr = super(NesterovMomentum, self).update(grads)
    for key, p in self._train_vars.items():
      m = self.implicit_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= lr * (g + self.momentum * m.value)


class Adam(Optimizer):
  """Adam optimizer.

  Adam [1]_ - a stochastic gradient descent method (SGD) that computes
  individual adaptive learning rates for different parameters from estimates of
  first- and second-order moments of the gradients.

  Parameters
  ----------
  beta1: optional, float
    A positive scalar value for beta_1, the exponential decay rate
    for the first moment estimates (default 0.9).
  beta2: optional, float
    A positive scalar value for beta_2, the exponential decay rate
    for the second moment estimates (default 0.999).
  eps: optional, float
    A positive scalar value for epsilon, a small constant for
    numerical stability (default 1e-8).
  name : optional, str
    The optimizer name.

  References
  ----------
  .. [1] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  """

  def __init__(self, lr, train_vars, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x))) for key, x in self._train_vars.items())
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x))) for key, x in self._train_vars.items())
    self.register_variables(ms)
    self.register_variables(vs)

  def update(self, grads: dict, **kwargs):
    lr = super(Adam, self).update(grads)
    lr *= jn.sqrt(1 - self.beta2 ** self.step[0]) / (1 - self.beta1 ** self.step[0])
    for key, p in self._train_vars.items():
      m = self.implicit_vars[key + '_m']
      v = self.implicit_vars[key + '_v']
      g = grads[key]
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      p.value -= lr * m.value / jn.sqrt(v.value + self.eps)


# learning rate schedules #
# ----------------------- #


def _make_schedule(scalar_or_schedule):
  if callable(scalar_or_schedule):
    return scalar_or_schedule
  elif isinstance(scalar_or_schedule, (int, float)):
    return constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))


def constant(lr):
  def schedule(i):
    return lr

  return schedule


def exponential_decay(lr, decay_steps, decay_rate):
  def schedule(i):
    return lr * decay_rate ** (i / decay_steps)

  return schedule


def inverse_time_decay(lr, decay_steps, decay_rate, staircase=False):
  if staircase:
    def schedule(i):
      return lr / (1 + decay_rate * jn.floor(i / decay_steps))
  else:
    def schedule(i):
      return lr / (1 + decay_rate * i / decay_steps)
  return schedule


def polynomial_decay(lr, decay_steps, final_lr, power=1.0):
  def schedule(i):
    i = jn.minimum(i, decay_steps)
    step_mult = (1 - i / decay_steps) ** power
    return step_mult * (lr - final_lr) + final_lr

  return schedule


def piecewise_constant(boundaries, values):
  boundaries = jn.array(boundaries)
  values = jn.array(values)
  if not boundaries.ndim == values.ndim == 1:
    raise ValueError("boundaries and values must be sequences")
  if not boundaries.shape[0] == values.shape[0] - 1:
    raise ValueError("boundaries length must be one shorter than values length")

  def schedule(i):
    return values[jn.sum(i > boundaries)]

  return schedule
