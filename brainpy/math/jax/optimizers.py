# -*- coding: utf-8 -*-

import jax.numpy as jn

import brainpy.math.jax as bm
from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import TensorCollector

__all__ = [
  # optimizers
  'Optimizer',
  'SGD',
  'Momentum',
  'MomentumNesterov',
  'Adam',

  # schedules
  'make_schedule',
  'Scheduler',
  'Constant',
  'ExponentialDecay',
  'InverseTimeDecay',
  'PolynomialDecay',
  'PiecewiseConstant',
]


class Optimizer(Base):
  """Base Optimizer Class.
  """
  target_backend = 'jax'

  def __init__(self, train_vars: dict, lr, name):
    super(Optimizer, self).__init__(name=name)

    assert isinstance(train_vars, dict), '"train_vars" must be a dict of JaxArray.'
    self.lr = make_schedule(lr)
    self.vars_to_train = train_vars
    self.implicit_vars = TensorCollector()

  def register_variables(self, variables: dict):
    if self.implicit_vars is None:
      raise ValueError(
        'Please super initialize the Optimizer first, '
        'then call "register_variables()".')
    for key, var in variables.items():
      if key in self.implicit_vars:
        if id(self.implicit_vars[key]) != id(var):
          raise ValueError(
            f'Name "{key}" conflicts: same name for {var} '
            f'and {self.implicit_vars[key]}.')
      self.implicit_vars[key] = var

  def check_grads(self, grads):
    if len(grads) != len(self.vars_to_train):
      raise errors.BrainPyError(
        f'The length of "grads" must be equal to "self.vars_to_train", '
        f'while we got {len(grads)} != {len(self.vars_to_train)}!')


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.
  """

  def __init__(self, lr, train_vars, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      p.value -= self.lr() * grads[key]
    self.lr.update()


class Momentum(Optimizer):
  """Momentum optimizer.
  """

  def __init__(self, lr, train_vars, momentum, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    self.register_variables(ms)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= self.lr() * m.value
    self.lr.update()


class MomentumNesterov(Optimizer):
  def __init__(self, lr, train_vars, momentum, name=None):
    super(MomentumNesterov, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    self.register_variables(ms)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      g = grads[key]
      m.value = g + self.momentum * m.value
      p.value -= self.lr() * (g + self.momentum * m.value)
    self.lr.update()


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
    ms = dict((key + '_m', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    self.register_variables(ms)
    self.register_variables(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    lr /= (1 - self.beta1 ** (self.lr.step[0] + 1))
    lr *= jn.sqrt(1 - self.beta2 ** (self.lr.step[0] + 1))
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      v = self.implicit_vars[key + '_v']
      g = grads[key]
      # First  moment estimate.
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      # Second moment estimate.
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      # Bias correction.
      p.value -= lr * m.value / (jn.sqrt(v.value) + self.eps)
    self.lr.update()


# learning rate schedules #
# ----------------------- #


def make_schedule(scalar_or_schedule):
  if isinstance(scalar_or_schedule, Scheduler):
    return scalar_or_schedule
  elif isinstance(scalar_or_schedule, (int, float)):
    return Constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))


class Scheduler(Base):
  def __init__(self, lr):
    super(Scheduler, self).__init__()

    assert isinstance(lr, (float, int))
    self.lr = lr
    self.step = bm.Variable(bm.array([0]))

  def update(self):
    self.step += 1

  def __call__(self, i=None):
    raise NotImplementedError


class Constant(Scheduler):
  def __call__(self, i=None):
    return self.lr


class ExponentialDecay(Scheduler):
  def __init__(self, lr, decay_steps, decay_rate):
    super(ExponentialDecay, self).__init__(lr=lr)
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

  def __call__(self, i=None):
    i = i if i else self.step[0]
    return self.lr * self.decay_rate ** (i / self.decay_steps)


class InverseTimeDecay(ExponentialDecay):
  def __init__(self, lr, decay_steps, decay_rate, staircase=False):
    super(InverseTimeDecay, self).__init__(lr, decay_steps, decay_rate)
    self.staircase = staircase

  def __call__(self, i=None):
    i = i if i else self.step[0]
    if self.staircase:
      return self.lr / (1 + self.decay_rate * jn.floor(i / self.decay_steps))
    else:
      return self.lr / (1 + self.decay_rate * i / self.decay_steps)


class PolynomialDecay(Scheduler):
  def __init__(self, lr, decay_steps, final_lr, power=1.0):
    super(PolynomialDecay, self).__init__(lr)
    self.decay_steps = decay_steps
    self.final_lr = final_lr
    self.power = power

  def __call__(self, i=None):
    i = i if i else self.step[0]
    i = jn.minimum(i, self.decay_steps)
    step_mult = (1 - i / self.decay_steps) ** self.power
    return step_mult * (self.lr - self.final_lr) + self.final_lr


class PiecewiseConstant(Scheduler):
  def __init__(self, boundaries, values):
    super(PiecewiseConstant, self).__init__(0.)

    boundaries = jn.array(boundaries)
    values = jn.array(values)
    if not boundaries.ndim == values.ndim == 1:
      raise ValueError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
      raise ValueError("boundaries length must be one shorter than values length")
    self.boundaries = boundaries
    self.values = values

  def __call__(self, i=None):
    i = i if i else self.step[0]
    return self.values[jn.sum(i > self.boundaries)]
