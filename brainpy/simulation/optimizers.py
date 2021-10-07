# -*- coding: utf-8 -*-

from brainpy.base.base import Base
from brainpy.base.collector import ArrayCollector
from brainpy.simulation._imports import mjax, jax

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'NesterovMomentum',
  'Adam',
]


class Optimizer(Base):
  """Base Optimizer Class.

  """
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
    if not hasattr(self, 'dynamic_vars'):
      raise ValueError('Please super initialize the Optimizer first, then call "register_dynamical_vars()".')
    for key, var in vars.items():
      if key in self.dynamic_vars:
        if id(self.dynamic_vars[key]) != id(var):
          raise ValueError(f'Name "{key}" conflicts: same name for {var} and {self.dynamic_vars[key]}.')
      self.dynamic_vars[key] = var

  def vars(self, method='absolute'):
    gather = ArrayCollector(self.dynamic_vars)
    gather.update(super(Optimizer, self).vars(method=method))
    return gather


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.
  """
  def __init__(self, lr, train_vars, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def update(self, grads: dict, **kwargs):
    if len(grads) != len(self._train_vars):
      raise ValueError('Expecting as many gradients as trainable variables')
    for key, p in self._train_vars.items():
      p.value -= self.lr * grads[key]


class Momentum(Optimizer):
  """Momentum optimizer.

  """
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
