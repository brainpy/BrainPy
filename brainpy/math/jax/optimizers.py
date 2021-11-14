# -*- coding: utf-8 -*-

import jax.numpy as jnp

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
  'Adagrad',
  'Adadelta',
  'RMSProp',
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
  r"""Stochastic gradient descent optimizer.

  SGD performs a parameter update for training examples :math:`x` and label
  :math:`y`:

  .. math::

      \theta = \theta - \eta \cdot \nabla_\theta J(\theta; x; y)

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


  Momentum [1]_ is a method that helps accelerate SGD in the relevant direction
  and dampens oscillations. It does this by adding a fraction :math:`\gamma`
  of the update vector of the past time step to the current update vector:

  .. math::

    \begin{align}
    \begin{split}
    v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\
    \theta &= \theta - v_t
    \end{split}
    \end{align}


  References
  ----------

  .. [1] Qian, N. (1999). On the momentum term in gradient descent learning
         algorithms. Neural Networks : The Official Journal of the International
         Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6

  """

  def __init__(self, lr, train_vars, momentum=0.9, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    self.register_variables(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      p.value += v.value
    self.lr.update()


class MomentumNesterov(Optimizer):
  """Nesterov accelerated gradient optimizer [2]_.

  .. math::

      \begin{align}
      \begin{split}
      v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\
      \theta &= \theta - v_t
      \end{split}
      \end{align}


  References
  ----------
  .. [2] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.

  """
  def __init__(self, lr, train_vars, momentum=0.9, name=None):
    super(MomentumNesterov, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x)))
              for key, x in self.vars_to_train.items())
    self.register_variables(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      p.value += (self.momentum * v.value - lr * g)
    self.lr.update()


class Adagrad(Optimizer):
  """Optimizer that implements the Adagrad algorithm.

  Adagrad [3]_ is an optimizer with parameter-specific learning rates, which are
  adapted relative to how frequently a parameter gets updated during training.
  The more updates a parameter receives, the smaller the updates.

  .. math::

      \theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}

  where :math:`G(t)` contains the sum of the squares of the past gradients

  One of Adagrad's main benefits is that it eliminates the need to manually tune
  the learning rate. Most implementations use a default value of 0.01 and leave it at that.
  Adagrad's main weakness is its accumulation of the squared gradients in the denominator:
  Since every added term is positive, the accumulated sum keeps growing during training.
  This in turn causes the learning rate to shrink and eventually become infinitesimally
  small, at which point the algorithm is no longer able to acquire additional knowledge.

  References
  ----------
  .. [3] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html

  """
  def __init__(self, lr, train_vars, epsilon=1e-6, name=None):
    super(Adagrad, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.epsilon = epsilon
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x)))
                  for key, x in self.vars_to_train.items())
    self.register_variables(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      c = self.implicit_vars[key + '_cache']
      c.value += g ** 2
      p.value -= lr * g / bm.sqrt(c + self.epsilon)
    self.lr.update()


class Adadelta(Optimizer):
  """Optimizer that implements the Adadelta algorithm.

  Adadelta [4]_ optimization is a stochastic gradient descent method that is based
  on adaptive learning rate per dimension to address two drawbacks:

  - The continual decay of learning rates throughout training.
  - The need for a manually selected global learning rate.

  Adadelta is a more robust extension of Adagrad that adapts learning rates based on
  a moving window of gradient updates, instead of accumulating all past gradients.
  This way, Adadelta continues learning even when many updates have been done. Compared
  to Adagrad, in the original version of Adadelta you don't have to set an initial
  learning rate.

  .. math::

    \boldsymbol{s}_t \leftarrow \rho \boldsymbol{s}_{t-1} + (1 - \rho) \boldsymbol{g}_t \odot \boldsymbol{g}_t, \\
    \boldsymbol{g}_t' \leftarrow \sqrt{\frac{\Delta\boldsymbol{x}_{t-1} + \epsilon}{\boldsymbol{s}_t + \epsilon}}   \odot \boldsymbol{g}_t, \\
    \boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t, \\
    \Delta\boldsymbol{x}_t \leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1 - \rho) \boldsymbol{g}'_t \odot \boldsymbol{g}'_t.

  :math:`\rho` should be between 0 and 1. A value of rho close to 1 will decay the
  moving average slowly and a value close to 0 will decay the moving average fast.

  :math:`\rho` = 0.95 and :math:`\epsilon`=1e-6 are suggested in the paper and reported
  to work for multiple datasets (MNIST, speech).

  In the paper, no learning rate is considered (so learning_rate=1.0). Probably best to
  keep it at this value. epsilon is important for the very first update (so the
  numerator does not become 0).

  References
  ----------
  .. [4] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701

  """
  def __init__(self, train_vars, lr=0.01, epsilon=1e-6, rho=0.95, name=None):
    super(Adadelta, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.epsilon = epsilon
    self.rho = rho
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x))) for key, x in self.vars_to_train.items())
    deltas = dict((key + '_delta', bm.Variable(bm.zeros_like(x))) for key, x in self.vars_to_train.items())
    self.register_variables(caches)
    self.register_variables(deltas)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      g = grads[key]
      c = self.implicit_vars[key + '_cache']
      d = self.implicit_vars[key + '_delta']
      c.value = self.rho * c.value + (1 - self.rho) * g ** 2
      update = g * jnp.sqrt(d.value + self.epsilon) / jnp.sqrt(c + self.epsilon)
      p.value -= update
      d.value = self.rho * d.value + (1- self.rho) * update ** 2


class RMSProp(Optimizer):
  """Optimizer that implements the RMSprop algorithm.

  RMSprop [5]_ and Adadelta have both been developed independently around the same time
  stemming from the need to resolve Adagrad's radically diminishing learning rates.

  The gist of RMSprop is to:

  - Maintain a moving (discounted) average of the square of gradients
  - Divide the gradient by the root of this average

  .. math::

    \begin{split}c_t &= \rho c_{t-1} + (1-\rho)*g^2\\
    p_t &= \frac{\eta}{\sqrt{c_t + \epsilon}} * g \end{split}

  The centered version additionally maintains a moving average of the gradients,
  and uses that average to estimate the variance.

  References
  ----------
  .. [5] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """
  def __init__(self, lr, train_vars, epsilon=1e-6, rho=0.9, name=None):
    super(RMSProp, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.epsilon = epsilon
    self.rho = rho
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x))) for key, x in self.vars_to_train.items())
    self.register_variables(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for k, p in self.vars_to_train.items():
      g = grads[k]
      c = self.implicit_vars[k + '_cache']
      c.value = self.rho * c.value + (1 - self.rho) * g ** 2
      p.value -= (lr * g / jnp.sqrt(c.value + self.epsilon))
    self.lr.update()


class Adam(Optimizer):
  """Optimizer that implements the Adam algorithm.

  Adam [6]_ - a stochastic gradient descent method (SGD) that computes
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
  .. [6] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
  """

  def __init__(self, lr, train_vars, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    ms = dict((k + '_m', bm.Variable(bm.zeros_like(x))) for k, x in self.vars_to_train.items())
    vs = dict((k + '_v', bm.Variable(bm.zeros_like(x))) for k, x in self.vars_to_train.items())
    self.register_variables(ms)
    self.register_variables(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    lr /= (1 - self.beta1 ** (self.lr.step[0] + 1))
    lr *= jnp.sqrt(1 - self.beta2 ** (self.lr.step[0] + 1))
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      v = self.implicit_vars[key + '_v']
      g = grads[key]
      # First  moment estimate.
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      # Second moment estimate.
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      # Bias correction.
      p.value -= lr * m.value / (jnp.sqrt(v.value) + self.eps)
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
  """The learning rate scheduler."""
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
      return self.lr / (1 + self.decay_rate * jnp.floor(i / self.decay_steps))
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
    i = jnp.minimum(i, self.decay_steps)
    step_mult = (1 - i / self.decay_steps) ** self.power
    return step_mult * (self.lr - self.final_lr) + self.final_lr


class PiecewiseConstant(Scheduler):
  def __init__(self, boundaries, values):
    super(PiecewiseConstant, self).__init__(0.)

    boundaries = jnp.array(boundaries)
    values = jnp.array(values)
    if not boundaries.ndim == values.ndim == 1:
      raise ValueError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
      raise ValueError("boundaries length must be one shorter than values length")
    self.boundaries = boundaries
    self.values = values

  def __call__(self, i=None):
    i = i if i else self.step[0]
    return self.values[jnp.sum(i > self.boundaries)]
