# -*- coding: utf-8 -*-

from typing import Union, Sequence, Dict, Optional

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.base.base import Base
from brainpy.base.collector import TensorCollector
from brainpy.errors import MathError
from brainpy.math.jaxarray import Variable
from .scheduler import make_schedule, Scheduler

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'MomentumNesterov',
  'Adagrad',
  'Adadelta',
  'RMSProp',
  'Adam',
  'LARS',
]


class Optimizer(Base):
  """Base Optimizer Class.
  """

  def __init__(
      self,
      lr: Union[float, int, Scheduler],
      train_vars: Union[Sequence[Variable], Dict[str, Variable]] = None,
      name: str = None
  ):
    super(Optimizer, self).__init__(name=name)
    self.lr = make_schedule(lr)
    self.vars_to_train = TensorCollector()
    self.register_vars(train_vars)

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    raise NotImplementedError

  def check_grads(self, grads):
    if len(grads) != len(self.vars_to_train):
      raise MathError(
        f'The length of "grads" must be equal to "self.vars_to_train", '
        f'while we got {len(grads)} != {len(self.vars_to_train)}!')

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr})"

  def update(self, grads: dict):
    raise NotImplementedError


class SGD(Optimizer):
  r"""Stochastic gradient descent optimizer.

  SGD performs a parameter update for training examples :math:`x` and label
  :math:`y`:

  .. math::

      \theta = \theta - \eta \cdot \nabla_\theta J(\theta; x; y)

  """

  def __init__(self, lr, train_vars=None, name=None):
    super(SGD, self).__init__(lr=lr, train_vars=train_vars, name=name)

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      p.value -= self.lr() * grads[key]
    self.lr.update()


class Momentum(Optimizer):
  r"""Momentum optimizer.

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

  def __init__(self, lr, train_vars=None, momentum=0.9, name=None):
    super(Momentum, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    vs = dict((key + '_v', Variable(bm.zeros_like(x)))
              for key, x in train_vars.items())
    self.register_implicit_vars(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      p.value += v.value
    self.lr.update()

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr}, momentum={self.momentum})"


class MomentumNesterov(Optimizer):
  r"""Nesterov accelerated gradient optimizer [2]_.

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

  def __init__(self, lr, train_vars=None, momentum=0.9, name=None):
    super(MomentumNesterov, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    vs = dict((key + '_v', Variable(bm.zeros_like(x)))
              for key, x in train_vars.items())
    self.register_implicit_vars(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      p.value += (self.momentum * v.value - lr * g)
    self.lr.update()

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr}, momentum={self.momentum})"


class Adagrad(Optimizer):
  r"""Optimizer that implements the Adagrad algorithm.

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

  def __init__(self, lr, train_vars=None, epsilon=1e-6, name=None):
    super(Adagrad, self).__init__(lr=lr, train_vars=train_vars, name=name)
    self.epsilon = epsilon

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    self.register_implicit_vars(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      c = self.implicit_vars[key + '_cache']
      c.value += g ** 2
      p.value -= lr * g / bm.sqrt(c + self.epsilon)
    self.lr.update()

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr}, epsilon={self.epsilon})"


class Adadelta(Optimizer):
  r"""Optimizer that implements the Adadelta algorithm.

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

  def __init__(self, train_vars=None, lr=0.01, epsilon=1e-6, rho=0.95, name=None):
    super(Adadelta, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.epsilon = epsilon
    self.rho = rho

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    deltas = dict((key + '_delta', Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    self.register_implicit_vars(caches)
    self.register_implicit_vars(deltas)

  def update(self, grads: dict):
    self.check_grads(grads)
    for key, p in self.vars_to_train.items():
      g = grads[key]
      c = self.implicit_vars[key + '_cache']
      d = self.implicit_vars[key + '_delta']
      c.value = self.rho * c.value + (1 - self.rho) * g ** 2
      update = g * jnp.sqrt(d.value + self.epsilon) / jnp.sqrt(c + self.epsilon)
      p.value -= update
      d.value = self.rho * d.value + (1 - self.rho) * update ** 2

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"epsilon={self.epsilon}, rho={self.rho})")


class RMSProp(Optimizer):
  r"""Optimizer that implements the RMSprop algorithm.

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

  def __init__(self, lr, train_vars=None, epsilon=1e-6, rho=0.9, name=None):
    super(RMSProp, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.epsilon = epsilon
    self.rho = rho

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    self.register_implicit_vars(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for k, p in self.vars_to_train.items():
      g = grads[k]
      c = self.implicit_vars[k + '_cache']
      c.value = self.rho * c.value + (1 - self.rho) * g ** 2
      p.value -= (lr * g / jnp.sqrt(c.value + self.epsilon))
    self.lr.update()

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"epsilon={self.epsilon}, rho={self.rho})")


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

  def __init__(self, lr, train_vars=None, beta1=0.9, beta2=0.999, eps=1e-8, name=None):
    super(Adam, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})")

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    ms = dict((k + '_m', Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(ms)
    vs = dict((k + '_v', Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(vs)

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


class LARS(Optimizer):
  """Layer-wise adaptive rate scaling (LARS) optimizer.

  Parameters
  ----------
  momentum: float
    coefficient used for the moving average of the gradient.
  weight_decay: float
    weight decay coefficient.
  tc: float
    trust coefficient eta ( < 1) for trust ratio computation.
  eps: float
    epsilon used for trust ratio computation.
  """

  def __init__(self,
               lr: Union[float, int, Scheduler],
               train_vars: Dict[str, Variable] = None,
               momentum: float = 0.9,
               weight_decay: float = 1e-4,
               tc: float = 1e-3,
               eps: float = 1e-5,
               name: str = None):
    super(LARS, self).__init__(lr=lr, train_vars=train_vars, name=name)

    self.momentum = momentum
    self.weight_decay = weight_decay
    self.tc = tc
    self.eps = eps

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"momentum={self.momentum}, weight_decay={self.weight_decay}, "
            f"tc={self.tc}, eps={self.eps})")

  def register_vars(self, train_vars: Optional[Dict[str, Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    ms = dict((k + '_m', Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(ms)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for k, p in self.vars_to_train.items():
      g = grads[k]
      m = self.implicit_vars[k + '_m']
      p_norm = jnp.linalg.norm(bm.as_device_array(p))
      g_norm = jnp.linalg.norm(bm.as_device_array(g))
      trust_ratio = self.tc * p_norm / (g_norm + self.weight_decay * p_norm + self.eps)
      local_lr = lr * jnp.maximum(jnp.logical_or(p_norm == 0, g_norm == 0), trust_ratio)
      m.value = self.momentum * m.value + local_lr * (g + self.weight_decay * p.value)
      p.value -= m.value
    self.lr.update()
