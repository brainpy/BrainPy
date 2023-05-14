# -*- coding: utf-8 -*-

import warnings
from typing import Union, Sequence, Dict, Optional, Tuple

import jax.numpy as jnp
from jax.lax import cond

import brainpy.math as bm
from brainpy import check
from brainpy._src.math.object_transform.base import BrainPyObject, ArrayCollector
from brainpy.errors import MathError
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
  'Adan',
  'AdamW',
]


class Optimizer(BrainPyObject):
  """Base Optimizer Class.

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.
  """

  lr: Scheduler  # learning rate
  '''Learning rate'''

  vars_to_train: ArrayCollector  # variables to train
  '''Variables to train.'''

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Union[Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      name: Optional[str] = None
  ):
    super(Optimizer, self).__init__(name=name)
    self.lr: Scheduler = make_schedule(lr)
    self.vars_to_train = ArrayCollector()
    self.register_train_vars(train_vars)

  def register_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    warnings.warn('Using "register_train_vars()" instead.', UserWarning)
    self.register_train_vars(train_vars)

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    raise NotImplementedError

  def check_grads(self, grads):
    if len(grads) != len(self.vars_to_train):
      raise MathError(f'The length of "grads" must be equal to "self.vars_to_train", '
                      f'while we got {len(grads)} != {len(self.vars_to_train)}!')

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr})"

  def update(self, grads: dict):
    raise NotImplementedError


class CommonOpt(Optimizer):
  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Union[Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      weight_decay: Optional[float] = None,
      name: Optional[str] = None
  ):
    super(Optimizer, self).__init__(name=name)
    self.lr: Scheduler = make_schedule(lr)
    self.vars_to_train = ArrayCollector()
    self.register_train_vars(train_vars)
    self.weight_decay = check.is_float(weight_decay, min_bound=0., max_bound=1., allow_none=True)


class SGD(CommonOpt):
  r"""Stochastic gradient descent optimizer.

  SGD performs a parameter update for training examples :math:`x` and label
  :math:`y`:

  .. math::

      \theta = \theta - \eta \cdot \nabla_\theta J(\theta; x; y)


  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      weight_decay: Optional[float] = None,
      name: Optional[str] = None
  ):
    super(SGD, self).__init__(lr=lr,
                              train_vars=train_vars,
                              weight_decay=weight_decay,
                              name=name)

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr})'

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      if self.weight_decay is None:
        p.value -= lr * grads[key]
      else:
        p.value = (1 - self.weight_decay) * p - lr * grads[key]
    self.lr.step_call()


class Momentum(CommonOpt):
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

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  References
  ----------

  .. [1] Qian, N. (1999). On the momentum term in gradient descent learning
         algorithms. Neural Networks : The Official Journal of the International
         Neural Network Society, 12(1), 145–151. http://doi.org/10.1016/S0893-6080(98)00116-6

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      momentum: float = 0.9,
      weight_decay: Optional[float] = None,
      name: Optional[str] = None
  ):
    super(Momentum, self).__init__(lr=lr,
                                   train_vars=train_vars,
                                   weight_decay=weight_decay,
                                   name=name)

    self.momentum = momentum

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr}, momentum={self.momentum})'

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x)))
              for key, x in train_vars.items())
    self.register_implicit_vars(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      if self.weight_decay is None:
        p.value += v.value
      else:
        p.value = (1 - self.weight_decay) * p + v
    self.lr.step_call()


class MomentumNesterov(CommonOpt):
  r"""Nesterov accelerated gradient optimizer [2]_.

  .. math::

      \begin{align}
      \begin{split}
      v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\
      \theta &= \theta - v_t
      \end{split}
      \end{align}

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  References
  ----------
  .. [2] Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2). Doklady ANSSSR (translated as Soviet.Math.Docl.), vol. 269, pp. 543– 547.

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      weight_decay: Optional[float] = None,
      momentum: float = 0.9,
      name: Optional[str] = None
  ):
    super(MomentumNesterov, self).__init__(lr=lr,
                                           train_vars=train_vars,
                                           weight_decay=weight_decay,
                                           name=name)

    self.momentum = momentum

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr}, momentum={self.momentum})'

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    vs = dict((key + '_v', bm.Variable(bm.zeros_like(x)))
              for key, x in train_vars.items())
    self.register_implicit_vars(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      v = self.implicit_vars[key + '_v']
      v.value = self.momentum * v.value - lr * g
      if self.weight_decay is None:
        p.value += v
      else:
        p.value = (1 - self.weight_decay) * p + v
    self.lr.step_call()


class Adagrad(CommonOpt):
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

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  References
  ----------
  .. [3] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, 2121–2159. Retrieved from http://jmlr.org/papers/v12/duchi11a.html

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      weight_decay: Optional[float] = None,
      epsilon: float = 1e-6,
      name: Optional[str] = None
  ):
    super(Adagrad, self).__init__(lr=lr,
                                  train_vars=train_vars,
                                  weight_decay=weight_decay,
                                  name=name)
    self.epsilon = epsilon

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    self.register_implicit_vars(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for key, p in self.vars_to_train.items():
      g = grads[key]
      c = self.implicit_vars[key + '_cache']
      c.value += g ** 2
      update = lr * g / jnp.sqrt(c.value + self.epsilon)
      if self.weight_decay is None:
        p.value -= update
      else:
        p.value = (1 - self.weight_decay) * p - update
    self.lr.step_call()

  def __repr__(self):
    return f"{self.__class__.__name__}(lr={self.lr}, epsilon={self.epsilon})"


class Adadelta(CommonOpt):
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

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  References
  ----------
  .. [4] Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. Retrieved from http://arxiv.org/abs/1212.5701

  """

  def __init__(
      self,
      lr: Union[float, Scheduler] = 0.01,
      train_vars: Dict[str, bm.Variable] = None,
      weight_decay: Optional[float] = None,
      epsilon: float = 1e-6,
      rho: float = 0.95,
      name: Optional[str] = None
  ):
    super(Adadelta, self).__init__(lr=lr,
                                   train_vars=train_vars,
                                   weight_decay=weight_decay,
                                   name=name)

    self.epsilon = epsilon
    self.rho = rho

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    deltas = dict((key + '_delta', bm.Variable(bm.zeros_like(x)))
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
      d.value = self.rho * d.value + (1 - self.rho) * update ** 2
      if self.weight_decay is None:
        p.value -= update
      else:
        p.value = (1 - self.weight_decay) * p - update
    self.lr.step_call()

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"epsilon={self.epsilon}, rho={self.rho})")


class RMSProp(CommonOpt):
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

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.

  References
  ----------
  .. [5] Tieleman, T. and Hinton, G. (2012):
         Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
         Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      weight_decay: Optional[float] = None,
      epsilon: float = 1e-6,
      rho: float = 0.9,
      name: Optional[str] = None
  ):
    super(RMSProp, self).__init__(lr=lr,
                                  train_vars=train_vars,
                                  weight_decay=weight_decay,
                                  name=name)

    self.epsilon = epsilon
    self.rho = rho

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    caches = dict((key + '_cache', bm.Variable(bm.zeros_like(x)))
                  for key, x in train_vars.items())
    self.register_implicit_vars(caches)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for k, p in self.vars_to_train.items():
      g = grads[k]
      c = self.implicit_vars[k + '_cache']
      c.value = self.rho * c.value + (1 - self.rho) * g ** 2
      update = (lr * g / jnp.sqrt(c.value + self.epsilon))
      if self.weight_decay is None:
        p.value -= update
      else:
        p.value = (1 - self.weight_decay) * p - update
    self.lr.step_call()

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"epsilon={self.epsilon}, rho={self.rho})")


class Adam(CommonOpt):
  """Optimizer that implements the Adam algorithm.

  Adam [6]_ - a stochastic gradient descent method (SGD) that computes
  individual adaptive learning rates for different parameters from estimates of
  first- and second-order moments of the gradients.

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.
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

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      beta1: float = 0.9,
      beta2: float = 0.999,
      eps: float = 1e-8,
      weight_decay: Optional[float] = None,
      name: Optional[str] = None
  ):
    super(Adam, self).__init__(lr=lr,
                               train_vars=train_vars,
                               weight_decay=weight_decay,
                               name=name)

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={str(self.lr)}, "
            f"beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})")

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    ms = dict((k + '_m', bm.Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(ms)
    vs = dict((k + '_v', bm.Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(vs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    lr /= (1 - self.beta1 ** (self.lr.last_epoch.value + 2))
    lr *= jnp.sqrt(1 - self.beta2 ** (self.lr.last_epoch.value + 2))
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      v = self.implicit_vars[key + '_v']
      g = grads[key]
      # First  moment estimate.
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      # Second moment estimate.
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      # Bias correction.
      update = lr * m.value / (jnp.sqrt(v.value) + self.eps)
      if self.weight_decay is None:
        p.value -= update
      else:
        p.value = (1 - self.weight_decay) * p - update
    self.lr.step_call()


class LARS(CommonOpt):
  r"""Layer-wise adaptive rate scaling (LARS) optimizer [1]_.

  Layer-wise Adaptive Rate Scaling, or LARS, is a large batch
  optimization technique. There are two notable differences
  between LARS and other adaptive algorithms such as `Adam` or `RMSProp`:
  first, LARS uses a separate learning rate for each layer and not for
  each weight. And second, the magnitude of the update is controlled
  with respect to the weight norm for better control of training speed.

  .. math::

     m_{t} = \beta_{1}m_{t-1} + \left(1-\beta_{1}\right)\left(g_{t} + \lambda{x_{t}}\right) \\
     x_{t+1}^{\left(i\right)} = x_{t}^{\left(i\right)}  - \eta_{t}\frac{\phi\left(|| x_{t}^{\left(i\right)} ||\right)}{|| m_{t}^{\left(i\right)} || }m_{t}^{\left(i\right)}

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.
  momentum: float
    coefficient used for the moving average of the gradient.
  weight_decay: float
    weight decay coefficient.
  tc: float
    trust coefficient eta ( < 1) for trust ratio computation.
  eps: float
    epsilon used for trust ratio computation.

  References
  ----------
  .. [1] You, Yang, Igor Gitman and Boris Ginsburg. “Large Batch Training of Convolutional Networks.” arXiv: Computer Vision and Pattern Recognition (2017): n. pag.
  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      momentum: float = 0.9,
      weight_decay: float = 1e-4,
      tc: float = 1e-3,
      eps: float = 1e-5,
      name: Optional[str] = None
  ):
    super(LARS, self).__init__(lr=lr,
                               train_vars=train_vars,
                               weight_decay=weight_decay,
                               name=name)

    self.momentum = momentum
    self.tc = tc
    self.eps = eps

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"momentum={self.momentum}, weight_decay={self.weight_decay}, "
            f"tc={self.tc}, eps={self.eps})")

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    self.register_implicit_vars({k + '_m': bm.Variable(bm.zeros_like(x))
                                 for k, x in train_vars.items()})

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    for k, p in self.vars_to_train.items():
      g = bm.as_jax(grads[k])
      m = self.implicit_vars[k + '_m']
      p_norm = jnp.linalg.norm(p.value)
      g_norm = jnp.linalg.norm(g)
      trust_ratio = self.tc * p_norm / (g_norm + self.weight_decay * p_norm + self.eps)
      local_lr = lr * jnp.maximum(jnp.logical_or(p_norm == 0, g_norm == 0), trust_ratio)
      m.value = self.momentum * m.value + local_lr * (g + self.weight_decay * p.value)
      p.value -= m.value
    self.lr.step_call()


class Adan(CommonOpt):
  r"""Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models [1]_.

  .. math::

     \begin{equation}
      \begin{aligned}
      & \mathbf{m}_k=\left(1-\beta_1\right) \mathbf{m}_{k-1}+\beta_1 \mathbf{g}_k \\
      & \mathbf{v}_k=\left(1-\beta_2\right) \mathbf{v}_{k-1}+\beta_2\left(\mathbf{g}_k-\mathbf{g}_{k-1}\right)  \\
      & \mathbf{n}_k=\left(1-\beta_3\right) \mathbf{n}_{k-1}+\beta_3\left[\mathbf{g}_k+\left(1-\beta_2\right)\left(\mathbf{g}_k-\mathbf{g}_{k-1}\right)\right]^2  \\
      & \boldsymbol{\eta}_k=\eta /\left(\sqrt{\mathbf{n}_k+\varepsilon}\right)  \\
      & \boldsymbol{\theta}_{k+1}=\left(1+\lambda_k \eta\right)^{-1}\left[\boldsymbol{\theta}_k-\boldsymbol{\eta}_k \circ\left(\mathbf{m}_k+\left(1-\beta_2\right) \mathbf{v}_k\right)\right] \\
      \end{aligned}
      \end{equation}

  Parameters
  ----------
  lr: float, Scheduler
    learning rate. Can be much higher than Adam, up to 5-10x. (default: 1e-3)
  betas : tuple
     Coefficients used for computing running averages of gradient and its norm. (default: (0.02, 0.08, 0.01))
  eps : float
    The term added to the denominator to improve numerical stability. (default: 1e-8)
  weight_decay : float
    decoupled weight decay (L2 penalty) (default: 0)
  no_prox: bool
    how to perform the decoupled weight decay (default: False).
    It determines the update rule of parameters with weight decay.
    By default, Adan updates the parameters in the way presented in Algorithm 1 in the paper:

    .. math::
       \boldsymbol{\theta}_{k+1} = ( 1+\lambda \eta)^{-1}\left[\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}k)\right],

    But one also can update the parameter like Adamw:

    .. math::
       \boldsymbol{\theta}_{k+1} = ( 1-\lambda \eta)\boldsymbol{\theta}_k - \boldsymbol{\eta}_k \circ (\mathbf{m}_k+(1-{\color{blue}\beta_2})\mathbf{v}_k).

  References
  ----------
  .. [1] Xie, Xingyu, Pan Zhou, Huan Li, Zhouchen Lin and Shuicheng Yan. 
         “Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing 
         Deep Models.” ArXiv abs/2208.06677 (2022): n. pag.
  """

  def __init__(
      self,
      lr: Union[float, Scheduler] = 1e-3,
      train_vars: Dict[str, bm.Variable] = None,
      betas: Tuple[float, float, float] = (0.02, 0.08, 0.01),
      eps: float = 1e-8,
      weight_decay: float = 0.02,
      no_prox: bool = False,
      name: Optional[str] = None,
  ):
    super(Adan, self).__init__(lr=lr,
                               train_vars=train_vars,
                               weight_decay=weight_decay,
                               name=name)

    assert len(betas) == 3
    if eps < 0.:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= betas[2] < 1.0:
      raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))

    self.betas = betas
    self.eps = eps
    self.weight_decay = weight_decay
    self.no_prox = no_prox

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"betas={self.betas}, "
            f"weight_decay={self.weight_decay}, "
            f"no_prox={self.no_prox}, "
            f"eps={self.eps}")

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    # Exponential moving average of gradient values
    exp_avg = {k + '_m': bm.Variable(bm.zeros_like(x)) for k, x in train_vars.items()}
    # Exponential moving average of squared gradient values
    exp_avg_sq = {k + '_v': bm.Variable(bm.zeros_like(x)) for k, x in train_vars.items()}
    # Exponential moving average of gradient difference
    exp_avg_diff = {k + '_n': bm.Variable(bm.zeros_like(x)) for k, x in train_vars.items()}
    # previous gradient
    pre_grad = {k + '_prev_grad': bm.Variable(bm.zeros_like(x)) for k, x in train_vars.items()}
    self.register_implicit_vars(exp_avg, exp_avg_sq, exp_avg_diff, pre_grad)

  def _update_moments(self, m, n, v, pre_g, g):
    m = m * (1 - self.betas[0]) + self.betas[0] * g
    gd = g - pre_g
    v = v * (1 - self.betas[1]) + self.betas[1] * gd
    n = n * (1 - self.betas[2]) + self.betas[2] * (g + (1 - self.betas[1]) * gd) ** 2
    return m, n, v

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()
    step = self.lr.last_epoch.value + 1
    correct_m = 1 / (1 - (1 - self.betas[0]) ** (step + 1))
    correct_v = 1 / (1 - (1 - self.betas[1]) ** (step + 1))
    correct_n = 1 / (1 - (1 - self.betas[2]) ** (step + 1))
    for key, p_var in self.vars_to_train.items():
      m_var = self.implicit_vars[key + '_m']
      n_var = self.implicit_vars[key + '_n']
      v_var = self.implicit_vars[key + '_v']
      prev_g_var = self.implicit_vars[key + '_prev_grad']
      g = grads[key]
      pre_g = cond(step == 0, lambda pg, g: g, lambda pg, g: pg, (prev_g_var.value, g))
      diff = g - pre_g
      m = m_var.value * (1 - self.betas[0]) + self.betas[0] * g
      v = v_var.value * (1 - self.betas[1]) + self.betas[1] * diff
      n = n_var.value * (1 - self.betas[2]) + self.betas[2] * (g + (1 - self.betas[1]) * diff) ** 2
      weighted_step_size = lr / (jnp.sqrt(n * correct_n) + self.eps)
      if self.no_prox:
        p = (p_var.value * (1 - self.weight_decay * lr) -
             weighted_step_size * (m * correct_m + (1 - self.betas[1]) * v * correct_v))
      else:
        p = (
            (p_var.value - weighted_step_size * (m * correct_m + (1 - self.betas[1]) * v * correct_v))
            / (1 + self.weight_decay * lr)
        )
      m_var.value = m
      n_var.value = n
      v_var.value = v
      prev_g_var.value = g
      p_var.value = p
    self.lr.step_call()


class AdamW(CommonOpt):
  r"""Adam with weight decay regularization [1]_.

  AdamW uses weight decay to regularize learning towards small weights, as
  this leads to better generalization. In SGD you can also use L2 regularization
  to implement this as an additive loss term, however L2 regularization
  does not behave as intended for adaptive gradient algorithms such as Adam.

  .. math::

     \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
            \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
            \: \epsilon \text{ (epsilon)}                                                    \\
        &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
            \: \textit{maximize}                                                             \\
        &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
            \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

        &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
        &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
        &\hspace{5mm}\textbf{else}                                                           \\
        &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
        &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
        &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
        &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
        &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
        &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
        &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
        &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
            \widehat{v_t})                                                                   \\
        &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
            \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
        &\hspace{5mm}\textbf{else}                                                           \\
        &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
            \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        &\bf{return} \:  \theta_t                                                     \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
     \end{aligned}


  Parameters
  ----------
  lr: float, Scheduler
    learning rate.
  beta1: optional, float
    A positive scalar value for beta_1, the exponential decay rate
    for the first moment estimates. Generally close to 1.
  beta2: optional, float
    A positive scalar value for beta_2, the exponential decay rate
    for the second moment estimates. Generally close to 1.
  eps: optional, float
    A positive scalar value for epsilon, a small constant for
    numerical stability.
  weight_decay: float
    Strength of the weight decay regularization. Note that this
    weight decay is multiplied with the learning rate.
  amsgrad: bool
    whether to use the AMSGrad variant of this algorithm
    from the paper `On the Convergence of Adam and Beyond`.
  name : optional, str
    The optimizer name.

  References
  ----------
  .. [1] Loshchilov, Ilya and Frank Hutter. “Decoupled Weight Decay Regularization.” International Conference on Learning Representations (2019).

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      beta1: float = 0.9,
      beta2: float = 0.999,
      eps: float = 1e-8,
      weight_decay: float = 1e-2,
      amsgrad: bool = False,
      name: Optional[str] = None,
  ):
    super(AdamW, self).__init__(lr=lr,
                                train_vars=train_vars,
                                weight_decay=weight_decay,
                                name=name)

    if eps < 0.:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= beta1 < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
    if not 0.0 <= beta2 < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
    if weight_decay < 0.:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.weight_decay = weight_decay
    self.amsgrad = amsgrad

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"beta1={self.beta1}, "
            f"beta2={self.beta2}, "
            f"weight_decay={self.weight_decay}, "
            f"eps={self.eps}, "
            f"amsgrad={self.amsgrad})")

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    # Exponential moving average of gradient values
    ms = dict((k + '_m', bm.Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    # Exponential moving average of squared gradient values
    vs = dict((k + '_v', bm.Variable(bm.zeros_like(x)))
              for k, x in train_vars.items())
    self.register_implicit_vars(ms, vs)
    # Maintains max of all exp. moving avg. of sq. grad. values
    if self.amsgrad:
      gs = {k + '_vmax': bm.Variable(bm.zeros_like(x))
            for k, x in train_vars.items()}
      self.register_implicit_vars(gs)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr_old = self.lr()
    step = self.lr.last_epoch.value + 2
    bias_correction1 = 1 - self.beta1 ** step
    bias_correction2 = 1 - self.beta2 ** step
    lr = lr_old * jnp.sqrt(bias_correction2) / bias_correction1
    for key, p in self.vars_to_train.items():
      m = self.implicit_vars[key + '_m']
      v = self.implicit_vars[key + '_v']
      g = grads[key]
      if self.weight_decay != 0:
        p *= (1 - lr_old * self.weight_decay)
      # First  moment estimate.
      m.value = self.beta1 * m.value + (1 - self.beta1) * g
      # Second moment estimate.
      v.value = self.beta2 * v.value + (1 - self.beta2) * g ** 2
      if self.amsgrad:
        # Maintains the maximum of all 2nd moment running avg. till now
        vmax = self.implicit_vars[key + '_vmax']
        vmax.value = jnp.maximum(vmax.value, v)
        # Use the max. for normalizing running avg. of gradient
        denom = jnp.sqrt(vmax) + self.eps
      else:
        denom = jnp.sqrt(v.value) + self.eps
      # Bias correction.
      p.value -= lr * m / denom
    self.lr.step_call()


class SM3(CommonOpt):
  """SM3 algorithm [1]_.

  The 'Square-root of Minima of Sums of Maxima of Squared-gradients Method'
  (SM3) algorithm is a memory-efficient adaptive optimization algorithm similar
  to Adam and Adagrad with greatly reduced memory usage for history tensors.
  For an `n x m` matrix, Adam and Adagrad use `O(nm)` memory for history
  tensors, while SM3 uses `O(n+m)` due to the chosen cover. In general, a tensor
  of shape `(n_1, n_2, ..., n_k)` optimized using Adam will use `O(prod n_i)`
  memory for storage tensors, while the optimization using SM3 will use
  `O(sum n_i)` memory. Despite storing fewer parameters, this optimization
  algorithm manages to be comparably effective.

  This advantage drastically shrinks when `momentum > 0`. The momentum is
  tracked using a tensor of the same shape as the tensor being optimized. With
  momentum, SM3 will use just over half as much memory as Adam, and a bit more
  than Adagrad.

  Parameters
  ----------
  lr: float, Scheduler
    learning rate.
  momentum: float
    coefficient used to scale prior updates
    before adding. This drastically increases memory usage if
    `momentum > 0.0`. (default: 0.0)
  beta: float
    coefficient used for exponential moving averages (default: 0.0)
  eps: float
    Term added to square-root in denominator to
    improve numerical stability (default: 1e-30).

  References
  ----------
  .. [1] Anil, Rohan, Vineet Gupta, Tomer Koren and Yoram Singer. “Memory Efficient Adaptive Optimization.” Neural Information Processing Systems (2019).

  """

  def __init__(
      self,
      lr: Union[float, Scheduler],
      train_vars: Dict[str, bm.Variable] = None,
      beta: float = 0.,
      momentum: float = 0.,
      eps: float = 1e-30,
      weight_decay: Optional[float] = None,
      name: Optional[str] = None,
  ):
    super(SM3, self).__init__(lr=lr,
                              weight_decay=weight_decay,
                              train_vars=train_vars,
                              name=name)

    if not 0.0 <= momentum < 1.0:
      raise ValueError("Invalid momentum: {0}".format(momentum))
    if not 0.0 <= beta < 1.0:
      raise ValueError("Invalid beta: {0}".format(beta))
    if not 0.0 <= eps:
      raise ValueError("Invalid eps: {0}".format(eps))

    self.eps = eps
    self.beta = beta
    self.momentum = momentum

  def __repr__(self):
    return (f"{self.__class__.__name__}(lr={self.lr}, "
            f"beta={self.beta}, eps={self.eps}, momentum={self.momentum})")

  def register_train_vars(self, train_vars: Optional[Dict[str, bm.Variable]] = None):
    train_vars = dict() if train_vars is None else train_vars
    if not isinstance(train_vars, dict):
      raise MathError('"train_vars" must be a dict of Variable.')
    self.vars_to_train.update(train_vars)
    vs = dict()
    for k, v in train_vars.items():
      rank, ndim = v.shape, v.ndim
      for i in range(ndim):
        shape = [1] * ndim
        shape[i] = rank[i]
        vs[f'{k}_m{i}'] = bm.Variable(bm.zeros(shape, dtype=v.dtype))
    self.register_implicit_vars(vs)
    if self.momentum > 0.:
      ms = {k + '_mbuffer': bm.Variable(bm.zeros_like(v))
            for k, v in train_vars.items()}
      self.register_implicit_vars(ms)

  def update(self, grads: dict):
    self.check_grads(grads)
    lr = self.lr()

    for k, p in self.vars_to_train.items():
      g = grads[k]
      ndim = p.ndim
      update = self.implicit_vars[f'{k}_m0']
      for i in range(1, ndim):
        update = bm.minimum(update, self.implicit_vars[f'{k}_m{i}'])
      if self.beta > 0.:
        update *= self.beta
      update += g * g * (1 - self.beta)
      # Computes max along all dimensions except the given dim.
      # If tensor is a scalar, it returns tensor.
      for i in range(ndim):
        result = update
        for j in range(ndim):
          if i != j:
            result = result.max(axis=j, keepdim=True)
        acc = self.implicit_vars[f'{k}_m{i}']
        if self.beta > 0.:
          acc.value = bm.maximum(acc, result)
        else:
          # No need to compare - nu_max is bigger because of grad ** 2
          acc.value = result
      update = g / bm.sqrt(update + self.eps)
      if self.momentum > 0.:
        m_buffer = self.implicit_vars[f'{k}_mbuffer']
        update = update * (1. - self.momentum) + m_buffer * self.momentum
        m_buffer.value = update
      if self.weight_decay is None:
        p -= lr * update
      else:
        p.value = (1 - self.weight_decay) * p - lr * update
    self.lr.step_call()
