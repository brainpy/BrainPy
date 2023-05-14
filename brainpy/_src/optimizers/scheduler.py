# -*- coding: utf-8 -*-


from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

import brainpy.math as bm
from brainpy import check
from brainpy._src.math.object_transform.base import BrainPyObject
from brainpy.errors import MathError


# learning rate schedules #
# ----------------------- #


def make_schedule(scalar_or_schedule):
  if isinstance(scalar_or_schedule, Scheduler):
    return scalar_or_schedule
  elif isinstance(scalar_or_schedule, (int, float)):
    return Constant(scalar_or_schedule)
  else:
    raise TypeError(type(scalar_or_schedule))


class Scheduler(BrainPyObject):
  """The learning rate scheduler."""

  def __init__(self, lr: float, last_epoch: int = -1):
    super(Scheduler, self).__init__()
    self.lr = check.is_float(lr, )
    check.is_integer(last_epoch, allow_none=False, min_bound=-1)
    self.last_epoch = bm.Variable(jnp.asarray(last_epoch))

  def step_epoch(self):
    self.last_epoch += 1

  def step_call(self):
    pass

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr}, last_epoch={self.last_epoch.value})'

  def __call__(self, i=None):
    raise NotImplementedError


class Constant(Scheduler):
  def __call__(self, i=None):
    return self.lr


class CallBasedScheduler(Scheduler):
  def __init__(self, lr: float, last_epoch: int = -1, last_call: int = -1):
    super().__init__(lr=lr, last_epoch=last_epoch)

    self.lr = check.is_float(lr, )
    check.is_integer(last_call, allow_none=False, min_bound=-1)
    self.last_call = bm.Variable(jnp.asarray(last_call))

  def step_call(self):
    self.last_call += 1

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr}, last_call={self.last_call.value})'


class StepLR(Scheduler):
  """Decays the learning rate of each parameter group by gamma every
  `step_size` epochs.

  Parameters
  ----------
  lr: float
    Initial learning rate.
  step_size: int
    Period of learning rate decay.
  gamma: float
    Multiplicative factor of learning rate decay.
    Default: 0.1.
  last_epoch: int
    The index of last epoch. Default: -1.
  """

  def __init__(
      self,
      lr: float,
      step_size: int,
      gamma: float = 0.1,
      last_epoch: int = -1
  ):
    super().__init__(lr=lr, last_epoch=last_epoch)

    self.step_size = check.is_integer(step_size, min_bound=1, allow_none=False)
    self.gamma = check.is_float(gamma, min_bound=0., max_bound=1., allow_int=False)

  def __call__(self, i=None):
    i = (self.last_epoch.value + 1) if i is None else i
    return self.lr * self.gamma ** (jnp.floor_divide(i, self.step_size))

  def __repr__(self):
    return (f'{self.__class__.__name__}(lr={self.lr}, '
            f'step_size={self.step_size}, gamma={self.gamma}, '
            f'last_epoch={self.last_epoch})')


class MultiStepLR(Scheduler):
  """Decays the learning rate of each parameter group by gamma once the
  number of epoch reaches one of the milestones. Notice that such decay can
  happen simultaneously with other changes to the learning rate from outside
  this scheduler. When last_epoch=-1, sets initial lr as lr.

  Parameters
  ----------
  lr: float
    Initial learning rate.
  milestones: sequence of int
    List of epoch indices. Must be increasing.
  gamma: float
    Multiplicative factor of learning rate decay.
    Default: 0.1.
  last_epoch: int
    The index of last epoch. Default: -1.
  """

  def __init__(
      self,
      lr: float,
      milestones: Sequence[int],
      gamma: float = 0.1,
      last_epoch: int = -1
  ):
    super().__init__(lr=lr, last_epoch=last_epoch)

    self.milestones = check.is_sequence(milestones, elem_type=int, allow_none=False)
    self.gamma = check.is_float(gamma, min_bound=0., max_bound=1., allow_int=False)

  @bm.cls_jit(inline=True)
  def __call__(self, i=None):
    i = (self.last_epoch.value + 1) if i is None else i
    p = bm.ifelse([i < m for m in self.milestones],
                  list(range(0, len(self.milestones))) + [len(self.milestones)])
    return self.lr * self.gamma ** p

  def __repr__(self):
    return (f'{self.__class__.__name__}(lr={self.lr}, '
            f'milestones={self.milestones}, gamma={self.gamma}, '
            f'last_epoch={self.last_epoch})')


class CosineAnnealingLR(Scheduler):
  r"""Set the learning rate of each parameter group using a cosine annealing
  schedule, where :math:`\eta_{max}` is set to the initial lr and
  :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

  .. math::
      \begin{aligned}
          \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
          + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
          & T_{cur} \neq (2k+1)T_{max}; \\
          \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
          \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
          & T_{cur} = (2k+1)T_{max}.
      \end{aligned}

  When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
  is defined recursively, the learning rate can be simultaneously modified
  outside this scheduler by other operators. If the learning rate is set
  solely by this scheduler, the learning rate at each step becomes:

  .. math::
      \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
      \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

  It has been proposed in
  `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
  implements the cosine annealing part of SGDR, and not the restarts.

  Parameters
  ----------
  lr: float
    Initial learning rate.
  T_max: int
    Maximum number of iterations.
  eta_min: float
    Minimum learning rate. Default: 0.
  last_epoch: int
    The index of last epoch. Default: -1.

  .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
      https://arxiv.org/abs/1608.03983
  """

  def __init__(self,
               lr: float,
               T_max: int,
               eta_min: float = 0.,
               last_epoch: int = -1, ):
    super().__init__(lr=lr, last_epoch=last_epoch)

    self._init_epoch = last_epoch
    self.T_max = check.is_integer(T_max, min_bound=1)
    self.eta_min = eta_min

  @bm.cls_jit(inline=True)
  def __call__(self, i=None):
    i = (self.last_epoch + 1) if i is None else i
    return (self.eta_min + (self.lr - self.eta_min) *
            (1 + jnp.cos(jnp.pi * i / self.T_max)) / 2)


class CosineAnnealingWarmRestarts(CallBasedScheduler):
  """Set the learning rate of each parameter group using a cosine annealing
  schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
  is the number of epochs since the last restart and :math:`T_{i}` is the number
  of epochs between two warm restarts in SGDR:

  .. math::
      \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
      \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

  When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
  When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

  It has been proposed in
  `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

  Parameters
  ----------
  lr: float
    Initial learning rate.
  num_call_per_epoch: int
    The number the scheduler to call in each epoch.
    This usually means the number of batch in each epoch training.
  T_0: int
    Number of iterations for the first restart.
  T_mult: int
    A factor increases :math:`T_{i}` after a restart. Default: 1.
  eta_min: float
    Minimum learning rate. Default: 0.
  last_call: int
    The index of last call. Default: -1.

  .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
      https://arxiv.org/abs/1608.03983
  """

  def __init__(self,
               lr: float,
               num_call_per_epoch: int,
               T_0: int,
               T_mult: int = 1,
               eta_min: float = 0.,
               last_epoch: int = -1,
               last_call: int = -1):
    super().__init__(lr=lr, last_call=last_call, last_epoch=last_epoch)
    if T_0 <= 0 or not isinstance(T_0, int):
      raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
    if T_mult < 1 or not isinstance(T_mult, int):
      raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))

    self.T_mult = T_mult
    self.eta_min = eta_min
    self.T_0 = T_0
    self.num_call_per_epoch = num_call_per_epoch

  def _cond1(self, epoch):
    if self.T_mult == 1:
      T_cur = epoch % self.T_0
      T_i = self.T_0
    else:
      n = jnp.floor(jnp.log(epoch / self.T_0 * (self.T_mult - 1) + 1) / jnp.log(self.T_mult))
      T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
      T_i = self.T_0 * self.T_mult ** n
    return T_cur, T_i

  def _cond2(self, epoch):
    return epoch, self.T_0

  @bm.cls_jit(inline=True)
  def __call__(self, i=None):
    i = (self.last_call + 1) if i is None else i
    epoch = i / self.num_call_per_epoch
    T_cur, T_i = jax.lax.cond(epoch >= self.T_0,
                              self._cond1,
                              self._cond2,
                              epoch)
    return self.eta_min + (self.lr - self.eta_min) * (1 + jnp.cos(jnp.pi * T_cur / T_i)) / 2

  @bm.cls_jit(inline=True)
  def current_epoch(self, i=None):
    i = (self.last_call + 1) if i is None else i
    return jnp.floor(i / self.num_call_per_epoch)


class ExponentialLR(Scheduler):
  """Decays the learning rate of each parameter group by gamma every epoch.
  When last_epoch=-1, sets initial lr as lr.

  Parameters
  ----------
  lr: float
    Initial learning rate.
  gamma: float
    Multiplicative factor of learning rate decay.
  last_epoch: int
    The index of last epoch. Default: -1.
  """

  def __init__(self,
               lr: float,
               gamma: float,
               last_epoch: int = -1):
    super(ExponentialLR, self).__init__(lr=lr, last_epoch=last_epoch)
    self.gamma = check.is_float(gamma, min_bound=0., max_bound=1.)

  def __call__(self, i: int = None):
    i = (self.last_epoch + 1) if i is None else i
    return self.lr * self.gamma ** i

  def __repr__(self):
    return f'{self.__class__.__name__}(lr={self.lr}, last_epoch={self.last_epoch}, gamma={self.gamma})'


class ExponentialDecay(CallBasedScheduler):
  def __init__(self, lr, decay_steps, decay_rate, last_epoch: int = -1, last_call: int = -1):
    super(ExponentialDecay, self).__init__(lr=lr, last_epoch=last_epoch, last_call=last_call)
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

  def __call__(self, i=None):
    i = (self.last_call.value + 1) if i is None else i
    return self.lr * self.decay_rate ** (i / self.decay_steps)

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.lr}, '
            f'decay_steps={self.decay_steps}, '
            f'decay_rate={self.decay_rate}), '
            f'last_call={self.last_call.value})')


class InverseTimeDecay(ExponentialDecay):
  def __init__(self, lr, decay_steps, decay_rate, staircase=False,
               last_epoch: int = -1, last_call: int = -1):
    super(InverseTimeDecay, self).__init__(lr, decay_steps, decay_rate,
                                           last_epoch=last_epoch,
                                           last_call=last_call)
    self.staircase = staircase

  def __call__(self, i=None):
    i = (self.last_call.value + 1) if i is None else i
    if self.staircase:
      return self.lr / (1 + self.decay_rate * jnp.floor(i / self.decay_steps))
    else:
      return self.lr / (1 + self.decay_rate * i / self.decay_steps)

  def __repr__(self):
    return f'{self.__class__.__name__}({self.lr}, staircase={self.staircase})'


class PolynomialDecay(CallBasedScheduler):
  def __init__(self, lr, decay_steps, final_lr, power=1.0, last_epoch: int = -1, last_call: int = -1):
    super(PolynomialDecay, self).__init__(lr, last_epoch=last_epoch, last_call=last_call)
    self.decay_steps = decay_steps
    self.final_lr = final_lr
    self.power = power

  def __call__(self, i=None):
    i = (self.last_call.value + 1) if i is None else i
    i = jnp.minimum(i, self.decay_steps)
    step_mult = (1 - i / self.decay_steps) ** self.power
    return step_mult * (self.lr - self.final_lr) + self.final_lr

  def __repr__(self):
    return (f'{self.__class__.__name__}({self.lr}, '
            f'last_call={self.last_call.value}, '
            f'decay_steps={self.decay_steps}, '
            f'final_lr={self.final_lr}, '
            f'power={self.power})')


class PiecewiseConstant(CallBasedScheduler):
  def __init__(self, boundaries, values, last_epoch: int = -1, last_call: int = -1):
    super(PiecewiseConstant, self).__init__(0., last_epoch=last_epoch, last_call=last_call)

    boundaries = jnp.array(boundaries)
    values = jnp.array(values)
    if not boundaries.ndim == values.ndim == 1:
      raise MathError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
      raise MathError("boundaries length must be one shorter than values length")
    self.boundaries = boundaries
    self.values = values

  def __call__(self, i=None):
    i = (self.last_call.value + 1) if i is None else i
    return self.values[jnp.sum(i > self.boundaries)]
