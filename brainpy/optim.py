# -*- coding: utf-8 -*-


# optimizers #
# ---------- #


from brainpy._src.optimizers.optimizer import (
  Optimizer as Optimizer,
)
from brainpy._src.optimizers.optimizer import (
  SGD as SGD,
  Momentum as Momentum,
  MomentumNesterov as MomentumNesterov,
  Adagrad as Adagrad,
  Adadelta as Adadelta,
  RMSProp as RMSProp,
  Adam as Adam,
  LARS as LARS,
  Adan as Adan,
  AdamW as AdamW,
)


# schedulers #
# ---------- #


from brainpy._src.optimizers.scheduler import (
  make_schedule as make_schedule,
  Scheduler as Scheduler,
)
from brainpy._src.optimizers.scheduler import (
  Constant as Constant,
  ExponentialDecay as ExponentialDecay,
  InverseTimeDecay as InverseTimeDecay,
  PolynomialDecay as PolynomialDecay,
  PiecewiseConstant as PiecewiseConstant,
)
from brainpy._src.optimizers.scheduler import (
  StepLR as StepLR,
  MultiStepLR as MultiStepLR,
  ExponentialLR as ExponentialLR,
  CosineAnnealingLR as CosineAnnealingLR,
  CosineAnnealingWarmRestarts as CosineAnnealingWarmRestarts,
)

