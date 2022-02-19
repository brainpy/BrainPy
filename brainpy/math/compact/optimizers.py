# -*- coding: utf-8 -*-

import warnings

from brainpy import optimizers

__all__ = [
  'Optimizer',
  'SGD',
  'Momentum',
  'MomentumNesterov',
  'Adagrad',
  'Adadelta',
  'RMSProp',
  'Adam',

  'Scheduler',
  'Constant',
  'ExponentialDecay',
  'InverseTimeDecay',
  'PolynomialDecay',
  'PiecewiseConstant',
]


class Optimizer(optimizers.Optimizer):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Optimizer" instead. '
                  '"brainpy.math.optimizers.Optimizer" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Optimizer, self).__init__(*args, **kwargs)


class SGD(optimizers.SGD):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.SGD" instead. '
                  '"brainpy.math.optimizers.SGD" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(SGD, self).__init__(*args, **kwargs)


class Momentum(optimizers.Momentum):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Momentum" instead. '
                  '"brainpy.math.optimizers.Momentum" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Momentum, self).__init__(*args, **kwargs)


class MomentumNesterov(optimizers.MomentumNesterov):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.MomentumNesterov" instead. '
                  '"brainpy.math.optimizers.MomentumNesterov" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(MomentumNesterov, self).__init__(*args, **kwargs)


class Adagrad(optimizers.Adagrad):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Adagrad" instead. '
                  '"brainpy.math.optimizers.Adagrad" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Adagrad, self).__init__(*args, **kwargs)


class Adadelta(optimizers.Adadelta):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Adadelta" instead. '
                  '"brainpy.math.optimizers.Adadelta" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Adadelta, self).__init__(*args, **kwargs)


class RMSProp(optimizers.RMSProp):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.RMSProp" instead. '
                  '"brainpy.math.optimizers.RMSProp" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(RMSProp, self).__init__(*args, **kwargs)


class Adam(optimizers.Adam):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Adam" instead. '
                  '"brainpy.math.optimizers.Adam" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Adam, self).__init__(*args, **kwargs)


class Scheduler(optimizers.Scheduler):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Scheduler" instead. '
                  '"brainpy.math.optimizers.Scheduler" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Scheduler, self).__init__(*args, **kwargs)


class Constant(optimizers.Constant):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.Constant" instead. '
                  '"brainpy.math.optimizers.Constant" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(Constant, self).__init__(*args, **kwargs)


class ExponentialDecay(optimizers.ExponentialDecay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.ExponentialDecay" instead. '
                  '"brainpy.math.optimizers.ExponentialDecay" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(ExponentialDecay, self).__init__(*args, **kwargs)


class InverseTimeDecay(optimizers.InverseTimeDecay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.InverseTimeDecay" instead. '
                  '"brainpy.math.optimizers.InverseTimeDecay" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(InverseTimeDecay, self).__init__(*args, **kwargs)


class PolynomialDecay(optimizers.PolynomialDecay):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.PolynomialDecay" instead. '
                  '"brainpy.math.optimizers.PolynomialDecay" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(PolynomialDecay, self).__init__(*args, **kwargs)


class PiecewiseConstant(optimizers.PiecewiseConstant):
  def __init__(self, *args, **kwargs):
    warnings.warn('Please use "brainpy.opt.PiecewiseConstant" instead. '
                  '"brainpy.math.optimizers.PiecewiseConstant" will be removed since version 2.1.0. ',
                  DeprecationWarning)
    super(PiecewiseConstant, self).__init__(*args, **kwargs)
