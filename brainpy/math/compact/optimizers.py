# -*- coding: utf-8 -*-

import warnings

from brainpy import optimizers

__all__ = [
  'SGD',
  'Momentum',
  'MomentumNesterov',
  'Adagrad',
  'Adadelta',
  'RMSProp',
  'Adam',

  'Constant',
  'ExponentialDecay',
  'InverseTimeDecay',
  'PolynomialDecay',
  'PiecewiseConstant',
]


def SGD(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.SGD" instead. '
                '"brainpy.math.optimizers.SGD" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.SGD(*args, **kwargs)


def Momentum(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.Momentum" instead. '
                '"brainpy.math.optimizers.Momentum" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.Momentum(*args, **kwargs)


def MomentumNesterov(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.MomentumNesterov" instead. '
                '"brainpy.math.optimizers.MomentumNesterov" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.MomentumNesterov(*args, **kwargs)


def Adagrad(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.Adagrad" instead. '
                '"brainpy.math.optimizers.Adagrad" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.Adagrad(*args, **kwargs)


def Adadelta(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.Adadelta" instead. '
                '"brainpy.math.optimizers.Adadelta" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.Adadelta(*args, **kwargs)


def RMSProp(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.RMSProp" instead. '
                '"brainpy.math.optimizers.RMSProp" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.RMSProp(*args, **kwargs)


def Adam(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.Adam" instead. '
                '"brainpy.math.optimizers.Adam" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.Adam(*args, **kwargs)


def Constant(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.Constant" instead. '
                '"brainpy.math.optimizers.Constant" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.Constant(*args, **kwargs)


def ExponentialDecay(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.ExponentialDecay" instead. '
                '"brainpy.math.optimizers.ExponentialDecay" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.ExponentialDecay(*args, **kwargs)


def InverseTimeDecay(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.InverseTimeDecay" instead. '
                '"brainpy.math.optimizers.InverseTimeDecay" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.InverseTimeDecay(*args, **kwargs)


def PolynomialDecay(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.PolynomialDecay" instead. '
                '"brainpy.math.optimizers.PolynomialDecay" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.PolynomialDecay(*args, **kwargs)


def PiecewiseConstant(*args, **kwargs):
  warnings.warn('Please use "brainpy.optim.PiecewiseConstant" instead. '
                '"brainpy.math.optimizers.PiecewiseConstant" is '
                'deprecated since version 2.0.3. ',
                DeprecationWarning)
  return optimizers.PiecewiseConstant(*args, **kwargs)
