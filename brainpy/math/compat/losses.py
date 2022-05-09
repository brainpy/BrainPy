# -*- coding: utf-8 -*-

import warnings

from brainpy import losses

__all__ = [
  'cross_entropy_loss',
  'l1_loos',
  'l2_loss',
  'l2_norm',
  'huber_loss',
  'mean_absolute_error',
  'mean_squared_error',
  'mean_squared_log_error',
]


def cross_entropy_loss(*args, **kwargs):
  """Cross entropy loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.cross_entropy_loss" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.cross_entropy_loss(*args, **kwargs)


def l1_loos(*args, **kwargs):
  """L1 loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.l1_loss" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.l1_loos(*args, **kwargs)


def l2_loss(*args, **kwargs):
  """L2 loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.l2_loss" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.l2_loss(*args, **kwargs)


def l2_norm(*args, **kwargs):
  """L2 normal.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.l2_norm" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.l2_norm(*args, **kwargs)


def huber_loss(*args, **kwargs):
  """Huber loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.huber_loss" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.huber_loss(*args, **kwargs)


def mean_absolute_error(*args, **kwargs):
  """mean absolute error loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.mean_absolute_error" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.mean_absolute_error(*args, **kwargs)


def mean_squared_error(*args, **kwargs):
  """Mean squared error loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.mean_squared_error" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.mean_squared_error(*args, **kwargs)


def mean_squared_log_error(*args, **kwargs):
  """Mean squared log error loss.

  .. deprecated:: 2.1.0
     Please use "brainpy.losses.mean_squared_log_error" instead.
  """
  warnings.warn('Please use "brainpy.losses.XXX" instead. '
                '"brainpy.math.losses.XXX" is deprecated since version 2.0.3. ',
                DeprecationWarning)
  return losses.mean_squared_log_error(*args, **kwargs)
