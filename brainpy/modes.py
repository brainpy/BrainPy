# -*- coding: utf-8 -*-

"""
This module is deprecated since version 2.3.1.
Please use ``brainpy.math.*`` instead.
"""


import numpy as np

import brainpy.math as bm


__all__ = [
  'Mode',
  'NormalMode',
  'BatchingMode',
  'TrainingMode',

  'normal',
  'batching',
  'training',

  'check_mode',
]

Mode = bm.Mode

NormalMode = bm.NonBatchingMode
BatchingMode = bm.BatchingMode
TrainingMode = bm.TrainingMode

normal = bm.nonbatching_mode
batching = bm.batching_mode
training = bm.training_mode


def check_mode(mode, supported_modes, name=''):
  """Check whether the used mode is in the list of the supported models.

  Parameters
  ----------
  mode: Mode
    The mode used.
  supported_modes: type, list of type, tuple of type
    The list of all types to support.
  name: Any
    The name.
  """
  if isinstance(supported_modes, type):
    supported_modes = (supported_modes,)
  if not isinstance(supported_modes, (tuple, list)):
    raise TypeError(f'supported_modes must be a tuple/list of type. But wwe got {type(supported_modes)}')
  for smode in supported_modes:
    if not isinstance(smode, type):
      raise TypeError(f'supported_modes must be a tuple/list of type. But wwe got {smode}')
  checking = np.asarray([issubclass(smode, type(mode)) for smode in supported_modes])
  if not np.isin(True, checking):
    raise NotImplementedError(f"{name} does not support {mode}. We only support "
                              f"{', '.join([mode.__name__ for mode in supported_modes])}. ")
