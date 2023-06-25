# -*- coding: utf-8 -*-

"""
This module is deprecated since version 2.3.1.
Please use ``brainpy.math.*`` instead.
"""

from brainpy._src.math import modes
from brainpy import check
from brainpy._src.deprecations import deprecation_getattr2

__deprecations = {
  'Mode': ('brainpy.modes.Mode', 'brainpy.math.Mode', modes.Mode),
  'NormalMode': ('brainpy.modes.NormalMode', 'brainpy.math.NonBatchingMode', modes.NonBatchingMode),
  'BatchingMode': ('brainpy.modes.BatchingMode', 'brainpy.math.BatchingMode', modes.BatchingMode),
  'TrainingMode': ('brainpy.modes.TrainingMode', 'brainpy.math.TrainingMode', modes.TrainingMode),
  'normal': ('brainpy.modes.normal', 'brainpy.math.nonbatching_mode', modes.nonbatching_mode),
  'batching': ('brainpy.modes.batching', 'brainpy.math.batching_mode', modes.batching_mode),
  'training': ('brainpy.modes.training', 'brainpy.math.training_mode', modes.training_mode),
  'check_mode': ('brainpy.modes.check_mode', 'brainpy.check.is_subclass', check.is_subclass),
}
__getattr__ = deprecation_getattr2('brainpy.modes', __deprecations)
del deprecation_getattr2




