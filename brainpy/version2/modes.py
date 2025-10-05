# -*- coding: utf-8 -*-

"""
This module is deprecated since version 2.3.1.
Please use ``brainpy.version2.math.*`` instead.
"""

from brainpy.version2 import check
from brainpy.version2.deprecations import deprecation_getattr2
from brainpy.version2.math import modes

__deprecations = {
    'Mode': ('brainpy.version2.modes.Mode', 'brainpy.version2.math.Mode', modes.Mode),
    'NormalMode': ('brainpy.version2.modes.NormalMode', 'brainpy.version2.math.NonBatchingMode', modes.NonBatchingMode),
    'BatchingMode': ('brainpy.version2.modes.BatchingMode', 'brainpy.version2.math.BatchingMode', modes.BatchingMode),
    'TrainingMode': ('brainpy.version2.modes.TrainingMode', 'brainpy.version2.math.TrainingMode', modes.TrainingMode),
    'normal': ('brainpy.version2.modes.normal', 'brainpy.version2.math.nonbatching_mode', modes.nonbatching_mode),
    'batching': ('brainpy.version2.modes.batching', 'brainpy.version2.math.batching_mode', modes.batching_mode),
    'training': ('brainpy.version2.modes.training', 'brainpy.version2.math.training_mode', modes.training_mode),
    'check_mode': ('brainpy.version2.modes.check_mode', 'brainpy.version2.check.is_subclass', check.is_subclass),
}
__getattr__ = deprecation_getattr2('brainpy.version2.modes', __deprecations)
del deprecation_getattr2
