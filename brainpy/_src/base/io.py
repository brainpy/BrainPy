# -*- coding: utf-8 -*-

from brainpy._src.checkpoints import io

__deprecations = {
  'save_as_h5': ('brainpy.base.io.save_as_h5', 'brainpy.checkpoints.save_as_h5', io.save_as_h5),
  'load_by_h5': ('brainpy.base.io.load_by_h5', 'brainpy.checkpoints.load_by_h5', io.load_by_h5),
  'save_as_npz': ('brainpy.base.io.save_as_npz', 'brainpy.checkpoints.save_as_npz', io.save_as_npz),
  'load_by_npz': ('brainpy.base.io.load_by_npz', 'brainpy.checkpoints.load_by_npz', io.load_by_npz),
  'save_as_pkl': ('brainpy.base.io.save_as_pkl', 'brainpy.checkpoints.save_as_pkl', io.save_as_pkl),
  'load_by_pkl': ('brainpy.base.io.load_by_pkl', 'brainpy.checkpoints.load_by_pkl', io.load_by_pkl),
  'save_as_mat': ('brainpy.base.io.save_as_mat', 'brainpy.checkpoints.save_as_mat', io.save_as_mat),
  'load_by_mat': ('brainpy.base.io.load_by_mat', 'brainpy.checkpoints.load_by_mat', io.load_by_mat),
}
from brainpy._src.deprecations import deprecation_getattr2
__getattr__ = deprecation_getattr2('', __deprecations)
del deprecation_getattr2



