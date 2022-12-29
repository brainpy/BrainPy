# -*- coding: utf-8 -*-

from brainpy import checkpoints

__all__ = [
  'SUPPORTED_FORMATS',
  'save_as_h5', 'load_by_h5',
  'save_as_npz', 'load_by_npz',
  'save_as_pkl', 'load_by_pkl',
  'save_as_mat', 'load_by_mat',
]


SUPPORTED_FORMATS = checkpoints.SUPPORTED_FORMATS
save_as_h5 = checkpoints.save_as_h5
load_by_h5 = checkpoints.load_by_h5
save_as_npz = checkpoints.save_as_npz
load_by_npz = checkpoints.load_by_npz
save_as_pkl = checkpoints.save_as_pkl
load_by_pkl = checkpoints.load_by_pkl
save_as_mat = checkpoints.save_as_mat
load_by_mat = checkpoints.load_by_mat


