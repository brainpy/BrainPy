# -*- coding: utf-8 -*-

from brainpy._src.checkpoints import io

__all__ = [
  'SUPPORTED_FORMATS',
  'save_as_h5', 'load_by_h5',
  'save_as_npz', 'load_by_npz',
  'save_as_pkl', 'load_by_pkl',
  'save_as_mat', 'load_by_mat',
]


SUPPORTED_FORMATS = io.SUPPORTED_FORMATS
save_as_h5 = io.save_as_h5
load_by_h5 = io.load_by_h5
save_as_npz = io.save_as_npz
load_by_npz = io.load_by_npz
save_as_pkl = io.save_as_pkl
load_by_pkl = io.load_by_pkl
save_as_mat = io.save_as_mat
load_by_mat = io.load_by_mat


