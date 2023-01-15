# -*- coding: utf-8 -*-

from brainpy._src.math.arrayoperation import (
  flatten as flatten,
  fill_diagonal as fill_diagonal,
  remove_diag as remove_diag,
  clip_by_norm as clip_by_norm,
)
from brainpy._src.math.arraycreation import (
  empty as empty,
  empty_like as empty_like,
  ones as ones,
  ones_like as ones_like,
  zeros as zeros,
  zeros_like as zeros_like,
  array as array,
  asarray as asarray,
  arange as arange,
  linspace as linspace,
  logspace as logspace,
)
from brainpy._src.math.arrayinterporate import (
  as_device_array as as_device_array,
  as_jax as as_jax,
  as_ndarray as as_ndarray,
  as_numpy as as_numpy,
  as_variable as as_variable,
)

