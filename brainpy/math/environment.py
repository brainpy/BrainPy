# -*- coding: utf-8 -*-

from brainpy._src.math.environment import (
  environment as environment,
  batching_environment as batching_environment,
  training_environment as training_environment,
  set as set,
  set_environment as set_environment,

  set_float as set_float,
  get_float as get_float,
  set_int as set_int,
  get_int as get_int,
  set_bool as set_bool,
  get_bool as get_bool,
  set_complex as set_complex,
  get_complex as get_complex,
  set_dt as set_dt,
  get_dt as get_dt,
  set_mode as set_mode,
  get_mode as get_mode,

  enable_x64 as enable_x64,
  disable_x64 as disable_x64,
  set_platform as set_platform,
  get_platform as get_platform,
  set_host_device_count as set_host_device_count,
  clear_buffer_memory as clear_buffer_memory,
  enable_gpu_memory_preallocation as enable_gpu_memory_preallocation,
  disable_gpu_memory_preallocation as disable_gpu_memory_preallocation,
  ditype as ditype,
  dftype as dftype,
)
