# -*- coding: utf-8 -*-

from brainpy._src.running.jax_multiprocessing import (
  jax_vectorize_map as jax_vectorize_map,
  jax_parallelize_map as jax_parallelize_map,
)

from brainpy._src.running.native_multiprocessing import (
  process_pool as process_pool,
  process_pool_lock as process_pool_lock,
)

from brainpy._src.running.pathos_multiprocessing import (
  cpu_ordered_parallel as cpu_ordered_parallel,
  cpu_unordered_parallel as cpu_unordered_parallel,
)
