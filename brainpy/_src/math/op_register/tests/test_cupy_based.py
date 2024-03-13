import jax.numpy as jnp
import jax
import cupy as cp
from time import time

import brainpy.math as bm
from brainpy._src.math import as_jax
bm.set_platform('gpu')

time1 = time()
a = bm.random.rand(4, 4)
time2 = time()
c = cp.from_dlpack(jax.dlpack.to_dlpack(as_jax(a)))
time3 = time()

c *= c
print(f'c: {c}')
print(f'a: {a}')
