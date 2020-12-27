# -*- coding: utf-8 -*-

import math

from numba import cuda


def numba_example(number_of_maximum_loop, gs, ts, bs):
    result = cuda.device_array([3, ])

    @cuda.jit(device=True)
    def BesselJ0(x):
        return math.sqrt(2 / math.pi / x)

    @cuda.jit
    def cuda_kernel(number_of_maximum_loop, result, gs, ts, bs):
        i = cuda.grid(1)
        if i < number_of_maximum_loop:
            cuda.atomic.add(result, 0, BesselJ0(i / 100 + gs))
            cuda.atomic.add(result, 1, BesselJ0(i / 100 + ts))
            cuda.atomic.add(result, 2, BesselJ0(i / 100 + bs))

    # Configure the blocks
    threadsperblock = 128
    blockspergrid = (number_of_maximum_loop + (threadsperblock - 1)) // threadsperblock

    # Start the kernel
    init = [0.0, 0.0, 0.0]
    result = cuda.to_device(init)
    cuda_kernel[blockspergrid, threadsperblock](number_of_maximum_loop, result, gs, ts, bs)

    return result.copy_to_host()
