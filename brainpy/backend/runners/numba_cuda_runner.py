# -*- coding: utf-8 -*-

from .numba_cpu_runner import NumbaCPUNodeRunner

__all__ = [
    'NumbaCudaNodeRunner',
]



class NumbaCudaNodeRunner(NumbaCPUNodeRunner):
    pass

