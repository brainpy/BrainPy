# -*- coding: utf-8 -*-

from brainpy import numpy

print(numpy.array)
print(numpy.ones)
print(numpy.random.uniform)
print(numpy.linalg.det)
print('-' * 30)

print('numba')
numpy._reload('numba')
print(numpy.array)
print(numpy.ones)
print(numpy.random.uniform)
print(numpy.random.seed)
print(numpy.linalg.det)
print('-' * 30)
