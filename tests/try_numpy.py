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
print(numpy.isreal)
print(numpy.isreal(numpy.zeros(10)))
print(numpy.isreal(numpy.zeros(10) + .1j))
print('-' * 30)

print('tensorflow')
numpy._reload('tf-numpy')
print(numpy.array)
print(numpy.ones)
print(numpy.random.uniform)
print(numpy.random.seed)
print(numpy.linalg.det)
print('-' * 30)



