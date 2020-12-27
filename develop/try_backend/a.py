# -*- coding: utf-8 -*-
from numba import cuda
import numpy
import math
import time


num_pre = 100
num_post = 300

A = numpy.ones((num_pre, num_post))
pre_ids, post_ids = numpy.where(A > 0.)

pre_ids = numpy.ascontiguousarray(pre_ids, dtype=int)
post_ids = numpy.ascontiguousarray(post_ids, dtype=int)
pre_ids_cuda = cuda.to_device(pre_ids)
post_ids_cuda = cuda.to_device(post_ids)


pre_val = numpy.random.randint(2, size=(num_pre))
post_val = numpy.zeros(num_post)
pre_val_cuda = cuda.to_device(pre_val)
post_val_cuda = cuda.to_device(post_val)

num = len(pre_ids)


@cuda.jit
def sum_reduce(post_val, pre_val, pre_ids, post_ids):
    i = cuda.grid(1)
    if i < num:
        pre_id = pre_ids[i]
        post_id = post_ids[i]
        cuda.atomic.add(post_val, post_id, pre_val[pre_id])
        # post_val[post_id] += pre_val[pre_id]

# pre_ids_cuda_const = cuda.const.array_like(pre_ids)
# post_ids_cuda_const = cuda.const.array_like(post_ids)


@cuda.jit
def sum_reduce2(post_val, pre_val):
    pre_ids1 = cuda.const.array_like(pre_ids)
    post_ids1 = cuda.const.array_like(post_ids)

    i = cuda.grid(1)
    if i < num:
        pre_id = pre_ids1[i]
        post_id = post_ids1[i]
        # pre_id = pre_ids_cuda_const[i]
        # post_id = post_ids_cuda_const[i]
        cuda.atomic.add(post_val, post_id, pre_val[pre_id])
        # post_val[post_id] += pre_val[pre_id]


t0 = time.time()
# sum_reduce[math.ceil(num / 1024), 1024](post_val_cuda, pre_val_cuda, pre_ids_cuda, post_ids_cuda)
sum_reduce2[math.ceil(num / 1024), 1024](post_val_cuda, pre_val_cuda)
print('GPU time: ', time.time() - t0)
post_val_cuda.to_host()



t0 = time.time()
cpu_val = numpy.dot(pre_val, A)
print('CPU time: ', time.time() - t0)

assert numpy.array_equal(post_val, cpu_val)

