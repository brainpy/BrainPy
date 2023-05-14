# -*- coding: utf-8 -*-

import time

import brainpy as bp
import brainpy.math as bm

import brainpylib


def compare(platform='cpu'):
  """

  CPU
  ---

  shape = (1000, 1000)
  cuSPARSE 0.02663278579711914 s
  brainpylib 0.028490781784057617 s

  shape = (1000, 10000)
  cuSPARSE 0.06195855140686035 s
  brainpylib 0.04008936882019043 s

  shape = (10000, 1000)
  cuSPARSE 0.04706525802612305 s
  brainpylib 0.04366803169250488 s

  shape = (10000, 10000)
  cuSPARSE 0.1891341209411621 s
  brainpylib 0.177717924118042 s

  shape = (100000, 10000)
  cuSPARSE 1.3123579025268555 s
  brainpylib 1.3357517719268799 s

  shape = (100000, 100000)
  cuSPARSE 13.544525384902954 s
  brainpylib 14.612009048461914 s


  GPU
  ---
  shape = (1000, 1000)
  cuSPARSE 0.04015922546386719 s
  brainpylib 0.024152517318725586 s

  shape = (1000, 10000)
  cuSPARSE 0.04857826232910156 s
  brainpylib 0.15707015991210938 s

  shape = (10000, 1000)
  cuSPARSE 0.04973483085632324 s
  brainpylib 0.14293313026428223 s

  shape = (10000, 10000)
  cuSPARSE 0.17399168014526367 s
  brainpylib 0.17151856422424316 s

  shape = (100000, 10000)
  cuSPARSE 0.5249958038330078 s
  brainpylib 0.3427560329437256 s

  shape = (50000, 50000)
  cuSPARSE 1.4121572971343994 s
  brainpylib 0.9002335071563721 s

  shape = (100000, 50000)
  cuSPARSE 2.697688341140747 s
  brainpylib 1.6211459636688232 s
  """


  bm.set_platform(platform)

  for shape in [
    (1000, 1000),
    (1000, 10000),
    (10000, 1000),
    (10000, 10000),
    (100000, 10000),
    (50000, 50000),
    (100000, 50000),
  ]:
    print(f'shape = {shape}')

    rng = bm.random.RandomState(123)
    conn = bp.conn.FixedProb(0.1)(*shape)
    indices, indptr = conn.require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    data = rng.random(indices.shape).value
    vector = rng.random(shape[1]).value

    r1 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
    r1.block_until_ready()
    r2 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)
    r2.block_until_ready()

    t0 = time.time()
    for _ in range(100):
      r1 = brainpylib.cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
      r1.block_until_ready()
    print(f'cuSPARSE {time.time() - t0} s')

    t0 = time.time()
    for _ in range(100):
      r1 = brainpylib.csr_matvec(data, indices, indptr, vector, shape=shape)
      r1.block_until_ready()
    print(f'brainpylib {time.time() - t0} s')
    print()


if __name__ == '__main__':
  # compare('cpu')
  compare('gpu')
