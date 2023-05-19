
# -*- coding: utf-8 -*-

import time

import brainpy as bp
import brainpy.math as bm
import numpy as np

from brainpy._src.math.sparse import cusparse_bcsr_matvec
# from brainpy._src.math.sparse import cusparse_csr_matvec
from brainpy._src.math.sparse import csrmv
from scipy.sparse import csr_matrix

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

    r1 = bm.sparse.csrmv(data, indices, indptr, vector, shape=shape, method='cusparse')
    r1.block_until_ready()
    r2 = bm.sparse.csrmv(data, indices, indptr, vector, shape=shape, method='vector')
    r2.block_until_ready()

    t0 = time.time()
    for _ in range(100):
      r1 = bm.sparse.csrmv(data, indices, indptr, vector, shape=shape, method='cusparse')
      r1.block_until_ready()
    print(f'cuSPARSE {time.time() - t0} s')

    t0 = time.time()
    for _ in range(100):
      r1 = bm.sparse.csrmv(data, indices, indptr, vector, shape=shape, method='vector')
      r1.block_until_ready()
    print(f'brainpylib {time.time() - t0} s')
    print()



def compare2(platform='cpu'):
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
    p = 0.1

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

        rng = bm.random.RandomState()
        conn = bp.conn.FixedProb(p)(*shape)
        indices, indptr = conn.require('pre2post')
        data = rng.random(indices.shape)
        vector = rng.random(shape[1])




        bs_bsr = 16
        conn = bp.conn.FixedProb(p)(shape[0] // bs_bsr , shape[1] // bs_bsr)
        indices_bsr, indptr_bsr = conn.require('pre2post')
        data_bsr = rng.rand(len(indices_bsr)*bs_bsr, bs_bsr )
        shape_bsr = (shape[0] // bs_bsr, shape[1] // bs_bsr)

        # Mcsr = csr_matrix((data, indices, indptr), shape=shape)
        # Mbsr = Mcsr.tobsr(blocksize=(8,8))
        # bs_bsr = 8
        # indices_bsr = Mbsr.indices
        # indptr_bsr = Mbsr.indptr
        # data_bsr_2 = Mbsr.data
        # data_bsr = list(np.array(data_bsr_2).flatten())
        # indices_bsr = bm.as_jax(indices_bsr)
        # indptr_bsr = bm.as_jax(indptr_bsr)
        # data_bsr = bm.as_jax(data_bsr)
        # shape_bsr = (shape[0]//bs_bsr,shape[1]//bs_bsr)

        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()

        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        r2 = csrmv(data, indices, indptr, vector, shape=shape)
        r2.block_until_ready()

        # print(r1[980:1000])
        # print(r2[980:1000])
        # print(r3[900:1000])
        # print(len(indptr_bsr))
        # print(shape_bsr)

        t0 = time.time()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape_bsr)
        r3.block_until_ready()
        print(f'bsrSPARSE {time.time() - t0} s')

        # t0 = time.time()
        # for _ in range(100):
        #     r3 = cusparse_bcsr_matvec(data_bsr, indices_bsr, indptr_bsr, vector, blocksize=bs_bsr,nnzb=len(indices_bsr), shape=shape)
        #     r3.block_until_ready()
        # print(f'bsrSPARSE {time.time() - t0} s')


        # t0 = time.time()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        # r1.block_until_ready()
        # print(f'cuSPARSE {time.time() - t0} s')
        # t0 = time.time()
        # for _ in range(100):
        #     r1 = cusparse_csr_matvec(data, indices, indptr, vector, shape=shape)
        #     r1.block_until_ready()
        # print(f'cuSPARSE {time.time() - t0} s')

        t0 = time.time()
        for _ in range(100):
            r1 = csrmv(data, indices, indptr, vector, shape=shape)
            r1.block_until_ready()
        print(f'brainpylib {time.time() - t0} s')
        print()

        bm.clear_buffer_memory()


if __name__ == '__main__':
    compare('cpu')
    # compare('gpu')
