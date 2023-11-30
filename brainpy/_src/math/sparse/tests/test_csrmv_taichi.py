# -*- coding: utf-8 -*-

from functools import partial

import jax
import pytest
# from absl.testing import parameterized
import platform
import brainpy as bp
import brainpy.math as bm

# is_manual_test = False
# if platform.system() == 'Windows' and not is_manual_test:
#   pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)

vector_csr_matvec = partial(bm.sparse.csrmv, method='vector')

homo_datas=[-1., 0., 0.1, 1.]
shapes=[(100, 200), (10, 1000), (2, 2000)]

def test_homo(shape, homo_data):
    print(f'test_homo: shape = {shape}, homo_data = {homo_data}')
    conn = bp.conn.FixedProb(0.1)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[1])
    vector = bm.as_jax(vector)

    r1 = vector_csr_matvec(homo_data, indices, indptr, vector, shape=shape)
    r2 = bm.sparse.csrmv_taichi(homo_data, indices, indptr, vector, shape=shape)

    assert(bm.allclose(r1, r2[0]))

def test_heter(shape):
    print(f'test_homo: shape = {shape}')
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    heter_data = bm.as_jax(rng.random(indices.shape))
    vector = bm.as_jax(rng.random(shape[1]))

    r1 = vector_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r2 = bm.sparse.csrmv_taichi(heter_data, indices, indptr, vector, shape=shape)

    assert(bm.allclose(r1, r2[0]))

# for shape in shapes:
#     for homo_data in homo_datas:
#         test_homo(shape, homo_data)

for shape in shapes:
    test_heter(shape)