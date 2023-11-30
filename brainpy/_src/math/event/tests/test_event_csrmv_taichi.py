# -*- coding: utf-8 -*-


from functools import partial

import jax
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
import platform

import pytest

is_manual_test = False
if platform.system() == 'Windows' and not is_manual_test:
  pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)


# transposes = [True, False]
# shapes = [(100, 200),
#          (200, 200),
#          (200, 100),
#          (10, 1000),
#          (2, 10000),
#          (1000, 10),
#          (10000, 2)]
# homo_datas = [-1., 0., 1.]

class Test_event_csr_matvec(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_event_csr_matvec, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.named_parameters(
    dict(
    testcase_name=f'transpose={transpose}, shape={shape}, homo_data={homo_data}',
    transpose=transpose,
    shape=shape,
    homo_data=homo_data,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                (200, 200),
                (200, 100),
                (10, 1000),
                (2, 10000),
                (1000, 10),
                (10000, 2)]
    for homo_data in [-1., 0., 1.]
  )
  def test_homo(self, shape, transpose, homo_data):
    print(f'test_homo: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')
    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    heter_data = bm.ones(indices.shape) * homo_data
    
    r1 = bm.event.csrmv(homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = bm.event.csrmv_taichi(homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    
    assert(bm.allclose(r1, r2[0]))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=f'transpose={transpose}, shape={shape}',
      shape=shape,
      transpose=transpose,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (10000, 2)]
  )
  def test_heter(self, shape, transpose):
    print(f'test_heter: shape = {shape}, transpose = {transpose}')
    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    heter_data = bm.as_jax(rng.random(indices.shape))

    r1 = bm.event.csrmv(heter_data, indices, indptr, events,
                            shape=shape, transpose=transpose)
    r2 = bm.event.csrmv_taichi(heter_data, indices, indptr, events,
                                shape=shape, transpose=transpose)
    
    assert(bm.allclose(r1, r2[0]))

    bm.clear_buffer_memory()

# for transpose in transposes:
#   for shape in shapes:
#     for homo_data in homo_datas:
#       test_homo(shape, transpose, homo_data) 

# for transpose in transposes:
#   for shape in shapes:
#       test_heter(shape, transpose) 

