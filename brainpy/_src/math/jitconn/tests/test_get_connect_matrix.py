# -*- coding: utf-8 -*-
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainpy.math as bm
from brainpy._src.dependency_check import import_taichi

if import_taichi(error_if_not_found=False) is None:
  pytest.skip('no taichi', allow_module_level=True)

import platform

force_test = False  # turn on to force test on windows locally
if platform.system() == 'Windows' and not force_test:
  pytest.skip('skip windows', allow_module_level=True)

shapes = [(100, 200), (1000, 10)]


# SEED = 1234

class TestGetConnectMatrix(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestGetConnectMatrix, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.product(
    transpose=[True, False],
    outdim_parallel=[True, False],
    shape=shapes,
    prob=[0.1],
  )
  def test_get_connect_matrix(self, transpose, outdim_parallel, shape, prob):
    print(
      f'test_get_connect_matrix: transpose={transpose}, outdim_parallel={outdim_parallel}, shape={shape}, prob={prob}')
    conn = bm.jitconn.get_connect_matrix(prob, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
    shape = (shape[1], shape[0]) if transpose else shape
    assert conn.shape == shape
    assert conn.dtype == jnp.bool_
    # sum all true values
    # assert jnp.sum(conn) == jnp.round(prob * shape[0] * shape[1])
    print(
      f'jnp.sum(conn): {jnp.sum(conn)}, jnp.round(prob * shape[0] * shape[1]): {jnp.round(prob * shape[0] * shape[1])}')
    # print(f'conn: {conn}')
