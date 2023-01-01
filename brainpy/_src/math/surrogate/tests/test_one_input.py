# -*- coding: utf-8 -*-

import jax
from absl.testing import parameterized

import brainpy.math as bm
from brainpy._src.math.surrogate import one_input


class TestOneInputGrad(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestOneInputGrad, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.named_parameters(
    dict(testcase_name=f'{name}_x64={x64}',
         func=getattr(one_input, name),
         x64=x64)
    for name in one_input.__all__
    for x64 in [True, False]
  )
  def test_bm_grad(self, func, x64):
    if x64:
      bm.enable_x64()

    xs = bm.arange(-3, 3, 0.005)
    grads = bm.vector_grad(func)(xs)
    self.assertTrue(grads.size == xs.size)

    if x64:
      bm.disable_x64()

  @parameterized.named_parameters(
    dict(testcase_name=f'{name}_x64={x64}',
         func=getattr(one_input, name),
         x64=x64, )
    for name in one_input.__all__
    for x64 in [True, False]
  )
  def test_jax_vjp(self, func, x64):
    if x64:
      bm.enable_x64()

    xs = bm.arange(-3, 3, 0.005)
    primals, f_vjp = jax.vjp(func, xs)
    grad2 = f_vjp(jax.numpy.ones_like(xs))
    self.assertTrue(grad2[0].size == xs.size)

    if x64:
      bm.disable_x64()
