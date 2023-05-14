# -*- coding: utf-8 -*-

import jax.numpy as jnp
import unittest

import brainpy.math as bm
from jax import vmap

from brainpylib import event_info


class Test_event_info(unittest.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_event_info, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  def _base_test(self, length):
    print(f'{self._base_test.__name__}: length = {length}')

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(length)) < 0.1
    event_ids, event_num = event_info(events)
    self.assertTrue(jnp.allclose(jnp.sum(events, keepdims=True), event_num))

    bm.clear_buffer_memory()

  def _base_vmap(self, length):
    print(f'{self._base_vmap.__name__}: length = {length}')

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random((10, length))) < 0.1
    event_ids, event_num = vmap(event_info)(events)
    self.assertTrue(jnp.allclose(jnp.sum(events, axis=-1), event_num))

    bm.clear_buffer_memory()

  def _base_vmap_vmap(self, length):
    print(f'{self._base_vmap_vmap.__name__}: length = {length}')

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random((10, length))) < 0.1
    event_ids, event_num = vmap(vmap(event_info))(events)
    self.assertTrue(jnp.allclose(jnp.sum(events, axis=-1), event_num))

    bm.clear_buffer_memory()

  def test(self):
    for length in [1, 3, 8, 10, 100, 200, 500, 1000, 10000, 100000]:
      self._base_test(length)

  def test_vmap(self):
    for length in [1, 3, 8, 10, 100, 200, 500, 1000, 10000, 100000]:
      self._base_test(length)

  def test_vmap_vmap(self):
    for length in [1, 3, 8, 10, 100, 200, 500, 1000, 10000, 100000]:
      self._base_test(length)



