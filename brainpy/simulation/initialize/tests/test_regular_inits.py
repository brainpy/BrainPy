# -*- coding: utf-8 -*-

import brainpy as bp


def test_zero_init():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    init = bp.initialize.ZeroInit()
    for size in [(100,), (10, 20), (10, 20, 30)]:
      weights = init(size)
      assert weights.shape == size
      assert isinstance(weights, bp.math.ndarray)


def test_one_init():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for size in [(100,), (10, 20), (10, 20, 30)]:
      for value in [0., 1., -1.]:
        init = bp.initialize.OneInit(value=value)
        weights = init(size)
        assert weights.shape == size
        assert isinstance(weights, bp.math.ndarray)
        assert (weights == value).all()


def test_identity_init():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for size in [(100,), (10, 20), ]:
      for value in [0., 1., -1.]:
        init = bp.initialize.Identity(value=value)
        weights = init(size)
        if len(size) == 1:
          assert weights.shape == (size[0], size[0])
        else:
          assert weights.shape == size
        assert isinstance(weights, bp.math.ndarray)
