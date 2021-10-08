# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as bp


def test_GaussianDecay1():
  bp.math.use_backend('numpy')
  conn = bp.initialize.GaussianDecay(sigma=4, max_w=1., periodic_boundary=True)
  weights = conn(100)

  # plt.imshow(weights)
  # plt.show()


def test_GaussianDecay2():
  bp.math.use_backend('numpy')

  size = (30, 30)
  conn = bp.initialize.GaussianDecay(sigma=4, max_w=1., periodic_boundary=True)
  weights = conn(size)

  # plt.imshow(weights)
  # plt.show()
  #
  # plt.imshow(weights[0].reshape(size))
  # plt.show()


def test_GaussianDecay3():
  bp.math.use_backend('numpy')

  size = (20, 20, 20)
  conn = bp.initialize.GaussianDecay(sigma=4, max_w=1., periodic_boundary=True)
  weights = conn(size)

  # plt.imshow(weights)
  # plt.show()


def test_DOGDecay1():
  bp.math.use_backend('numpy')
  conn = bp.initialize.DOGDecay(sigmas=(1., 2.5), max_ws=(1.0, 0.7), periodic_boundary=True)
  weights = conn(100)

  # pos = plt.imshow(weights, cmap='cool', interpolation='none')
  # plt.colorbar(pos)
  # plt.show()


def test_DOGDecay2():
  bp.math.use_backend('numpy')

  size = (30, 30)
  conn = bp.initialize.DOGDecay(sigmas=(1., 2.5), max_ws=(1.0, 0.7), periodic_boundary=True)
  weights = conn(size)

  # pos = plt.imshow(weights, cmap='cool', interpolation='none')
  # plt.colorbar(pos)
  # plt.show()

  # pos = plt.imshow(weights[0].reshape(size), cmap='cool', interpolation='none')
  # plt.colorbar(pos)
  # plt.show()


def test_DOGDecay3():
  bp.math.use_backend('numpy')

  size = (10, 10, 10)
  conn = bp.initialize.DOGDecay(sigmas=(1., 2.5), max_ws=(1.0, 0.7), periodic_boundary=True)
  weights = conn(size)

  # pos = plt.imshow(weights, cmap='cool', interpolation='none')
  # plt.colorbar(pos)
  # plt.show()
