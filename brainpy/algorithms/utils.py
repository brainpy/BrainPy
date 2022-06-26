# -*- coding: utf-8 -*-

import brainpy.math as bm

from itertools import combinations_with_replacement

__all__ = [
  'Sigmoid',
  'Regularization',
  'L1Regularization',
  'L2Regularization',
  'L1L2Regularization',

  'polynomial_features',
  'normalize',
]


class Sigmoid(object):
  def __call__(self, x):
    return 1 / (1 + bm.exp(-x))

  def grad(self, x):
    exp = bm.exp(-x)
    return exp / (1 + exp) ** 2


class Regularization(object):
  def __init__(self, alpha):
    self.alpha = alpha

  def __call__(self, x):
    return 0

  def grad(self, x):
    return 0


class L1Regularization(Regularization):
  """L1 Regularization."""

  def __init__(self, alpha):
    super(L1Regularization, self).__init__(alpha=alpha)

  def __call__(self, w):
    return self.alpha * bm.linalg.norm(w)

  def grad(self, w):
    return self.alpha * bm.sign(w)


class L2Regularization(Regularization):
  """L2 Regularization."""

  def __init__(self, alpha):
    super(L2Regularization, self).__init__(alpha=alpha)

  def __call__(self, w):
    return self.alpha * 0.5 * w.T.dot(w)

  def grad(self, w):
    return self.alpha * w


class L1L2Regularization(Regularization):
  """L1 and L2 Regularization."""

  def __init__(self, alpha, l1_ratio=0.5):
    super(L1L2Regularization, self).__init__(alpha=alpha)
    self.l1_ratio = l1_ratio

  def __call__(self, w):
    l1_contr = self.l1_ratio * bm.linalg.norm(w)
    l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w)
    return self.alpha * (l1_contr + l2_contr)

  def grad(self, w):
    l1_contr = self.l1_ratio * bm.sign(w)
    l2_contr = (1 - self.l1_ratio) * w
    return self.alpha * (l1_contr + l2_contr)


def index_combinations(n_features, degree):
  combs = [combinations_with_replacement(range(n_features), i) for i in range(2, degree + 1)]
  flat_combs = [item for sublist in combs for item in sublist]
  return flat_combs


def polynomial_features(X, degree: int, add_bias: bool = True):
  n_samples, n_features = X.shape
  combinations = index_combinations(n_features, degree)
  if len(combinations) == 0:
    return bm.insert(X, 0, 1, axis=1) if add_bias else X
  if add_bias:
    n_features += 1
  X_new = bm.zeros((n_samples, 1 + n_features + len(combinations)))
  if add_bias:
    X_new[:, 0] = 1
    X_new[:, 1:n_features] = X
  else:
    X_new[:, :n_features] = X
  for i, index_combs in enumerate(combinations):
    X_new[:, n_features + i] = bm.prod(X[:, index_combs], axis=1)
  return X_new


def normalize(X, axis=-1, order=2):
  """ Normalize the dataset X """
  l2 = bm.atleast_1d(bm.linalg.norm(X, order, axis))
  l2 = bm.where(l2 == 0, 1, l2)
  return X / bm.expand_dims(l2, axis)
