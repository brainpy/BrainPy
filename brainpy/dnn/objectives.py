# -*- coding: utf-8 -*-


from brainpy.dnn.imports import jax, jmath

__all__ = [
  'cross_entropy',
  'cross_entropy_sparse',
  'cross_entropy_sigmoid',
  'l2',
  'mean_absolute_error',
  'mean_squared_error',
  'mean_squared_log_error',

]


def cross_entropy(logits, labels):
  """Computes the softmax cross-entropy loss on n-dimensional data.

  Args:
      logits: (batch, ..., #class) tensor of logits.
      labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

  Returns:
      (batch, ...) tensor of the cross-entropies for each entry.
  """
  logits = logits.value if isinstance(logits, jmath.JaxArray) else logits
  if jmath.ndim(labels) == 1:
    labels2 = jmath.zeros_like(logits)
    rows = jmath.arange(len(logits))
    labels2[rows, labels] = 1.
    labels = labels2
  labels = labels.value if isinstance(labels, jmath.JaxArray) else labels
  return jax.scipy.special.logsumexp(logits, axis=-1) - (logits * labels).sum(-1)


def cross_entropy_sparse(logits, labels):
  """Computes the softmax cross-entropy loss.

  Args:
      logits: (batch, ..., #class) tensor of logits.
      labels: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

  Returns:
      (batch, ...) tensor of the cross-entropies for each entry.
  """
  if isinstance(labels, int):
    labeled_logits = logits[..., labels]
  else:
    labeled_logits = jmath.take_along_axis(logits, labels[..., None], -1).squeeze(-1)

  return jax.scipy.special.logsumexp(logits, axis=-1) - labeled_logits


def l2(x):
  """Computes the L2 loss.

  Args:
      x: n-dimensional tensor of floats.

  Returns:
      scalar tensor containing the l2 loss of x.
  """
  return 0.5 * (x ** 2).sum()


def mean_absolute_error(x, y, keep_axis=(0,)):
  """Computes the mean absolute error between x and y.

  Args:
      x: a tensor of shape (d0, .. dN-1).
      y: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean absolute error.
  """
  loss = jax.numpy.abs(x - y)
  axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
  return loss.mean(axis)


def mean_squared_error(x, y, keep_axis=(0,)):
  """Computes the mean squared error between x and y.

  Args:
      x: a tensor of shape (d0, .. dN-1).
      y: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  loss = (x - y) ** 2
  axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
  return loss.mean(axis)


def mean_squared_log_error(y_true, y_pred, keep_axis=(0,)):
  """Computes the mean squared logarithmic error between y_true and y_pred.

  Args:
      y_true: a tensor of shape (d0, .. dN-1).
      y_pred: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  loss = (jax.numpy.log1p(y_true) - jax.numpy.log1p(y_pred)) ** 2
  axis = [i for i in range(loss.ndim) if i not in (keep_axis or ())]
  return loss.mean(axis)


def cross_entropy_sigmoid(logits, labels):
  """Computes the sigmoid cross-entropy loss.

  Args:
      logits: (batch, ..., #class) tensor of logits.
      labels: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

  Returns:
      (batch, ...) tensor of the cross-entropies for each entry.
  """
  return jax.numpy.maximum(logits, 0) - logits * labels + \
         jax.numpy.log(1 + jax.numpy.exp(-jax.numpy.abs(logits)))
