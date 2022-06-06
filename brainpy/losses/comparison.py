# -*- coding: utf-8 -*-

"""
This module implements several loss functions.
"""

# - https://github.com/deepmind/optax/blob/master/optax/_src/loss.py
# - https://github.com/google/jaxopt/blob/main/jaxopt/_src/loss.py

import jax.numpy as jnp
import jax.scipy
from jax.tree_util import tree_map

import brainpy.math as bm
from .utils import _return, _multi_return, _is_leaf

__all__ = [
  'cross_entropy_loss',
  'cross_entropy_sparse',
  'cross_entropy_sigmoid',
  'l1_loos',
  'l2_loss',
  'huber_loss',
  'mean_absolute_error',
  'mean_squared_error',
  'mean_squared_log_error',
  'binary_logistic_loss',
  'multiclass_logistic_loss',
  'smooth_labels',
  'sigmoid_binary_cross_entropy',
  'softmax_cross_entropy',
  'log_cosh_loss',
]


def cross_entropy_loss(predicts, targets, weight=None, reduction='mean'):
  r"""This criterion combines ``LogSoftmax`` and `NLLLoss`` in one single class.

  It is useful when training a classification problem with `C` classes.
  If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
  assigning weight to each of the classes. This is particularly useful when
  you have an unbalanced training set.

  The ``input`` is expected to contain raw, unnormalized scores for each class.
  ``input`` has to be an array of size either :math:`(minibatch, C)` or
  :math:`(d_1, d_2, ..., d_K, minibatch, C)` with :math:`K \geq 1` for the
  `K`-dimensional case (described later).

  This criterion expects a class index in the range :math:`[0, C-1]` as the
  `target` for each value of a 1D tensor of size `minibatch`.

  The loss can be described as:

  .. math::
      \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                     = -x[class] + \log\left(\sum_j \exp(x[j])\right)

  or in the case of the :attr:`weight` argument being specified:

  .. math::
      \text{loss}(x, class) = weight[class] \left(-x[class] + \log\left(\sum_j \exp(x[j])\right)\right)

  Can also be used for higher dimension inputs, such as 2D images, by providing
  an input of size :math:`(d_1, d_2, ..., d_K, minibatch, C)` with :math:`K \geq 1`,
  where :math:`K` is the number of dimensions, and a target of appropriate shape.

  Parameters
  ----------
  predicts : jmath.JaxArray
    :math:`(N, C)` where `C = number of classes`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` with :math:`K \geq 1`
    in the case of `K`-dimensional loss.
  targets : jmath.JaxArray
    :math:`(N, C)` or :math:`(N)`  where each value is
    :math:`0 \leq \text{targets}[i] \leq C-1`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` or :math:`(d_1, d_2, ..., d_K, N)`
    with :math:`K \geq 1` in the case of K-dimensional loss.
  weight : JaxArray, optional
    A manual rescaling weight given to each class. If given, has to be an array of size `C`.
  reduction : str, optional
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
    - ``'none'``: no reduction will be applied,
    - ``'mean'``: the weighted mean of the output is taken,
    - ``'sum'``: the output will be summed.

  Returns
  -------
  output : scalar, mjax.JaxArray
    If :attr:`reduction` is ``'none'``, then the same size as the target:
    :math:`(N)`, or  :math:`(d_1, d_2, ..., d_K, N)` with :math:`K \geq 1`
    in the case of K-dimensional loss.
  """
  targets = bm.as_device_array(targets)
  predicts = bm.as_device_array(predicts)

  # loss
  if bm.ndim(targets) + 1 == bm.ndim(predicts):
    # targets_old = targets.reshape((-1,))
    # length = targets_old.shape[0]
    # rows = jn.arange(length)
    # targets = ops.zeros((length, logits.shape[-1]))
    # targets[rows, targets_old] = 1.
    # targets = targets.reshape(logits.shape).value
    targets = bm.activations.one_hot(targets, predicts.shape[-1])
  loss = jax.scipy.special.logsumexp(predicts, axis=-1) - (predicts * targets).sum(axis=-1)

  # weighted loss
  if weight:
    loss *= weight[targets]
    raise NotImplementedError

  return _return(outputs=loss, reduction=reduction)


def cross_entropy_sparse(predicts, targets):
  r"""Computes the softmax cross-entropy loss.

  Args:
      predicts: (batch, ..., #class) tensor of logits.
      targets: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

  Returns:
      (batch, ...) tensor of the cross-entropy for each entry.
  """
  predicts = bm.as_device_array(predicts)
  targets = bm.as_device_array(targets)
  if isinstance(targets, int):
    labeled_logits = predicts[..., targets]
  else:
    labeled_logits = jnp.take_along_axis(predicts, targets, -1).squeeze(-1)
  loss = jax.scipy.special.logsumexp(predicts, axis=-1) - labeled_logits
  return loss


def cross_entropy_sigmoid(predicts, targets):
  """Computes the sigmoid cross-entropy loss.

  Args:
      predicts: (batch, ..., #class) tensor of logits.
      targets: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

  Returns:
      (batch, ...) tensor of the cross-entropies for each entry.
  """
  predicts = bm.as_device_array(predicts)
  targets = bm.as_device_array(targets)
  return jnp.maximum(predicts, 0) - predicts * targets + jnp.log(1 + jnp.exp(-jnp.abs(predicts)))


def l1_loos(logits, targets, reduction='sum'):
  r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
  the logits :math:`x` and targets :math:`y`. It is useful in regression problems.

  The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

  .. math::
      \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
      l_n = \left| x_n - y_n \right|,

  where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
  (default ``'mean'``), then:

  .. math::
      \ell(x, y) =
      \begin{cases}
          \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
          \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
      \end{cases}

  :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
  of :math:`n` elements each.

  The sum operation still operates over all the elements, and divides by :math:`n`.

  The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

  Supports real-valued and complex-valued inputs.

  Parameters
  ----------
  logits : jmath.JaxArray
    :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
  targets : jmath.JaxArray
    :math:`(N, *)`, same shape as the input.
  reduction : str
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
    Default: ``'mean'``.
    - ``'none'``: no reduction will be applied,
    - ``'mean'``: the sum of the output will be divided by the number of elements in the output,
    - ``'sum'``: the output will be summed. Note: :attr:`size_average`

  Returns
  -------
  output : scalar.
    If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same shape as the input.
  """
  diff = (logits - targets).reshape((logits.shape[0], -1))
  norm = jnp.linalg.norm(bm.as_device_array(diff), ord=1, axis=1, keepdims=False)
  return _return(outputs=norm, reduction=reduction)


def l2_loss(predicts, targets):
  r"""Computes the L2 loss.

  The 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop [1]_, but not "The Elements of Statistical Learning" by Tibshirani.

  Parameters
  ----------

  predicts: JaxArray
    A vector of arbitrary shape.
  targets: JaxArray
    A vector of shape compatible with predictions.

  Returns
  -------
  loss : float
    A scalar value containing the l2 loss.

  References
  ----------
  .. [1] Bishop, Christopher M. 2006. Pattern Recognition and Machine Learning.
  """
  return bm.as_device_array(0.5 * (predicts - targets) ** 2)


def mean_absolute_error(x, y, axis=None):
  r"""Computes the mean absolute error between x and y.

  Args:
      x: a tensor of shape (d0, .. dN-1).
      y: a tensor of shape (d0, .. dN-1).
      axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean absolute error.
  """
  r = tree_map(lambda a, b: bm.mean(bm.abs(a - b), axis=axis), x, y, is_leaf=_is_leaf)
  return _multi_return(r)


def mean_squared_error(predicts, targets, axis=None):
  r"""Computes the mean squared error between x and y.

  Args:
      predicts: a tensor of shape (d0, .. dN-1).
      targets: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  r = tree_map(lambda a, b: bm.mean((a - b) ** 2, axis=axis), predicts, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def mean_squared_log_error(predicts, targets, axis=None):
  r"""Computes the mean squared logarithmic error between y_true and y_pred.

  Args:
      targets: a tensor of shape (d0, .. dN-1).
      predicts: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  r = tree_map(lambda a, b: bm.mean((bm.log1p(a) - bm.log1p(b)) ** 2, axis=axis),
               predicts, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def huber_loss(predicts, targets, delta: float = 1.0):
  r"""Huber loss.

  Huber loss is similar to L2 loss close to zero, L1 loss away from zero.
  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  Parameters
  ----------
  predicts: JaxArray
    predictions
  targets: JaxArray
    ground truth
  delta: float
    radius of quadratic behavior

  Returns
  -------
  loss : float
    The loss value.

  References
  ----------
  .. [1] https://en.wikipedia.org/wiki/Huber_loss
  """
  def _loss(y_predict, y_target):
    # 0.5 * err^2                  if |err| <= d
    # 0.5 * d^2 + d * (|err| - d)  if |err| > d
    diff = bm.abs(y_predict - y_target)
    return bm.where(diff > delta,
                    delta * (diff - .5 * delta),
                    0.5 * diff ** 2)

  return tree_map(_loss, targets, predicts, is_leaf=_is_leaf)


def binary_logistic_loss(predicts: float, targets: int, ) -> float:
  """Binary logistic loss.

  Args:
    targets: ground-truth integer label (0 or 1).
    predicts: score produced by the model (float).
  Returns:
    loss value
  """
  # Softplus is the Fenchel conjugate of the Fermi-Dirac negentropy on [0, 1].
  # softplus = proba * logit - xlogx(proba) - xlogx(1 - proba),
  # where xlogx(proba) = proba * log(proba).
  return bm.as_device_array(bm.activations.softplus(predicts) - targets * predicts)


def multiclass_logistic_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass logistic loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).
  Returns:
    loss value
  """
  logits = bm.as_device_array(logits)
  n_classes = logits.shape[0]
  one_hot = bm.one_hot(label, n_classes)
  # Logsumexp is the Fenchel conjugate of the Shannon negentropy on the simplex.
  # logsumexp = jnp.dot(proba, logits) - jnp.dot(proba, jnp.log(proba))
  return jax.scipy.special.logsumexp(logits) - bm.dot(logits, one_hot)


def smooth_labels(labels, alpha: float) -> jnp.ndarray:
  r"""Apply label smoothing.
  Label smoothing is often used in combination with a cross-entropy loss.
  Smoothed labels favour small logit gaps, and it has been shown that this can
  provide better model calibration by preventing overconfident predictions.
  References:
    [Müller et al, 2019](https://arxiv.org/pdf/1906.02629.pdf)
  Args:
    labels: one hot labels to be smoothed.
    alpha: the smoothing factor, the greedy category with be assigned
      probability `(1-alpha) + alpha / num_categories`
  Returns:
    a smoothed version of the one hot input labels.
  """
  num_categories = labels.shape[-1]
  return (1.0 - alpha) * labels + alpha / num_categories


def sigmoid_binary_cross_entropy(logits, labels):
  """Computes sigmoid cross entropy given logits and multiple class labels.
  Measures the probability error in discrete classification tasks in which
  each class is an independent binary prediction and different classes are
  not mutually exclusive. This may be used for multilabel image classification
  for instance a model may predict that an image contains both a cat and a dog.
  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
  Args:
    logits: unnormalized log probabilities.
    labels: the probability for that class.
  Returns:
    a sigmoid cross entropy loss.
  """
  log_p = bm.log_sigmoid(logits)
  # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
  log_not_p = bm.log_sigmoid(-logits)
  return -labels * log_p - (1. - labels) * log_not_p


def softmax_cross_entropy(logits, labels):
  """Computes the softmax cross entropy between sets of logits and labels.
  Measures the probability error in discrete classification tasks in which
  the classes are mutually exclusive (each entry is in exactly one class).
  For example, each CIFAR-10 image is labeled with one and only one label:
  an image can be a dog or a truck, but not both.
  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
  Args:
    logits: unnormalized log probabilities.
    labels: a valid probability distribution (non-negative, sum to 1), e.g a
      one hot encoding of which class is the correct one for each input.
  Returns:
    the cross entropy loss.
  """
  logits = bm.as_device_array(logits)
  labels = bm.as_device_array(labels)
  return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def log_cosh_loss(predicts, targets):
  r"""Calculates the log-cosh loss for a set of predictions.

  log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
  for large x.  It is a twice differentiable alternative to the Huber loss.
  References:
    [Chen et al, 2019](https://openreview.net/pdf?id=rkglvsC9Ym)
  Args:
    predicts: a vector of arbitrary shape.
    targets: a vector of shape compatible with predictions; if not provided
      then it is assumed to be zero.
  Returns:
    the log-cosh loss.
  """
  errors = bm.as_device_array(predicts - targets)
  return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)
