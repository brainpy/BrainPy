# -*- coding: utf-8 -*-

"""
This module implements many commonly used loss functions.

The references used are included:

- https://zhuanlan.zhihu.com/p/61379965
- https://pytorch.org/docs/stable/nn.html#loss-functions
- https://github.com/ddbourgin/numpy-ml

"""

from brainpy import errors
from brainpy.dnn.imports import jax, jmath

__all__ = [
  'cross_entropy_loss',
  'l1_loos',
  'mean_absolute_error',
  'mean_squared_error',
  'mean_squared_log_error',

]

_reduction_error = 'Only support reduction of "mean", "sum" and "none", but we got "%s".'


def _return(outputs, reduction):
  if reduction == 'mean':
    return outputs.value.mean()
  elif reduction == 'sum':
    return outputs.value.sum()
  elif reduction == 'none':
    return outputs
  else:
    raise errors.UnsupportedError(_reduction_error % reduction)


def cross_entropy_loss(logits, targets, weight=None, reduction='mean'):
  """This criterion combines ``LogSoftmax`` and `NLLLoss`` in one single class.

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
  logits : jmath.JaxArray
    :math:`(N, C)` where `C = number of classes`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` with :math:`K \geq 1`
    in the case of `K`-dimensional loss.
  targets : jmath.JaxArray
    :math:`(N, C)` or :math:`(N)`  where each value is
    :math:`0 \leq \text{targets}[i] \leq C-1`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` or :math:`(d_1, d_2, ..., d_K, N)`
    with :math:`K \geq 1` in the case of K-dimensional loss.
  weight : jmath.JaxArray, optional
    A manual rescaling weight given to each class. If given, has to be an array of size `C`.
  reduction : str, optional
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
    - ``'none'``: no reduction will be applied,
    - ``'mean'``: the weighted mean of the output is taken,
    - ``'sum'``: the output will be summed.

  Returns
  -------
  output : scalar, jmath.JaxArray
    If :attr:`reduction` is ``'none'``, then the same size as the target:
    :math:`(N)`, or  :math:`(d_1, d_2, ..., d_K, N)` with :math:`K \geq 1`
    in the case of K-dimensional loss.
  """
  if jmath.ndim(targets) + 1 == jmath.ndim(logits):
    targets_old = targets.reshape((-1,))
    length = targets_old.shape[0]
    rows = jmath.arange(length)
    targets = jmath.zeros((length, logits.shape[-1]))
    targets[rows, targets_old] = 1.

  # loss
  logits = logits.value if isinstance(logits, jmath.JaxArray) else logits
  loss = jax.scipy.special.logsumexp(logits, axis=-1) - (logits * targets).sum(axis=-1)

  # weighted loss
  if weight:
    loss *= weight[targets]
    raise NotImplementedError

  return _return(outputs=loss, reduction=reduction)


def cross_entropy_sparse(logits, labels):
  """Computes the softmax cross-entropy loss.

  Args:
      logits: (batch, ..., #class) tensor of logits.
      labels: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

  Returns:
      (batch, ...) tensor of the cross-entropy for each entry.
  """

  if isinstance(labels, int):
    labeled_logits = logits[..., labels]
  else:
    logits = logits.value if isinstance(logits, jmath.JaxArray) else logits
    labels = labels.value if isinstance(labels, jmath.JaxArray) else labels
    # labeled_logits = jmath.take_along_axis(logits, labels[..., None], -1).squeeze(-1)
    labeled_logits = jmath.take_along_axis(logits, labels, -1).squeeze(-1)
  loss = jax.scipy.special.logsumexp(logits, axis=-1) - labeled_logits
  return jmath.JaxArray(loss)


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


def l1_loos(logits, targets, reduction='sum'):
  """Creates a criterion that measures the mean absolute error (MAE) between each element in
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
  norm = jmath.linalg.norm(diff, ord=1, axis=1, keepdims=False)
  return _return(outputs=norm, reduction=reduction)


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


def nll_loss(logits, targets, weight=None, reduction='mean'):
  pass
