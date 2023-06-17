# -*- coding: utf-8 -*-

"""
This module implements several loss functions.
"""

from typing import Tuple, Optional

import jax.numpy as jnp
from jax.lax import scan
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

import brainpy.math as bm
from brainpy.types import ArrayType
from .base import Loss, WeightedLoss
from .utils import _reduce, _multi_return, _is_leaf

__all__ = [
  'CrossEntropyLoss', 'cross_entropy_loss',

  'cross_entropy_sparse',
  'cross_entropy_sigmoid',

  'NLLLoss', 'nll_loss',
  'L1Loss', 'l1_loss',

  'l2_loss',
  'huber_loss',

  'MAELoss', 'mean_absolute_error',
  'MSELoss', 'mean_squared_error',

  'mean_squared_log_error',
  'binary_logistic_loss',
  'multiclass_logistic_loss',
  'sigmoid_binary_cross_entropy',
  'softmax_cross_entropy',
  'log_cosh_loss',
  'ctc_loss_with_forward_probs',
  'ctc_loss',
]


class CrossEntropyLoss(WeightedLoss):
  r"""This criterion computes the cross entropy loss between input logits
  and target.

  It is useful when training a classification problem with `C` classes.
  If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
  assigning weight to each of the classes.
  This is particularly useful when you have an unbalanced training set.

  The `input` is expected to contain the unnormalized logits for each class (which do `not` need
  to be positive or sum to 1, in general).
  `input` has to be a Tensor of size :math:`(C)` for unbatched input,
  :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
  `K`-dimensional case. The last being useful for higher dimension inputs, such
  as computing cross entropy loss per-pixel for 2D images.

  The `target` that this criterion expects should contain either:

  - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
    `ignore_index` is specified, this loss also accepts this class index (this index
    may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
    set to ``'none'``) loss for this case can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
        \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
    :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
    :math:`d_1, ..., d_k` for the `K`-dimensional case. If
    :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}} l_n, &
             \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
          \end{cases}

    Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
    :class:`~torch.nn.NLLLoss`.

  - Probabilities for each class; useful when labels beyond a single class per minibatch item
    are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
    :attr:`reduction` set to ``'none'``) loss for this case can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
    :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
    :math:`d_1, ..., d_k` for the `K`-dimensional case. If
    :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \frac{\sum_{n=1}^N l_n}{N}, &
             \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
          \end{cases}

  .. note::
      The performance of this criterion is generally better when `target` contains class
      indices, as this allows for optimized computation. Consider providing `target` as
      class probabilities only when a single class label per minibatch item is too restrictive.

  Args:
      weight (Tensor, optional): a manual rescaling weight given to each class.
          If given, has to be a Tensor of size `C`
      size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
          the losses are averaged over each loss element in the batch. Note that for
          some losses, there are multiple elements per sample. If the field :attr:`size_average`
          is set to ``False``, the losses are instead summed for each minibatch. Ignored
          when :attr:`reduce` is ``False``. Default: ``True``
      ignore_index (int, optional): Specifies a target value that is ignored
          and does not contribute to the input gradient. When :attr:`size_average` is
          ``True``, the loss is averaged over non-ignored targets. Note that
          :attr:`ignore_index` is only applicable when the target contains class indices.
      reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
          losses are averaged or summed over observations for each minibatch depending
          on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
          batch element instead and ignores :attr:`size_average`. Default: ``True``
      reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
          be applied, ``'mean'``: the weighted mean of the output is taken,
          ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in
          the meantime, specifying either of those two args will override
          :attr:`reduction`. Default: ``'mean'``
      label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
          of smoothing when computing the loss, where 0.0 means no smoothing. The targets
          become a mixture of the original ground truth and a uniform distribution as described in
          `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

  Shape:
      - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
        in the case of `K`-dimensional loss.
      - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
        :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
        If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
      - Output: If reduction is 'none', shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
        in the case of K-dimensional loss, depending on the shape of the input. Otherwise, scalar.


      where:

      .. math::
          \begin{aligned}
              C ={} & \text{number of classes} \\
              N ={} & \text{batch size} \\
          \end{aligned}

  Examples::

      >>> # Example of target with class indices
      >>> loss = nn.CrossEntropyLoss()
      >>> input = torch.randn(3, 5, requires_grad=True)
      >>> target = torch.empty(3, dtype=torch.long).random_(5)
      >>> output = loss(input, target)
      >>> output.backward()
      >>>
      >>> # Example of target with class probabilities
      >>> input = torch.randn(3, 5, requires_grad=True)
      >>> target = torch.randn(3, 5).softmax(dim=1)
      >>> output = loss(input, target)
      >>> output.backward()
  """
  __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
  ignore_index: int
  label_smoothing: float

  def __init__(self, weight: Optional[ArrayType] = None, ignore_index: int = -100,
               reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
    super().__init__(weight, reduction)
    self.ignore_index = ignore_index
    self.label_smoothing = label_smoothing

  def update(self, input: ArrayType, target: ArrayType) -> ArrayType:
    return cross_entropy_loss(input, target, weight=self.weight, reduction=self.reduction)


def cross_entropy_loss(predicts, targets, weight=None, reduction='mean'):
  r"""This criterion combines ``LogSoftmax`` and `NLLLoss`` in one single class.

  It is useful when training a classification problem with `C` classes.
  If provided, the optional argument :attr:`weight` should be a 1D `Array`
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
  predicts : ArrayType
    :math:`(N, C)` where `C = number of classes`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` with :math:`K \geq 1`
    in the case of `K`-dimensional loss.
  targets : ArrayType
    :math:`(N, C)` or :math:`(N)`  where each value is
    :math:`0 \leq \text{targets}[i] \leq C-1`, or
    :math:`(d_1, d_2, ..., d_K, N, C)` or :math:`(d_1, d_2, ..., d_K, N)`
    with :math:`K \geq 1` in the case of K-dimensional loss.
  weight : ArrayType, optional
    A manual rescaling weight given to each class. If given, has to be an array of size `C`.
  reduction : str, optional
    Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
    - ``'none'``: no reduction will be applied,
    - ``'mean'``: the weighted mean of the output is taken,
    - ``'sum'``: the output will be summed.

  Returns
  -------
  output : scalar, ArrayType
    If :attr:`reduction` is ``'none'``, then the same size as the target:
    :math:`(N)`, or  :math:`(d_1, d_2, ..., d_K, N)` with :math:`K \geq 1`
    in the case of K-dimensional loss.
  """

  def _cel(_pred, _tar):
    if bm.ndim(_tar) + 1 == bm.ndim(_pred):
      _tar = bm.one_hot(_tar, _pred.shape[-1])
    loss = logsumexp(bm.as_jax(_pred), axis=-1) - (_pred * _tar).sum(axis=-1)
    if weight is not None:
      loss *= weight
    return _reduce(outputs=loss, reduction=reduction)

  r = tree_map(_cel, predicts, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def cross_entropy_sparse(predicts, targets):
  r"""Computes the softmax cross-entropy loss.

  Args:
      predicts: (batch, ..., #class) tensor of logits.
      targets: (batch, ...) integer tensor of label indexes in {0, ...,#nclass-1} or just a single integer.

  Returns:
      (batch, ...) tensor of the cross-entropy for each entry.
  """

  def crs(_prd, _tar):
    if isinstance(_tar, int):
      logits = _prd[..., _tar]
    else:
      logits = jnp.take_along_axis(_prd, _tar, -1).squeeze(-1)
    return logsumexp(bm.as_jax(_prd), axis=-1) - logits

  r = tree_map(crs, predicts, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def cross_entropy_sigmoid(predicts, targets):
  """Computes the sigmoid cross-entropy loss.

  Args:
      predicts: (batch, ..., #class) tensor of logits.
      targets: (batch, ..., #class) tensor of label probabilities (e.g. labels.sum(axis=-1) must be 1)

  Returns:
      (batch, ...) tensor of the cross-entropies for each entry.
  """
  r = tree_map(
    lambda pred, tar: bm.as_jax(
      bm.maximum(pred, 0) - pred * tar + bm.log(1 + bm.exp(-bm.abs(pred)))
    ),
    predicts,
    targets,
    is_leaf=_is_leaf
  )
  return _multi_return(r)


class NLLLoss(Loss):
  r"""The negative log likelihood loss.

  The negative log likelihood loss. It is useful to train a classification
  problem with `C` classes.

  If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
  weight to each of the classes. This is particularly useful when you have an
  unbalanced training set.

  The `input` given through a forward call is expected to contain
  log-probabilities of each class. `input` has to be a Tensor of size either
  :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
  with :math:`K \geq 1` for the `K`-dimensional case. The latter is useful for
  higher dimension inputs, such as computing NLL loss per-pixel for 2D images.

  Obtaining log-probabilities in a neural network is easily achieved by
  adding a  `LogSoftmax`  layer in the last layer of your network.
  You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
  layer.

  The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
  where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
  this class index (this index may not necessarily be in the class range).

  The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

  .. math::
      \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
      l_n = - w_{y_n} x_{n,y_n}, \quad
      w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

  where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
  :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
  (default ``'mean'``), then

  .. math::
      \ell(x, y) = \begin{cases}
          \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
          \text{if reduction} = \text{`mean';}\\
          \sum_{n=1}^N l_n,  &
          \text{if reduction} = \text{`sum'.}
      \end{cases}

  Args:
      reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
          be applied, ``'mean'``: the weighted mean of the output is taken,
          ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in
          the meantime, specifying either of those two args will override
          :attr:`reduction`. Default: ``'mean'``

  Shape:
      - Input: :math:`(N, C)` or :math:`(C)`, where `C = number of classes`, or
        :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
        in the case of `K`-dimensional loss.
      - Target: :math:`(N)` or :math:`()`, where each value is
        :math:`0 \leq \text{targets}[i] \leq C-1`, or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
        K-dimensional loss.
      - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)` or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
        Otherwise, scalar.

  """

  def __init__(self, reduction: str = 'mean'):
    super().__init__(reduction=reduction)

  def update(self, input, target):
    return nll_loss(input, target, reduction=self.reduction)


def nll_loss(input, target, reduction: str = 'mean'):
  r"""The negative log likelihood loss.

  The negative log likelihood loss. It is useful to train a classification
  problem with `C` classes.

  If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
  weight to each of the classes. This is particularly useful when you have an
  unbalanced training set.

  The `input` given through a forward call is expected to contain
  log-probabilities of each class. `input` has to be a Tensor of size either
  :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
  with :math:`K \geq 1` for the `K`-dimensional case. The latter is useful for
  higher dimension inputs, such as computing NLL loss per-pixel for 2D images.

  Obtaining log-probabilities in a neural network is easily achieved by
  adding a  `LogSoftmax`  layer in the last layer of your network.
  You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
  layer.

  The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
  where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
  this class index (this index may not necessarily be in the class range).

  The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

  .. math::
      \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
      l_n = - w_{y_n} x_{n,y_n}, \quad
      w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore\_index}\},

  where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
  :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
  (default ``'mean'``), then

  .. math::
      \ell(x, y) = \begin{cases}
          \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
          \text{if reduction} = \text{`mean';}\\
          \sum_{n=1}^N l_n,  &
          \text{if reduction} = \text{`sum'.}
      \end{cases}

  Args:
      reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
          be applied, ``'mean'``: the weighted mean of the output is taken,
          ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in
          the meantime, specifying either of those two args will override
          :attr:`reduction`. Default: ``'mean'``

  Shape:
      - Input: :math:`(N, C)` or :math:`(C)`, where `C = number of classes`, or
        :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
        in the case of `K`-dimensional loss.
      - Target: :math:`(N)` or :math:`()`, where each value is
        :math:`0 \leq \text{targets}[i] \leq C-1`, or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
        K-dimensional loss.
      - Output: If :attr:`reduction` is ``'none'``, shape :math:`(N)` or
        :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of K-dimensional loss.
        Otherwise, scalar.

  """
  assert target.ndim + 1 == input.ndim
  input = bm.as_jax(input)
  target = bm.as_jax(target)
  loss = input[jnp.arange(len(target)), target]
  if reduction == 'mean':
    return loss.mean()
  elif reduction == 'sum':
    return loss.sum()
  elif reduction == 'none':
    return loss
  elif reduction is None:
    return loss
  else:
    raise ValueError


class L1Loss(Loss):
  r"""Creates a criterion that measures the mean absolute error (MAE) between each element in
  the input :math:`x` and target :math:`y`.

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

  Args:
      reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in the meantime,
          specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Target: :math:`(*)`, same shape as the input.
      - Output: scalar. If :attr:`reduction` is ``'none'``, then
        :math:`(*)`, same shape as the input.

  Examples::

      >>> loss = nn.L1Loss()
      >>> input = bm.random.randn(3, 5)
      >>> target = bm.random.randn(3, 5)
      >>> output = loss(input, target)
      >>> output.backward()
  """
  __constants__ = ['reduction']

  def __init__(self, reduction: str = 'mean') -> None:
    super().__init__(reduction=reduction)

  def update(self, input: ArrayType, target: ArrayType) -> ArrayType:
    return l1_loss(input, target, reduction=self.reduction)


def l1_loss(logits, targets, reduction='sum'):
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
  logits : ArrayType
    :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
  targets : ArrayType
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

  def loss(pred, tar):
    diff = (pred - tar).reshape((pred.shape[0], -1))
    norm = jnp.linalg.norm(bm.as_jax(diff), ord=1, axis=1, keepdims=False)
    return _reduce(outputs=norm, reduction=reduction)

  r = tree_map(loss, logits, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def l2_loss(predicts, targets):
  r"""Computes the L2 loss.

  The 0.5 term is standard in "Pattern Recognition and Machine Learning"
  by Bishop [1]_, but not "The Elements of Statistical Learning" by Tibshirani.

  Parameters
  ----------

  predicts: ArrayType
    A vector of arbitrary shape.
  targets: ArrayType
    A vector of shape compatible with predictions.

  Returns
  -------
  loss : float
    A scalar value containing the l2 loss.

  References
  ----------
  .. [1] Bishop, Christopher M. 2006. Pattern Recognition and Machine Learning.
  """
  r = tree_map(lambda pred, tar: 0.5 * (pred - tar) ** 2,
               predicts,
               targets)
  return _multi_return(r)



class MAELoss(Loss):
  def __init__(self, axis=None, reduction: str = 'mean'):
    super().__init__(reduction=reduction)
    self.axis = axis

  def update(self, input, target):
    return mean_absolute_error(input, target, self.axis, reduction=self.reduction)




def mean_absolute_error(x, y, axis=None, reduction: str = 'mean'):
  r"""Computes the mean absolute error between x and y.

  Args:
      x: a tensor of shape (d0, .. dN-1).
      y: a tensor of shape (d0, .. dN-1).
      axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean absolute error.
  """
  r = tree_map(lambda a, b: _reduce(bm.abs(a - b), reduction=reduction, axis=axis),
               x,
               y,
               is_leaf=_is_leaf)
  return _multi_return(r)


class MSELoss(Loss):
  r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
  each element in the input :math:`x` and target :math:`y`.

  The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

  .. math::
      \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
      l_n = \left( x_n - y_n \right)^2,

  where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
  (default ``'mean'``), then:

  .. math::
      \ell(x, y) =
      \begin{cases}
          \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
          \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
      \end{cases}

  :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
  of :math:`n` elements each.

  The mean operation still operates over all the elements, and divides by :math:`n`.

  The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

  Args:
      reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
          and :attr:`reduce` are in the process of being deprecated, and in the meantime,
          specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Target: :math:`(*)`, same shape as the input.

  Examples::

      >>> loss = nn.MSELoss()
      >>> input = torch.randn(3, 5, requires_grad=True)
      >>> target = torch.randn(3, 5)
      >>> output = loss(input, target)
      >>> output.backward()
  """
  __constants__ = ['reduction']

  def __init__(self, reduction: str = 'mean') -> None:
    super().__init__(reduction=reduction)

  def update(self, input: ArrayType, target: ArrayType) -> ArrayType:
    return mean_squared_error(input, target, reduction=self.reduction)


def mean_squared_error(predicts, targets, axis=None, reduction: str = 'mean'):
  r"""Computes the mean squared error between x and y.

  Args:
      predicts: a tensor of shape (d0, .. dN-1).
      targets: a tensor of shape (d0, .. dN-1).
      axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  r = tree_map(lambda a, b: _reduce((a - b) ** 2, reduction, axis=axis),
               predicts,
               targets,
               is_leaf=_is_leaf)
  return _multi_return(r)


def mean_squared_log_error(predicts, targets, axis=None, reduction: str = 'mean'):
  r"""Computes the mean squared logarithmic error between y_true and y_pred.

  Args:
      targets: a tensor of shape (d0, .. dN-1).
      predicts: a tensor of shape (d0, .. dN-1).
      keep_axis: a sequence of the dimensions to keep, use `None` to return a scalar value.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean squared error.
  """
  r = tree_map(lambda a, b: _reduce((jnp.log1p(a) - jnp.log1p(b)) ** 2, reduction, axis=axis),
               predicts,
               targets,
               is_leaf=_is_leaf)
  return _multi_return(r)


def huber_loss(predicts, targets, delta: float = 1.0):
  r"""Huber loss.

  Huber loss is similar to L2 loss close to zero, L1 loss away from zero.
  If gradient descent is applied to the `huber loss`, it is equivalent to
  clipping gradients of an `l2_loss` to `[-delta, delta]` in the backward pass.

  Parameters
  ----------
  predicts: ArrayType
    predictions
  targets: ArrayType
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
    r = bm.where(diff > delta,
                 delta * (diff - .5 * delta),
                 0.5 * diff ** 2)
    return bm.as_jax(r)

  r = tree_map(_loss, targets, predicts, is_leaf=_is_leaf)
  return _multi_return(r)


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
  r = tree_map(lambda a, b: bm.softplus(a) - b * a,
               predicts,
               targets,
               is_leaf=_is_leaf)
  return _multi_return(r)


def multiclass_logistic_loss(label: int, logits: jnp.ndarray) -> float:
  """Multiclass logistic loss.

  Args:
    label: ground-truth integer label, between 0 and n_classes - 1.
    logits: scores produced by the model, shape = (n_classes, ).

  Returns:
    loss value
  """

  def loss(pred, tar):
    pred = bm.as_jax(pred)
    one_hot = bm.one_hot(tar, pred.shape[0])
    return logsumexp(pred) - jnp.dot(pred, one_hot)

  r = tree_map(loss, logits, label, is_leaf=_is_leaf)
  return _multi_return(r)


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

  def loss(pred, tar):
    log_p = bm.log_sigmoid(pred)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    log_not_p = bm.log_sigmoid(-pred)
    return -tar * log_p - (1. - tar) * log_not_p

  r = tree_map(loss, logits, labels, is_leaf=_is_leaf)
  return _multi_return(r)


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
  r = tree_map(lambda pred, tar: -jnp.sum(tar * bm.log_softmax(pred, axis=-1), axis=-1),
               logits,
               labels,
               is_leaf=_is_leaf)
  return _multi_return(r)


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

  def loss(pred, tar):
    errors = bm.as_jax(pred - tar)
    return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)

  r = tree_map(loss, predicts, targets, is_leaf=_is_leaf)
  return _multi_return(r)


def ctc_loss_with_forward_probs(
    logits: ArrayType,
    logit_paddings: ArrayType,
    labels: ArrayType,
    label_paddings: ArrayType,
    blank_id: int = 0,
    log_epsilon: float = -1e5
) -> Tuple[ArrayType, ArrayType, ArrayType]:
  r"""Computes CTC loss and CTC forward-probabilities.

  The CTC loss is a loss function based on log-likelihoods of the model that
  introduces a special blank symbol :math:`\phi` to represent variable-length
  output sequences.

  Forward probabilities returned by this function, as auxiliary results, are
  grouped into two part: blank alpha-probability and non-blank alpha
  probability. Those are defined as follows:

  .. math::
    \alpha_{\mathrm{BLANK}}(t, n) =
    \sum_{\pi_{1:t-1}} p(\pi_t = \phi | \pi_{1:t-1}, y_{1:n-1}, \cdots), \\
    \alpha_{\mathrm{LABEL}}(t, n) =
    \sum_{\pi_{1:t-1}} p(\pi_t = y_n | \pi_{1:t-1}, y_{1:n-1}, \cdots).

  Here, :math:`\pi` denotes the alignment sequence in the reference
  [Graves et al, 2006] that is blank-inserted representations of ``labels``.
  The return values are the logarithms of the above probabilities.

  References:
    [Graves et al, 2006](https://dl.acm.org/doi/abs/10.1145/1143844.1143891)

  Args:
    logits: (B, T, K)-array containing logits of each class where B denotes
      the batch size, T denotes the max time frames in ``logits``, and K
      denotes the number of classes including a class for blanks.
    logit_paddings: (B, T)-array. Padding indicators for ``logits``. Each
      element must be either 1.0 or 0.0, and ``logitpaddings[b, t] == 1.0``
      denotes that ``logits[b, t, :]`` are padded values.
    labels: (B, N)-array containing reference integer labels where N denotes
      the max time frames in the label sequence.
    label_paddings: (B, N)-array. Padding indicators for ``labels``. Each
      element must be either 1.0 or 0.0, and ``labelpaddings[b, n] == 1.0``
      denotes that ``labels[b, n]`` is a padded label. In the current
      implementation, ``labels`` must be right-padded, i.e. each row
      ``labelpaddings[b, :]`` must be repetition of zeroes, followed by
      repetition of ones.
    blank_id: Id for blank token. ``logits[b, :, blank_id]`` are used as
      probabilities of blank symbols.
    log_epsilon: Numerically-stable approximation of log(+0).

  Returns:
    A tuple ``(loss_value, logalpha_blank, logalpha_nonblank)``. Here,
    ``loss_value`` is a (B,)-array containing the loss values for each sequence
    in the batch, ``logalpha_blank`` and ``logalpha_nonblank`` are
    (T, B, N+1)-arrays where the (t, b, n)-th element denotes
    \log \alpha_B(t, n) and \log \alpha_L(t, n), respectively, for ``b``-th
    sequence in the batch.
  """
  assert logits.ndim == 3
  assert labels.ndim == 2
  batchsize, unused_maxinputlen, num_classes = logits.shape
  batchsize_of_labels, maxlabellen = labels.shape
  assert (batchsize == batchsize_of_labels)
  assert (labels.shape == label_paddings.shape)
  assert (logits.shape[:2] == logit_paddings.shape)

  logits = logits.value if isinstance(logits, bm.Array) else logits
  logit_paddings = logit_paddings.value if isinstance(logit_paddings, bm.Array) else logit_paddings
  labels = labels.value if isinstance(labels, bm.Array) else labels
  label_paddings = label_paddings.value if isinstance(label_paddings, bm.Array) else label_paddings

  logprobs = bm.log_softmax(logits).value
  labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)

  # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
  repeat = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)
  repeat = jnp.pad(repeat, ((0, 0), (0, 1)))

  logprobs_phi = logprobs[:, :, blank_id:blank_id + 1]  # [B, T, 1]
  logprobs_phi = jnp.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

  one_hot = bm.one_hot(labels, num_classes=num_classes)  # [B, N, K]
  logprobs_emit = jnp.einsum('btk,bnk->btn', logprobs, one_hot)
  logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

  logalpha_phi_init = jnp.ones(
    (batchsize, maxlabellen + 1)) * log_epsilon  # [B, N]
  logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
  logalpha_emit_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon

  def update_phi_score(phi, added_score):
    # Update `phi[:, 1:]`` with adding `added_score` in log space.
    return jnp.concatenate(
      [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1)

  def loop_body(prev, x):
    prev_phi, prev_emit = prev
    # emit-to-phi epsilon transition, except if the next label is repetition
    prev_phi_orig = prev_phi
    prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

    logprob_emit, logprob_phi, pad = x

    # phi-to-emit transition
    next_emit = jnp.logaddexp(prev_phi[:, :-1] + logprob_emit,
                              prev_emit + logprob_emit)
    # self-loop transition
    next_phi = prev_phi + logprob_phi
    # emit-to-phi blank transition only when the next label is repetition
    next_phi = update_phi_score(
      next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat))

    pad = pad.reshape((batchsize, 1))
    next_emit = pad * prev_emit + (1.0 - pad) * next_emit
    next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

    return (next_phi, next_emit), (next_phi, next_emit)

  xs = (logprobs_emit, logprobs_phi, logit_paddings.transpose((1, 0)))
  _, (logalpha_phi, logalpha_emit) = scan(loop_body, (logalpha_phi_init, logalpha_emit_init), xs)

  # last row needs to be updated with the last epsilon transition
  logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
  logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

  # extract per_seq_loss
  one_hot = bm.one_hot(labellens, num_classes=maxlabellen + 1).value  # [B, N+1]
  per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)

  return per_seq_loss, logalpha_phi, logalpha_emit


def ctc_loss(logits: ArrayType,
             logit_paddings: ArrayType,
             labels: ArrayType,
             label_paddings: ArrayType,
             blank_id: int = 0,
             log_epsilon: float = -1e5) -> ArrayType:
  """Computes CTC loss.

  See docstring for ``ctc_loss_with_forward_probs`` for details.

  Args:
    logits: (B, T, K)-array containing logits of each class where B denotes
      the batch size, T denotes the max time frames in ``logits``, and K
      denotes the number of classes including a class for blanks.
    logit_paddings: (B, T)-array. Padding indicators for ``logits``. Each
      element must be either 1.0 or 0.0, and ``logitpaddings[b, t] == 1.0``
      denotes that ``logits[b, t, :]`` are padded values.
    labels: (B, N)-array containing reference integer labels where N denotes
      the max time frames in the label sequence.
    label_paddings: (B, N)-array. Padding indicators for ``labels``. Each
      element must be either 1.0 or 0.0, and ``labelpaddings[b, n] == 1.0``
      denotes that ``labels[b, n]`` is a padded label. In the current
      implementation, ``labels`` must be right-padded, i.e. each row
      ``labelpaddings[b, :]`` must be repetition of zeroes, followed by
      repetition of ones.
    blank_id: Id for blank token. ``logits[b, :, blank_id]`` are used as
      probabilities of blank symbols.
    log_epsilon: Numerically-stable approximation of log(+0).

  Returns:
    (B,)-array containing loss values for each sequence in the batch.
  """
  per_seq_loss, _, _ = ctc_loss_with_forward_probs(
    logits, logit_paddings, labels, label_paddings,
    blank_id=blank_id, log_epsilon=log_epsilon)
  return per_seq_loss
