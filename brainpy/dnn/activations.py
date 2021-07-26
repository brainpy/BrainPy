# -*- coding: utf-8 -*-

from brainpy.dnn.imports import jax, jmath


__all__ = [
  'elu',
  'celu',
  'selu',
  'relu',
  'leaky_relu',
  'sigmoid',
  'softmax',
  'softplus',
  'tanh',
  'log_sigmoid',
  'log_softmax',
  'log_sumexp',
]


def _get(activation):
  global_vars = globals()

  if activation not in global_vars:
    raise ValueError(f'Unknown activation function: {activation}, \nwe only support: '
                     f'{[k for k, v in global_vars.items() if not k.startswith("_") and callable(v)]}')
  return global_vars[activation]


def celu(x, alpha=1.0):
  """Continuously-differentiable exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/pdf/1704.07483.pdf>`_.

  Parameters
  ----------
  x : ndarray, jnp.ndarray
    The input array.
  alpha : ndarray, float
    The default is 1.0.
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  alpha = alpha.value if isinstance(alpha, jmath.ndarray) else alpha
  return jmath.ndarray(jax.numpy.where(x > 0, x, alpha * jax.numpy.expm1(x / alpha)))


def elu(x, alpha=1.0):
  """Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Args:
    x : input array
    alpha : scalar or array of alpha values (default: 1.0)
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  alpha = alpha.value if isinstance(alpha, jmath.ndarray) else alpha
  safe_x = jax.numpy.where(x > 0, 0., x)
  return jmath.ndarray(jax.numpy.where(x > 0, x, alpha * jax.numpy.expm1(safe_x)))


def leaky_relu(x, negative_slope=1e-2):
  """Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = :code:`negative_slope`.

  Args:
    x : input array
    negative_slope  or scalar specifying the negative slope (default: 0.01)
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(jax.numpy.where(x >= 0, x, negative_slope * x))


def softplus(x):
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)

  Args:
    x : input array
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(jax.numpy.logaddexp(x, 0))


def log_sigmoid(x):
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Args:
    x : input array
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(-jax.numpy.logaddexp(-x, 0))


def selu(x):
  """Scaled exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{selu}(x) = \lambda \begin{cases}
      x, & x > 0\\
      \alpha e^x - \alpha, & x \le 0
    \end{cases}

  where :math:`\lambda = 1.0507009873554804934193349852946` and
  :math:`\alpha = 1.6732632423543772848170429916717`.

  For more information, see
  `Self-Normalizing Neural Networks
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.

  Args:
    x : input array
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  x = x.value if isinstance(x, jmath.ndarray) else x
  safe_x = jax.numpy.where(x > 0, 0., x)
  return jmath.ndarray(scale * jax.numpy.where(x > 0, x, alpha * jax.numpy.expm1(safe_x)))


def sigmoid(x):
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x : input array
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(jax.scipy.special.expit(x))


def tanh(x):
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(jax.lax.tanh(x))


def relu(x):
  x = x.value if isinstance(x, jmath.ndarray) else x
  return jmath.ndarray(jax.nn.relu(x))


def log_softmax(x, axis=-1):
  """Log-Softmax function.

  Computes the logarithm of the :code:`softmax` function, which rescales
  elements to the range :math:`[-\infty, 0)`.

  .. math ::
    \mathrm{log\_softmax}(x) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    \right)

  Args:
    x : input array
    axis: the axis or axes along which the :code:`log_softmax` should be
      computed. Either an integer or a tuple of integers.
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  shifted = x - jax.lax.stop_gradient(x.max(axis, keepdims=True))
  return jmath.ndarray(shifted - jax.numpy.log(jax.numpy.sum(jax.numpy.exp(shifted), axis, keepdims=True)))


def softmax(x, axis=-1):
  """Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
  """
  x = x.value if isinstance(x, jmath.ndarray) else x
  unnormalized = jax.numpy.exp(x - jax.lax.stop_gradient(x.max(axis, keepdims=True)))
  return jmath.ndarray(unnormalized / unnormalized.sum(axis, keepdims=True))


def log_sumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  """Compute the log of the sum of exponentials of input elements.

  Parameters
  ----------
  a : array_like
      Input array.
  axis : None or int or tuple of ints, optional
      Axis or axes over which the sum is taken. By default `axis` is None,
      and all elements are summed.

      .. versionadded:: 0.11.0
  keepdims : bool, optional
      If this is set to True, the axes which are reduced are left in the
      result as dimensions with size one. With this option, the result
      will broadcast correctly against the original array.

      .. versionadded:: 0.15.0
  b : array-like, optional
      Scaling factor for exp(`a`) must be of the same shape as `a` or
      broadcastable to `a`. These values may be negative in order to
      implement subtraction.

      .. versionadded:: 0.12.0
  return_sign : bool, optional
      If this is set to True, the result will be a pair containing sign
      information; if False, results that are negative will be returned
      as NaN. Default is False (no sign information).

      .. versionadded:: 0.16.0

  Returns
  -------
  res : ndarray
      The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
      more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
      is returned.
  sgn : ndarray
      If return_sign is True, this will be an array of floating-point
      numbers matching res and +1, 0, or -1 depending on the sign
      of the result. If False, only one result is returned.

  See Also
  --------
  numpy.logaddexp, numpy.logaddexp2

  Notes
  -----
  NumPy has a logaddexp function which is very similar to `logsumexp`, but
  only handles two arguments. `logaddexp.reduce` is similar to this
  function, but may be less stable.

  Examples
  --------
  >>> from scipy.special import logsumexp
  >>> a = np.arange(10)
  >>> np.log(np.sum(np.exp(a)))
  9.4586297444267107
  >>> logsumexp(a)
  9.4586297444267107

  With weights

  >>> a = np.arange(10)
  >>> b = np.arange(10, 0, -1)
  >>> logsumexp(a, b=b)
  9.9170178533034665
  >>> np.log(np.sum(b*np.exp(a)))
  9.9170178533034647

  Returning a sign flag

  >>> logsumexp([1,2],b=[1,-1],return_sign=True)
  (1.5413248546129181, -1.0)

  Notice that `logsumexp` does not directly support masked arrays. To use it
  on a masked array, convert the mask into zero weights:

  >>> a = np.ma.array([np.log(2), 2, np.log(3)],
  ...                  mask=[False, True, False])
  >>> b = (~a.mask).astype(int)
  >>> logsumexp(a.data, b=b), np.log(5)
  1.6094379124341005, 1.6094379124341005

  """
  a = a.value if isinstance(a, jmath.ndarray) else a
  b = b.value if isinstance(b, jmath.ndarray) else b
  res = jax.scipy.special.logsumexp(a=a, axis=axis, b=b,
                                    keepdims=keepdims,
                                    return_sign=return_sign)
  if return_sign:
    if isinstance(res[0], jax.numpy.ndarray):
      return jmath.ndarray(res[0]), jmath.ndarray(res[1])
    else:
      return res
  else:
    if isinstance(res, jax.numpy.ndarray):
      return jmath.ndarray(res)
    else:
      return res
