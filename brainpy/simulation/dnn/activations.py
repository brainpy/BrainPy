# -*- coding: utf-8 -*-

from brainpy.simulation.dnn.imports import lax, nn, jnp, scipy, ndarray

__all__ = [
  'celu',
  'elu',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'logsumexp',
  'selu',
  'sigmoid',
  'softmax',
  'softplus',
  'tanh',
  'relu',
]


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
  x = x.value if isinstance(x, ndarray) else x
  alpha = alpha.value if isinstance(alpha, ndarray) else alpha
  return ndarray(jnp.where(x > 0, x, alpha * jnp.expm1(x / alpha)))


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
  x = x.value if isinstance(x, ndarray) else x
  alpha = alpha.value if isinstance(alpha, ndarray) else alpha
  safe_x = jnp.where(x > 0, 0., x)
  return ndarray(jnp.where(x > 0, x, alpha * jnp.expm1(safe_x)))


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
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(jnp.where(x >= 0, x, negative_slope * x))


def softplus(x):
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)

  Args:
    x : input array
  """
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(jnp.logaddexp(x, 0))


def log_sigmoid(x):
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Args:
    x : input array
  """
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(-jnp.logaddexp(-x, 0))


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
  x = x.value if isinstance(x, ndarray) else x
  safe_x = jnp.where(x > 0, 0., x)
  return ndarray(scale * jnp.where(x > 0, x, alpha * jnp.expm1(safe_x)))


def sigmoid(x):
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x : input array
  """
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(scipy.special.expit(x))


log_softmax = nn.log_softmax
logsumexp = scipy.special.logsumexp
softmax = nn.softmax
tanh = lax.tanh
relu = nn.relu
