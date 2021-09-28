# -*- coding: utf-8 -*-


r"""This module provides commonly used activation functions.

Activation functions are a critical part of the design of a neural network.
The choice of activation function in the hidden layer will control how well
the network model learns the training dataset. The choice of activation
function in the output layer will define the type of predictions the model
can make.
"""

import operator

import numpy as np

from brainpy import math
from brainpy.math import numpy as mjax

__all__ = [
  'celu',
  'elu',
  'gelu',
  'glu',
  'hard_tanh',
  'hard_sigmoid',
  'hard_silu',
  'hard_swish',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'one_hot',
  'normalize',
  'relu',
  'relu6',
  'sigmoid',
  'soft_sign',
  'softmax',
  'softplus',
  'silu',
  'swish',
  'selu',
  'tanh',
]

MAXVAL = 1e50


def _erf(x):
  """
  Port of cephes ``ndtr.c`` ``erf`` function.
  See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtr.c
  """
  T = [9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
       7.00332514112805075473E3, 5.55923013010394962768E4, ]

  U = [3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
       2.26290000613890934246E4, 4.92673942608635921086E4, ]

  # Shorcut special cases
  if x == 0: return 0
  if x >= MAXVAL: return 1
  if x <= -MAXVAL: return -1
  if abs(x) > 1: return 1 - _erfc(x)
  z = x * x
  return x * _polevl(z, T, 4) / _p1evl(z, U, 5)


def _erfc(a):
  """
  Port of cephes ``ndtr.c`` ``erfc`` function.
  See https://github.com/jeremybarnes/cephes/blob/master/cprob/ndtr.c
  """
  # approximation for abs(a) < 8 and abs(a) >= 1
  P = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
  ]

  Q = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
  ]

  # approximation for abs(a) >= 8
  R = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
  ]

  S = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
  ]

  # Shortcut special cases
  if a == 0: return 1
  if a >= MAXVAL: return 0
  if a <= -MAXVAL:  return 2

  x = a
  if a < 0: x = -a

  # computationally cheaper to calculate erf for small values, I guess.
  if x < 1: return 1 - _erf(a)

  z = -a * a

  z = math.exp(z)

  if x < 8:
    p = _polevl(x, P, 8)
    q = _p1evl(x, Q, 8)
  else:
    p = _polevl(x, R, 5)
    q = _p1evl(x, S, 6)

  y = (z * p) / q

  if a < 0: y = 2 - y

  return y


def _polevl(x, coefs, N):
  """
  Port of cephes ``polevl.c``: evaluate polynomial
  See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
  """
  ans = 0
  power = len(coefs) - 1
  for coef in coefs:
    try:
      ans += coef * x ** power
    except OverflowError:
      pass
    power -= 1
  return ans


def _p1evl(x, coefs, N):
  """
  Port of cephes ``polevl.c``: evaluate polynomial, assuming coef[N] = 1
  See https://github.com/jeremybarnes/cephes/blob/master/cprob/polevl.c
  """
  return _polevl(x, [1] + coefs, N)


def get(activation):
  global_vars = globals()

  if activation not in global_vars:
    raise ValueError(f'Unknown activation function: {activation}, \nwe only support: '
                     f'{[k for k, v in global_vars.items() if not k.startswith("_") and callable(v)]}')
  return global_vars[activation]


def celu(x, alpha=1.0):
  r"""Continuously-differentiable exponential linear unit activation.

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
  return math.where(x > 0, x, alpha * math.expm1(x / alpha))


def elu(x, alpha=1.0):
  r"""Exponential linear unit activation function.

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
  safe_x = math.where(x > 0, 0., x)
  return math.where(x > 0, x, alpha * math.expm1(safe_x))


def gelu(x, approximate=True):
  r"""Gaussian error linear unit activation function.

  If ``approximate=False``, computes the element-wise function:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
      \frac{x}{\sqrt{2}} \right) \right)

  If ``approximate=True``, uses the approximate formulation of GELU:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
      \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

  For more information, see `Gaussian Error Linear Units (GELUs)
  <https://arxiv.org/abs/1606.08415>`_, section 2.

  Args:
    x : input array
    approximate: whether to use the approximate or exact formulation.
  """
  x = x.value if isinstance(x, mjax.JaxArray) else x
  if approximate:
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
    cdf = 0.5 * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
    y = x * cdf
  else:
    # TODO
    y = math.asarray(x * (jax.lax.erf(x / np.sqrt(2)) + 1) / 2, dtype=x.dtype)
  return y


def glu(x, axis=-1):
  r"""Gated linear unit activation function.

  Args:
    x : input array
    axis: the axis along which the split should be computed (default: -1)
  """
  size = x.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  x1, x2 = math.split(x, 2, axis)
  return x1 * sigmoid(x2)


def hard_tanh(x):
  r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}

  Args:
    x : input array
  """
  return math.where(x > 1, 1, math.where(x < -1, -1, x))


def hard_sigmoid(x):
  r"""Hard Sigmoid activation function.

  Computes the element-wise function

  .. math::
    \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

  Args:
    x : input array
  """
  return relu6(x + 3.) / 6.


def hard_silu(x):
  r"""Hard SiLU activation function

  Computes the element-wise function

  .. math::
    \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

  Args:
    x : input array
  """
  return x * hard_sigmoid(x)


hard_swish = hard_silu


def leaky_relu(x, negative_slope=1e-2):
  r"""Leaky rectified linear unit activation function.

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
  return math.where(x >= 0, x, negative_slope * x)


def softplus(x):
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)

  Args:
    x : input array
  """
  return math.logaddexp(x, 0.)


def log_sigmoid(x):
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Args:
    x : input array
  """
  return -softplus(-x)


def log_softmax(x, axis=-1):
  r"""Log-Softmax function.

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
  x = x.value if isinstance(x, mjax.JaxArray) else x
  shifted = x - jax.lax.stop_gradient(x.max(axis, keepdims=True))
  return shifted - jax.numpy.log(jax.numpy.sum(jax.numpy.exp(shifted), axis, keepdims=True))


def _canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(
      "axis {} is out of bounds for array of dimension {}".format(
        axis, num_dims))
  if axis < 0: axis = axis + num_dims
  return axis


def one_hot(x, num_classes, *, dtype=None, axis=-1):
  r"""One-hot encodes the given indicies.

  Each index in the input ``x`` is encoded as a vector of zeros of length
  ``num_classes`` with the element at ``index`` set to one::

    >>> jax.nn.one_hot(mjax.array([0, 1, 2]), 3)
    DeviceArray([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=float32)

  Indicies outside the range [0, num_classes) will be encoded as zeros::

    >>> jax.nn.one_hot(mjax.array([-1, 3]), 3)
    DeviceArray([[0., 0., 0.],
                 [0., 0., 0.]], dtype=float32)

  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    axis: the axis or axes along which the function should be
      computed.
  """
  # TODO
  num_classes = jax.core.concrete_or_error(
    int, num_classes, "The error arose in 'one_hot' argument `num_classes`.")
  dtype = jax.dtypes.canonicalize_dtype(mjax.float64 if dtype is None else dtype)
  x = jax.numpy.asarray(x)
  try:
    output_pos_axis = _canonicalize_axis(axis, x.ndim + 1)
  except TypeError:
    axis_size = jax.lax.psum(1, axis)
    if num_classes != axis_size:
      raise ValueError(f"Expected num_classes to match the size of axis {axis}, "
                       f"but {num_classes} != {axis_size}") from None
    axis_idx = jax.lax.axis_index(axis)
    return jax.numpy.asarray(x == axis_idx, dtype=dtype)
  axis = operator.index(axis)
  lhs = jax.lax.expand_dims(x, (axis,))
  rhs_shape = [1] * x.ndim
  rhs_shape.insert(output_pos_axis, num_classes)
  rhs = jax.lax.broadcast_in_dim(jax.numpy.arange(num_classes, dtype=x.dtype),
                                 rhs_shape, (output_pos_axis,))
  return mjax.JaxArray(jax.numpy.asarray(lhs == rhs, dtype=dtype))


def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5):
  """Normalizes an array by subtracting mean and dividing by sqrt(var)."""
  # TODO
  x = x.value if isinstance(x, mjax.JaxArray) else x
  if mean is None:
    mean = math.mean(x, axis, keepdims=True)
  if variance is None:
    # this definition is traditionally seen as less accurate than jnp.var's
    # mean((x - mean(x))**2) but may be faster and even, given typical
    # activation distributions and low-precision arithmetic, more accurate
    # when used in neural network normalization layers
    variance = math.mean(math.square(x), axis, keepdims=True) - math.square(mean)
  y = (x - mean) * jax.lax.rsqrt(variance + epsilon)
  return y


def relu(x):
  return math.maximum(x, 0.)


def relu6(x):
  r"""Rectified Linear Unit 6 activation function.

  Computes the element-wise function

  .. math::
    \mathrm{relu6}(x) = \min(\max(x, 0), 6)

  Args:
    x : input array
  """
  return math.minimum(math.maximum(x, 0), 6.)


def sigmoid(x):
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x : input array
  """
  return 1. / (1. + math.exp(-x))


def soft_sign(x):
  r"""Soft-sign activation function.

  Computes the element-wise function

  .. math::
    \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

  Args:
    x : input array
  """
  return x / (math.abs(x) + 1)


def softmax(x, axis=-1):
  r"""Softmax function.

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
  # TODO
  x = x.value if isinstance(x, mjax.JaxArray) else x
  unnormalized = math.exp(x - jax.lax.stop_gradient(x.max(axis, keepdims=True)))
  return mjax.JaxArray(unnormalized / unnormalized.sum(axis, keepdims=True))


def silu(x):
  r"""SiLU activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

  Args:
    x : input array
  """
  return x * sigmoid(x)


swish = silu


def selu(x):
  r"""Scaled exponential linear unit activation.

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
  safe_x = math.where(x > 0, 0., x)
  return scale * math.where(x > 0, x, alpha * math.expm1(safe_x))


def tanh(x):
  return math.tanh(x)
