# -*- coding: utf-8 -*-


r"""This module provides commonly used activation functions.

Activation functions are a critical part of the design of a neural network.
The choice of activation function in the hidden layer will control how well
the network model learns the training dataset. The choice of activation
function in the output layer will define the type of predictions the model
can make.

"""

import operator

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np

from .ndarray import Array
from .random import uniform

__all__ = [
  'celu',
  'elu',
  'gelu',
  'glu',
  'prelu',
  'hard_tanh',
  'hard_sigmoid',
  'tanh_shrink',
  'hard_silu',
  'hard_swish',
  'hard_shrink',
  'leaky_relu',
  'log_sigmoid',
  'log_softmax',
  'one_hot',
  'normalize',
  'relu',
  'relu6',
  'rrelu',
  'sigmoid',
  'soft_sign',
  'softmax',
  'softmin',
  'softplus',
  'soft_shrink',
  'silu',
  'mish',
  'swish',
  'selu',
  'identity',
]


def get(activation):
  global_vars = globals()

  if activation is None:
    return None

  if isinstance(activation, str):
    if activation not in global_vars:
      raise ValueError(f'Unknown activation function: {activation}, \nwe only support: '
                       f'{[k for k, v in global_vars.items() if not k.startswith("_") and callable(v)]}')
  elif callable(activation):
    return activation
  else:
    raise ValueError(f'Unknown activation function {activation}. ')
  return global_vars[activation]


def identity(x):
  return x.value if isinstance(x, Array) else x


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
  x : ArrayType
    The input array.
  alpha : ndarray, float
    The default is 1.0.
  """
  x = x.value if isinstance(x, Array) else x
  alpha = alpha.value if isinstance(alpha, Array) else alpha
  return jnp.where(x > 0, x, alpha * jnp.expm1(x / alpha))


def elu(x, alpha=1.0):
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Parameters
  ----------
  x: JaxArray, jnp.ndarray
    The input array.
  alpha : scalar or Array
    default: 1.0.
  """
  x = x.value if isinstance(x, Array) else x
  alpha = alpha.value if isinstance(alpha, Array) else alpha
  safe_x = jnp.where(x > 0, 0., x)
  return jnp.where(x > 0, x, alpha * jnp.expm1(safe_x))


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

  Parameters
  ----------
  x: ArrayType
    The input array.
  approximate: bool
    whether to use the approximate or exact formulation.
  """
  x = x.value if isinstance(x, Array) else x
  if approximate:
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x ** 3))))
    y = x * cdf
  else:
    y = jnp.array(x * (jax.lax.erf(x / np.sqrt(2)) + 1) / 2, dtype=x.dtype)
  return y


def glu(x, axis=-1):
  r"""Gated linear unit activation function.

  Parameters
  ----------
  x: ArrayType
    The input array.
  axis: int
    The axis along which the split should be computed (default: -1)
  """
  size = x.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  x = x.value if isinstance(x, Array) else x
  x1, x2 = jnp.split(x, 2, axis)
  return x1 * sigmoid(x2)


def hard_tanh(x, min_val=- 1.0, max_val=1.0):
  r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}

  Parameters
  ----------
  x: ArrayType
    The input array.
  min_val: float
    minimum value of the linear region range. Default: -1
  max_val: float
    maximum value of the linear region range. Default: 1
  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x > max_val, max_val, jnp.where(x < min_val, min_val, x))


def hard_sigmoid(x):
  r"""Hard Sigmoid activation function.

  Computes the element-wise function

  .. math::
    \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  return relu6(x + 3.) / 6.


def tanh_shrink(x):
  r"""Applies the element-wise function:

  .. math::
      \text{Tanhshrink}(x) = x - \tanh(x)
  """
  x = x.value if isinstance(x, Array) else x
  return x - jnp.tanh(x)


def hard_silu(x):
  r"""Hard SiLU activation function

  Computes the element-wise function

  .. math::
    \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  return x * hard_sigmoid(x)


hard_swish = hard_silu


def hard_shrink(x, lambd=0.5):
  r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

  Hardshrink is defined as:

  .. math::
      \text{HardShrink}(x) =
      \begin{cases}
      x, & \text{ if } x > \lambda \\
      x, & \text{ if } x < -\lambda \\
      0, & \text{ otherwise }
      \end{cases}

  Args:
      lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x > lambd, x, jnp.where(x < -lambd, x, 0.))


def leaky_relu(x, negative_slope=1e-2):
  r"""Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = :code:`negative_slope`.

  Parameters
  ----------
  x: ArrayType
    The input array.
  negative_slope : float
    The scalar specifying the negative slope (default: 0.01)
  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x >= 0, x, negative_slope * x)


def softplus(x, beta=1, threshold=20):
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

  SoftPlus is a smooth approximation to the ReLU function and can be used
  to constrain the output of a machine to always be positive.

  For numerical stability the implementation reverts to the linear function
  when :math:`input \times \beta > threshold`.

  Parameters
  ----------
  x: The input array.
  beta: the :math:`\beta` value for the Softplus formulation. Default: 1
  threshold: values above this revert to a linear function. Default: 20

  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x > threshold, x * beta, 1 / beta * jnp.logaddexp(beta * x, 0))


def log_sigmoid(x):
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  return -softplus(-x)


def soft_shrink(x, lambd=0.5):
  r"""Applies the soft shrinkage function elementwise:

  .. math::
      \text{SoftShrinkage}(x) =
      \begin{cases}
      x - \lambda, & \text{ if } x > \lambda \\
      x + \lambda, & \text{ if } x < -\lambda \\
      0, & \text{ otherwise }
      \end{cases}

  Args:
      lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.
  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x > lambd, x - lambd, jnp.where(x < -lambd, x + lambd, 0.))


def log_softmax(x, axis=-1):
  r"""Log-Softmax function.

  Computes the logarithm of the :code:`softmax` function, which rescales
  elements to the range :math:`[-\infty, 0)`.

  .. math ::
    \mathrm{log\_softmax}(x) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    \right)

  Parameters
  ----------
  x: ArrayType
    The input array.
  axis: int
    The axis or axes along which the :code:`log_softmax` should be
    computed. Either an integer or a tuple of integers.
  """
  x = x.value if isinstance(x, Array) else x
  shifted = x - jax.lax.stop_gradient(x.max(axis, keepdims=True))
  return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis, keepdims=True))
  # exp = jnp.exp(x)
  # return jnp.log(exp / exp.sum(axis=axis, keepdims=True))


def _canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(
      "axis {} is out of bounds for array of dimension {}".format(
        axis, num_dims))
  if axis < 0:
    axis = axis + num_dims
  return axis


def one_hot(x, num_classes, *, dtype=None, axis=-1):
  r"""One-hot encodes the given indicies.

  Each index in the input ``x`` is encoded as a vector of zeros of length
  ``num_classes`` with the element at ``index`` set to one::

    >>> import jax.numpy as jnp
    >>> one_hot(jnp.array([0, 1, 2]), 3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

  Indicies outside the range [0, num_classes) will be encoded as zeros::

    >>> import jax.numpy as jnp
    >>> one_hot(jnp.array([-1, 3]), 3)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    axis: the axis or axes along which the function should be
      computed.
  """
  num_classes = jax.core.concrete_or_error(
    int, num_classes, "The error arose in jax.nn.one_hot argument `num_classes`.")
  dtype = jax.dtypes.canonicalize_dtype(jnp.float64 if dtype is None else dtype)
  x = jnp.asarray(x.value if isinstance(x, Array) else x)
  try:
    output_pos_axis = _canonicalize_axis(axis, x.ndim + 1)
  except TypeError:
    axis_size = jax.lax.psum(1, axis)
    if num_classes != axis_size:
      raise ValueError(f"Expected num_classes to match the size of axis {axis}, "
                       f"but {num_classes} != {axis_size}") from None
    axis_idx = jax.lax.axis_index(axis)
    return jnp.asarray(x == axis_idx, dtype=dtype)
  axis = operator.index(axis)
  lhs = jax.lax.expand_dims(x, (axis,))
  rhs_shape = [1] * x.ndim
  rhs_shape.insert(output_pos_axis, num_classes)
  rhs = jax.lax.broadcast_in_dim(jnp.arange(num_classes, dtype=x.dtype),
                                 rhs_shape,
                                 (output_pos_axis,))
  return jnp.asarray(lhs == rhs, dtype=dtype)


def normalize(x, axis=-1, mean=None, variance=None, epsilon=1e-5):
  """Normalizes an array by subtracting mean and dividing by sqrt(var)."""
  x = x.value if isinstance(x, Array) else x
  if mean is None:
    mean = jnp.mean(x, axis, keepdims=True)
  if variance is None:
    # this definition is traditionally seen as less accurate than jnp.var's
    # mean((x - mean(x))**2) but may be faster and even, given typical
    # activation distributions and low-precision arithmetic, more accurate
    # when used in neural network normalization layers
    variance = jnp.mean(jnp.square(x), axis, keepdims=True) - jnp.square(mean)
  y = (x - mean) * jax.lax.rsqrt(variance + epsilon)
  return y


def relu(x):
  x = x.value if isinstance(x, Array) else x
  return _relu(x)


@jax.custom_jvp
def _relu(x: Array) -> Array:
  r"""Rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{relu}(x) = \max(x, 0)

  except under differentiation, we take:

  .. math::
    \nabla \mathrm{relu}(0) = 0

  For more information see
  `Numerical influence of ReLU’(0) on backpropagation
  <https://openreview.net/forum?id=urrcVI-_jRm>`_.

  Args:
    x : input array
  """
  return jnp.maximum(x, 0)


# For behavior at 0, see https://openreview.net/forum?id=urrcVI-_jRm
_relu.defjvps(lambda g, ans, x: jax.lax.select(x > 0, g, jax.lax.full_like(g, 0)))


def relu6(x):
  r"""Rectified Linear Unit 6 activation function.

  Computes the element-wise function

  .. math::
    \mathrm{relu6}(x) = \min(\max(x, 0), 6)

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  x = x.value if isinstance(x, Array) else x
  return jnp.minimum(jnp.maximum(x, 0), 6.)


def rrelu(x, lower=0.125, upper=0.3333333333333333, ):
  r"""Applies the randomized leaky rectified liner unit function, element-wise,
  as described in the paper:

  `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

  The function is defined as:

  .. math::
      \text{RReLU}(x) =
      \begin{cases}
          x & \text{if } x \geq 0 \\
          ax & \text{ otherwise }
      \end{cases}

  where :math:`a` is randomly sampled from uniform distribution
  :math:`\mathcal{U}(\text{lower}, \text{upper})`.

   See: https://arxiv.org/pdf/1505.00853.pdf

  Args:
      lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
      upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
      https://arxiv.org/abs/1505.00853
  """
  x = x.value if isinstance(x, Array) else x
  a = uniform(lower, upper, size=x.shape)
  return jnp.where(x >= 0., x, a * x)


def prelu(x, a=0.25):
  r"""Applies the element-wise function:

  .. math::
      \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

  or

  .. math::
      \text{PReLU}(x) =
      \begin{cases}
      x, & \text{ if } x \geq 0 \\
      ax, & \text{ otherwise }
      \end{cases}

  Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
  parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
  a separate :math:`a` is used for each input channel.
  """
  x = x.value if isinstance(x, Array) else x
  return jnp.where(x >= 0., x, a * x)


def sigmoid(x):
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  x = x.value if isinstance(x, Array) else x
  return jax.scipy.special.expit(x)


def soft_sign(x):
  r"""Soft-sign activation function.

  Computes the element-wise function

  .. math::
    \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  x = x.value if isinstance(x, Array) else x
  return x / (jnp.abs(x) + 1)


def softmax(x, axis=-1):
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Parameters
  ----------
  x: ArrayType
    The input array.
  axis: int
    The axis or axes along which the softmax should be computed. The
    softmax output summed across these dimensions should sum to :math:`1`.
    Either an integer or a tuple of integers.
  """
  x = x.value if isinstance(x, Array) else x
  unnormalized = jnp.exp(x - jax.lax.stop_gradient(x.max(axis, keepdims=True)))
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def softmin(x, axis=-1):
  r"""Applies the Softmin function to an n-dimensional input Tensor
  rescaling them so that the elements of the n-dimensional output Tensor
  lie in the range `[0, 1]` and sum to 1.

  Softmin is defined as:

  .. math::
      \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

  Shape:
      - Input: :math:`(*)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(*)`, same shape as the input

  Args:
      axis (int): A dimension along which Softmin will be computed (so every slice
          along dim will sum to 1).
  """
  x = x.value if isinstance(x, Array) else x
  unnormalized = jnp.exp(-x)
  return unnormalized / unnormalized.sum(axis, keepdims=True)


soft_max = softmax


def silu(x):
  r"""SiLU activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  x = x.value if isinstance(x, Array) else x
  return x * sigmoid(x)


swish = silu


def mish(x):
  r"""Applies the Mish function, element-wise.

  Mish: A Self Regularized Non-Monotonic Neural Activation Function.

  .. math::
      \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

  .. note::
      See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.
  """
  x = x.value if isinstance(x, Array) else x
  return x * jnp.tanh(softplus(x))


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

  Parameters
  ----------
  x: ArrayType
    The input array.
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  x = x.value if isinstance(x, Array) else x
  safe_x = jnp.where(x > 0, 0., x)
  return scale * jnp.where(x > 0, x, alpha * jnp.expm1(safe_x))
