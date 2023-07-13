from typing import Optional

from brainpy import math as bm
from brainpy._src.dnn.base import Layer
from brainpy.types import ArrayType

__all__ = [
  'Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh',
  'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU',
  'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU', 'Softsign', 'Tanhshrink',
  'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax'
]


def _inplace(inp, val, inplace):
  if inplace:
    assert isinstance(inp, bm.Array), 'input must be instance of brainpy.math.Array if inplace=True'
    inp.value = val
    return inp
  else:
    return val


class Threshold(Layer):
  r"""Thresholds each element of the input Tensor.

  Threshold is defined as:

  .. math::
      y =
      \begin{cases}
      x, &\text{ if } x > \text{threshold} \\
      \text{value}, &\text{ otherwise }
      \end{cases}

  Args:
      threshold: The value to threshold at
      value: The value to replace with
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Threshold(0.1, 20)
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['threshold', 'value', 'inplace']

  threshold: float
  value: float
  inplace: bool

  def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
    super().__init__()
    self.threshold = threshold
    self.value = value
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    v = bm.where(input > self.threshold, input, self.value)
    return _inplace(input, v, self.inplace)

  def extra_repr(self):
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'threshold={}, value={}{}'.format(
      self.threshold, self.value, inplace_str
    )


class ReLU(Layer):
  r"""Applies the rectified linear unit function element-wise:

  :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.ReLU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)


    An implementation of CReLU - https://arxiv.org/abs/1603.05201

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.ReLU()
      >>> input = bm.random.randn(2).unsqueeze(0)
      >>> output = bm.cat((m(input), m(-input)))
  """
  __constants__ = ['inplace']
  inplace: bool

  def __init__(self, inplace: bool = False):
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    v = bm.relu(input)
    return _inplace(input, v, self.inplace)

  def extra_repr(self) -> str:
    inplace_str = 'inplace=True' if self.inplace else ''
    return inplace_str


class RReLU(Layer):
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
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.RReLU(0.1, 0.3)
      >>> input = bm.random.randn(2)
      >>> output = m(input)

  .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
      https://arxiv.org/abs/1505.00853
  """
  __constants__ = ['lower', 'upper', 'inplace']

  lower: float
  upper: float
  inplace: bool

  def __init__(
      self,
      lower: float = 1. / 8,
      upper: float = 1. / 3,
      inplace: bool = False
  ):
    super().__init__()
    self.lower = lower
    self.upper = upper
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    x = bm.rrelu(input, self.lower, self.upper)
    return _inplace(input, x, self.inplace)

  def extra_repr(self):
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'lower={}, upper={}{}'.format(self.lower, self.upper, inplace_str)


class Hardtanh(Layer):
  r"""Applies the HardTanh function element-wise.

  HardTanh is defined as:

  .. math::
      \text{HardTanh}(x) = \begin{cases}
          \text{max\_val} & \text{ if } x > \text{ max\_val } \\
          \text{min\_val} & \text{ if } x < \text{ min\_val } \\
          x & \text{ otherwise } \\
      \end{cases}

  Args:
      min_val: minimum value of the linear region range. Default: -1
      max_val: maximum value of the linear region range. Default: 1
      inplace: can optionally do the operation in-place. Default: ``False``

  Keyword arguments :attr:`min_value` and :attr:`max_value`
  have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Hardtanh(-2, 2)
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['min_val', 'max_val', 'inplace']

  min_val: float
  max_val: float
  inplace: bool

  def __init__(
      self,
      min_val: float = -1.,
      max_val: float = 1.,
      inplace: bool = False,
  ) -> None:
    super().__init__()
    self.min_val = min_val
    self.max_val = max_val
    self.inplace = inplace
    assert self.max_val > self.min_val

  def update(self, input: ArrayType) -> ArrayType:
    x = bm.hard_tanh(input, self.min_val, self.max_val)
    return _inplace(input, x, self.inplace)

  def extra_repr(self) -> str:
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'min_val={}, max_val={}{}'.format(
      self.min_val, self.max_val, inplace_str
    )


class ReLU6(Hardtanh):
  r"""Applies the element-wise function:

  .. math::
      \text{ReLU6}(x) = \min(\max(0,x), 6)

  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.test_ReLU6()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def __init__(self, inplace: bool = False):
    super().__init__(0., 6., inplace)

  def extra_repr(self) -> str:
    inplace_str = 'inplace=True' if self.inplace else ''
    return inplace_str


class Sigmoid(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Sigmoid()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    return bm.sigmoid(input)


class Hardsigmoid(Layer):
  r"""Applies the Hardsigmoid function element-wise.

  Hardsigmoid is defined as:

  .. math::
      \text{Hardsigmoid}(x) = \begin{cases}
          0 & \text{if~} x \le -3, \\
          1 & \text{if~} x \ge +3, \\
          x / 6 + 1 / 2 & \text{otherwise}
      \end{cases}

  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Hardsigmoid()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['inplace']

  inplace: bool

  def __init__(self, inplace: bool = False) -> None:
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    x = bm.hard_sigmoid(input)
    return _inplace(input, x, self.inplace)


class Tanh(Layer):
  r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

  Tanh is defined as:

  .. math::
      \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Tanh()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    return bm.tanh(input)


class SiLU(Layer):
  r"""Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
  The SiLU function is also known as the swish function.

  .. math::
      \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

  .. note::
      See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
      where the SiLU (Sigmoid Linear Unit) was originally coined, and see
      `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
      in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
      a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
      where the SiLU was experimented with later.
  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.SiLU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['inplace']
  inplace: bool

  def __init__(self, inplace: bool = False):
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.silu(input), self.inplace)

  def extra_repr(self) -> str:
    inplace_str = 'inplace=True' if self.inplace else ''
    return inplace_str


class Mish(Layer):
  r"""Applies the Mish function, element-wise.
  Mish: A Self Regularized Non-Monotonic Neural Activation Function.

  .. math::
      \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

  .. note::
      See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Mish()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['inplace']
  inplace: bool

  def __init__(self, inplace: bool = False):
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.mish(input), inplace=self.inplace)

  def extra_repr(self) -> str:
    inplace_str = 'inplace=True' if self.inplace else ''
    return inplace_str


class Hardswish(Layer):
  r"""Applies the Hardswish function, element-wise, as described in the paper:
  `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

  Hardswish is defined as:

  .. math::
      \text{Hardswish}(x) = \begin{cases}
          0 & \text{if~} x \le -3, \\
          x & \text{if~} x \ge +3, \\
          x \cdot (x + 3) /6 & \text{otherwise}
      \end{cases}

  Args:
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Hardswish()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['inplace']

  inplace: bool

  def __init__(self, inplace: bool = False) -> None:
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.hard_swish(input), self.inplace)


class ELU(Layer):
  r"""Applies the Exponential Linear Unit (ELU) function, element-wise, as described
  in the paper: `Fast and Accurate Deep Network Learning by Exponential Linear
  Units (ELUs) <https://arxiv.org/abs/1511.07289>`__.

  ELU is defined as:

  .. math::
      \text{ELU}(x) = \begin{cases}
      x, & \text{ if } x > 0\\
      \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
      \end{cases}

  Args:
      alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.ELU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['alpha', 'inplace']
  alpha: float
  inplace: bool

  def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
    super().__init__()
    self.alpha = alpha
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.elu(input, self.alpha), self.inplace)

  def extra_repr(self) -> str:
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'alpha={}{}'.format(self.alpha, inplace_str)


class CELU(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

  More details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .

  Args:
      alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.CELU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)

  .. _`Continuously Differentiable Exponential Linear Units`:
      https://arxiv.org/abs/1704.07483
  """
  __constants__ = ['alpha', 'inplace']
  alpha: float
  inplace: bool

  def __init__(self, alpha: float = 1., inplace: bool = False) -> None:
    super().__init__()
    self.alpha = alpha
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.celu(input, self.alpha), self.inplace)

  def extra_repr(self) -> str:
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'alpha={}{}'.format(self.alpha, inplace_str)


class SELU(Layer):
  r"""Applied element-wise, as:

  .. math::
      \text{SELU}(x) = \text{scale} * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))

  with :math:`\alpha = 1.6732632423543772848170429916717` and
  :math:`\text{scale} = 1.0507009873554804934193349852946`.

  More details can be found in the paper `Self-Normalizing Neural Networks`_ .

  Args:
      inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.SELU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)

  .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
  """
  __constants__ = ['inplace']
  inplace: bool

  def __init__(self, inplace: bool = False) -> None:
    super().__init__()
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.selu(input), self.inplace)

  def extra_repr(self) -> str:
    inplace_str = 'inplace=True' if self.inplace else ''
    return inplace_str


class GLU(Layer):
  r"""Applies the gated linear unit function
  :math:`{GLU}(a, b)= a \otimes \sigma(b)` where :math:`a` is the first half
  of the input matrices and :math:`b` is the second half.

  Args:
      dim (int): the dimension on which to split the input. Default: -1

  Shape:
      - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.GLU()
      >>> input = bm.random.randn(4, 2)
      >>> output = m(input)
  """
  __constants__ = ['dim']
  dim: int

  def __init__(self, dim: int = -1) -> None:
    super().__init__()
    self.dim = dim

  def update(self, input: ArrayType) -> ArrayType:
    return bm.glu(input, self.dim)

  def extra_repr(self) -> str:
    return 'dim={}'.format(self.dim)


class GELU(Layer):
  r"""Applies the Gaussian Error Linear Units function:

  .. math:: \text{GELU}(x) = x * \Phi(x)

  where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

  When the approximate argument is 'tanh', Gelu is estimated with:

  .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

  Args:
      approximate (str, optional): the gelu approximation algorithm to use:
          ``'none'`` | ``'tanh'``. Default: ``'none'``

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.GELU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['approximate']
  approximate: bool

  def __init__(self, approximate: bool = False) -> None:
    super().__init__()
    self.approximate = approximate

  def update(self, input: ArrayType) -> ArrayType:
    return bm.gelu(input, approximate=self.approximate)

  def extra_repr(self) -> str:
    return 'approximate={}'.format(repr(self.approximate))


class Hardshrink(Layer):
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

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Hardshrink()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['lambd']
  lambd: float

  def __init__(self, lambd: float = 0.5) -> None:
    super().__init__()
    self.lambd = lambd

  def update(self, input: ArrayType) -> ArrayType:
    return bm.hard_shrink(input, self.lambd)

  def extra_repr(self) -> str:
    return '{}'.format(self.lambd)


class LeakyReLU(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)


  or

  .. math::
      \text{LeakyReLU}(x) =
      \begin{cases}
      x, & \text{ if } x \geq 0 \\
      \text{negative\_slope} \times x, & \text{ otherwise }
      \end{cases}

  Args:
      negative_slope: Controls the angle of the negative slope (which is used for
        negative input values). Default: 1e-2
      inplace: can optionally do the operation in-place. Default: ``False``

  Shape:
      - Input: :math:`(*)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(*)`, same shape as the input

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.LeakyReLU(0.1)
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['inplace', 'negative_slope']
  inplace: bool
  negative_slope: float

  def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
    super().__init__()
    self.negative_slope = negative_slope
    self.inplace = inplace

  def update(self, input: ArrayType) -> ArrayType:
    return _inplace(input, bm.leaky_relu(input, self.negative_slope), self.inplace)

  def extra_repr(self) -> str:
    inplace_str = ', inplace=True' if self.inplace else ''
    return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


class LogSigmoid(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right)

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.LogSigmoid()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    return bm.log_sigmoid(input)


class Softplus(Layer):
  r"""Applies the Softplus function :math:`\text{Softplus}(x) = \frac{1}{\beta} *
  \log(1 + \exp(\beta * x))` element-wise.

  SoftPlus is a smooth approximation to the ReLU function and can be used
  to constrain the output of a machine to always be positive.

  For numerical stability the implementation reverts to the linear function
  when :math:`input \times \beta > threshold`.

  Args:
      beta: the :math:`\beta` value for the Softplus formulation. Default: 1
      threshold: values above this revert to a linear function. Default: 20

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softplus()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['beta', 'threshold']
  beta: int
  threshold: int

  def __init__(self, beta: int = 1, threshold: int = 20) -> None:
    super().__init__()
    self.beta = beta
    self.threshold = threshold

  def update(self, x: ArrayType) -> ArrayType:
    return bm.softplus(x, self.beta, self.threshold)

  def extra_repr(self) -> str:
    return 'beta={}, threshold={}'.format(self.beta, self.threshold)


class Softshrink(Layer):
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

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softshrink()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['lambd']
  lambd: float

  def __init__(self, lambd: float = 0.5) -> None:
    super().__init__()
    self.lambd = lambd

  def update(self, input: ArrayType) -> ArrayType:
    return bm.soft_shrink(input, self.lambd)

  def extra_repr(self) -> str:
    return str(self.lambd)


class PReLU(Layer):
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

  Here :math:`a` is a learnable parameter. When called without arguments, `bp.dnn.PReLU()` uses a single
  parameter :math:`a` across all input channels. If called with `bp.dnn.PReLU(nChannels)`,
  a separate :math:`a` is used for each input channel.


  .. note::
      weight decay should not be used when learning :math:`a` for good performance.

  .. note::
      Channel dim is the 2nd dim of input. When input has dims < 2, then there is
      no channel dim and the number of channels = 1.

  Args:
      num_parameters (int): number of :math:`a` to learn.
          Although it takes an int as input, there is only two values are legitimate:
          1, or the number of channels at input. Default: 1
      init (float): the initial value of :math:`a`. Default: 0.25

  Shape:
      - Input: :math:`( *)` where `*` means, any number of additional
        dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Attributes:
      weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.PReLU()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """
  __constants__ = ['num_parameters']
  num_parameters: int

  def __init__(self, num_parameters: int = 1, init: float = 0.25, dtype=None) -> None:
    self.num_parameters = num_parameters
    super().__init__()
    self.weight = bm.TrainVar(bm.ones(num_parameters, dtype=dtype) * init)

  def update(self, input: ArrayType) -> ArrayType:
    return bm.prelu(input, self.weight)

  def extra_repr(self) -> str:
    return 'num_parameters={}'.format(self.num_parameters)


class Softsign(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softsign()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    return bm.soft_sign(input)


class Tanhshrink(Layer):
  r"""Applies the element-wise function:

  .. math::
      \text{Tanhshrink}(x) = x - \tanh(x)

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Tanhshrink()
      >>> input = bm.random.randn(2)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    return bm.tanh_shrink(input)


class Softmin(Layer):
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
      dim (int): A dimension along which Softmin will be computed (so every slice
          along dim will sum to 1).

  Returns:
      a Tensor of the same dimension and shape as the input, with
      values in the range [0, 1]

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softmin(dim=1)
      >>> input = bm.random.randn(2, 3)
      >>> output = m(input)
  """
  __constants__ = ['dim']
  dim: Optional[int]

  def __init__(self, dim: Optional[int] = None) -> None:
    super().__init__()
    self.dim = dim

  def update(self, input: ArrayType) -> ArrayType:
    return bm.softmin(input, self.dim)

  def extra_repr(self):
    return 'dim={dim}'.format(dim=self.dim)


class Softmax(Layer):
  r"""Applies the Softmax function to an n-dimensional input Tensor
  rescaling them so that the elements of the n-dimensional output Tensor
  lie in the range [0,1] and sum to 1.

  Softmax is defined as:

  .. math::
      \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  When the input Tensor is a sparse tensor then the unspecified
  values are treated as ``-inf``.

  Shape:
      - Input: :math:`(*)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(*)`, same shape as the input

  Returns:
      a Tensor of the same dimension and shape as the input with
      values in the range [0, 1]

  Args:
      dim (int): A dimension along which Softmax will be computed (so every slice
          along dim will sum to 1).

  .. note::
      This module doesn't work directly with NLLLoss,
      which expects the Log to be computed between the Softmax and itself.
      Use `LogSoftmax` instead (it's faster and has better numerical properties).

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softmax(dim=1)
      >>> input = bm.random.randn(2, 3)
      >>> output = m(input)

  """
  __constants__ = ['dim']
  dim: Optional[int]

  def __init__(self, dim: Optional[int] = None) -> None:
    super().__init__()
    self.dim = dim

  def update(self, input: ArrayType) -> ArrayType:
    return bm.softmax(input, self.dim)

  def extra_repr(self) -> str:
    return 'dim={dim}'.format(dim=self.dim)


class Softmax2d(Layer):
  r"""Applies SoftMax over features to each spatial location.

  When given an image of ``Channels x Height x Width``, it will
  apply `Softmax` to each location :math:`(Channels, h_i, w_j)`

  Shape:
      - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
      - Output: :math:`(N, C, H, W)` or :math:`(C, H, W)` (same shape as input)

  Returns:
      a Tensor of the same dimension and shape as the input with
      values in the range [0, 1]

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.Softmax2d()
      >>> # you softmax over the 2nd dimension
      >>> input = bm.random.randn(2, 3, 12, 13)
      >>> output = m(input)
  """

  def update(self, input: ArrayType) -> ArrayType:
    assert input.ndim == 4 or input.ndim == 3, 'Softmax2d requires a 3D or 4D tensor as input'
    return bm.softmax(input, -3)


class LogSoftmax(Layer):
  r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
  input Tensor. The LogSoftmax formulation can be simplified as:

  .. math::
      \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

  Shape:
      - Input: :math:`(*)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(*)`, same shape as the input

  Args:
      dim (int): A dimension along which LogSoftmax will be computed.

  Returns:
      a Tensor of the same dimension and shape as the input with
      values in the range [-inf, 0)

  Examples::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>> m = bp.dnn.LogSoftmax(dim=1)
      >>> input = bm.random.randn(2, 3)
      >>> output = m(input)
  """
  __constants__ = ['dim']
  dim: Optional[int]

  def __init__(self, dim: Optional[int] = None) -> None:
    super().__init__()
    self.dim = dim

  def update(self, input: ArrayType) -> ArrayType:
    return bm.log_softmax(input, self.dim)

  def extra_repr(self):
    return 'dim={dim}'.format(dim=self.dim)
