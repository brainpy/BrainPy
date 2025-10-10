# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Union

import jax
import jax.numpy as jnp
import jax.scipy as sci

if jax.__version__ >= '0.5.0':
    from jax.extend.core import Primitive
else:
    from jax.core import Primitive
from jax.interpreters import batching, ad, mlir

from brainpy.math.interoperability import as_jax
from brainpy.math.ndarray import Array as Array

__all__ = [
    'Surrogate',
    'Sigmoid',
    'sigmoid',
    'PiecewiseQuadratic',
    'piecewise_quadratic',
    'PiecewiseExp',
    'piecewise_exp',
    'SoftSign',
    'soft_sign',
    'Arctan',
    'arctan',
    'NonzeroSignLog',
    'nonzero_sign_log',
    'ERF',
    'erf',
    'PiecewiseLeakyRelu',
    'piecewise_leaky_relu',
    'SquarewaveFourierSeries',
    'squarewave_fourier_series',
    'S2NN',
    's2nn',
    'QPseudoSpike',
    'q_pseudo_spike',
    'LeakyRelu',
    'leaky_relu',
    'LogTailedRelu',
    'log_tailed_relu',
    'ReluGrad',
    'relu_grad',
    'GaussianGrad',
    'gaussian_grad',
    'InvSquareGrad',
    'inv_square_grad',
    'MultiGaussianGrad',
    'multi_gaussian_grad',
    'SlayerGrad',
    'slayer_grad',
]


def _heaviside_abstract(x, dx):
    return [x]


def _heaviside_imp(x, dx):
    z = jnp.asarray(x >= 0, dtype=x.dtype)
    return [z]


def _heaviside_batching(args, axes):
    return heaviside_p.bind(*args), [axes[0]]


def _heaviside_jvp(primals, tangents):
    x, dx = primals
    tx, tdx = tangents
    primal_outs = heaviside_p.bind(x, dx)
    tangent_outs = [dx * tx, ]
    return primal_outs, tangent_outs


heaviside_p = Primitive('heaviside_p')
heaviside_p.multiple_results = True
heaviside_p.def_abstract_eval(_heaviside_abstract)
heaviside_p.def_impl(_heaviside_imp)
batching.primitive_batchers[heaviside_p] = _heaviside_batching
ad.primitive_jvps[heaviside_p] = _heaviside_jvp
mlir.register_lowering(heaviside_p, mlir.lower_fun(_heaviside_imp, multiple_results=True))


def _is_bp_array(x):
    return isinstance(x, Array)


def _as_jax(x):
    return x.value if _is_bp_array(x) else x


class Surrogate(object):
    """The base surrograte gradient function.

    To customize a surrogate gradient function, you can inherit this class and
    implement the `surrogate_fun` and `surrogate_grad` methods.

    Examples::

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>> import jax.numpy as jnp

    >>> class MySurrogate(bm.Surrogate):
    ...   def __init__(self, alpha=1.):
    ...     super().__init__()
    ...     self.alpha = alpha
    ...
    ...   def surrogate_fun(self, x):
    ...     return jnp.sin(x) * self.alpha
    ...
    ...   def surrogate_grad(self, x):
    ...     return jnp.cos(x) * self.alpha

    """

    def __call__(self, x):
        x = _as_jax(x)
        dx = self.surrogate_grad(x)
        return heaviside_p.bind(x, dx)[0]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def surrogate_fun(self, x) -> jax.Array:
        """The surrogate function."""
        raise NotImplementedError

    def surrogate_grad(self, x) -> jax.Array:
        """The gradient function of the surrogate function."""
        raise NotImplementedError


class Sigmoid(Surrogate):
    """Spike function with the sigmoid-shaped surrogate gradient.

    See Also::

    sigmoid

    """

    def __init__(self, alpha: float = 4.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        return sci.special.expit(self.alpha * x)

    def surrogate_grad(self, x):
        sgax = sci.special.expit(x * self.alpha)
        dx = (1. - sgax) * sgax * self.alpha
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def sigmoid(
    x: Union[jax.Array, Array],
    alpha: float = 4.,
):
    r"""Spike function with the sigmoid-shaped surrogate gradient.

    If `origin=False`, return the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = \mathrm{sigmoid}(\alpha x) = \frac{1}{1+e^{-\alpha x}}

    Backward function:

    .. math::

       g'(x) = \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x)

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-2, 2, 1000)
       >>> for alpha in [1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.sigmoid)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.
    """
    return Sigmoid(alpha=alpha)(x)


class PiecewiseQuadratic(Surrogate):
    """Judge spiking state with a piecewise quadratic function.

    See Also::

    piecewise_quadratic

    """

    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        x = as_jax(x)
        z = jnp.where(x < -1 / self.alpha,
                      0.,
                      jnp.where(x > 1 / self.alpha,
                                1.,
                                (-self.alpha * jnp.abs(x) / 2 + 1) * self.alpha * x + 0.5))
        return z

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.where(jnp.abs(x) > 1 / self.alpha, 0., (-(self.alpha * x) ** 2 + self.alpha))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def piecewise_quadratic(
    x: Union[jax.Array, Array],
    alpha: float = 1.,
):
    r"""Judge spiking state with a piecewise quadratic function [1]_ [2]_ [3]_ [4]_ [5]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) =
          \begin{cases}
          0, & x < -\frac{1}{\alpha} \\
          -\frac{1}{2}\alpha^2|x|x + \alpha x + \frac{1}{2}, & |x| \leq \frac{1}{\alpha}  \\
          1, & x > \frac{1}{\alpha} \\
          \end{cases}

    Backward function:

    .. math::

       g'(x) =
          \begin{cases}
          0, & |x| > \frac{1}{\alpha} \\
          -\alpha^2|x|+\alpha, & |x| \leq \frac{1}{\alpha}
          \end{cases}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.piecewise_quadratic)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Esser S K, Merolla P A, Arthur J V, et al. Convolutional networks for fast, energy-efficient neuromorphic computing[J]. Proceedings of the national academy of sciences, 2016, 113(41): 11441-11446.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Bellec G, Salaj D, Subramoney A, et al. Long short-term memory and learning-to-learn in networks of spiking neurons[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 795-805.
    .. [4] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.
    .. [5] Panda P, Aketi S A, Roy K. Toward scalable, efficient, and accurate deep spiking neural networks with backward residual connections, stochastic softmax, and hybridization[J]. Frontiers in Neuroscience, 2020, 14.
    """
    return PiecewiseQuadratic(alpha=alpha)(x)


class PiecewiseExp(Surrogate):
    """Judge spiking state with a piecewise exponential function.

    See Also::

    piecewise_exp
    """

    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = (self.alpha / 2) * jnp.exp(-self.alpha * jnp.abs(x))
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        return jnp.where(x < 0, jnp.exp(self.alpha * x) / 2, 1 - jnp.exp(-self.alpha * x) / 2)

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def piecewise_exp(
    x: Union[jax.Array, Array],
    alpha: float = 1.,

):
    r"""Judge spiking state with a piecewise exponential function [1]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = \begin{cases}
              \frac{1}{2}e^{\alpha x}, & x < 0 \\
              1 - \frac{1}{2}e^{-\alpha x}, & x \geq 0
              \end{cases}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{2}e^{-\alpha |x|}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.piecewise_exp)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.
    """
    return PiecewiseExp(alpha=alpha)(x)


class SoftSign(Surrogate):
    """Judge spiking state with a soft sign function.

    See Also::

    soft_sign
    """

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = self.alpha * 0.5 / (1 + jnp.abs(self.alpha * x)) ** 2
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        return x / (2 / self.alpha + 2 * jnp.abs(x)) + 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def soft_sign(
    x: Union[jax.Array, Array],
    alpha: float = 1.,

):
    r"""Judge spiking state with a soft sign function.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = \frac{1}{2} (\frac{\alpha x}{1 + |\alpha x|} + 1)
              = \frac{1}{2} (\frac{x}{\frac{1}{\alpha} + |x|} + 1)

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{2(1 + |\alpha x|)^{2}} = \frac{1}{2\alpha(\frac{1}{\alpha} + |x|)^{2}}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.soft_sign)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    """
    return SoftSign(alpha=alpha)(x)


class Arctan(Surrogate):
    """Judge spiking state with an arctan function.

    See Also::

    arctan
    """

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = self.alpha * 0.5 / (1 + (jnp.pi / 2 * self.alpha * x) ** 2)
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        return jnp.arctan2(jnp.pi / 2 * self.alpha * x) / jnp.pi + 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def arctan(
    x: Union[jax.Array, Array],
    alpha: float = 1.,

):
    r"""Judge spiking state with an arctan function.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = \frac{1}{\pi} \arctan(\frac{\pi}{2}\alpha x) + \frac{1}{2}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{2(1 + (\frac{\pi}{2}\alpha x)^2)}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.arctan)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    """
    return Arctan(alpha=alpha)(x)


class NonzeroSignLog(Surrogate):
    """Judge spiking state with a nonzero sign log function.

    See Also::

    nonzero_sign_log
    """

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = 1. / (1 / self.alpha + jnp.abs(x))
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        return jnp.where(x < 0, -1., 1.) * jnp.log(jnp.abs(self.alpha * x) + 1)

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def nonzero_sign_log(
    x: Union[jax.Array, Array],
    alpha: float = 1.,

):
    r"""Judge spiking state with a nonzero sign log function.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = \mathrm{NonzeroSign}(x) \log (|\alpha x| + 1)

    where

    .. math::

       \begin{split}\mathrm{NonzeroSign}(x) =
        \begin{cases}
        1, & x \geq 0 \\
        -1, & x < 0 \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{1 + |\alpha x|} = \frac{1}{\frac{1}{\alpha} + |x|}

    This surrogate function has the advantage of low computation cost during the backward.


    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.nonzero_sign_log)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    """
    return NonzeroSignLog(alpha=alpha)(x)


class ERF(Surrogate):
    """Judge spiking state with an erf function.

    See Also::

    erf
    """

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = (self.alpha / jnp.sqrt(jnp.pi)) * jnp.exp(-jnp.power(self.alpha, 2) * x * x)
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        return sci.special.erf(-self.alpha * x) * 0.5

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def erf(
    x: Union[jax.Array, Array],
    alpha: float = 1.,

):
    r"""Judge spiking state with an erf function [1]_ [2]_ [3]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}
        g(x) &= \frac{1}{2}(1-\text{erf}(-\alpha x)) \\
        &= \frac{1}{2} \text{erfc}(-\alpha x) \\
        &= \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\alpha x}e^{-t^2}dt
        \end{split}

    Backward function:

    .. math::

       g'(x) = \frac{\alpha}{\sqrt{\pi}}e^{-\alpha^2x^2}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.nonzero_sign_log)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Esser S K, Appuswamy R, Merolla P, et al. Backpropagation for energy-efficient neuromorphic computing[J]. Advances in neural information processing systems, 2015, 28: 1117-1125.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

    """
    return ERF(alpha=alpha)(x)


class PiecewiseLeakyRelu(Surrogate):
    """Judge spiking state with a piecewise leaky relu function.

    See Also::

    piecewise_leaky_relu
    """

    def __init__(self, c=0.01, w=1.):
        super().__init__()
        self.c = c
        self.w = w

    def surrogate_fun(self, x):
        x = as_jax(x)
        z = jnp.where(x < -self.w,
                      self.c * x + self.c * self.w,
                      jnp.where(x > self.w,
                                self.c * x - self.c * self.w + 1,
                                0.5 * x / self.w + 0.5))
        return z

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.where(jnp.abs(x) > self.w, self.c, 1 / self.w)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(c={self.c}, w={self.w})'


def piecewise_leaky_relu(
    x: Union[jax.Array, Array],
    c: float = 0.01,
    w: float = 1.,

):
    r"""Judge spiking state with a piecewise leaky relu function [1]_ [2]_ [3]_ [4]_ [5]_ [6]_ [7]_ [8]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}g(x) =
        \begin{cases}
        cx + cw, & x < -w \\
        \frac{1}{2w}x + \frac{1}{2}, & -w \leq x \leq w \\
        cx - cw + 1, & x > w \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \frac{1}{w}, & |x| \leq w \\
        c, & |x| > w
        \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for c in [0.01, 0.05, 0.1]:
       >>>   for w in [1., 2.]:
       >>>     grads1 = bm.vector_grad(bm.surrogate.piecewise_leaky_relu)(xs, c=c, w=w)
       >>>     plt.plot(bm.as_numpy(xs), bm.as_numpy(grads1), label=f'x={c}, w={w}')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    c: float
      When :math:`|x| > w` the gradient is `c`.
    w: float
      When :math:`|x| <= w` the gradient is `1 / w`.


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Yin S, Venkataramanaiah S K, Chen G K, et al. Algorithm and hardware design of discrete-time spiking neural networks based on back propagation with binary activations[C]//2017 IEEE Biomedical Circuits and Systems Conference (BioCAS). IEEE, 2017: 1-5.
    .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
    .. [3] Huh D, Sejnowski T J. Gradient descent for spiking neural networks[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 1440-1450.
    .. [4] Wu Y, Deng L, Li G, et al. Direct training for spiking neural networks: Faster, larger, better[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 1311-1318.
    .. [5] Gu P, Xiao R, Pan G, et al. STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks[C]//IJCAI. 2019: 1366-1372.
    .. [6] Roy D, Chakraborty I, Roy K. Scaling deep spiking neural networks with binary stochastic activations[C]//2019 IEEE International Conference on Cognitive Computing (ICCC). IEEE, 2019: 50-58.
    .. [7] Cheng X, Hao Y, Xu J, et al. LISNN: Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition[C]//IJCAI. 1519-1525.
    .. [8] Kaiser J, Mostafa H, Neftci E. Synaptic plasticity dynamics for deep continuous local learning (DECOLLE)[J]. Frontiers in Neuroscience, 2020, 14: 424.

    """
    return PiecewiseLeakyRelu(c=c, w=w)(x)


class SquarewaveFourierSeries(Surrogate):
    """Judge spiking state with a squarewave fourier series.

    See Also::

    squarewave_fourier_series
    """

    def __init__(self, n=2, t_period=8.):
        super().__init__()
        self.n = n
        self.t_period = t_period

    def surrogate_grad(self, x):
        x = as_jax(x)
        w = jnp.pi * 2. / self.t_period
        dx = jnp.cos(w * x)
        for i in range(2, self.n):
            dx += jnp.cos((2 * i - 1.) * w * x)
        dx *= 4. / self.t_period
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        w = jnp.pi * 2. / self.t_period
        ret = jnp.sin(w * x)
        for i in range(2, self.n):
            c = (2 * i - 1.)
            ret += jnp.sin(c * w * x) / c
        z = 0.5 + 2. / jnp.pi * ret
        return z

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self.n}, t_period={self.t_period})'


def squarewave_fourier_series(
    x: Union[jax.Array, Array],
    n: int = 2,
    t_period: float = 8.,

):
    r"""Judge spiking state with a squarewave fourier series.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       g(x) = 0.5 + \frac{1}{\pi}*\sum_{i=1}^n {\sin\left({(2i-1)*2\pi}*x/T\right) \over 2i-1 }

    Backward function:

    .. math::

       g'(x) = \sum_{i=1}^n\frac{4\cos\left((2 * i - 1.) * 2\pi * x / T\right)}{T}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for n in [2, 4, 8]:
       >>>   f = bm.surrogate.SquarewaveFourierSeries(n=n)
       >>>   grads1 = bm.vector_grad(f)(xs)
       >>>   plt.plot(bm.as_numpy(xs), bm.as_numpy(grads1), label=f'n={n}')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    n: int
    t_period: float


    Returns::

    out: jax.Array
      The spiking state.

    """

    return SquarewaveFourierSeries(n=n, t_period=t_period)(x)


class S2NN(Surrogate):
    """Judge spiking state with the S2NN surrogate spiking function.

    See Also::

    s2nn
    """

    def __init__(self, alpha=4., beta=1., epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def surrogate_fun(self, x):
        x = as_jax(x)
        z = jnp.where(x < 0.,
                      sci.special.expit(x * self.alpha),
                      self.beta * jnp.log(jnp.abs((x + 1.)) + self.epsilon) + 0.5)
        return z

    def surrogate_grad(self, x):
        x = as_jax(x)
        sg = sci.special.expit(self.alpha * x)
        dx = jnp.where(x < 0., self.alpha * sg * (1. - sg), self.beta / (x + 1.))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta}, epsilon={self.epsilon})'


def s2nn(
    x: Union[jax.Array, Array],
    alpha: float = 4.,
    beta: float = 1.,
    epsilon: float = 1e-8,

):
    r"""Judge spiking state with the S2NN surrogate spiking function [1]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}g(x) = \begin{cases}
          \mathrm{sigmoid} (\alpha x), x < 0 \\
          \beta \ln(|x + 1|) + 0.5, x \ge 0
      \end{cases}\end{split}

    Backward function:

    .. math::

       \begin{split}g'(x) = \begin{cases}
          \alpha * (1 - \mathrm{sigmoid} (\alpha x)) \mathrm{sigmoid} (\alpha x), x < 0 \\
          \frac{\beta}{(x + 1)}, x \ge 0
      \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> grads = bm.vector_grad(bm.surrogate.s2nn)(xs, 4., 1.)
       >>> plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=4, \beta=1$')
       >>> grads = bm.vector_grad(bm.surrogate.s2nn)(xs, 8., 2.)
       >>> plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=8, \beta=2$')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      The param that controls the gradient when ``x < 0``.
    beta: float
      The param that controls the gradient when ``x >= 0``
    epsilon: float
      Avoid nan


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Suetake, Kazuma et al. “S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks.” ArXiv abs/2201.10879 (2022): n. pag.

    """
    return S2NN(alpha=alpha, beta=beta, epsilon=epsilon)(x)


class QPseudoSpike(Surrogate):
    """Judge spiking state with the q-PseudoSpike surrogate function.

    See Also::

    q_pseudo_spike
    """

    def __init__(self, alpha=2.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.power(1 + 2 / (self.alpha + 1) * jnp.abs(x), -self.alpha)
        return dx

    def surrogate_fun(self, x):
        x = as_jax(x)
        z = jnp.where(x < 0.,
                      0.5 * jnp.power(1 - 2 / (self.alpha - 1) * jnp.abs(x), 1 - self.alpha),
                      1. - 0.5 * jnp.power(1 + 2 / (self.alpha - 1) * jnp.abs(x), 1 - self.alpha))
        return z

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def q_pseudo_spike(
    x: Union[jax.Array, Array],
    alpha: float = 2.,

):
    r"""Judge spiking state with the q-PseudoSpike surrogate function [1]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}g(x) =
        \begin{cases}
        \frac{1}{2}(1-\frac{2x}{\alpha-1})^{1-\alpha}, & x < 0 \\
        1 - \frac{1}{2}(1+\frac{2x}{\alpha-1})^{1-\alpha}, & x \geq 0.
        \end{cases}\end{split}

    Backward function:

    .. math::

       g'(x) = (1+\frac{2|x|}{\alpha-1})^{-\alpha}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.q_pseudo_spike)(xs, alpha)
       >>>   plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=$' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      The parameter to control tail fatness of gradient.


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Herranz-Celotti, Luca and Jean Rouat. “Surrogate Gradients Design.” ArXiv abs/2202.00282 (2022): n. pag.
    """
    return QPseudoSpike(alpha=alpha)(x)


class LeakyRelu(Surrogate):
    """Judge spiking state with the Leaky ReLU function.

    See Also::

    leaky_relu
    """

    def __init__(self, alpha=0.1, beta=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def surrogate_fun(self, x):
        x = as_jax(x)
        return jnp.where(x < 0., self.alpha * x, self.beta * x)

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.where(x < 0., self.alpha, self.beta)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'


def leaky_relu(
    x: Union[jax.Array, Array],
    alpha: float = 0.1,
    beta: float = 1.,

):
    r"""Judge spiking state with the Leaky ReLU function.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}g(x) =
        \begin{cases}
        \beta \cdot x, & x \geq 0 \\
        \alpha \cdot x, & x < 0 \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \beta, & x \geq 0 \\
        \alpha, & x < 0 \\
        \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> grads = bm.vector_grad(bm.surrogate.leaky_relu)(xs, 0., 1.)
       >>> plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=0., \beta=1.$')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      The parameter to control the gradient when :math:`x < 0`.
    beta: float
      The parameter to control the  gradient when :math:`x >= 0`.


    Returns::

    out: jax.Array
      The spiking state.
    """
    return LeakyRelu(alpha=alpha, beta=beta)(x)


class LogTailedRelu(Surrogate):
    """Judge spiking state with the Log-tailed ReLU function.

    See Also::

    log_tailed_relu
    """

    def __init__(self, alpha=0.):
        super().__init__()
        self.alpha = alpha

    def surrogate_fun(self, x):
        x = as_jax(x)
        z = jnp.where(x > 1,
                      jnp.log(x),
                      jnp.where(x > 0,
                                x,
                                self.alpha * x))
        return z

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.where(x > 1,
                       1 / x,
                       jnp.where(x > 0,
                                 1.,
                                 self.alpha))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def log_tailed_relu(
    x: Union[jax.Array, Array],
    alpha: float = 0.,

):
    r"""Judge spiking state with the Log-tailed ReLU function [1]_.

    If `origin=False`, computes the forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    If `origin=True`, computes the original function:

    .. math::

       \begin{split}g(x) =
        \begin{cases}
        \alpha x, & x \leq 0 \\
        x, & 0 < x \leq 0 \\
        log(x), x > 1 \\
        \end{cases}\end{split}

    Backward function:

    .. math::

       \begin{split}g'(x) =
        \begin{cases}
        \alpha, & x \leq 0 \\
        1, & 0 < x \leq 0 \\
        \frac{1}{x}, x > 1 \\
        \end{cases}\end{split}

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> grads = bm.vector_grad(bm.surrogate.leaky_relu)(xs, 0., 1.)
       >>> plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=0., \beta=1.$')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      The parameter to control the gradient.


    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Cai, Zhaowei et al. “Deep Learning with Low Precision by Half-Wave Gaussian Quantization.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 5406-5414.
    """
    return LogTailedRelu(alpha=alpha)(x)


class ReluGrad(Surrogate):
    """Judge spiking state with the ReLU gradient function.

    See Also::

    relu_grad
    """

    def __init__(self, alpha=0.3, width=1.):
        super().__init__()
        self.alpha = alpha
        self.width = width

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.maximum(self.alpha * self.width - jnp.abs(x) * self.alpha, 0)
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, width={self.width})'


def relu_grad(
    x: Union[jax.Array, Array],
    alpha: float = 0.3,
    width: float = 1.,
):
    r"""Spike function with the ReLU gradient function [1]_.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward function:

    .. math::

       g'(x) = \text{ReLU}(\alpha * (\mathrm{width}-|x|))

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> for s in [0.5, 1.]:
       >>>   for w in [1, 2.]:
       >>>     grads = bm.vector_grad(bm.surrogate.relu_grad)(xs, s, w)
       >>>     plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=$' + f'{s}, width={w}')
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      The parameter to control the gradient.
    width: float
      The parameter to control the width of the gradient.

    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Neftci, E. O., Mostafa, H. & Zenke, F. Surrogate gradient learning in spiking neural networks. IEEE Signal Process. Mag. 36, 61–63 (2019).
    """
    return ReluGrad(alpha=alpha, width=width)(x)


class GaussianGrad(Surrogate):
    """Judge spiking state with the Gaussian gradient function.

    See Also::

    gaussian_grad
    """

    def __init__(self, sigma=0.5, alpha=0.5):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def surrogate_grad(self, x):
        x = as_jax(x)
        dx = jnp.exp(-(x ** 2) / 2 * jnp.power(self.sigma, 2)) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
        return self.alpha * dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})'


def gaussian_grad(
    x: Union[jax.Array, Array],
    sigma: float = 0.5,
    alpha: float = 0.5,
):
    r"""Spike function with the Gaussian gradient function [1]_.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward function:

    .. math::

       g'(x) = \alpha * \text{gaussian}(x, 0., \sigma)

    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> for s in [0.5, 1., 2.]:
       >>>   grads = bm.vector_grad(bm.surrogate.gaussian_grad)(xs, s, 0.5)
       >>>   plt.plot(bm.as_numpy(xs), bm.as_numpy(grads), label=r'$\alpha=0.5, \sigma=$' + str(s))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    sigma: float
      The parameter to control the variance of gaussian distribution.
    alpha: float
      The parameter to control the scale of the gradient.

    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nat Mach Intell 3, 905–913 (2021).
    """
    return GaussianGrad(sigma=sigma, alpha=alpha)(x)


class MultiGaussianGrad(Surrogate):
    """Judge spiking state with the multi-Gaussian gradient function.

    See Also::

    multi_gaussian_grad
    """

    def __init__(self, h=0.15, s=6.0, sigma=0.5, scale=0.5):
        super().__init__()
        self.h = h
        self.s = s
        self.sigma = sigma
        self.scale = scale

    def surrogate_grad(self, x):
        x = as_jax(x)
        g1 = jnp.exp(-x ** 2 / (2 * jnp.power(self.sigma, 2))) / (jnp.sqrt(2 * jnp.pi) * self.sigma)
        g2 = jnp.exp(-(x - self.sigma) ** 2 / (2 * jnp.power(self.s * self.sigma, 2))
                     ) / (jnp.sqrt(2 * jnp.pi) * self.s * self.sigma)
        g3 = jnp.exp(-(x + self.sigma) ** 2 / (2 * jnp.power(self.s * self.sigma, 2))
                     ) / (jnp.sqrt(2 * jnp.pi) * self.s * self.sigma)
        dx = g1 * (1. + self.h) - g2 * self.h - g3 * self.h
        return self.scale * dx

    def __repr__(self):
        return f'{self.__class__.__name__}(h={self.h}, s={self.s}, sigma={self.sigma}, scale={self.scale})'


def multi_gaussian_grad(
    x: Union[jax.Array, Array],
    h: float = 0.15,
    s: float = 6.0,
    sigma: float = 0.5,
    scale: float = 0.5,
):
    r"""Spike function with the multi-Gaussian gradient function [1]_.

    The forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward function:

    .. math::

       \begin{array}{l}
       g'(x)=(1+h){{{\mathcal{N}}}}(x, 0, {\sigma }^{2})
       -h{{{\mathcal{N}}}}(x, \sigma,{(s\sigma )}^{2})-
       h{{{\mathcal{N}}}}(x, -\sigma ,{(s\sigma )}^{2})
       \end{array}


    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> grads = bm.vector_grad(bm.surrogate.multi_gaussian_grad)(xs)
       >>> plt.plot(bm.as_numpy(xs), bm.as_numpy(grads))
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    h: float
      The hyper-parameters of approximate function
    s: float
      The hyper-parameters of approximate function
    sigma: float
      The gaussian sigma.
    scale: float
      The gradient scale.

    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nat Mach Intell 3, 905–913 (2021).
    """
    return MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale)(x)


class InvSquareGrad(Surrogate):
    """Judge spiking state with the inverse-square surrogate gradient function.

    See Also::

    inv_square_grad
    """

    def __init__(self, alpha=100.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = 1. / (self.alpha * jnp.abs(x) + 1.0) ** 2
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def inv_square_grad(
    x: Union[jax.Array, Array],
    alpha: float = 100.
):
    r"""Spike function with the inverse-square surrogate gradient.

    Forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward function:

    .. math::

       g'(x) = \frac{1}{(\alpha * |x| + 1.) ^ 2}


    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> xs = bm.linspace(-1, 1, 1000)
       >>> for alpha in [1., 10., 100.]:
       >>>   grads = bm.vector_grad(bm.surrogate.inv_square_grad)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient

    Returns::

    out: jax.Array
      The spiking state.
    """
    return InvSquareGrad(alpha=alpha)(x)


class SlayerGrad(Surrogate):
    """Judge spiking state with the slayer surrogate gradient function.

    See Also::

    slayer_grad
    """

    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def surrogate_grad(self, x):
        dx = jnp.exp(-self.alpha * jnp.abs(x))
        return dx

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha})'


def slayer_grad(
    x: Union[jax.Array, Array],
    alpha: float = 1.
):
    r"""Spike function with the slayer surrogate gradient function.

    Forward function:

    .. math::

       g(x) = \begin{cases}
          1, & x \geq 0 \\
          0, & x < 0 \\
          \end{cases}

    Backward function:

    .. math::

       g'(x) = \exp(-\alpha |x|)


    .. plot::
       :include-source: True

       >>> import brainpy as bp
       >>> import brainpy.math as bm
       >>> import matplotlib.pyplot as plt
       >>> bp.visualize.get_figure(1, 1, 4, 6)
       >>> xs = bm.linspace(-3, 3, 1000)
       >>> for alpha in [0.5, 1., 2., 4.]:
       >>>   grads = bm.vector_grad(bm.surrogate.slayer_grad)(xs, alpha)
       >>>   plt.plot(xs, grads, label=r'$\alpha$=' + str(alpha))
       >>> plt.legend()
       >>> plt.show()

    Parameters::

    x: jax.Array, Array
      The input data.
    alpha: float
      Parameter to control smoothness of gradient

    Returns::

    out: jax.Array
      The spiking state.

    References::

    .. [1] Shrestha, S. B. & Orchard, G. Slayer: spike layer error reassignment in time. In Advances in Neural Information Processing Systems Vol. 31, 1412–1421 (NeurIPS, 2018).
    """
    return SlayerGrad(alpha=alpha)(x)
