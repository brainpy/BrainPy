# -*- coding: utf-8 -*-


import math
from typing import Union

import jax
import jax.numpy as jnp
import jax.scipy as sci
from jax.core import Primitive
from jax.interpreters import ad, batching, mlir

import brainpy.math.numpy_ops as bm
from ._utils import generalized_vjp_custom

__all__ = [
  'sigmoid',
  'piecewise_quadratic',
  'piecewise_exp',
  'soft_sign',
  'arctan',
  'nonzero_sign_log',
  'erf',
  'piecewise_leaky_relu',
  'squarewave_fourier_series',
  's2nn',
  'q_pseudo_spike',
  'leaky_relu',
  'log_tailed_relu',
  'relu_grad',
  'gaussian_grad',
  'inv_square_grad',
  'multi_gaussian_grad',
  'slayer_grad',
]


def _heaviside_abstract(x, *, alpha):
  return [x]


def _heaviside_imp(x, *, alpha):
  z = jnp.asarray(x >= 0, dtype=x.dtype)
  return [z]


def _heaviside_batching(args, axes, *, alpha):
  return heaviside_p.bind(*args, alpha=alpha), axes


def _heaviside_jvp(primals, tangents, *, alpha):
  x, = primals
  xt, = tangents
  primal_outs = heaviside_p.bind(x, alpha=alpha)
  tangent_outs = heaviside_p.bind(bm.as_jax(xt), alpha=alpha)
  return primal_outs, tangent_outs


def _heaviside_transpose(ct, x, *, alpha):
  dE_dx = bm.as_jax(ct[0]) / (alpha * jnp.abs(x.aval) + 1.0) ** 2
  return [dE_dx]


heaviside_p = Primitive('heaviside_p')
heaviside_p.multiple_results = True
heaviside_p.def_abstract_eval(_heaviside_abstract)
heaviside_p.def_impl(_heaviside_imp)
batching.primitive_batchers[heaviside_p] = _heaviside_batching
ad.primitive_jvps[heaviside_p] = _heaviside_jvp
ad.primitive_transposes[heaviside_p] = _heaviside_transpose
mlir.register_lowering(heaviside_p, mlir.lower_fun(_heaviside_imp))


@generalized_vjp_custom('x', alpha=4., origin=False)
def sigmoid(
    x: Union[jax.Array, bm.Array],
    alpha: float = None,
    origin: bool = None,
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.
  """
  # return sigmoid_p.bind(bm.as_jax(x), alpha=alpha, origin=origin)[0]
  if origin:
    z = sci.special.expit(x)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    sgax = sci.special.expit(x * alpha)
    dx = bm.as_jax(dz) * (1. - sgax) * sgax * alpha
    return dx

  return z, grad


def _sigmoid_abstract(x, *, alpha, origin):
  return [x]


def _sigmoid_imp(x, *, alpha, origin):
  if origin:
    z = sci.special.expit(x)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)
  return [z]


def _sigmoid_batching(args, axes, *, alpha, origin):
  return sigmoid_p.bind(*args, alpha=alpha, origin=origin), axes


def _sigmoid_jvp(primals, tangents, *, alpha, origin):
  x, = primals
  xt, = tangents
  primal_outs = sigmoid_p.bind(x, alpha=alpha, origin=origin)
  tangent_outs = sigmoid_p.bind(bm.as_jax(xt), alpha=alpha, origin=origin)
  return primal_outs, tangent_outs


def _sigmoid_transpose(ct, x, *, alpha, origin):
  sgax = sci.special.expit(x.aval * alpha)
  dx = bm.as_jax(ct[0]) * (1. - sgax) * sgax * alpha
  return [dx]


sigmoid_p = Primitive('sigmoid_p')
sigmoid_p.multiple_results = True
sigmoid_p.def_abstract_eval(_sigmoid_abstract)
sigmoid_p.def_impl(_sigmoid_imp)
batching.primitive_batchers[sigmoid_p] = _sigmoid_batching
mlir.register_lowering(sigmoid_p, mlir.lower_fun(_sigmoid_imp))
ad.primitive_jvps[sigmoid_p] = _sigmoid_jvp
ad.primitive_transposes[sigmoid_p] = _sigmoid_transpose


@generalized_vjp_custom('x', alpha=1., origin=False)
def piecewise_quadratic(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Esser S K, Merolla P A, Arthur J V, et al. Convolutional networks for fast, energy-efficient neuromorphic computing[J]. Proceedings of the national academy of sciences, 2016, 113(41): 11441-11446.
  .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
  .. [3] Bellec G, Salaj D, Subramoney A, et al. Long short-term memory and learning-to-learn in networks of spiking neurons[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 795-805.
  .. [4] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.
  .. [5] Panda P, Aketi S A, Roy K. Toward scalable, efficient, and accurate deep spiking neural networks with backward residual connections, stochastic softmax, and hybridization[J]. Frontiers in Neuroscience, 2020, 14.
  """
  if origin:
    z = jnp.where(x < -1 / alpha,
                  0.,
                  jnp.where(x > 1 / alpha,
                            1.,
                            (-alpha * jnp.abs(x) / 2 + 1) * alpha * x + 0.5))
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.where(jnp.abs(x) > 1 / alpha, 0., dz * (-(alpha * x) ** 2 + alpha))
    return dx

  return z, grad


@generalized_vjp_custom('x', alpha=1., origin=False)
def piecewise_exp(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Neftci E O, Mostafa H, Zenke F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63.
  """
  if origin:
    z = jnp.where(x < 0, jnp.exp(alpha * x) / 2, 1 - jnp.exp(-alpha * x) / 2)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = (alpha / 2) * jnp.exp(-alpha * jnp.abs(x))
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=1., origin=False)
def soft_sign(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  """
  if origin:
    z = x / (2 / alpha + 2 * jnp.abs(x)) + 0.5
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = alpha * 0.5 / (1 + jnp.abs(alpha * x)) ** 2
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=1., origin=False)
def arctan(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  """
  if origin:
    z = jnp.arctan2(jnp.pi / 2 * alpha * x) / jnp.pi + 0.5
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = alpha * 0.5 / (1 + (jnp.pi / 2 * alpha * x) ** 2)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=1., origin=False)
def nonzero_sign_log(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  """
  if origin:
    z = jnp.where(x < 0, -1., 1.) * jnp.log(jnp.abs(alpha * x) + 1)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = bm.as_jax(dz) / (1 / alpha + jnp.abs(x))
    return dx

  return z, grad


@generalized_vjp_custom('x', alpha=1., origin=False)
def erf(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Esser S K, Appuswamy R, Merolla P, et al. Backpropagation for energy-efficient neuromorphic computing[J]. Advances in neural information processing systems, 2015, 28: 1117-1125.
  .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
  .. [3] Yin B, Corradi F, Bohté S M. Effective and efficient computation with multiple-timescale spiking recurrent neural networks[C]//International Conference on Neuromorphic Systems 2020. 2020: 1-8.

  """
  if origin:
    z = sci.special.erf(-alpha * x) * 0.5
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = (alpha / math.sqrt(math.pi)) * jnp.exp(-math.pow(alpha, 2) * x * x)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', c=0.01, w=1., origin=False)
def piecewise_leaky_relu(
    x: Union[jax.Array, bm.Array],
    c: float,
    w: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  c: float
    When :math:`|x| > w` the gradient is `c`.
  w: float
    When :math:`|x| <= w` the gradient is `1 / w`.
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Yin S, Venkataramanaiah S K, Chen G K, et al. Algorithm and hardware design of discrete-time spiking neural networks based on back propagation with binary activations[C]//2017 IEEE Biomedical Circuits and Systems Conference (BioCAS). IEEE, 2017: 1-5.
  .. [2] Wu Y, Deng L, Li G, et al. Spatio-temporal backpropagation for training high-performance spiking neural networks[J]. Frontiers in neuroscience, 2018, 12: 331.
  .. [3] Huh D, Sejnowski T J. Gradient descent for spiking neural networks[C]//Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018: 1440-1450.
  .. [4] Wu Y, Deng L, Li G, et al. Direct training for spiking neural networks: Faster, larger, better[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 1311-1318.
  .. [5] Gu P, Xiao R, Pan G, et al. STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks[C]//IJCAI. 2019: 1366-1372.
  .. [6] Roy D, Chakraborty I, Roy K. Scaling deep spiking neural networks with binary stochastic activations[C]//2019 IEEE International Conference on Cognitive Computing (ICCC). IEEE, 2019: 50-58.
  .. [7] Cheng X, Hao Y, Xu J, et al. LISNN: Improving Spiking Neural Networks with Lateral Interactions for Robust Object Recognition[C]//IJCAI. 1519-1525.
  .. [8] Kaiser J, Mostafa H, Neftci E. Synaptic plasticity dynamics for deep continuous local learning (DECOLLE)[J]. Frontiers in Neuroscience, 2020, 14: 424.

  """
  if origin:
    z = jnp.where(x < -w,
                  c * x + c * w,
                  jnp.where(x > w,
                            c * x - c * w + 1,
                            0.5 * x / w + 0.5))
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.where(jnp.abs(x) > w, c, 1 / w)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', n=2, t_period=8., origin=False)
def squarewave_fourier_series(
    x: Union[jax.Array, bm.Array],
    n: int,
    t_period: float,
    origin: bool
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
     >>>   grads1 = bm.vector_grad(bm.surrogate.squarewave_fourier_series)(xs, n=n)
     >>>   plt.plot(bm.as_numpy(xs), bm.as_numpy(grads1), label=f'n={n}')
     >>> plt.legend()
     >>> plt.show()

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  n: int
  t_period: float
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  """
  w = math.pi * 2. / t_period
  if origin:
    ret = jnp.sin(w * x)
    for i in range(2, n):
      c = (2 * i - 1.)
      ret += jnp.sin(c * w * x) / c
    z = 0.5 + 2. / math.pi * ret
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.cos(w * x)
    for i in range(2, n):
      dx += jnp.cos((2 * i - 1.) * w * x)
    dx *= 4. / t_period
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=4., beta=1., epsilon=1e-8, origin=False)
def s2nn(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    beta: float,
    epsilon: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    The param that controls the gradient when ``x < 0``.
  beta: float
    The param that controls the gradient when ``x >= 0``
  epsilon: float
    Avoid nan
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Suetake, Kazuma et al. “S2NN: Time Step Reduction of Spiking Surrogate Gradients for Training Energy Efficient Single-Step Neural Networks.” ArXiv abs/2201.10879 (2022): n. pag.

  """
  if origin:
    z = jnp.where(x < 0.,
                  sci.special.expit(x * alpha),
                  beta * jnp.log(jnp.abs((x + 1.)) + epsilon) + 0.5)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    sg = sci.special.expit(alpha * x)
    dx = jnp.where(x < 0., alpha * sg * (1. - sg), beta / (x + 1.))
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=2., origin=False)
def q_pseudo_spike(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    The parameter to control tail fatness of gradient.
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Herranz-Celotti, Luca and Jean Rouat. “Surrogate Gradients Design.” ArXiv abs/2202.00282 (2022): n. pag.
  """
  if origin:
    z = jnp.where(x < 0.,
                  0.5 * jnp.power(1 - 2 / (alpha - 1) * jnp.abs(x), 1 - alpha),
                  1. - 0.5 * jnp.power(1 + 2 / (alpha - 1) * jnp.abs(x), 1 - alpha))
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.power(1 + 2 / (alpha + 1) * jnp.abs(x), -alpha)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=0.1, beta=1., origin=False)
def leaky_relu(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    beta: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    The parameter to control the gradient when :math:`x < 0`.
  beta: float
    The parameter to control the  gradient when :math:`x >= 0`.
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.
  """
  if origin:
    z = jnp.where(x < 0., alpha * x, beta * x)
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.where(x < 0., alpha, beta)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=0., origin=False)
def log_tailed_relu(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    origin: bool
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    The parameter to control the gradient.
  origin: bool
    Whether to compute the original function as the feedfoward output.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Cai, Zhaowei et al. “Deep Learning with Low Precision by Half-Wave Gaussian Quantization.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 5406-5414.
  """
  if origin:
    z = jnp.where(x > 1,
                  jnp.log(x),
                  jnp.where(x > 0,
                            x,
                            alpha * x))
  else:
    z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.where(x > 1,
                   1 / x,
                   jnp.where(x > 0,
                             1.,
                             alpha))
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=0.3, width=1.)
def relu_grad(
    x: Union[jax.Array, bm.Array],
    alpha: float,
    width: float,
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    The parameter to control the gradient.
  width: float
    The parameter to control the width of the gradient.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Neftci, E. O., Mostafa, H. & Zenke, F. Surrogate gradient learning in spiking neural networks. IEEE Signal Process. Mag. 36, 61–63 (2019).
  """
  z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.maximum(alpha * width - jnp.abs(x) * alpha, 0)
    return dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', sigma=0.5, alpha=0.5)
def gaussian_grad(
    x: Union[jax.Array, bm.Array],
    sigma: float,
    alpha: float,
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  sigma: float
    The parameter to control the variance of gaussian distribution.
  alpha: float
    The parameter to control the scale of the gradient.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nat Mach Intell 3, 905–913 (2021).
  """
  z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = jnp.exp(-(x ** 2) / 2 * math.pow(sigma, 2)) / (math.sqrt(2 * bm.pi) * sigma)
    return alpha * dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', h=0.15, s=6.0, sigma=0.5, scale=0.5)
def multi_gaussian_grad(
    x: Union[jax.Array, bm.Array],
    h: float,
    s: float,
    sigma: float,
    scale: float,
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  h: float
    The hyper-parameters of approximate function
  s: float
    The hyper-parameters of approximate function
  sigma: float
    The gaussian sigma.
  scale: float
    The gradient scale.

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Yin, B., Corradi, F. & Bohté, S.M. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. Nat Mach Intell 3, 905–913 (2021).
  """
  z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    g1 = jnp.exp(-x ** 2 / (2 * math.pow(sigma, 2))) / (math.sqrt(2 * math.pi) * sigma)
    g2 = jnp.exp(-(x - sigma) ** 2 / (2 * math.pow(s * sigma, 2))) / (math.sqrt(2 * math.pi) * s * sigma)
    g3 = jnp.exp(-(x + sigma) ** 2 / (2 * math.pow(s * sigma, 2))) / (math.sqrt(2 * math.pi) * s * sigma)
    dx = g1 * (1. + h) - g2 * h - g3 * h
    return scale * dx * bm.as_jax(dz)

  return z, grad


@generalized_vjp_custom('x', alpha=100.)
def inv_square_grad(
    x: Union[jax.Array, bm.Array],
    alpha: float
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient

  Returns
  -------
  out: jax.Array
    The spiking state.
  """
  z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = bm.as_jax(dz) / (alpha * jnp.abs(x) + 1.0) ** 2
    return dx

  return z, grad


@generalized_vjp_custom('x', alpha=1.)
def slayer_grad(
    x: Union[jax.Array, bm.Array],
    alpha: float
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

  Parameters
  ----------
  x: jax.Array, bm.Array
    The input data.
  alpha: float
    Parameter to control smoothness of gradient

  Returns
  -------
  out: jax.Array
    The spiking state.

  References
  ----------
  .. [1] Shrestha, S. B. & Orchard, G. Slayer: spike layer error reassignment in time. In Advances in Neural Information Processing Systems Vol. 31, 1412–1421 (NeurIPS, 2018).
  """
  z = jnp.asarray(x >= 0, dtype=x.dtype)

  def grad(dz):
    dx = bm.as_jax(dz) * jnp.exp(-alpha * jnp.abs(x))
    return dx

  return z, grad
