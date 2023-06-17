# -*- coding: utf-8 -*-


# from brainpy._src.math.surrogate._utils import (
#   vjp_custom as vjp_custom
# )

from brainpy._src.math.surrogate.base import (
  Surrogate
)

from brainpy._src.math.surrogate._one_input import (
  Sigmoid,
  sigmoid as sigmoid,

  PiecewiseQuadratic,
  piecewise_quadratic as piecewise_quadratic,

  PiecewiseExp,
  piecewise_exp as piecewise_exp,

  SoftSign,
  soft_sign as soft_sign,

  Arctan,
  arctan as arctan,

  NonzeroSignLog,
  nonzero_sign_log as nonzero_sign_log,

  ERF,
  erf as erf,

  PiecewiseLeakyRelu,
  piecewise_leaky_relu as piecewise_leaky_relu,

  SquarewaveFourierSeries,
  squarewave_fourier_series as squarewave_fourier_series,

  S2NN,
  s2nn as s2nn,

  QPseudoSpike,
  q_pseudo_spike as q_pseudo_spike,

  LeakyRelu,
  leaky_relu as leaky_relu,

  LogTailedRelu,
  log_tailed_relu as log_tailed_relu,

  ReluGrad,
  relu_grad as relu_grad,

  GaussianGrad,
  gaussian_grad as gaussian_grad,

  InvSquareGrad,
  inv_square_grad as inv_square_grad,

  MultiGaussianGrad,
  multi_gaussian_grad as multi_gaussian_grad,

  SlayerGrad,
  slayer_grad as slayer_grad,
)
from brainpy._src.math.surrogate._two_inputs import (
  inv_square_grad2 as inv_square_grad2,
  relu_grad2 as relu_grad2,
)

