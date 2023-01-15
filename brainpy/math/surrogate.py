# -*- coding: utf-8 -*-


# from brainpy._src.math.surrogate._utils import (
#   vjp_custom as vjp_custom
# )
from brainpy._src.math.surrogate.one_input import (
  sigmoid as sigmoid,
  piecewise_quadratic as piecewise_quadratic,
  piecewise_exp as piecewise_exp,
  soft_sign as soft_sign,
  arctan as arctan,
  nonzero_sign_log as nonzero_sign_log,
  erf as erf,
  piecewise_leaky_relu as piecewise_leaky_relu,
  squarewave_fourier_series as squarewave_fourier_series,
  s2nn as s2nn,
  q_pseudo_spike as q_pseudo_spike,
  leaky_relu as leaky_relu,
  log_tailed_relu as log_tailed_relu,
  relu_grad as relu_grad,
  gaussian_grad as gaussian_grad,
  inv_square_grad as inv_square_grad,
  multi_gaussian_grad as multi_gaussian_grad,
  slayer_grad as slayer_grad,
)
from brainpy._src.math.surrogate.two_inputs import (
  inv_square_grad2 as inv_square_grad2,
  relu_grad2 as relu_grad2,
)

