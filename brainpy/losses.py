# -*- coding: utf-8 -*-

from brainpy._src.losses.comparison import (
  cross_entropy_loss as cross_entropy_loss,
  cross_entropy_sparse as cross_entropy_sparse,
  cross_entropy_sigmoid as cross_entropy_sigmoid,
  nll_loss,
  l1_loss as l1_loss,
  l2_loss as l2_loss,
  huber_loss as huber_loss,
  mean_absolute_error as mean_absolute_error,
  mean_squared_error as mean_squared_error,
  mean_squared_log_error as mean_squared_log_error,
  binary_logistic_loss as binary_logistic_loss,
  multiclass_logistic_loss as multiclass_logistic_loss,
  sigmoid_binary_cross_entropy as sigmoid_binary_cross_entropy,
  softmax_cross_entropy as softmax_cross_entropy,
  log_cosh_loss as log_cosh_loss,
  ctc_loss_with_forward_probs as ctc_loss_with_forward_probs,
  ctc_loss as ctc_loss,
)

from brainpy._src.losses.comparison import (
  CrossEntropyLoss,
  NLLLoss,
  L1Loss,
  MAELoss,
  MSELoss,
)

from brainpy._src.losses.regularization import (
  l2_norm as l2_norm,
  mean_absolute as mean_absolute,
  mean_square as mean_square,
  log_cosh as log_cosh,
  smooth_labels as smooth_labels,
)

