# -*- coding: utf-8 -*-

from typing import Dict, Sequence, Any, Union

import jax.numpy as jnp
import tqdm.auto
from jax.experimental.host_callback import id_tap

import brainpy.math as bm
from brainpy.errors import UnsupportedError, NoImplementationError
from brainpy.nn.base import Node, Network
from brainpy.nn.utils import serialize_kwargs
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'OnlineTrainer',
  'FORCELearning',
]


class OnlineTrainer(RNNTrainer):
  pass


class FORCELearning(OnlineTrainer):
  """Force learning."""

  def __init__(self, target, alpha=1., **kwargs):
    super(FORCELearning, self).__init__(target=target, **kwargs)

    self.train_nodes = self._get_trainable_nodes()
    self._check_interface('__force_init__')
    self._check_interface('__force_train__')
    self.train_pars = dict(alpha=alpha)

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],
          ys: Union[Tensor, Dict[str, Tensor]],
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None,
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          reset=False, ):
    pass
