# -*- coding: utf-8 -*-

from typing import Union, Dict

from brainpy.types import Tensor
from .rnn_trainer import RNNTrainer

__all__ = [
  'BPTrainer'
]


class BPTrainer(RNNTrainer):
  """Back-propagation trainer."""

  def __init__(self, target, lr=0.01, **kwargs):
    super(BPTrainer, self).__init__(target=target, **kwargs)

    self.train_pars = dict(lr=lr)

  def fit(self,
          xs: Union[Tensor, Dict[str, Tensor]],
          ys: Union[Tensor, Dict[str, Tensor]],
          forced_states: Dict[str, Tensor] = None,
          forced_feedbacks: Dict[str, Tensor] = None,
          initial_states: Dict[str, Tensor] = None,
          initial_feedbacks: Dict[str, Tensor] = None,
          reset=False):

    self._init_target(xs)
    if reset:  # reset the model states
      self.target.reset_state()
    self._set_initial_states(initial_states)
    self._set_initial_feedbacks(initial_feedbacks)
    iter_forced_states, fixed_forced_states = self._check_forced_states(forced_states)
    iter_forced_feedbacks, fixed_forced_feedbacks = self._check_forced_feedbacks(forced_feedbacks)
    return self._fit_func(xs, ys,
                          iter_forced_states, fixed_forced_states,
                          iter_forced_feedbacks, fixed_forced_feedbacks)
