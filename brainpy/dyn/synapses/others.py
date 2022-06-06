# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

import brainpy.math as bm
from brainpy.connect import TwoEndConnector, All2All, One2One
from brainpy.dyn.base import NeuGroup, SynapseOutput, SynapsePlasticity, TwoEndConn
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor
from ..synouts import CUBA, MgBlock

__all__ = [
  'WeightedSum',
]


class WeightedSum(TwoEndConn):
  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      output: Optional[SynapseOutput] = None,
      plasticity: Optional[SynapsePlasticity] = None,
      conn_type: str = 'sparse',
      weights: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[float, Tensor, Initializer, Callable] = None,
      post_key: str = 'V',
      post_has_ref: bool = False,
      name: str = None,
  ):
    super(WeightedSum, self).__init__(pre, post, conn, name=name)


