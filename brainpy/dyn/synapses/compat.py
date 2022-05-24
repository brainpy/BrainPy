# -*- coding: utf-8 -*-
import warnings
from typing import Union, Dict, Callable

from brainpy.connect import TwoEndConnector
from brainpy.dyn.base import NeuGroup
from brainpy.initialize import Initializer
from brainpy.types import Tensor
from .abstract_models import Delta, Exponential, DualExponential, NMDA
from ..synouts import COBA, CUBA

__all__ = [
  'DeltaSynapse',
  'ExpCUBA',
  'ExpCOBA',
  'DualExpCUBA',
  'DualExpCOBA',
  'AlphaCUBA',
  'AlphaCOBA',
  'NMDA',
]


class DeltaSynapse(Delta):
  """Delta synapse.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.Delta" instead.

  """
  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'sparse',
      weights: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[float, Tensor, Initializer, Callable] = None,
      post_key: str = 'V',
      post_has_ref: bool = False,
      name: str = None,
  ):
    warnings.warn('Please use "brainpy.dyn.synapses.Delta" instead.', DeprecationWarning)
    super(DeltaSynapse, self).__init__(pre=pre,
                                       post=post,
                                       conn=conn,
                                       output=CUBA(),
                                       name=name,
                                       conn_type=conn_type,
                                       weights=weights,
                                       delay_step=delay_step,
                                       post_key=post_key,
                                       post_has_ref=post_has_ref)


class ExpCUBA(Exponential):
  r"""Current-based exponential decay synapse model.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.Exponential" instead.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'sparse',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau: Union[float, Tensor] = 8.0,
      name: str = None,
      method: str = 'exp_auto',
  ):
    super(ExpCUBA, self).__init__(pre=pre,
                                  post=post,
                                  conn=conn,
                                  name=name,
                                  conn_type=conn_type,
                                  g_max=g_max,
                                  delay_step=delay_step,
                                  tau=tau,
                                  method=method,
                                  output=CUBA())


class ExpCOBA(Exponential):
  """Conductance-based exponential decay synapse model.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.Exponential" instead.
  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      # connection
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'sparse',
      # connection strength
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      # synapse parameter
      tau: Union[float, Tensor] = 8.0,
      E: Union[float, Tensor] = 0.,
      # synapse delay
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      # others
      method: str = 'exp_auto',
      name: str = None
  ):
    super(ExpCOBA, self).__init__(pre=pre,
                                  post=post,
                                  conn=conn,
                                  conn_type=conn_type,
                                  g_max=g_max,
                                  delay_step=delay_step,
                                  tau=tau,
                                  method=method,
                                  name=name,
                                  output=COBA(E=E))


class DualExpCUBA(DualExponential):
  r"""Current-based dual exponential synapse model.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.DualExponential" instead.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      tau_decay: Union[float, Tensor] = 10.0,
      tau_rise: Union[float, Tensor] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(DualExpCUBA, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      conn_type=conn_type,
                                      g_max=g_max,
                                      tau_decay=tau_decay,
                                      tau_rise=tau_rise,
                                      delay_step=delay_step,
                                      method=method,
                                      name=name,
                                      output=CUBA())


class DualExpCOBA(DualExponential):
  """Conductance-based dual exponential synapse model.


  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.DualExponential" instead.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau_decay: Union[float, Tensor] = 10.0,
      tau_rise: Union[float, Tensor] = 1.,
      E: Union[float, Tensor] = 0.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(DualExpCOBA, self).__init__(pre=pre,
                                      post=post,
                                      conn=conn,
                                      conn_type=conn_type,
                                      g_max=g_max,
                                      tau_decay=tau_decay,
                                      tau_rise=tau_rise,
                                      delay_step=delay_step,
                                      method=method,
                                      name=name,
                                      output=COBA(E=E))


class AlphaCUBA(DualExpCUBA):
  r"""Current-based alpha synapse model.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.Alpha" instead.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau_decay: Union[float, Tensor] = 10.0,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(AlphaCUBA, self).__init__(pre=pre,
                                    post=post,
                                    conn=conn,
                                    conn_type=conn_type,
                                    delay_step=delay_step,
                                    g_max=g_max,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    method=method,
                                    name=name)


class AlphaCOBA(DualExpCOBA):
  """Conductance-based alpha synapse model.

  .. deprecated:: 2.1.13
     Please use "brainpy.dyn.synapses.Alpha" instead.

  """

  def __init__(
      self,
      pre: NeuGroup,
      post: NeuGroup,
      conn: Union[TwoEndConnector, Tensor, Dict[str, Tensor]],
      conn_type: str = 'dense',
      g_max: Union[float, Tensor, Callable, Initializer] = 1.,
      delay_step: Union[int, Tensor, Initializer, Callable] = None,
      tau_decay: Union[float, Tensor] = 10.0,
      E: Union[float, Tensor] = 0.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(AlphaCOBA, self).__init__(pre=pre,
                                    post=post,
                                    conn=conn,
                                    conn_type=conn_type,
                                    delay_step=delay_step,
                                    g_max=g_max, E=E,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    method=method,
                                    name=name)