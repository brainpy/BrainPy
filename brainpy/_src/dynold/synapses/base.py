from typing import Union, Dict, Callable, Optional, Tuple

import jax

from brainpy import math as bm
from brainpy._src.connect import TwoEndConnector, One2One, All2All
from brainpy._src.dnn import linear
from brainpy._src.dyn import projections
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.initialize import parameter
from brainpy._src.mixin import (ParamDesc, JointType,
                                AutoDelaySupp, BindCondData, ReturnInfo)
from brainpy.errors import UnsupportedError
from brainpy.types import ArrayType

__all__ = [
  '_SynSTP',
  '_SynOut',
  'TwoEndConn',
  '_TwoEndConnAlignPre',
]


class _SynapseComponent(DynamicalSystem):
  """Base class for modeling synaptic components,
  including synaptic output, synaptic short-term plasticity,
  synaptic long-term plasticity, and others. """

  '''Master of this component.'''
  master: projections.SynConn

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._registered = False

  @property
  def isregistered(self) -> bool:
    """State of the component, representing whether it has been registered."""
    return self._registered

  @isregistered.setter
  def isregistered(self, val: bool):
    if not isinstance(val, bool):
      raise ValueError('Must be an instance of bool.')
    self._registered = val

  def reset_state(self, batch_size=None):
    pass

  def register_master(self, master: projections.SynConn):
    if not isinstance(master, projections.SynConn):
      raise TypeError(f'master must be instance of {projections.SynConn.__name__}, but we got {type(master)}')
    if self.isregistered:
      raise ValueError(f'master has been registered, but we got another master going to be registered.')
    if hasattr(self, 'master') and self.master != master:
      raise ValueError(f'master has been registered, but we got another master going to be registered.')
    self.master = master
    self._registered = True

  def __repr__(self):
    return self.__class__.__name__

  def __call__(self, *args, **kwargs):
    return self.filter(*args, **kwargs)

  def clone(self) -> '_SynapseComponent':
    """The function useful to clone a new object when it has been used."""
    raise NotImplementedError

  def filter(self, g):
    raise NotImplementedError


class _SynOut(_SynapseComponent, ParamDesc):
  """Base class for synaptic current output."""

  def __init__(
      self,
      name: str = None,
      target_var: Union[str, bm.Variable] = None,
  ):
    super().__init__(name=name)
    # check target variable
    if target_var is not None:
      if not isinstance(target_var, (str, bm.Variable)):
        raise TypeError('"target_var" must be instance of string or Variable. '
                        f'But we got {type(target_var)}')
    self.target_var: Optional[bm.Variable] = target_var

  def register_master(self, master: projections.SynConn):
    super().register_master(master)

    # initialize target variable to output
    if isinstance(self.target_var, str):
      if not hasattr(self.master.post, self.target_var):
        raise KeyError(f'Post-synaptic group does not have target variable: {self.target_var}')
      self.target_var = getattr(self.master.post, self.target_var)

  def filter(self, g):
    if self.target_var is None:
      return g
    else:
      self.target_var += g

  def update(self):
    pass


class _SynSTP(_SynapseComponent, ParamDesc, AutoDelaySupp):
  """Base class for synaptic short-term plasticity."""

  def update(self, pre_spike):
    pass

  def return_info(self):
    assert self.isregistered
    return ReturnInfo(self.master.pre.varshape, None, self.master.pre.mode, bm.zeros)


class _NullSynOut(_SynOut):
  def clone(self):
    return _NullSynOut()


class TwoEndConn(projections.SynConn):
  """Base class to model synaptic connections.

  Parameters
  ----------
  pre : NeuGroup
    Pre-synaptic neuron group.
  post : NeuGroup
    Post-synaptic neuron group.
  conn : optional, ndarray, ArrayType, dict, TwoEndConnector
    The connection method between pre- and post-synaptic groups.
  output: Optional, SynOutput
    The output for the synaptic current.

    .. versionadded:: 2.1.13
       The output component for a two-end connection model.

  stp: Optional, SynSTP
    The short-term plasticity model for the synaptic variables.

    .. versionadded:: 2.1.13
       The short-term plasticity component for a two-end connection model.

  ltp: Optional, SynLTP
    The long-term plasticity model for the synaptic variables.

    .. versionadded:: 2.1.13
       The long-term plasticity component for a two-end connection model.

  name: Optional, str
    The name of the dynamic system.
  """

  def __init__(
      self,
      pre: DynamicalSystem,
      post: DynamicalSystem,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]] = None,
      output: _SynOut = _NullSynOut(),
      stp: Optional[_SynSTP] = None,
      ltp: Optional = None,
      mode: bm.Mode = None,
      name: str = None,
      init_stp: bool = True
  ):
    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     name=name,
                     mode=mode)

    # synaptic output
    output = _NullSynOut() if output is None else output
    if output.isregistered:
      output = output.clone()
    if not isinstance(output, _SynOut):
      raise TypeError(f'output must be instance of {_SynOut.__name__}, '
                      f'but we got {type(output)}')
    output.register_master(master=self)
    self.output: _SynOut = output

    # short-term synaptic plasticity
    if init_stp:
      stp = _init_stp(stp, self)
    self.stp: Optional[_SynSTP] = stp

  def _init_weights(
      self,
      weight: Union[float, ArrayType, Callable],
      comp_method: str,
      sparse_data: str = 'csr'
  ) -> Tuple[Union[float, ArrayType], ArrayType]:
    if comp_method not in ['sparse', 'dense']:
      raise ValueError(f'"comp_method" must be in "sparse" and "dense", but we got {comp_method}')
    if sparse_data not in ['csr', 'ij', 'coo']:
      raise ValueError(f'"sparse_data" must be in "csr" and "ij", but we got {sparse_data}')
    if self.conn is None:
      raise ValueError(f'Must provide "conn" when initialize the model {self.name}')

    # connections and weights
    if isinstance(self.conn, One2One):
      weight = parameter(weight, (self.pre.num,), allow_none=False)
      conn_mask = None

    elif isinstance(self.conn, All2All):
      weight = parameter(weight, (self.pre.num, self.post.num), allow_none=False)
      conn_mask = None

    else:
      if comp_method == 'sparse':
        if sparse_data == 'csr':
          conn_mask = self.conn.require('pre2post')
        elif sparse_data in ['ij', 'coo']:
          conn_mask = self.conn.require('post_ids', 'pre_ids')
        else:
          ValueError(f'Unknown sparse data type: {sparse_data}')
        weight = parameter(weight, conn_mask[0].shape, allow_none=False)
      elif comp_method == 'dense':
        weight = parameter(weight, (self.pre.num, self.post.num), allow_none=False)
        conn_mask = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {comp_method}')

    # training weights
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    return weight, conn_mask

  def _syn2post_with_all2all(self, syn_value, syn_weight):
    if bm.ndim(syn_weight) == 0:
      if isinstance(self.mode, bm.BatchingMode):
        post_vs = bm.sum(syn_value, keepdims=True, axis=tuple(range(syn_value.ndim))[1:])
      else:
        post_vs = bm.sum(syn_value)
      if not self.conn.include_self:
        post_vs = post_vs - syn_value
      post_vs = syn_weight * post_vs
    else:
      post_vs = syn_value @ syn_weight
    return post_vs

  def _syn2post_with_one2one(self, syn_value, syn_weight):
    return syn_value * syn_weight

  def _syn2post_with_dense(self, syn_value, syn_weight, conn_mat):
    if bm.ndim(syn_weight) == 0:
      post_vs = (syn_weight * syn_value) @ conn_mat
    else:
      post_vs = syn_value @ (syn_weight * conn_mat)
    return post_vs


def _init_stp(stp, master):
  if stp is not None:
    if stp.isregistered:
      stp = stp.clone()
    if not isinstance(stp, _SynSTP):
      raise TypeError(f'Short-term plasticity must be instance of {_SynSTP.__name__}, '
                      f'but we got {type(stp)}')
    stp.register_master(master=master)
  return stp


class _TwoEndConnAlignPre(TwoEndConn):
  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      syn: DynamicalSystem,
      conn: TwoEndConnector,
      g_max: Union[float, ArrayType, Callable],
      output: JointType[DynamicalSystem, BindCondData] = _NullSynOut(),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      delay_step: Union[int, ArrayType, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    assert isinstance(pre, NeuDyn)
    assert isinstance(post, NeuDyn)
    assert isinstance(syn, DynamicalSystem)

    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     output=output,
                     stp=stp,
                     name=name,
                     mode=mode)

    # delay
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

    # synaptic dynamics
    self.syn = syn

    # synaptic communications
    if isinstance(conn, All2All):
      self.comm = linear.AllToAll(pre.num, post.num, g_max)
    elif isinstance(conn, One2One):
      assert post.num == pre.num
      self.comm = linear.OneToOne(pre.num, g_max)
    else:
      if comp_method == 'dense':
        self.comm = linear.MaskedLinear(conn, g_max)
      elif comp_method == 'sparse':
        self.comm = linear.CSRLinear(conn, g_max)
      else:
        raise UnsupportedError(f'Does not support {comp_method}, only "sparse" or "dense".')

  def update(self, pre_spike=None, stop_spike_gradient: bool = False):
    if pre_spike is None:
      pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    if stop_spike_gradient:
      pre_spike = jax.lax.stop_gradient(pre_spike)
    if self.stp is not None:
      self.stp.update(pre_spike)
      pre_spike = self.stp(pre_spike)
    current = self.comm(self.syn(pre_spike))
    return self.output(current)


