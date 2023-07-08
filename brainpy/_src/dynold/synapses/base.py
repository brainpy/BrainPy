from typing import Union, Dict, Callable, Optional, Tuple

import jax
import numpy as np

from brainpy import math as bm
from brainpy._src.connect import TwoEndConnector, MatConn, IJConn, One2One, All2All
from brainpy._src.dnn import linear
from brainpy._src.dyn import projections
from brainpy._src.dynsys import Projection, DynamicalSystem, NeuDyn, Sequential
from brainpy._src.initialize import parameter
from brainpy._src.mixin import (ParamDesc, ParamDescInit, JointType,
                                AutoDelaySupp, BindCondData, AlignPost,
                                ReturnInfo)
from brainpy.errors import UnsupportedError
from brainpy.types import ArrayType

__all__ = [
  'SynConn',
  '_SynSTP',
  '_SynOut',
  'TwoEndConn',
  '_TwoEndConnAlignPre',
  '_TwoEndConnAlignPost',
]


class SynConn(Projection):
  """Base class to model two-end synaptic connections.

  Parameters
  ----------
  pre : NeuGroup
    Pre-synaptic neuron group.
  post : NeuGroup
    Post-synaptic neuron group.
  conn : optional, ndarray, ArrayType, dict, TwoEndConnector
    The connection method between pre- and post-synaptic groups.
  name : str, optional
    The name of the dynamic system.
  """

  def __init__(
      self,
      pre: DynamicalSystem,
      post: DynamicalSystem,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, DynamicalSystem):
      raise TypeError('"pre" must be an instance of DynamicalSystem.')
    if not isinstance(post, DynamicalSystem):
      raise TypeError('"post" must be an instance of DynamicalSystem.')
    self.pre = pre
    self.post = post

    # connectivity
    # ------------
    if isinstance(conn, TwoEndConnector):
      self.conn = conn(pre.size, post.size)
    elif isinstance(conn, (bm.Array, np.ndarray, jax.Array)):
      if (pre.num, post.num) != conn.shape:
        raise ValueError(f'"conn" is provided as a matrix, and it is expected '
                         f'to be an array with shape of (pre.num, post.num) = '
                         f'{(pre.num, post.num)}, however we got {conn.shape}')
      self.conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise ValueError(f'"conn" is provided as a dict, and it is expected to '
                         f'be a dictionary with "i" and "j" specification, '
                         f'however we got {conn}')
      self.conn = IJConn(i=conn['i'], j=conn['j'])
    elif isinstance(conn, str):
      self.conn = conn
    elif conn is None:
      self.conn = None
    else:
      raise ValueError(f'Unknown "conn" type: {conn}')

  def __repr__(self):
    names = self.__class__.__name__
    return (f'{names}(name={self.name}, mode={self.mode}, \n'
            f'{" " * len(names)} pre={self.pre}, \n'
            f'{" " * len(names)} post={self.post})')

  def check_pre_attrs(self, *attrs):
    """Check whether pre group satisfies the requirement."""
    if not hasattr(self, 'pre'):
      raise ValueError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise TypeError(f'Must be string. But got {attr}.')
      if not hasattr(self.pre, attr):
        raise ValueError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def check_post_attrs(self, *attrs):
    """Check whether post group satisfies the requirement."""
    if not hasattr(self, 'post'):
      raise ValueError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise TypeError(f'Must be string. But got {attr}.')
      if not hasattr(self.post, attr):
        raise ValueError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def update(self, *args, **kwargs):
    """The function to specify the updating rule.

    Assume any dynamical system depends on the shared variables (`sha`),
    like time variable ``t``, the step precision ``dt``, and the time step `i`.
    """
    raise NotImplementedError('Must implement "update" function by subclass self.')


class _SynapseComponent(DynamicalSystem):
  """Base class for modeling synaptic components,
  including synaptic output, synaptic short-term plasticity,
  synaptic long-term plasticity, and others. """

  '''Master of this component.'''
  master: SynConn

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

  def register_master(self, master: SynConn):
    if not isinstance(master, SynConn):
      raise TypeError(f'master must be instance of {SynConn.__name__}, but we got {type(master)}')
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

  def register_master(self, master: SynConn):
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
    return ReturnInfo(self.master.pre.varshape, None, self.master.pre.mode, init=bm.zeros)


class _NullSynOut(_SynOut):
  def clone(self):
    return _NullSynOut()


class TwoEndConn(SynConn):
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


def _get_delay(delay_step):
  if delay_step is None:
    return None
  elif callable(delay_step):
    raise UnsupportedError('Currently delay step supports integer.')
  else:
    return delay_step * bm.get_dt()


class _TempOut(DynamicalSystem, BindCondData, ParamDesc):
  def update(self, *args, **kwargs):
    pass


class _TwoEndConnAlignPre(TwoEndConn):
  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      syn: ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]],
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
    assert isinstance(syn, ParamDescInit[JointType[DynamicalSystem, AutoDelaySupp]])

    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     output=output,
                     stp=None,
                     name=name,
                     mode=mode,
                     init_stp=False)

    delay = _get_delay(delay_step)

    # Projection
    if isinstance(conn, All2All):
      proj = projections.ProjAlignPre(pre=pre,
                                      syn=syn,
                                      delay=delay,
                                      comm=linear.AllToAll(pre.num, post.num, g_max),
                                      out=_TempOut(),
                                      post=post)

    elif isinstance(conn, One2One):
      assert post.num == pre.num
      proj = projections.ProjAlignPre(pre=pre,
                                      syn=syn,
                                      delay=delay,
                                      comm=linear.OneToOne(pre.num, g_max),
                                      out=_TempOut(),
                                      post=post)

    else:
      if comp_method == 'dense':
        proj = projections.ProjAlignPre(pre=pre,
                                        syn=syn,
                                        delay=delay,
                                        comm=linear.MaskedLinear(conn, g_max),
                                        out=_TempOut(),
                                        post=post)

      elif comp_method == 'sparse':
        proj = projections.ProjAlignPre(pre=pre,
                                        syn=syn,
                                        delay=delay,
                                        comm=linear.CSRLinear(conn, g_max),
                                        out=_TempOut(),
                                        post=post)

      else:
        raise UnsupportedError(f'Does not support {comp_method}, only "sparse" or "dense".')
    self.proj = proj
    self.proj.post.cur_inputs.pop(self.proj.name)
    self.stp = self.pre.after_updates[self.proj._syn_id].syn.stp

  def update(self, pre_spike=None, stop_spike_gradient: bool = False):
    if pre_spike is None:
      pre_spike = self.pre.after_updates[self.proj._syn_id].delay.at(self.proj.name)
    if stop_spike_gradient:
      pre_spike = jax.lax.stop_gradient(pre_spike)
    current = self.proj.comm(pre_spike)
    return self.output(current)


class _TwoEndConnAlignPost(TwoEndConn):
  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      syn: ParamDescInit[JointType[DynamicalSystem, AlignPost]],
      conn: TwoEndConnector,
      g_max: Union[float, ArrayType, Callable],
      output: _SynOut = _NullSynOut(),
      stp: Optional[_SynSTP] = None,
      comp_method: str = 'dense',
      delay_step: Union[int, ArrayType, Callable] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(pre=pre,
                     post=post,
                     conn=conn,
                     output=output,
                     stp=stp,
                     name=name,
                     mode=mode,
                     init_stp=True)

    pre = _DelayedSyn(pre, self.stp)
    delay = _get_delay(delay_step)

    # make every synapse unique
    syn._identifier = syn._identifier + f' // {self.name}'

    # Projection
    if isinstance(conn, All2All):
      proj = projections.ProjAlignPost(pre=pre,
                                       delay=delay,
                                       comm=linear.AllToAll(self.pre.num, self.post.num, g_max),
                                       syn=syn,
                                       out=_TempOut.desc(),
                                       post=post)

    elif isinstance(conn, One2One):
      assert post.num == self.pre.num
      proj = projections.ProjAlignPost(pre=pre,
                                       delay=delay,
                                       comm=linear.OneToOne(self.pre.num, g_max),
                                       syn=syn,
                                       out=_TempOut.desc(),
                                       post=post)

    else:
      if comp_method == 'dense':
        proj = projections.ProjAlignPost(pre=pre,
                                         delay=delay,
                                         comm=linear.MaskedLinear(conn, g_max),
                                         syn=syn,
                                         out=_TempOut.desc(),
                                         post=post)

      elif comp_method == 'sparse':
        if self.stp is None:
          comm = linear.EventCSRLinear(conn, g_max)
        else:
          comm = linear.CSRLinear(conn, g_max)
        proj = projections.ProjAlignPost(pre=pre,
                                         delay=delay,
                                         comm=comm,
                                         syn=syn,
                                         out=_TempOut.desc(),
                                         post=post)

      else:
        raise UnsupportedError(f'Does not support {comp_method}, only "sparse" or "dense".')
    self.proj = proj
    self.proj.post.cur_inputs.pop(self.proj.name)

  def update(self, pre_spike=None, stop_spike_gradient: bool = False):
    if pre_spike is None:
      pre_spike = self.proj.pre.after_updates[self.proj._delay_repr].at(self.proj.name)
    if stop_spike_gradient:
      # TODO: if self.stp is not None
      pre_spike = jax.lax.stop_gradient(pre_spike)
    current = self.proj.comm(pre_spike)
    self.proj.post.before_updates[self.proj._post_repr].syn.add_current(current)  # synapse post current
    return self.output(current)


class _DelayedSyn(DynamicalSystem, ParamDesc, AutoDelaySupp):
  def __init__(self, syn, stp=None):
    super().__init__()
    self.syn = syn
    self.stp = stp

  def update(self, x):
    if self.stp is None:
      return self.syn(x)
    else:
      self.stp.update(x)
      return self.stp(self.syn(x))

  def return_info(self):
    if self.stp is None:
      return self.syn.return_info()
    else:
      return self.stp.return_info()

