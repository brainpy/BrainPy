from typing import Union, Dict, Callable, Optional, Tuple

import jax
import numpy as np

import brainpy.math as bm
from brainpy import check
from brainpy._src import tools
from brainpy._src.connect import TwoEndConnector, All2All, One2One, MatConn, IJConn
from brainpy._src.dyn.base import DynamicalSystem, not_pass_shargs
from brainpy._src.initialize import Initializer, variable_, parameter
from brainpy._src.integrators import odeint
from brainpy.types import ArrayType
from .synout import SynOut
from .synstp import SynSTP


class Synapse(DynamicalSystem):
  def __init__(
      self,
      conn: TwoEndConnector,
      out: Optional[SynOut] = None,
      stp: Optional[SynSTP] = None,
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(name=name, mode=mode)

    # parameters
    assert isinstance(conn, TwoEndConnector)
    self.conn = self._init_conn(conn)
    self.pre_size = conn.pre_size
    self.post_size = conn.post_size
    self.pre_num = conn.pre_num
    self.post_num = conn.post_num
    assert out is None or isinstance(out, SynOut)
    assert stp is None or isinstance(stp, SynSTP)
    self.out = out
    self.stp = stp

  def _init_conn(self, conn):
    if isinstance(conn, TwoEndConnector):
      pass
    elif isinstance(conn, (bm.ndarray, np.ndarray, jax.Array)):
      if (self.pre_num, self.post_num) != conn.shape:
        raise ValueError(f'"conn" is provided as a matrix, and it is expected '
                         f'to be an array with shape of (self.pre_num, self.post_num) = '
                         f'{(self.pre_num, self.post_num)}, however we got {conn.shape}')
      conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise ValueError(f'"conn" is provided as a dict, and it is expected to '
                         f'be a dictionary with "i" and "j" specification, '
                         f'however we got {conn}')
      conn = IJConn(i=conn['i'], j=conn['j'])
    elif conn is None:
      conn = None
    else:
      raise ValueError(f'Unknown "conn" type: {conn}')
    return conn

  def _init_weights(
      self,
      weight: Union[float, ArrayType, Initializer, Callable],
      comp_method: str,
      data_if_sparse: str = 'csr'
  ) -> Tuple[Union[float, ArrayType], ArrayType]:
    if comp_method not in ['sparse', 'dense']:
      raise ValueError(f'"comp_method" must be in "sparse" and "dense", but we got {comp_method}')
    if data_if_sparse not in ['csr', 'ij', 'coo']:
      raise ValueError(f'"sparse_data" must be in "csr" and "ij", but we got {data_if_sparse}')

    # connections and weights
    if isinstance(self.conn, One2One):
      weight = parameter(weight, (self.pre_num,), allow_none=False)
      conn_mask = None

    elif isinstance(self.conn, All2All):
      weight = parameter(weight, (self.pre_num, self.post_num), allow_none=False)
      conn_mask = None

    else:
      if comp_method == 'sparse':
        if data_if_sparse == 'csr':
          conn_mask = self.conn.require('pre2post')
        elif data_if_sparse in ['ij', 'coo']:
          conn_mask = self.conn.require('post_ids', 'pre_ids')
        else:
          ValueError(f'Unknown sparse data type: {data_if_sparse}')
        weight = parameter(weight, conn_mask[0].shape, allow_none=False)
      elif comp_method == 'dense':
        weight = parameter(weight, (self.pre_num, self.post_num), allow_none=False)
        conn_mask = self.conn.require('conn_mat')
      else:
        raise ValueError(f'Unknown connection type: {comp_method}')

    # training weights
    if isinstance(self.mode, bm.TrainingMode):
      weight = bm.TrainVar(weight)
    return weight, conn_mask

  def _syn2post_with_all2all(self, syn_value, syn_weight, include_self):
    if bm.ndim(syn_weight) == 0:
      if isinstance(self.mode, bm.BatchingMode):
        post_vs = bm.sum(syn_value, keepdims=True, axis=tuple(range(syn_value.ndim))[1:])
      else:
        post_vs = bm.sum(syn_value)
      if not include_self:
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


class Exponential(Synapse):
  r"""Exponential decay synapse model.

  **Model Descriptions**

  The single exponential decay synapse model assumes the release of neurotransmitter,
  its diffusion across the cleft, the receptor binding, and channel opening all happen
  very quickly, so that the channels instantaneously jump from the closed to the open state.
  Therefore, its expression is given by

  .. math::

      g_{\mathrm{syn}}(t)=g_{\mathrm{max}} e^{-\left(t-t_{0}\right) / \tau}

  where :math:`\tau_{delay}` is the time constant of the synaptic state decay,
  :math:`t_0` is the time of the pre-synaptic spike,
  :math:`g_{\mathrm{max}}` is the maximal conductance.

  Accordingly, the differential form of the exponential synapse is given by

  .. math::

      \begin{aligned}
       & g_{\mathrm{syn}}(t) = g_{max} g * \mathrm{STP} \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  where :math:`\mathrm{STP}` is used to model the short-term plasticity effect.

  Parameters
  ----------
  conn: optional, ArrayType, dict of (str, ndarray), TwoEndConnector
    The synaptic connections.
  comp_method: str
    The connection type used for model speed optimization. It can be
    `sparse` and `dense`. The default is `sparse`.
  tau: float, ArrayType
    The time constant of decay. [ms]
  g_max: float, ArrayType, Initializer, Callable
    The synaptic strength (the maximum conductance). Default is 1.
  name: str
    The name of this synaptic projection.
  method: str
    The numerical integration methods.

  References
  ----------

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(
      self,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      out: Optional[SynOut] = None,
      stp: Optional[SynSTP] = None,
      comp_method: str = 'sparse',
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      tau: Union[float, ArrayType] = 8.0,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super(Exponential, self).__init__(conn=conn,
                                      out=out,
                                      stp=stp,
                                      name=name,
                                      mode=mode)

    # parameters
    self.comp_method = comp_method
    self.tau = check.is_float(tau, allow_int=True)

    # connections and weights
    self.g_max, self.conn_mask = self._init_weights(g_max, comp_method, data_if_sparse='csr')

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.g = variable_(bm.zeros, self.post_num, batch_size)
    if self.out is not None:
      self.out.reset_state(batch_size)
    if self.stp is not None:
      self.stp.reset_state(batch_size)

  @not_pass_shargs
  def update(self, pre_spike):
    if self.stp is not None:
      syn_value = self.stp(pre_spike) * pre_spike
    else:
      syn_value = bm.asarray(pre_spike, dtype=bm.float_)

    # post values
    if isinstance(self.conn, All2All):
      post_vs = self._syn2post_with_all2all(syn_value, self.g_max, self.conn.include_self)
    elif isinstance(self.conn, One2One):
      post_vs = self._syn2post_with_one2one(syn_value, self.g_max)
    else:
      if self.comp_method == 'sparse':
        bl = tools.import_brainpylib()

        if self.stp is None:
          f = lambda s: bl.event_ops.event_csr_matvec(self.g_max,
                                                      self.conn_mask[0],
                                                      self.conn_mask[1],
                                                      s,
                                                      shape=(self.pre_num, self.post_num),
                                                      transpose=True)
          if isinstance(self.mode, bm.BatchingMode):
            f = jax.vmap(f)
          post_vs = f(pre_spike)
        else:
          f = lambda s: bl.sparse_ops.cusparse_csr_matvec(self.g_max,
                                                          self.conn_mask[0],
                                                          self.conn_mask[1],
                                                          s,
                                                          shape=(self.pre_num, self.post_num),
                                                          transpose=True)
          if isinstance(self.mode, bm.BatchingMode):
            f = jax.vmap(f)
          post_vs = f(syn_value)
      else:
        post_vs = self._syn2post_with_dense(syn_value, self.g_max, self.conn_mask)

    # updates
    self.g.value = self.integral(self.g.value, bm.share.get('t'), bm.dt) + post_vs

    # outputs
    if self.out is not None:
      return self.out(self.g.value)
    else:
      return self.g.value

