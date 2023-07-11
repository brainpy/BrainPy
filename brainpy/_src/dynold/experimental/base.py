from typing import Union, Callable, Optional, Tuple

import jax
import numpy as np

import brainpy.math as bm
from brainpy._src.connect import TwoEndConnector, All2All, One2One, MatConn, IJConn
from brainpy._src.dynsys import DynamicalSystem
from brainpy._src.initialize import Initializer, parameter
from brainpy.types import ArrayType


class SynConnNS(DynamicalSystem):
  def __init__(
      self,
      conn: TwoEndConnector,
      out: Optional['SynOutNS'] = None,
      stp: Optional['SynSTPNS'] = None,
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
    assert out is None or isinstance(out, SynOutNS)
    assert stp is None or isinstance(stp, SynSTPNS)
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


class SynOutNS(DynamicalSystem):
  def update(self, post_g, post_v):
    raise NotImplementedError

  def reset_state(self, batch_size: Optional[int] = None):
    pass


class SynSTPNS(DynamicalSystem):
  """Base class for synaptic short-term plasticity."""

  def update(self, pre_spike):
    raise NotImplementedError
