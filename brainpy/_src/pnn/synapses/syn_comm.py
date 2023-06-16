from typing import Optional, Union, Callable, Sequence

from brainpy import math as bm
from brainpy._src import connect, initialize as init
from brainpy._src.dynsys import DynamicalSystemNS
from brainpy._src.pnn.utils import POST_AXIS, PRE_AXIS
from brainpy.types import ArrayType

__all__ = [
  'All2allMM',
  'One2oneMM',
  'DenseMM',
  'CsrMM',
]


class SynComm(DynamicalSystemNS):
  pass


class All2allMM(SynComm):
  """Synaptic matrix multiplication with All2All connections.

  Args:
    num_pre: int. The number of neurons in the presynaptic neuron group.
    num_post: int. The number of neurons in the postsynaptic neuron group.
    weight: The synaptic weights.
    axis_names: sequence of str. The name for each axis.
    include_self: bool. Whether connect the neuron with at the same position.
    mode: Mode. The computing mode.
    name: str. The object name.
  """

  def __init__(
      self,
      num_pre: int,
      num_post: int,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      include_self: bool = True,

      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(mode=mode, name=name)

    self.num_pre = num_pre
    self.num_post = num_post
    self.include_self = include_self
    self.axis_names = axis_names

    self.weight = init.parameter(weight, (self.num_pre, self.num_post), axis_names=axis_names)
    if isinstance(self.mode, bm.TrainingMode):
      self.weight = bm.TrainVar(self.weight)

  def update(self, pre_val):
    if bm.ndim(self.weight) == 0:  # weight is a scalar
      if isinstance(self.mode, bm.BatchingMode):
        assert pre_val.ndim == 2
        post_val = bm.sum(pre_val, keepdims=True, axis=1)
      else:
        assert pre_val.ndim == 1
        post_val = bm.sum(pre_val)
      if not self.include_self:
        if self.num_pre == self.num_post:
          post_val = post_val - pre_val
        elif self.num_pre > self.num_post:
          val = pre_val[:self.num_post]
          post_val = post_val - val
        else:
          val = bm.concatenate([pre_val, bm.zeros(self.num_post - self.num_pre)])
          post_val = post_val - val
      post_val = self.weight * post_val

    else:  # weight is a matrix
      if not self.include_self:
        post_val = pre_val @ bm.fill_diagonal(self.weight, 0., inplace=False)
      else:
        post_val = pre_val @ self.weight
    return post_val


class One2oneMM(SynComm):
  """Synaptic matrix multiplication with One2One connection.

  Args:
    num: int. The number of neurons.
    weight: The synaptic weight.
    axis_names: The axis names.
    mode: The computing mode.
    name: The object name.

  """

  def __init__(
      self,
      num: int,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (POST_AXIS,),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(mode=mode, name=name)

    self.num = num
    self.axis_names = axis_names

    self.weight = init.parameter(weight, (self.num,), axis_names=axis_names)
    if isinstance(self.mode, bm.TrainingMode):
      self.weight = bm.TrainVar(self.weight)

  def update(self, pre_val):
    return pre_val * self.weight


class _SynMatMul(SynComm):
  def __init__(
      self,
      conn: connect.TwoEndConnector,
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode)

    assert isinstance(conn, connect.TwoEndConnector)
    self.conn = conn
    self.axis_names = axis_names


class DenseMM(_SynMatMul):
  r"""Synaptic matrix multiplication with dense computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a dense matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    axis_names: sequence of str. The synaptic weight axis.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode, conn=conn)

    # weight
    self.weight = init.parameter(weight, (conn.pre_num, conn.post_num), axis_names=axis_names)
    if isinstance(self.mode, bm.TrainingMode):
      self.weight = bm.TrainVar(self.weight)

    # connection
    self.mask = bm.sharding.partition_by_axname(self.conn.require('conn_mat'),
                                                axis_names=axis_names)

  def update(self, x):
    return x @ (self.weight * self.mask)


class CsrMM(_SynMatMul):
  r"""Synaptic matrix multiplication with CSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a CSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    axis_names: sequence of str. The synaptic weight axis.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode, conn=conn)

    # connection
    self.indices, self.indptr = self.conn.require('csr')

    # weight
    self.weight = init.parameter(weight, (conn.pre_num, conn.post_num), axis_names=axis_names)
    if isinstance(self.mode, bm.TrainingMode):
      self.weight = bm.TrainVar(self.weight)

  def update(self, x):
    raise NotImplementedError


class CscMM(_SynMatMul):
  r"""Synaptic matrix multiplication with CSC sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a CSC sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    axis_names: sequence of str. The synaptic weight axis.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode, conn=conn)


class EventCsrMM(_SynMatMul):
  pass


class BcsrMM(_SynMatMul):
  r"""Synaptic matrix multiplication with BCSR sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a BCSR sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    axis_names: sequence of str. The synaptic weight axis.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode, conn=conn)


class BcscMM(_SynMatMul):
  r"""Synaptic matrix multiplication with BCSC sparse computation.

  It performs the computation of:

  .. math::

     y = x @ M

  where :math:`y` is the postsynaptic value, :math:`x` the presynaptic value,
  :math:`M` the synaptic weight using a BCSC sparse matrix.

  Args:
    conn: TwoEndConnector. The connection.
    weight: Synaptic weights. Can be a scalar, array, or callable function.
    axis_names: sequence of str. The synaptic weight axis.
    mode: The synaptic computing mode.
    name: The synapse model name.
  """

  def __init__(
      self,
      conn: connect.TwoEndConnector,
      weight: Union[float, ArrayType, Callable],
      axis_names: Optional[Sequence[str]] = (PRE_AXIS, POST_AXIS),
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name, mode=mode, conn=conn)


class JitProbHomoMM(_SynMatMul):
  pass


class JitProbUniformMM(_SynMatMul):
  pass


class JitProbNormalMM(_SynMatMul):
  pass
