# -*- coding: utf-8 -*-

import abc
import logging
from typing import Union, List, Tuple

import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from brainpy import tools, math
from brainpy.errors import ConnectorError

logger = logging.getLogger('brainpy.simulation.connect')

__all__ = [
  # the connection types
  'CONN_MAT', 'WEIGHT_MAT',
  'PRE_IDS', 'POST_IDS',
  'PRE2POST', 'POST2PRE',
  'PRE2SYN', 'POST2SYN',
  'SUPPORTED_SYN_STRUCTURE',

  # the types to provide
  'PROVIDE_MAT', 'PROVIDE_IJ',

  # the types to store connections
  'set_default_dtype', 'MAT_DTYPE', 'IDX_DTYPE', 'WEIGHT_DTYPE',

  # base class
  'Connector', 'TwoEndConnector', 'OneEndConnector',
]

CONN_MAT = 'conn_mat'
WEIGHT_MAT = 'weight_mat'
PRE_IDS = 'pre_ids'
POST_IDS = 'post_ids'
PRE2POST = 'pre2post'
POST2PRE = 'post2pre'
PRE2SYN = 'pre2syn'
POST2SYN = 'post2syn'
PRE_SLICE = 'pre_slice'
POST_SLICE = 'post_slice'

SUPPORTED_SYN_STRUCTURE = [CONN_MAT, WEIGHT_MAT,
                           PRE_IDS, POST_IDS,
                           PRE2POST, POST2PRE,
                           PRE2SYN, POST2SYN,
                           PRE_SLICE, POST_SLICE]

PROVIDE_MAT = 'mat'
PROVIDE_IJ = 'ij'

MAT_DTYPE = np.bool_
IDX_DTYPE = np.uint32
WEIGHT_DTYPE = np.float32


def set_default_dtype(mat_dtype=None, idx_dtype=None):
  """Set the default dtype.

  Use this method, you can set the default dtype for connetion matrix and
  connection index.

  For examples:

  >>> import numpy as np
  >>> import brainpy as bp
  >>>
  >>> conn = bp.conn.GridFour()(4, 4)
  >>> conn.require('conn_mat')
  JaxArray(DeviceArray([[False,  True, False, False],
                        [ True, False,  True, False],
                        [False,  True, False,  True],
                        [False, False,  True, False]], dtype=bool))
  >>> bp.conn.set_default_dtype(mat_dtype=np.float32)
  >>> conn = bp.conn.GridFour()(4, 4)
  >>> conn.require('conn_mat')
  JaxArray(DeviceArray([[0., 1., 0., 0.],
                        [1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 0.]], dtype=float32))

  Parameters
  ----------
  mat_dtype : type
    The default dtype for connection matrix.
  idx_dtype : type
    The default dtype for connection index.
  """
  if mat_dtype is not None:
    global MAT_DTYPE
    MAT_DTYPE = mat_dtype
  if idx_dtype is not None:
    global IDX_DTYPE
    IDX_DTYPE = idx_dtype


class Connector(abc.ABC):
  """Base Synaptic Connector Class."""
  pass


class TwoEndConnector(Connector):
  """Synaptic connector to build synapse connections between two neuron groups."""
  def __init__(self, ):
    self.pre_size = None
    self.post_size = None
    self.pre_num = None
    self.post_num = None
    self.structures = None

  # @abc.abstractmethod
  # def __call__(self, pre_size, post_size):
  #   raise NotImplementedError

  def __call__(self, pre_size, post_size):
    """Create the concrete connections between two end objects.

    Parameters
    ----------
    pre_size : int, tuple of int, list of int
        The size of the pre-synaptic group.
    post_size : int, tuple of int, list of int
        The size of the post-synaptic group.

    Returns
    -------
    conn : TwoEndConnector
        Return the self.
    """
    if isinstance(pre_size, int):
      pre_size = (pre_size,)
    else:
      pre_size = tuple(pre_size)
    if isinstance(post_size, int):
      post_size = (post_size,)
    else:
      post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
    return self

  def _reset_conn(self, pre_size, post_size):
    """

    Parameters
    ----------
    pre_size : int, tuple of int, list of int
        The size of the pre-synaptic group.
    post_size : int, tuple of int, list of int
        The size of the post-synaptic group.

    Returns
    -------

    """
    self.__init__()

    self.__call__(pre_size, post_size)

  def check(self, structures: Union[Tuple, List, str]):
    if isinstance(structures, str):
      structures = [structures]

    if structures is None or len(structures) == 0:
      raise ConnectorError('No synaptic structure is received.')

    # check synaptic structures
    for n in structures:
      if n not in SUPPORTED_SYN_STRUCTURE:
        raise ConnectorError(f'Unknown synapse structure "{n}". Only {SUPPORTED_SYN_STRUCTURE} is supported.')

    self.structures = list(structures)
  
  def returns(self, ind, indptr):
    """
    calculate the desired properties
    """
    ind = math.asarray(ind)
    indptr = math.asarray(indptr)

    all_data = dict()

    if CONN_MAT in self.structures:
      all_data[CONN_MAT] = todense(ind, indptr)

    if PRE_IDS in self.structures:
      all_data[PRE_IDS] = math.repeat(math.arange(indptr.size - 1), math.diff(indptr))

    if POST_IDS in self.structures:
      all_data[POST_IDS] = ind

    if PRE2POST in self.structures:
      all_data[PRE2POST] = ind, indptr

    if POST2PRE in self.structures:
      indc, indptrc = tocsc(ind, indptr)
      all_data[POST2PRE] = indc, indptrc

    if PRE2SYN in self.structures:
      syn_seq = math.arange(ind.size)
      all_data[PRE2SYN] = syn_seq, indptr

    if POST2SYN in self.structures:
      syn_seq = math.arange(ind.size)
      _, indptrc, syn_seqc = tocsc(ind, indptr, syn_seq)
      all_data[POST2SYN] = syn_seqc, indptrc
  
  def require(self, *structures):
    raise NotImplementedError

  def requires(self, *structures):
    return self.require(*structures)


class OneEndConnector(TwoEndConnector):
  """Synaptic connector to build synapse connections within a population of neurons."""
  def __init__(self):
    super(OneEndConnector, self).__init__()

  def __call__(self, pre_size, post_size=None):
    raise NotImplementedError

  def _reset_conn(self, pre_size, post_size=None):
    self._data = None
    self._pre_ids = None
    self._post_ids = None
    self._conn_mat = None
    self._weight_mat = None
    self._pre2post = None
    self._post2pre = None
    self._pre2syn = None
    self._post2syn = None

    if post_size is None:
      post_size = pre_size
    else:
      assert pre_size == post_size
    if isinstance(pre_size, int):
      pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int):
      post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size

    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
    return self


def tocsc(ind, indptr, data=None):
  pre_ids = math.repeat(math.arange(indptr.size - 1), math.diff(indptr))

  sort_idx = math.argsort(ind, kind='mergesort') # to maintain the original order of the elements with the same value
  ind_new = pre_ids[sort_idx]

  post_idx = math.arange(math.max(ind) + 1)
  post_count = math.zeros_like(post_idx)
  for i in ind:
    post_count[i] += 1

  indptr_new = math.concatenate(([0], post_count)).cumsum()

  if data is None:
    return ind_new, indptr_new

  data_new = data[sort_idx]
  return ind_new, indptr_new, data_new


def todense(ind, indptr):
  d = math.zeros((indptr.size - 1, math.max(ind) + 1), dtype=math.bool_)  # num_pre, num_post
  pre_ids = math.repeat(math.arange(indptr.size - 1), math.diff(indptr))
  d[pre_ids, ind] = True

  return d
