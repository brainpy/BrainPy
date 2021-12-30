# -*- coding: utf-8 -*-

import abc
import logging
from typing import Union, List, Tuple

import numpy as np

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

  # the types to store connections
  'set_default_dtype', 'CONN_DTYPE', 'IDX_DTYPE', 'WEIGHT_DTYPE',

  # base class
  'Connector', 'TwoEndConnector', 'OneEndConnector',

  # methods
  'tocsc', 'todense', 'tocsr', 'toind'
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

CONN_DTYPE = np.bool_
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
    global CONN_DTYPE
    CONN_DTYPE = mat_dtype
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
    try:
      assert self.pre_num is not None and self.post_num is not None
    except AssertionError:
      raise ConnectorError(f'self.pre_num or self.post_num is not defined. '
                           f'Please use self.__call__(pre_size, post_size) '
                           f'before requiring properties.')

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
    # ind = np.asarray(ind)
    # indptr = np.asarray(indptr)
    assert isinstance(ind, np.ndarray)
    assert isinstance(indptr, np.ndarray)

    all_data = dict()

    if CONN_MAT in self.structures:
      conn_mat = todense(ind, indptr, self.pre_num, self.post_num)
      all_data[CONN_MAT] = math.asarray(conn_mat, dtype=math.bool_)

    if PRE_IDS in self.structures:
      pre_ids = np.repeat(np.arange(self.pre_num), np.diff(indptr))
      all_data[PRE_IDS] = math.asarray(pre_ids, dtype=math.int_)

    if POST_IDS in self.structures:
      all_data[POST_IDS] = math.asarray(ind, dtype=math.int_)

    if PRE2POST in self.structures:
      all_data[PRE2POST] = math.asarray(ind, dtype=math.int_), \
                           math.asarray(indptr, dtype=math.int_)

    if POST2PRE in self.structures:
      indc, indptrc = tocsc(ind, indptr, self.post_num)
      all_data[POST2PRE] = math.asarray(indc, dtype=math.int_), \
                           math.asarray(indptrc, dtype=math.int_)

    if PRE2SYN in self.structures:
      syn_seq = np.arange(ind.size)
      all_data[PRE2SYN] = math.asarray(syn_seq, dtype=math.int_), \
                          math.asarray(indptr, dtype=math.int_)

    if POST2SYN in self.structures:
      syn_seq = np.arange(ind.size)
      _, indptrc, syn_seqc = tocsc(ind, indptr, self.post_num, syn_seq)
      all_data[POST2SYN] = math.asarray(syn_seqc, dtype=math.int_), \
                           math.asarray(indptrc, dtype=math.int_)

    if len(self.structures) == 1:
      return all_data[self.structures[0]]
    else:
      return tuple([all_data[n] for n in self.structures])

  def require(self, *structures):
    raise NotImplementedError

  def requires(self, *structures):
    return self.require(*structures)


class OneEndConnector(TwoEndConnector):
  """Synaptic connector to build synapse connections within a population of neurons."""
  def __init__(self):
    super(OneEndConnector, self).__init__()

  def __call__(self, pre_size, post_size=None):
    if post_size is None:
      post_size = pre_size

    try:
      assert pre_size == post_size
    except AssertionError:
      raise ConnectorError(
        f'The shape of pre-synaptic group should be the same with the post group. '
        f'But we got {pre_size} != {post_size}.')

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

  def _reset_conn(self, pre_size, post_size=None):
    self.__init__()

    self.__call__(pre_size, post_size)


def tocsc(ind, indptr, post_num, data=None):
  pre_ids = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))

  sort_idx = np.argsort(ind, kind='mergesort')  # to maintain the original order of the elements with the same value
  ind_new = pre_ids[sort_idx]

  uni_idx, count = np.unique(ind, return_counts=True)
  post_count = np.zeros(post_num, dtype=IDX_DTYPE)
  post_count[uni_idx] = count

  indptr_new = np.concatenate(([0], post_count)).cumsum()

  if data is None:
    return ind_new, indptr_new

  data_new = data[sort_idx]
  return ind_new, indptr_new, data_new


def tocsr(dense):
  """convert a dense matrix to ind, indptr."""
  pre_ids, post_ids = np.where(dense)
  pre_num = dense.shape[0]

  uni_idx, count = np.unique(pre_ids, return_counts=True)
  pre_count = np.zeros(pre_num, dtype=IDX_DTYPE)
  pre_count[uni_idx] = count
  indptr = np.concatenate(([0], count)).cumsum()

  return post_ids, indptr


def todense(ind, indptr, num_pre, num_post):
  """convert ind, indptr to a dense matrix."""
  d = np.zeros((num_pre, num_post), dtype=CONN_DTYPE)  # num_pre, num_post
  pre_ids = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
  d[pre_ids, ind] = True

  return d


def toind(pre_ids, post_ids):
  """convert pre_ids, post_ids to ind, indptr."""
  # sorting
  sort_idx = np.argsort(pre_ids, kind='mergesort')
  pre_ids = pre_ids[sort_idx]
  post_ids = post_ids[sort_idx]

  ind = post_ids
  _, pre_count = np.unique(pre_ids, return_counts=True)
  indptr = np.concatenate(([0], pre_count)).cumsum()

  return ind, indptr
