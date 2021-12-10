# -*- coding: utf-8 -*-

import abc
import logging
from typing import Union, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from brainpy import tools, math
from brainpy.errors import ConnectorError

logger = logging.getLogger('brainpy.simulation.connect')

__all__ = [
  # the connection types
  'CONN_MAT',
  'PRE_IDS', 'POST_IDS',
  'PRE2POST', 'POST2PRE',
  'POST2PRE_MAT', 'PRE2POST_MAT',
  'PRE2SYN', 'POST2SYN',
  'PRE_SLICE', 'POST_SLICE',
  'SUPPORTED_SYN_STRUCTURE',

  # the types to provide
  'PROVIDE_MAT', 'PROVIDE_IJ',

  # the types to store connections
  'set_default_dtype', 'MAT_DTYPE', 'IDX_DTYPE',

  # base class
  'Connector', 'TwoEndConnector', 'OneEndConnector',

  # formatter functions
  'ij2mat', 'mat2ij',
  'pre2post', 'post2pre',
  'pre2post_mat', 'post2pre_mat',
  'pre2syn', 'post2syn',
  'pre_slice', 'post_slice',
]

CONN_MAT = 'conn_mat'
PRE_IDS = 'pre_ids'
POST_IDS = 'post_ids'
PRE2POST = 'pre2post'
POST2PRE = 'post2pre'
PRE2POST_MAT = 'pre2post_mat'
POST2PRE_MAT = 'post2pre_mat'
PRE2SYN = 'pre2syn'
POST2SYN = 'post2syn'
PRE_SLICE = 'pre_slice'
POST_SLICE = 'post_slice'

SUPPORTED_SYN_STRUCTURE = [CONN_MAT,
                           PRE_IDS, POST_IDS,
                           PRE2POST, POST2PRE,
                           PRE2POST_MAT, POST2PRE_MAT,
                           PRE2SYN, POST2SYN,
                           PRE_SLICE, POST_SLICE]

PROVIDE_MAT = 'mat'
PROVIDE_IJ = 'ij'

MAT_DTYPE = np.bool_
IDX_DTYPE = np.uint32


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
  """Base Synaptical Connector Class."""

  def __init__(self, ):
    self._data = None
    self._pre_ids = None
    self._post_ids = None
    self._conn_mat = None
    self._pre2post = None
    self._post2pre = None
    self._pre2syn = None
    self._post2syn = None


class TwoEndConnector(Connector):
  """Synaptical connector to build synapse connections between two neuron groups."""

  @abc.abstractmethod
  def __call__(self, pre_size, post_size):
    raise NotImplementedError

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
    self._data = None
    self._pre_ids = None
    self._post_ids = None
    self._conn_mat = None
    self._pre2post = None
    self._post2pre = None
    self._pre2syn = None
    self._post2syn = None

    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)

  def check(self, structures: Union[Tuple, List, str]):
    if (not hasattr(self, 'pre_size')) or (not hasattr(self, 'post_size')):
      raise ConnectorError(f'Please call "__call__" first to gather the size of the '
                           f'pre-synaptic and post-synaptic neuron groups for: {str(self)}')

    # get synaptic structures
    for n in structures:
      if n not in SUPPORTED_SYN_STRUCTURE:
        raise ConnectorError(f'Unknown synapse structure "{n}". We only '
                             f'support {SUPPORTED_SYN_STRUCTURE}.')

    # provide what synaptic structure?
    if len(structures) == 0:
      raise ConnectorError('Do not return any synaptic structures.')

  def _check(self):
    if self._data is None:
      raise ConnectorError('Please initialize the class first.')
    if not isinstance(self._data, csr_matrix):
      raise ConnectorError(f'Please call "__call__" first to gather the size of the '
                           f'pre-synaptic and post-synaptic neuron groups for: {str(self)}.')

  @property
  def conn_mat(self):
    self._check()
    if self._conn_mat is None:
      raise ConnectorError('Please require conn_mat first.')
    return self._conn_mat

  @property
  def pre_ids(self):
    self._check()
    if self._pre_ids is None:
      raise ConnectorError('Please require pre_ids first.')
    return self._pre_ids

  @property
  def post_ids(self):
    self._check()
    if self._post_ids is None:
      raise ConnectorError('Please require post_ids first.')
    return self._post_ids

  @property
  def pre2post(self):
    self._check()
    if self._pre2post is None:
      raise ConnectorError('Please require pre2post first.')
    return self._pre2post

  @property
  def post2pre(self):
    self._check()
    if self._post2pre is None:
      raise ConnectorError('Please require post2pre first.')
    return self._post2pre

  @property
  def pre2syn(self):
    self._check()
    if self._pre2syn is None:
      raise ConnectorError('Please require pre2syn first.')
    return self._pre2syn

  @property
  def post2syn(self):
    if self._post2syn is None:
      raise ConnectorError('Please require post2syn first.')
    return self._post2syn

  def require(self, *structures):
    self.check(structures)
    if isinstance(structures, str):
      structures = (structures,)
    all_data = dict()

    for n in structures:
      if n == CONN_MAT:
        self._conn_mat = all_data[CONN_MAT] = math.asarray(self._data.todense(), dtype=MAT_DTYPE)
      elif n == PRE_IDS:
        self._pre_ids = all_data[PRE_IDS] = math.asarray(self._data.nonzero()[0], dtype=IDX_DTYPE)
      elif n == POST_IDS:
        self._post_ids = all_data[POST_IDS] = math.asarray(self._data.nonzero()[1], dtype=IDX_DTYPE)
      elif n == PRE2POST:
        self._pre2post = all_data[PRE2POST] = pre2post(self._data)
      elif n == PRE2SYN:
        self._pre2syn = all_data[PRE2SYN] = pre2syn(self._data)
      elif n == POST2PRE:
        self._post2pre = all_data[POST2PRE] = post2pre(self._data)
      elif n == POST2SYN:
        self._post2syn = all_data[POST2SYN] = post2syn(self._data)
      else:
        raise ConnectorError(f'Unknown synaptic structures "{n}", only support {SUPPORTED_SYN_STRUCTURE}.')

    # data of the needed structures
    if len(structures) == 1:
      return all_data[structures[0]]
    else:
      return tuple([all_data[n] for n in structures])

  def requires(self, *structures):
    return self.require(*structures)


class OneEndConnector(TwoEndConnector):
  """Synaptical connector to build synapse connections within a population of neurons."""

  def __call__(self, pre_size, post_size=None):
    raise NotImplementedError

  def _reset_conn(self, pre_size, post_size=None):
    self._data = None
    self._pre_ids = None
    self._post_ids = None
    self._conn_mat = None
    self._pre2post = None
    self._post2pre = None
    self._pre2syn = None
    self._post2syn = None

    if post_size is None:
      post_size = pre_size
    else:
      assert pre_size == post_size
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
    return self


def pre2post(data: csr_matrix):
  """Get pre2post connections from `i` and `j` indexes.

  Parameters
  ----------
  data: class, csr_matrix
        The instance of class csr_matrix

  Returns
  -------
  conn : Tuple
        A tuple of two vectors: indices and indptr
  """
  return (math.asarray(data.indices, dtype=IDX_DTYPE),
          math.asarray(data.indptr, dtype=IDX_DTYPE))


def post2pre(data: csr_matrix):
  """Get post2pre connections from `i` and `j` indexes.

  Parameters
  ----------
  data: class, csr_matrix
        The instance of class csr_matrix

  Returns
  -------
  conn : Tuple
        A tuple of two vectors: indices and indptr
  """
  return (math.asarray(data.tocsc().indices, dtype=IDX_DTYPE),
          math.asarray(data.tocsc().indptr, dtype=IDX_DTYPE))


def pre2syn(data: csr_matrix):
  """Get pre2syn connections from `i` and `j` indexes.

  Parameters
  ----------
  data: class, csr_matrix
        The instance of class csr_matrix

  Returns
  -------
  conn : Tuple
        A tuple of two vectors: indices and indptr
  """
  syn_seq = np.arange(data.indices.shape[0])
  syn_csr_mat = csr_matrix((syn_seq, data.indices, data.indptr))
  return (math.asarray(syn_csr_mat.data, dtype=IDX_DTYPE),
          math.asarray(syn_csr_mat.indptr, dtype=IDX_DTYPE))


def post2syn(data: csr_matrix):
  """Get post2syn connections from `i` and `j` indexes.

  Parameters
  ----------
  data: class, csr_matrix
        The instance of class csr_matrix

  Returns
  -------
  conn : Tuple
        A tuple of two vectors: indices and indptr
  """
  syn_seq = np.arange(data.indices.shape[0])
  syn_csr_mat = csr_matrix((syn_seq, data.indices, data.indptr))
  return (math.asarray(syn_csr_mat.tocsc().data, dtype=IDX_DTYPE),
          math.asarray(syn_csr_mat.tocsc().indptr, dtype=IDX_DTYPE))
