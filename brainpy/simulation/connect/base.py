# -*- coding: utf-8 -*-

import abc
import logging
from typing import Union, List, Tuple

import jax.numpy as jnp
import numpy as np

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


def _check_structures(structures):
  if isinstance(structures, str):
    structures = (structures,)
  if 'pre_slice' in structures and 'post_slice' in structures:
    raise ConnectorError('Cannot use "pre_slice" and "post_slice" simultaneously. \n'
                         'We recommend you use "pre_slice + post2syn" or "post_slice + pre2syn".')
  return structures


def _check_connections(conn):
  if not isinstance(conn, (np.ndarray, math.JaxArray, jnp.ndarray)):
    raise ConnectorError(f'Synaptic connection must be a '
                         f'numpy.ndarray / brainpy.math.JaxArray / jax.numpy.ndarray, '
                         f'but got {type(conn)}')


class Connector(abc.ABC):
  """Base Synaptical Connector Class."""
  pass


class TwoEndConnector(Connector):
  """Synaptical connector to build synapse connections between two neuron groups."""

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
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
    return self

  def check(self, structures: Union[Tuple, List, str]):
    if (not hasattr(self, 'pre_size')) or (not hasattr(self, 'post_size')):
      raise ConnectorError(f'Please call "__call__" first to gather the size of the '
                           f'pre-synaptic and post-synaptic neuron groups for: {str(self)}')

    # check synaptic structures
    self.structures = _check_structures(structures)

    # get synaptic structures
    for n in self.structures:
      if n not in SUPPORTED_SYN_STRUCTURE:
        raise ConnectorError(f'Unknown synapse structure "{n}". We only '
                             f'support {SUPPORTED_SYN_STRUCTURE}.')

    # provide what synaptic structure?
    if len(self.structures) == 0:
      raise ConnectorError('Do not return any synaptic structures.')
    elif CONN_MAT in self.structures:
      return PROVIDE_MAT
    else:
      return PROVIDE_IJ

  def make_return(self, mat=None, ij=None, structures=None):
    if not hasattr(self, 'structures'):
      if structures is None:
        raise ConnectorError(f'Please call "self.check" first to get synaptic '
                             f'structures required. Error in {str(self)}')
      else:
        self.structures = _check_structures(structures)

    if mat is not None:
      _check_connections(mat)
    if ij is not None:
      _check_connections(ij[0])
      _check_connections(ij[1])

    if len(self.structures) == 1 and self.structures[0] == CONN_MAT:
      if mat is None:
        mat = ij2mat(i=ij[0], j=ij[1], num_pre=self.pre_num, num_post=self.post_num)
      return math.asarray(mat, dtype=MAT_DTYPE)
    else:
      all_data = dict()

      # check 'ij'
      if ij is None:
        if mat is None:
          raise ConnectorError(f'"mat" and "ij" are both none, please provide at least one of them.')
        ij = mat2ij(mat)
      all_data[PRE_IDS] = math.asarray(ij[0], dtype=IDX_DTYPE)
      all_data[POST_IDS] = math.asarray(ij[1], dtype=IDX_DTYPE)

      # check 'mat'
      if CONN_MAT in self.structures:
        if mat is None:
          mat = ij2mat(i=ij[0], j=ij[1], num_pre=self.pre_num, num_post=self.post_num)
      all_data[CONN_MAT] = math.asarray(mat, dtype=MAT_DTYPE)

      # names of the needed structures
      if PRE_SLICE in self.structures:
        r = pre_slice(i=ij[0], j=ij[1], num_pre=self.pre_num)
        all_data[PRE_IDS] = math.asarray(r[0], dtype=IDX_DTYPE)
        all_data[POST_IDS] = math.asarray(r[1], dtype=IDX_DTYPE)
        all_data[PRE_SLICE] = math.asarray(r[2], dtype=IDX_DTYPE)

      elif POST_SLICE in self.structures:
        r = post_slice(i=ij[0], j=ij[1], num_post=self.post_num)
        all_data[PRE_IDS] = math.asarray(r[0], dtype=IDX_DTYPE)
        all_data[POST_IDS] = math.asarray(r[1], dtype=IDX_DTYPE)
        all_data[POST_SLICE] = math.asarray(r[2], dtype=IDX_DTYPE)

      for n in self.structures:
        if n in [PRE_SLICE, POST_SLICE, PRE_IDS, POST_IDS, CONN_MAT]:
          continue
        elif n == PRE2POST:
          all_data[PRE2POST] = pre2post(i=ij[0], j=ij[1], num_pre=self.pre_num)
        elif n == PRE2POST_MAT:
          all_data[PRE2POST_MAT] = pre2post_mat(i=ij[0], j=ij[1], num_pre=self.pre_num)
        elif n == PRE2SYN:
          all_data[PRE2SYN] = pre2syn(i=ij[0], num_pre=self.pre_num)
        elif n == POST2PRE:
          all_data[POST2PRE] = post2pre(i=ij[0], j=ij[1], num_post=self.post_num)
        elif n == POST2PRE_MAT:
          all_data[POST2PRE_MAT] = post2pre_mat(i=ij[0], j=ij[1], num_post=self.post_num)
        elif n == POST2SYN:
          all_data[POST2SYN] = post2syn(j=ij[1], num_post=self.post_num)
        else:
          raise ConnectorError(f'Unknown synaptic structures "{n}", only support {SUPPORTED_SYN_STRUCTURE}.')

      # data of the needed structures
      if len(self.structures) == 1:
        return all_data[self.structures[0]]
      else:
        return tuple([all_data[n] for n in self.structures])

  def requires(self, *structures):
    return self.require(*structures)

  def require(self, *structures):
    raise NotImplementedError


class OneEndConnector(TwoEndConnector):
  """Synaptical connector to build synapse connections within a population of neurons."""

  def __call__(self, pre_size, post_size=None):
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


def ij2mat(i, j, num_pre=None, num_post=None):
  """Convert i-j connection to matrix connection.

  Parameters
  ----------
  i : list, np.ndarray
      Pre-synaptic neuron index.
  j : list, np.ndarray
      Post-synaptic neuron index.
  num_pre : int, optional
      The number of the pre-synaptic neurons.
  num_post : int, optional
      The number of the post-synaptic neurons.

  Returns
  -------
  conn_mat : np.ndarray
      A 2D ndarray connectivity matrix.
  """
  if len(i) != len(j):
    raise ConnectorError('"i" and "j" must be the equal length.')
  if num_pre is None:
    logger.warning('"num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)
  if num_post is None:
    logger.warning('"num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)
  conn_mat = np.zeros((num_pre, num_post), dtype=MAT_DTYPE)
  conn_mat[i, j] = True
  return math.asarray(conn_mat, dtype=MAT_DTYPE)


def mat2ij(conn_mat: np.ndarray):
  """Get the i-j connections from connectivity matrix.

  Parameters
  ----------
  conn_mat : np.ndarray
      Connectivity matrix with `(num_pre, num_post)` shape.

  Returns
  -------
  conn_tuple : tuple
      (Pre-synaptic neuron indexes,
       post-synaptic neuron indexes).
  """
  if len(np.shape(conn_mat)) != 2:
    raise ConnectorError('Connectivity matrix must be in the '
                         'shape of (num_pre, num_post).')
  pre_ids, post_ids = np.where(conn_mat > 0)
  return math.asarray(pre_ids, dtype=IDX_DTYPE), math.asarray(post_ids, dtype=IDX_DTYPE)


def pre2post(i: np.ndarray, j: np.ndarray, num_pre=None):
  """Get pre2post connections from `i` and `j` indexes.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_pre : int, None
      The number of the pre-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of pre2post.
  """
  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_pre is None:
    logger.warning('"num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)

  pre2post_list = [[] for _ in range(num_pre)]
  for pre_id, post_id in zip(i, j):
    pre2post_list[pre_id].append(post_id)
  pre2post_list = [math.array(l, dtype=IDX_DTYPE) for l in pre2post_list]

  return pre2post_list


def pre2post_mat(i: np.ndarray, j: np.ndarray, num_pre=None):
  """Get pre2post_mat connections from `i` and `j` indexes.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_pre : int, None
      The number of the pre-synaptic neurons.

  Returns
  -------
  conn : math.ndarray
      The conn of pre2post.
  """
  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_pre is None:
    logger.warning('"num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)

  pre2post_list = [[] for _ in range(num_pre)]
  for pre_id, post_id in zip(i, j):
    pre2post_list[pre_id].append(post_id)
  max_len = max([len(l) for l in pre2post_list])
  pre2post_list = [np.pad(np.array(l, dtype=IDX_DTYPE),
                          (0, max_len - len(l)),
                          constant_values=-1)
                   for l in pre2post_list]
  return math.asarray(np.array(pre2post_list), dtype=IDX_DTYPE)


def post2pre(i: np.ndarray, j: np.ndarray, num_post=None):
  """Get post2pre connections from `i` and `j` indexes.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_post : int, None
      The number of the post-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of post2pre.
  """

  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_post is None:
    logger.warning('WARNING: "num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)

  post2pre_list = [[] for _ in range(num_post)]
  for pre_id, post_id in zip(i, j):
    post2pre_list[post_id].append(pre_id)
  post2pre_list = [math.array(l, dtype=IDX_DTYPE) for l in post2pre_list]
  return post2pre_list


def post2pre_mat(i: np.ndarray, j: np.ndarray, num_post=None):
  """Get post2pre_mat connections from `i` and `j` indexes.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_post : int, None
      The number of the post-synaptic neurons.

  Returns
  -------
  conn : math.ndarray
      The conn of post2pre.
  """

  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_post is None:
    logger.warning('WARNING: "num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)

  post2pre_list = [[] for _ in range(num_post)]
  for pre_id, post_id in zip(i, j):
    post2pre_list[post_id].append(pre_id)

  max_len = max([len(l) for l in post2pre_list])
  post2pre_list = [np.pad(np.array(l, dtype=IDX_DTYPE),
                          (0, max_len - len(l)),
                          constant_values=-1)
                   for l in post2pre_list]
  return math.asarray(np.array(post2pre_list, dtype=IDX_DTYPE), dtype=IDX_DTYPE)


def pre2syn(i: np.ndarray, num_pre=None):
  """Get pre2syn connections from `i` and `j` indexes.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  num_pre : int
      The number of the pre-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of pre2syn.
  """
  if num_pre is None:
    logger.warning('WARNING: "num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)

  pre2syn_list = [[] for _ in range(num_pre)]
  for syn_id, pre_id in enumerate(i):
    pre2syn_list[pre_id].append(syn_id)
  pre2syn_list = [math.array(l, dtype=IDX_DTYPE) for l in pre2syn_list]
  return pre2syn_list


def post2syn(j: np.ndarray, num_post=None):
  """Get post2syn connections from `i` and `j` indexes.

  Parameters
  ----------
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_post : int
      The number of the post-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of post2syn.
  """
  if num_post is None:
    logger.warning('WARNING: "num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)

  post2syn_list = [[] for _ in range(num_post)]
  for syn_id, post_id in enumerate(j):
    post2syn_list[post_id].append(syn_id)
  post2syn_list = [math.array(l, dtype=IDX_DTYPE) for l in post2syn_list]
  return post2syn_list


def pre_slice(i: np.ndarray, j: np.ndarray, num_pre=None):
  """Get post slicing connections by pre-synaptic ids.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_pre : int
      The number of the pre-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of post2syn.
  """
  # check
  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_pre is None:
    logger.warning('WARNING: "num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)

  # pre2post connection
  pre2post_list = [[] for _ in range(num_pre)]
  for pre_id, post_id in zip(i, j):
    pre2post_list[pre_id].append(post_id)
  pre_ids, post_ids = [], []
  for pre_i, posts in enumerate(pre2post_list):
    post_ids.extend(posts)
    pre_ids.extend([pre_i] * len(posts))
  post_ids = np.array(post_ids, dtype=IDX_DTYPE)
  pre_ids = np.array(pre_ids, dtype=IDX_DTYPE)

  # pre2post slicing
  slicing = []
  start = 0
  for posts in pre2post_list:
    end = start + len(posts)
    slicing.append([start, end])
    start = end
  slicing = np.array(slicing, dtype=IDX_DTYPE)

  post_ids = math.array(post_ids, dtype=IDX_DTYPE)
  pre_ids = math.array(pre_ids, dtype=IDX_DTYPE)
  slicing = math.array(slicing, dtype=IDX_DTYPE)

  return pre_ids, post_ids, slicing


def post_slice(i: np.ndarray, j: np.ndarray, num_post=None):
  """Get pre slicing connections by post-synaptic ids.

  Parameters
  ----------
  i : list, np.ndarray
      The pre-synaptic neuron indexes.
  j : list, np.ndarray
      The post-synaptic neuron indexes.
  num_post : int
      The number of the post-synaptic neurons.

  Returns
  -------
  conn : list
      The conn list of post2syn.
  """
  if len(i) != len(j):
    raise ConnectorError('The length of "i" and "j" must be the same.')
  if num_post is None:
    logger.warning('WARNING: "num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)

  # post2pre connection
  post2pre_list = [[] for _ in range(num_post)]
  for pre_id, post_id in zip(i, j):
    post2pre_list[post_id].append(pre_id)
  pre_ids, post_ids = [], []
  for _post_id, _pre_ids in enumerate(post2pre_list):
    pre_ids.extend(_pre_ids)
    post_ids.extend([_post_id] * len(_pre_ids))
  post_ids = np.array(post_ids, dtype=IDX_DTYPE)
  pre_ids = np.array(pre_ids, dtype=IDX_DTYPE)

  # post2pre slicing
  slicing = []
  start = 0
  for pres in post2pre_list:
    end = start + len(pres)
    slicing.append([start, end])
    start = end
  slicing = np.array(slicing, dtype=IDX_DTYPE)

  post_ids = math.array(post_ids, dtype=IDX_DTYPE)
  pre_ids = math.array(pre_ids, dtype=IDX_DTYPE)
  slicing = math.array(slicing, dtype=IDX_DTYPE)

  return pre_ids, post_ids, slicing
