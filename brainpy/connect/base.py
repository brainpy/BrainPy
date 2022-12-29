# -*- coding: utf-8 -*-

import abc
from typing import Union, List, Tuple

import jax.numpy as jnp
import numpy as onp

from brainpy import tools, math as bm
from brainpy.errors import ConnectorError

__all__ = [
  # the connection types
  'CONN_MAT',
  'PRE_IDS', 'POST_IDS',
  'PRE2POST', 'POST2PRE',
  'PRE2SYN', 'POST2SYN',
  'SUPPORTED_SYN_STRUCTURE',

  # the connection dtypes
  'set_default_dtype', 'MAT_DTYPE', 'IDX_DTYPE',

  # brainpy_object class
  'Connector', 'TwoEndConnector', 'OneEndConnector',

  # methods
  'mat2coo', 'mat2csc', 'mat2csr',
  'csr2csc', 'csr2mat', 'csr2coo',
  'coo2csr', 'coo2csc', 'coo2mat',
]

CONN_MAT = 'conn_mat'
PRE_IDS = 'pre_ids'
POST_IDS = 'post_ids'
PRE2POST = 'pre2post'
POST2PRE = 'post2pre'
PRE2SYN = 'pre2syn'
POST2SYN = 'post2syn'
PRE_SLICE = 'pre_slice'
POST_SLICE = 'post_slice'
COO = 'coo'
CSR = 'csr'
CSC = 'csc'

SUPPORTED_SYN_STRUCTURE = [CONN_MAT,
                           PRE_IDS, POST_IDS,
                           PRE2POST, POST2PRE,
                           PRE2SYN, POST2SYN,
                           PRE_SLICE, POST_SLICE,
                           COO, CSR, CSC]

MAT_DTYPE = jnp.bool_
IDX_DTYPE = jnp.int32


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
  Array([[False,  True, False, False],
         [ True, False,  True, False],
         [False,  True, False,  True],
         [False, False,  True, False]], dtype=bool)
  >>> bp.conn.set_default_dtype(mat_dtype=np.float32)
  >>> conn = bp.conn.GridFour()(4, 4)
  >>> conn.require('conn_mat')
  Array([[0., 1., 0., 0.],
         [1., 0., 1., 0.],
         [0., 1., 0., 1.],
         [0., 0., 1., 0.]], dtype=float32)

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
  """Synaptic connector to build connections between two neuron groups.

  If users want to customize their `Connector`, there are two ways:

  1. Implementing ``build_conn(self)`` function, which returns one of
     the connection data ``csr`` (CSR sparse data, a tuple of <post_ids, inptr>),
     ``coo`` (COO sparse data, a tuple of <pre_ids, post_ids>), or ``mat``
     (a binary connection matrix). For instance,

     .. code-block:: python

        import brainpy as bp
        class MyConnector(bp.conn.TwoEndConnector):
          def build_conn(self):
            return dict(csr=, mat=, coo=)

  2. Implementing functions ``build_mat()``, ``build_csr()``, and
     ``build_coo()``. Users can provide all three functions, or one of them.

     .. code-block:: python

        import brainpy as bp
        class MyConnector(bp.conn.TwoEndConnector):
          def build_mat(self, ):
            return conn_matrix

          def build_csr(self, ):
            return post_ids, inptr

          def build_coo(self, ):
            return pre_ids, post_ids

  """

  def __init__(
      self,
      pre: Union[int, Tuple[int, ...]] = None,
      post: Union[int, Tuple[int, ...]] = None,
  ):
    self.pre_size = None
    self.post_size = None
    self.pre_num = None
    self.post_num = None
    if pre is not None:
      if isinstance(pre, int):
        pre = (pre,)
      else:
        pre = tuple(pre)
      self.pre_size = pre
      self.pre_num = tools.size2num(self.pre_size)
    if post is not None:
      if isinstance(post, int):
        post = (post,)
      else:
        post = tuple(post)
      self.post_size = post
      self.post_num = tools.size2num(self.post_size)
    

  def __repr__(self):
    return self.__class__.__name__

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
    """Reset connection attributes.

    Parameters
    ----------
    pre_size : int, tuple of int, list of int
        The size of the pre-synaptic group.
    post_size : int, tuple of int, list of int
        The size of the post-synaptic group.
    """
    self.__call__(pre_size, post_size)

  @property
  def is_version2_style(self):
    if ((hasattr(self.build_coo, 'not_customized') and self.build_coo.not_customized) and
        (hasattr(self.build_csr, 'not_customized') and self.build_csr.not_customized) and
        (hasattr(self.build_mat, 'not_customized') and self.build_mat.not_customized)):
      return False
    else:
      return True

  def _check(self, structures: Union[Tuple, List, str]):
    # check synaptic structures
    if isinstance(structures, str):
      structures = [structures]
    if structures is None or len(structures) == 0:
      raise ConnectorError('No synaptic structure is received.')
    for n in structures:
      if n not in SUPPORTED_SYN_STRUCTURE:
        raise ConnectorError(f'Unknown synapse structure "{n}". '
                             f'Only {SUPPORTED_SYN_STRUCTURE} is supported.')

  def _return_by_mat(self, structures, mat, all_data: dict):
    assert mat.ndim == 2
    if (CONN_MAT in structures) and (CONN_MAT not in all_data):
      all_data[CONN_MAT] = bm.asarray(mat, dtype=MAT_DTYPE)

    if len([s for s in structures
            if s not in [CONN_MAT]]) > 0:
      ij = mat2coo(mat)
      self._return_by_coo(structures, coo=ij, all_data=all_data)

  def _return_by_csr(self, structures, csr: tuple, all_data: dict):
    indices, indptr = csr
    np = onp if isinstance(indices, onp.ndarray) else bm
    assert self.pre_num == indptr.size - 1

    if (CONN_MAT in structures) and (CONN_MAT not in all_data):
      conn_mat = csr2mat((indices, indptr), self.pre_num, self.post_num)
      all_data[CONN_MAT] = bm.asarray(conn_mat, dtype=MAT_DTYPE)

    if (PRE_IDS in structures) and (PRE_IDS not in all_data):
      pre_ids = np.repeat(np.arange(self.pre_num), np.diff(indptr))
      all_data[PRE_IDS] = bm.asarray(pre_ids, dtype=IDX_DTYPE)

    if (POST_IDS in structures) and (POST_IDS not in all_data):
      all_data[POST_IDS] = bm.asarray(indices, dtype=IDX_DTYPE)

    if (COO in structures) and (COO not in all_data):
      pre_ids = np.repeat(np.arange(self.pre_num), np.diff(indptr))
      all_data[COO] = (bm.asarray(pre_ids, dtype=IDX_DTYPE),
                       bm.asarray(indices, dtype=IDX_DTYPE))

    if (PRE2POST in structures) and (PRE2POST not in all_data):
      all_data[PRE2POST] = (bm.asarray(indices, dtype=IDX_DTYPE),
                            bm.asarray(indptr, dtype=IDX_DTYPE))

    if (CSR in structures) and (CSR not in all_data):
      all_data[CSR] = (bm.asarray(indices, dtype=IDX_DTYPE),
                       bm.asarray(indptr, dtype=IDX_DTYPE))

    if (POST2PRE in structures) and (POST2PRE not in all_data):
      indc, indptrc = csr2csc((indices, indptr), self.post_num)
      all_data[POST2PRE] = (bm.asarray(indc, dtype=IDX_DTYPE),
                            bm.asarray(indptrc, dtype=IDX_DTYPE))

    if (CSC in structures) and (CSC not in all_data):
      indc, indptrc = csr2csc((indices, indptr), self.post_num)
      all_data[CSC] = (bm.asarray(indc, dtype=IDX_DTYPE),
                       bm.asarray(indptrc, dtype=IDX_DTYPE))

    if (PRE2SYN in structures) and (PRE2SYN not in all_data):
      syn_seq = np.arange(indices.size, dtype=IDX_DTYPE)
      all_data[PRE2SYN] = (bm.asarray(syn_seq, dtype=IDX_DTYPE),
                           bm.asarray(indptr, dtype=IDX_DTYPE))

    if (POST2SYN in structures) and (POST2SYN not in all_data):
      syn_seq = np.arange(indices.size, dtype=IDX_DTYPE)
      _, indptrc, syn_seqc = csr2csc((indices, indptr), self.post_num, syn_seq)
      all_data[POST2SYN] = (bm.asarray(syn_seqc, dtype=IDX_DTYPE),
                            bm.asarray(indptrc, dtype=IDX_DTYPE))

  def _return_by_coo(self, structures, coo: tuple, all_data: dict):
    pre_ids, post_ids = coo

    if (CONN_MAT in structures) and (CONN_MAT not in all_data):
      all_data[CONN_MAT] = bm.asarray(coo2mat(coo, self.pre_num, self.post_num), dtype=MAT_DTYPE)

    if (PRE_IDS in structures) and (PRE_IDS not in all_data):
      all_data[PRE_IDS] = bm.asarray(pre_ids, dtype=IDX_DTYPE)

    if (POST_IDS in structures) and (POST_IDS not in all_data):
      all_data[POST_IDS] = bm.asarray(post_ids, dtype=IDX_DTYPE)

    if (COO in structures) and (COO not in all_data):
      all_data[COO] = (bm.asarray(pre_ids, dtype=IDX_DTYPE),
                       bm.asarray(post_ids, dtype=IDX_DTYPE))

    if CSC in structures and CSC not in all_data:
      csc = coo2csc(coo, self.post_num)
      all_data[CSC] = (bm.asarray(csc[0], dtype=IDX_DTYPE),
                       bm.asarray(csc[1], dtype=IDX_DTYPE))

    if POST2PRE in structures and POST2PRE not in all_data:
      csc = coo2csc(coo, self.post_num)
      all_data[POST2PRE] = (bm.asarray(csc[0], dtype=IDX_DTYPE),
                            bm.asarray(csc[1], dtype=IDX_DTYPE))

    if (len([s for s in structures
             if s not in [CONN_MAT, PRE_IDS, POST_IDS,
                          COO, CSC, POST2PRE]]) > 0):
      csr = coo2csr(coo, self.pre_num)
      self._return_by_csr(structures, csr=csr, all_data=all_data)

  def _make_returns(self, structures, conn_data):
    """Make the desired synaptic structures and return them.
    """
    csr = None
    mat = None
    coo = None
    if isinstance(conn_data, dict):
      csr = conn_data.get('csr', None)
      mat = conn_data.get('mat', None)
      coo = conn_data.get('coo', None) or conn_data.get('ij', None)
    elif isinstance(conn_data, tuple):
      if conn_data[0] == 'csr':
        csr = conn_data[1]
      elif conn_data[0] == 'mat':
        mat = conn_data[1]
      elif conn_data[0] in ['coo', 'ij']:
        coo = conn_data[1]
      else:
        raise ConnectorError(f'Must provide one of "csr", "mat" or "coo". Got "{conn_data[0]}" instead.')
    else:
      raise ConnectorError('Unknown type')

    # checking
    if (csr is None) and (mat is None) and (coo is None):
      raise ConnectorError('Must provide one of "csr", "mat" or "coo".')
    structures = (structures,) if isinstance(structures, str) else structures
    assert isinstance(structures, (tuple, list))

    all_data = dict()
    # "csr" structure
    if csr is not None:
      if (PRE2POST in structures) and (PRE2POST not in all_data):
        all_data[PRE2POST] = (bm.asarray(csr[0], dtype=IDX_DTYPE),
                              bm.asarray(csr[1], dtype=IDX_DTYPE))
      self._return_by_csr(structures, csr=csr, all_data=all_data)

    # "mat" structure
    if mat is not None:
      assert mat.ndim == 2
      if (CONN_MAT in structures) and (CONN_MAT not in all_data):
        all_data[CONN_MAT] = bm.asarray(mat, dtype=MAT_DTYPE)
      self._return_by_mat(structures, mat=mat, all_data=all_data)

    # "coo" structure
    if coo is not None:
      if (PRE_IDS in structures) and (PRE_IDS not in structures):
        all_data[PRE_IDS] = bm.asarray(coo[0], dtype=IDX_DTYPE)
      if (POST_IDS in structures) and (POST_IDS not in structures):
        all_data[POST_IDS] = bm.asarray(coo[1], dtype=IDX_DTYPE)
      self._return_by_coo(structures, coo=coo, all_data=all_data)

    # return
    if len(structures) == 1:
      return all_data[structures[0]]
    else:
      return tuple([all_data[n] for n in structures])

  def require(self, *structures):
    """Require all the connection data needed.

    Examples
    --------

    >>> import brainpy as bp
    >>> conn = bp.connect.FixedProb(0.1)
    >>> mat = conn.require(10, 20, 'conn_mat')
    >>> mat.shape
    (10, 20)
    """

    if len(structures) > 0:
      pre_size = None
      post_size = None
      if not isinstance(structures[0], str):
        pre_size = structures[0]
        structures = structures[1:]
        if len(structures) > 0:
          if not isinstance(structures[0], str):
            post_size = structures[0]
            structures = structures[1:]
      if pre_size is not None:
        self.__call__(pre_size, post_size)
    else:
      return tuple()

    if self.pre_num is None or self.post_num is None:
      raise ConnectorError(f'self.pre_num or self.post_num is not defined. '
                           f'Please use "self.require(pre_size, post_size, DATA1, DATA2, ...)" ')

    _has_coo_imp = not hasattr(self.build_coo, 'not_customized')
    _has_csr_imp = not hasattr(self.build_csr, 'not_customized')
    _has_mat_imp = not hasattr(self.build_mat, 'not_customized')

    self._check(structures)
    if (_has_coo_imp or _has_csr_imp or _has_mat_imp):
      if len(structures) == 1:
        if PRE2POST in structures and _has_csr_imp:
          r = self.build_csr()
          return bm.asarray(r[0], dtype=IDX_DTYPE), bm.asarray(r[1], dtype=IDX_DTYPE)
        elif CSR in structures and _has_csr_imp:
          r = self.build_csr()
          return bm.asarray(r[0], dtype=IDX_DTYPE), bm.asarray(r[1], dtype=IDX_DTYPE)
        elif CONN_MAT in structures and _has_mat_imp:
          return bm.asarray(self.build_mat(), dtype=MAT_DTYPE)
        elif PRE_IDS in structures and _has_coo_imp:
          return bm.asarray(self.build_coo()[0], dtype=IDX_DTYPE)
        elif POST_IDS in structures and _has_coo_imp:
          return bm.asarray(self.build_coo()[1], dtype=IDX_DTYPE)
        elif COO in structures and not _has_coo_imp:
          return bm.asarray(self.build_coo(), dtype=IDX_DTYPE)

      elif len(structures) == 2:
        if (PRE_IDS in structures and POST_IDS in structures and _has_coo_imp):
          r = self.build_coo()
          if structures[0] == PRE_IDS:
            return bm.asarray(r[0], dtype=IDX_DTYPE), bm.asarray(r[1], dtype=IDX_DTYPE)
          else:
            return bm.asarray(r[1], dtype=IDX_DTYPE), bm.asarray(r[0], dtype=IDX_DTYPE)

        if ((CSR in structures or PRE2POST in structures)
            and _has_csr_imp and COO in structures and _has_coo_imp):
          csr = self.build_csr()
          csr = (bm.asarray(csr[0], dtype=IDX_DTYPE), bm.asarray(csr[1], dtype=IDX_DTYPE))
          coo = self.build_coo()
          coo = (bm.asarray(coo[0], dtype=IDX_DTYPE), bm.asarray(coo[1], dtype=IDX_DTYPE))
          if structures[0] == COO:
            return coo, csr
          else:
            return csr, coo

        if ((CSR in structures or PRE2POST in structures)
            and _has_csr_imp and CONN_MAT in structures and _has_mat_imp):
          csr = self.build_csr()
          csr = (bm.asarray(csr[0], dtype=IDX_DTYPE), bm.asarray(csr[1], dtype=IDX_DTYPE))
          mat = bm.asarray(self.build_mat(), dtype=MAT_DTYPE)
          if structures[0] == CONN_MAT:
            return mat, csr
          else:
            return csr, mat

        if (COO in structures and _has_coo_imp and CONN_MAT in structures and _has_mat_imp):
          coo = self.build_coo()
          coo = (bm.asarray(coo[0], dtype=IDX_DTYPE), bm.asarray(coo[1], dtype=IDX_DTYPE))
          mat = bm.asarray(self.build_mat(), dtype=MAT_DTYPE)
          if structures[0] == COO:
            return coo, mat
          else:
            return mat, coo

      conn_data = dict(csr=None, ij=None, mat=None)
      if _has_coo_imp:
        conn_data['coo'] = self.build_coo()
        # if (CSR in structures or PRE2POST in structures) and _has_csr_imp:
        #   conn_data['csr'] = self.build_csr()
        # if CONN_MAT in structures and _has_mat_imp:
        #   conn_data['mat'] = self.build_mat()
      elif _has_csr_imp:
        conn_data['csr'] = self.build_csr()
        # if COO in structures and _has_coo_imp:
        #   conn_data['coo'] = self.build_coo()
        # if CONN_MAT in structures and _has_mat_imp:
        #   conn_data['mat'] = self.build_mat()
      elif _has_mat_imp:
        conn_data['mat'] = self.build_mat()
        # if COO in structures and _has_coo_imp:
        #   conn_data['coo'] = self.build_coo()
        # if (CSR in structures or PRE2POST in structures) and _has_csr_imp:
        #   conn_data['csr'] = self.build_csr()
      else:
        raise ValueError

    else:
      conn_data = self.build_conn()
    return self._make_returns(structures, conn_data)

  def requires(self, *structures):
    """Require all the connection data needed."""
    return self.require(*structures)

  @tools.not_customized
  def build_conn(self):
    """build connections with certain data type.

    If users want to customize their connections, please provide one
    of the following functions:

    - ``build_mat()``: build a matrix binary connection matrix.
    - ``build_csr()``: build a csr sparse connection data.
    - ``build_coo()``: build a coo sparse connection data.
    - ``build_conn()``: deprecated.

    Returns
    -------
    conn: tuple, dict
      A tuple with two elements: connection type (str) and connection data.
      For example: ``return 'csr', (ind, indptr)``
      Or a dict with three elements: csr, mat and coo. For example:
      ``return dict(csr=(ind, indptr), mat=None, coo=None)``
    """
    pass

  @tools.not_customized
  def build_mat(self):
    """Build a binary matrix connection data.


    If users want to customize their connections, please provide one
    of the following functions:

    - ``build_mat()``: build a matrix binary connection matrix.
    - ``build_csr()``: build a csr sparse connection data.
    - ``build_coo()``: build a coo sparse connection data.
    - ``build_conn()``: deprecated.

    Returns
    -------
    conn: Array
      A binary matrix with the shape ``(num_pre, num_post)``.
    """
    pass

  @tools.not_customized
  def build_csr(self):
    """Build a csr sparse connection data.

    Returns
    -------
    conn: tuple
      A tuple denoting the ``(indices, indptr)``.
    """
    pass

  @tools.not_customized
  def build_coo(self):
    """Build a coo sparse connection data.

    Returns
    -------
    conn: tuple
      A tuple denoting the ``(pre_ids, post_ids)``.
    """
    pass


class OneEndConnector(TwoEndConnector):
  """Synaptic connector to build synapse connections within a population of neurons."""

  def __init__(self, *args, **kwargs):
    super(OneEndConnector, self).__init__(*args, **kwargs)

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


def mat2csr(dense):
  """convert a dense matrix to (indices, indptr)."""
  np = onp if isinstance(dense, onp.ndarray) else bm
  pre_ids, post_ids = np.where(dense > 0)
  return coo2csr((pre_ids, post_ids), dense.shape[0])


def mat2coo(dense):
  np = onp if isinstance(dense, onp.ndarray) else bm
  pre_ids, post_ids = np.where(dense > 0)
  return np.asarray(pre_ids, dtype=IDX_DTYPE), np.asarray(post_ids, dtype=IDX_DTYPE)


def mat2csc(dense):
  np = onp if isinstance(dense, onp.ndarray) else bm
  pre_ids, post_ids = np.where(dense > 0)
  return coo2csr((post_ids, pre_ids), dense.shape[1])


def csr2mat(csr, num_pre, num_post):
  """convert (indices, indptr) to a dense matrix."""
  indices, indptr = csr
  np = onp if isinstance(indices, onp.ndarray) else bm
  d = np.zeros((num_pre, num_post), dtype=MAT_DTYPE)  # num_pre, num_post
  pre_ids = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
  d[pre_ids, indices] = True
  return d


def csr2csc(csr, post_num, data=None):
  """Convert csr to csc."""
  return coo2csc(csr2coo(csr), post_num, data)


def csr2coo(csr):
  np = onp if isinstance(csr[0], onp.ndarray) else bm
  indices, indptr = csr
  pre_ids = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
  return pre_ids, indices


def coo2mat(ij, num_pre, num_post):
  """convert (indices, indptr) to a dense matrix."""
  pre_ids, post_ids = ij
  np = onp if isinstance(pre_ids, onp.ndarray) else bm
  d = np.zeros((num_pre, num_post), dtype=MAT_DTYPE)  # num_pre, num_post
  d[pre_ids, post_ids] = True
  return d


def coo2csr(coo, num_pre):
  """convert pre_ids, post_ids to (indices, indptr) when'jax_platform_name' = 'gpu'"""
  pre_ids, post_ids = coo
  np = onp if isinstance(pre_ids, onp.ndarray) else bm

  sort_ids = np.argsort(pre_ids)
  post_ids = np.asarray(post_ids)
  post_ids = post_ids[sort_ids]
  indices = post_ids
  unique_pre_ids, pre_count = np.unique(pre_ids, return_counts=True)
  final_pre_count = np.zeros(num_pre, dtype=jnp.uint32)
  final_pre_count[unique_pre_ids] = pre_count
  indptr = final_pre_count.cumsum()
  indptr = np.insert(indptr, 0, 0)
  return np.asarray(indices, dtype=IDX_DTYPE), np.asarray(indptr, dtype=IDX_DTYPE)


def coo2csc(coo, post_num, data=None):
  """Convert csr to csc."""
  pre_ids, indices = coo
  np = onp if isinstance(indices, onp.ndarray) else bm

  # to maintain the original order of the elements with the same value
  sort_ids = np.argsort(indices)
  pre_ids_new = np.asarray(pre_ids[sort_ids], dtype=IDX_DTYPE)

  unique_post_ids, count = np.unique(indices, return_counts=True)
  post_count = np.zeros(post_num, dtype=IDX_DTYPE)
  post_count[unique_post_ids] = count

  indptr_new = post_count.cumsum()
  indptr_new = np.insert(indptr_new, 0, 0)
  indptr_new = np.asarray(indptr_new, dtype=IDX_DTYPE)

  if data is None:
    return pre_ids_new, indptr_new
  else:
    data_new = data[sort_ids]
    return pre_ids_new, indptr_new, data_new
