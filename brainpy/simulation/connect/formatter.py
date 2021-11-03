# -*- coding: utf-8 -*-

import logging

import numpy as np

from brainpy import math, errors

try:
  import numba as nb
except ModuleNotFoundError:
  nb = None

logger = logging.getLogger('brainpy.simulation.connect')

__all__ = [
  'ij2mat',
  'mat2ij',
  'pre2post',
  'post2pre',
  'pre2syn',
  'post2syn',
  'pre_slice',
  'post_slice',
]


def _numpy_backend():
  r = math.get_backend_name().startswith('numpy')
  return r and (nb is not None)


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
    raise errors.BrainPyError('"i" and "j" must be the equal length.')
  if num_pre is None:
    logger.warning('"num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)
  if num_post is None:
    logger.warning('"num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)
  conn_mat = np.zeros((num_pre, num_post), dtype=np.bool_)
  conn_mat[i, j] = True
  return math.asarray(conn_mat, dtype=math.bool_)


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
    raise errors.BrainPyError('Connectivity matrix must be in the '
                              'shape of (num_pre, num_post).')
  pre_ids, post_ids = np.where(conn_mat > 0)
  return math.asarray(pre_ids, dtype=math.int_), math.asarray(post_ids, dtype=math.int_)


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
    raise errors.BrainPyError('The length of "i" and "j" must be the same.')
  if num_pre is None:
    logger.warning('"num_pre" is not provided, the result may not be accurate.')
    num_pre = np.max(i)

  pre2post_list = [[] for _ in range(num_pre)]
  for pre_id, post_id in zip(i, j):
    pre2post_list[pre_id].append(post_id)
  pre2post_list = [math.array(l, dtype=np.int_) for l in pre2post_list]

  if _numpy_backend():
    pre2post_list_nb = nb.typed.List()
    for pre_id in range(num_pre):
      pre2post_list_nb.append(pre2post_list[pre_id])
    pre2post_list = math.Variable(math.asarray(pre2post_list, dtype=object))
    pre2post_list.value = pre2post_list_nb
  return pre2post_list


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
    raise errors.BrainPyError('The length of "i" and "j" must be the same.')
  if num_post is None:
    logger.warning('WARNING: "num_post" is not provided, the result may not be accurate.')
    num_post = np.max(j)

  post2pre_list = [[] for _ in range(num_post)]
  for pre_id, post_id in zip(i, j):
    post2pre_list[post_id].append(pre_id)
  post2pre_list = [math.array(l, dtype=math.int_) for l in post2pre_list]

  if _numpy_backend():
    post2pre_list_nb = nb.typed.List()
    for post_id in range(num_post):
      post2pre_list_nb.append(post2pre_list[post_id])
    post2pre_list = math.Variable(math.asarray(post2pre_list, dtype=object),
                                  type='connection')
    post2pre_list.value = post2pre_list_nb
  return post2pre_list


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
  pre2syn_list = [math.array(l, dtype=math.int_) for l in pre2syn_list]

  if _numpy_backend():
    pre2syn_list_nb = nb.typed.List()
    for pre_ids in pre2syn_list:
      pre2syn_list_nb.append(pre_ids)
    pre2syn_list = math.Variable(math.asarray(pre2syn_list, dtype=object),
                                 type='connection')
    pre2syn_list.value = pre2syn_list_nb
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
  post2syn_list = [math.array(l, dtype=math.int_) for l in post2syn_list]

  if _numpy_backend():
    post2syn_list_nb = nb.typed.List()
    for pre_ids in post2syn_list:
      post2syn_list_nb.append(pre_ids)
    post2syn_list = math.Variable(math.asarray(post2syn_list, dtype=object),
                                  type='connection')
    post2syn_list.value = post2syn_list_nb

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
    raise errors.BrainPyError('The length of "i" and "j" must be the same.')
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
  post_ids = np.array(post_ids, dtype=np.int_)
  pre_ids = np.array(pre_ids, dtype=np.int_)

  # pre2post slicing
  slicing = []
  start = 0
  for posts in pre2post_list:
    end = start + len(posts)
    slicing.append([start, end])
    start = end
  slicing = np.array(slicing, dtype=np.int_)

  post_ids = math.array(post_ids, dtype=math.int_)
  pre_ids = math.array(pre_ids, dtype=math.int_)
  slicing = math.array(slicing, dtype=math.int_)

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
    raise errors.BrainPyError('The length of "i" and "j" must be the same.')
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
  post_ids = np.array(post_ids, dtype=np.int_)
  pre_ids = np.array(pre_ids, dtype=np.int_)

  # post2pre slicing
  slicing = []
  start = 0
  for pres in post2pre_list:
    end = start + len(pres)
    slicing.append([start, end])
    start = end
  slicing = np.array(slicing, dtype=np.int_)

  post_ids = math.array(post_ids, dtype=math.int_)
  pre_ids = math.array(pre_ids, dtype=math.int_)
  slicing = math.array(slicing, dtype=math.int_)

  return pre_ids, post_ids, slicing
