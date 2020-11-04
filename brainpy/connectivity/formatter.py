# -*- coding: utf-8 -*-

from .. import numpy as np
from .. import profile
from ..errors import ModelUseError

try:
    import numba as nb
except ImportError:
    nb = None

__all__ = [
    'mat2ij',
    'pre2post',
    'pre2syn',
    'post2syn',
    'post2pre',
    'post_slicing_by_pre',
    'pre_slicing_by_post',
]


def mat2ij(conn_mat):
    """Get the i-j connections from connectivity matrix.

    Parameters
    ----------
    conn_mat : numpy.ndarray
        Connectivity matrix with `(num_pre, num_post)` shape.

    Returns
    -------
    conn_tuple : tuple
        (Pre-synaptic neuron indexes,
         post-synaptic neuron indexes).
    """
    conn_mat = np.asarray(conn_mat)
    try:
        assert np.ndim(conn_mat) == 2
    except AssertionError:
        raise ModelUseError('Connectivity matrix must be in the shape of (num_pre, num_post).')
    pre_ids, post_ids = np.where(conn_mat > 0)
    return pre_ids, post_ids


def pre2post(i, j, num_pre):
    """Get pre2post connections from `i` and `j` indexes.

    Parameters
    ----------
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int, None
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of pre2post.
    """
    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('The length of "i" and "j" must be the same.')

    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)

    if profile.is_numba_bk():
        pre2post_list_nb = nb.typed.List()
        for pre_id in range(num_pre):
            pre2post_list_nb.append(np.int64(pre2post_list[pre_id]))
        pre2post_list = pre2post_list_nb
    return pre2post_list


def post2pre(i, j, num_post):
    """Get post2pre connections from `i` and `j` indexes.

    Parameters
    ----------
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int, None
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of post2pre.
    """

    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('The length of "i" and "j" must be the same.')

    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)

    if profile.is_numba_bk():
        post2pre_list_nb = nb.typed.List()
        for post_id in range(num_post):
            post2pre_list_nb.append(np.int64(post2pre_list[post_id]))
        post2pre_list = post2pre_list_nb
    return post2pre_list


def pre2syn(i, num_pre):
    """Get pre2syn connections from `i` and `j` indexes.

    Parameters
    ----------
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    num_pre : int
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of pre2syn.
    """
    pre2syn_list = [[] for _ in range(num_pre)]
    for syn_id, pre_id in enumerate(i):
        pre2syn_list[pre_id].append(syn_id)

    if profile.is_numba_bk():
        pre2syn_list_nb = nb.typed.List()
        for pre_ids in pre2syn_list:
            pre2syn_list_nb.append(np.int64(pre_ids))
        pre2syn_list = pre2syn_list_nb
    return pre2syn_list


def post2syn(j, num_post):
    """Get post2syn connections from `i` and `j` indexes.

    Parameters
    ----------
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of post2syn.
    """
    post2syn_list = [[] for _ in range(num_post)]
    for syn_id, post_id in enumerate(j):
        post2syn_list[post_id].append(syn_id)

    if profile.is_numba_bk():
        post2syn_list_nb = nb.typed.List()
        for pre_ids in post2syn_list:
            post2syn_list_nb.append(np.int64(pre_ids))
        post2syn_list = post2syn_list_nb
    return post2syn_list


def post_slicing_by_pre(i, j, num_pre):
    """Get post slicing connections by pre-synaptic ids.

    Parameters
    ----------
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of post2syn.
    """
    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('The length of "i" and "j" must be the same.')

    # pre2post connection
    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)
    post_indices = np.concatenate(pre2post_list)
    post_indices = np.asarray(post_indices, dtype=np.int_)

    # pre2post slicing
    post_slicing = []
    start = 0
    for post_ids in pre2post_list:
        end = start + len(post_ids)
        post_slicing.append([start, end])
        start = end
    post_slicing = np.asarray(post_slicing, dtype=np.int_)

    return post_indices, post_slicing


def pre_slicing_by_post(i, j, num_post):
    """Get pre slicing connections by post-synaptic ids.

    Parameters
    ----------
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The conn list of post2syn.
    """
    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('The length of "i" and "j" must be the same.')

    # post2pre connection
    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)
    pre_indices = np.concatenate(post2pre_list)
    pre_indices = np.asarray(pre_indices, dtype=np.int_)

    # post2pre slicing
    pre_slicing = []
    start = 0
    for pre_ids in post2pre_list:
        end = start + len(pre_ids)
        pre_slicing.append([start, end])
        start = end
    pre_slicing = np.asarray(pre_slicing, dtype=np.int_)

    return pre_indices, pre_slicing
