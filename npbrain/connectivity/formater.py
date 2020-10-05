# -*- coding: utf-8 -*-

from .. import _numpy as np
from .. import profile

try:
    import numba as nb
except ImportError:
    nb = None

__all__ = ['mat2ij', 'pre2post', 'pre2syn', 'post2syn', 'post2pre']


def mat2ij(conn_mat):
    """Get the i-j connections from connectivity matrix.

    Parameters
    ----------
    conn_mat : numpy.ndarray
        Connectivity matrix with `(num_pre x num_post)` shape.

    Returns
    -------
    conn_tuple : tuple
        (Pre-synaptic neuron indexes,
         post-synaptic neuron indexes).
    """
    pre_ids = []
    post_ids = []
    num_pre = conn_mat.shape[0]
    for pre_idx in range(num_pre):
        post_idxs = np.where(conn_mat[pre_idx] > 0)[0]
        post_ids.extend(post_idxs)
        pre_ids.extend([pre_idx] * len(post_idxs))
    pre_ids = np.array(pre_ids, dtype=np.uint64)
    post_ids = np.array(post_ids, dtype=np.uint64)
    return pre_ids, post_ids


def pre2post(i, j, num_pre):
    """Get pre2post connections from `i` and `j` indexes.

    Parameters
    ----------
    i : a_list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : a_list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int, None
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The conn a_list of pre2post.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        pre2post_list = nb.typed.List()

        for pre_i in range(num_pre):
            index = np.where(i == pre_i)[0]
            post_idx = j[index]
            pre2post_list.append(np.uint64(post_idx))
    else:
        pre2post_list = []

        for pre_i in range(num_pre):
            index = np.where(i == pre_i)[0]
            post_idx = j[index]
            pre2post_list.append(np.uint64(post_idx))
    return pre2post_list


def post2pre(i, j, num_post):
    """Get post2pre connections from `i` and `j` indexes.

    Parameters
    ----------
    i : a_list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : a_list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int, None
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The conn a_list of post2pre.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        post2pre_list = nb.typed.List()
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            pre_idx = i[index]
            post2pre_list.append(np.uint64(pre_idx))
    else:
        post2pre_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            pre_idx = i[index]
            post2pre_list.append(np.uint64(pre_idx))
    return post2pre_list


def pre2syn(i, j, num_pre):
    """Get pre2syn connections from `i` and `j` indexes.

    Parameters
    ----------
    i : a_list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : a_list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The conn a_list of pre2syn.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        post2syn_list = nb.typed.List()
        for pre_i in range(num_pre):
            index = np.where(i == pre_i)[0]
            post2syn_list.append(np.uint64(index))
    else:
        post2syn_list = []
        for pre_i in range(num_pre):
            index = np.where(j == pre_i)[0]
            post2syn_list.append(np.uint64(index))
    return post2syn_list


def post2syn(i, j, num_post):
    """Get post2syn connections from `i` and `j` indexes.

    Parameters
    ----------
    i : a_list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : a_list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The conn a_list of post2syn.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        post2syn_list = nb.typed.List()
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            post2syn_list.append(np.uint64(index))
    else:
        post2syn_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            post2syn_list.append(np.uint64(index))
    return post2syn_list
