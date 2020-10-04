# -*- coding: utf-8 -*-

from .. import _numpy as np
from .. import profile

try:
    import numba as nb
except ImportError:
    nb = None

__all__ = ['from_matrix', 'from_ij', 'pre2post', 'pre2syn', 'post2syn', 'post2pre']


def from_matrix(conn_mat):
    """Get the connections from connectivity matrix.

    This function which create three arrays. The first one is the connected
    pre-synaptic neurons, a 1-D array. The second one is the connected
    post-synaptic neurons, another 1-D array. The third one is the start
    and the end indexes at the 1-D array for each pre-synaptic neurons.

    Parameters
    ----------
    conn_mat : numpy.ndarray
        Connectivity matrix with `(num_pre x num_post)` shape.

    Returns
    -------
    conn_tuple : tuple
        (Pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic
         neuron for each pre-synaptic neuron).
    """
    pre_ids = []
    post_ids = []
    anchors = []
    ii = 0
    num_pre = conn_mat.shape[0]
    for pre_idx in range(num_pre):
        post_idxs = np.where(conn_mat[pre_idx] > 0)[0]
        post_ids.extend(post_idxs)
        len_idx = len(post_idxs)
        anchors.append([ii, ii + len_idx])
        pre_ids.extend([pre_idx] * len(post_idxs))
        ii += len_idx
    post_ids = np.array(post_ids)
    anchors = np.array(anchors).T
    return pre_ids, post_ids, anchors


def from_ij(i, j, num_pre=None, others=()):
    """Format complete connections from `i` and `j` indexes.

    Parameters
    ----------
    i : a_list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : a_list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int, None
        The number of the pre-synaptic neurons.
    others : tuple, a_list, numpy.array
        The other parameters.

    Returns
    -------
    conn_tuple : tuple
        (pre_ids, post_ids, anchors).
    """
    conn_i = np.array(i)
    conn_j = np.array(j)
    num_pre = np.max(i) + 1 if num_pre is None else num_pre
    pre_ids, post_ids, anchors = [], [], []
    assert isinstance(others, (list, tuple)), '"others" must be a a_list/tuple of arrays.'
    others = [np.asarray(o) for o in others]
    other_arrays = [[] for _ in range(len(others))]
    ii = 0
    for i in range(num_pre):
        indexes = np.where(conn_i == i)[0]
        post_idx = conn_j[indexes]
        post_len = len(post_idx)
        pre_ids.extend([i] * post_len)
        post_ids.extend(post_idx)
        anchors.append([ii, ii + post_len])
        for iii, o in enumerate(others):
            other_arrays[iii].extend(o[indexes])
        ii += post_len
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    other_arrays = [np.asarray(arr) for arr in other_arrays]
    return (pre_ids, post_ids, anchors) + tuple(other_arrays)


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
            l = nb.typed.List.empty_list(nb.types.int32)
            for idx in post_idx:
                l.append(np.int32(idx))
            pre2post_list.append(l)
    else:
        pre2post_list = []

        for pre_i in range(num_pre):
            index = np.where(i == pre_i)[0]
            post_idx = j[index]
            pre2post_list.append(post_idx)
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
            l = nb.typed.List.empty_list(nb.types.int32)
            for idx in pre_idx:
                l.append(np.int32(idx))
            post2pre_list.append(l)
    else:
        post2pre_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            pre_idx = i[index]
            post2pre_list.append(pre_idx)
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
            l = nb.typed.List.empty_list(nb.types.uint64)
            for idx in index:
                l.append(np.uint64(idx))
            post2syn_list.append(l)
    else:
        post2syn_list = []
        for pre_i in range(num_pre):
            index = np.where(j == pre_i)[0]
            post2syn_list.append(index)
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
            l = nb.typed.List.empty_list(nb.types.uint64)
            for idx in index:
                l.append(np.uint64(idx))
            post2syn_list.append(l)
    else:
        post2syn_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            post2syn_list.append(index)
    return post2syn_list
