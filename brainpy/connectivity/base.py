# -*- coding: utf-8 -*-


from .. import numpy as np
from .. import profile
from ..errors import ModelUseError

try:
    import numba as nb
except ImportError:
    nb = None

__all__ = [
    'Connector',
    'ij2mat',
    'mat2ij',
    'pre2post',
    'post2pre',
    'pre2syn',
    'post2syn',
    'pre_slice_syn',
    'post_slice_syn',
]


def ij2mat(i, j, num_pre=None, num_post=None):
    """Convert i-j connection to matrix connection.

    Parameters
    ----------
    i : list, np.ndarray
        Pre-synaptic neuron index.
    j : list, np.ndarray
        Post-synaptic neuron index.
    num_pre : int
        The number of the pre-synaptic neurons.
    num_post : int
        The number of the post-synaptic neurons.

    Returns
    -------
    conn_mat : np.ndarray
        A 2D ndarray connectivity matrix.
    """
    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('"i" and "j" must be the equal length.')
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = np.max(i)
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = np.max(j)

    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    conn_mat = np.zeros((num_pre, num_post), dtype=np.float_)
    conn_mat[i, j] = 1.
    return conn_mat


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


def pre2post(i, j, num_pre=None):
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
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = np.max(i)

    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)

    if profile.is_numba_bk():
        pre2post_list_nb = nb.typed.List()
        for pre_id in range(num_pre):
            pre2post_list_nb.append(np.int64(pre2post_list[pre_id]))
        pre2post_list = pre2post_list_nb
    return pre2post_list


def post2pre(i, j, num_post=None):
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
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = np.max(j)

    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)

    if profile.is_numba_bk():
        post2pre_list_nb = nb.typed.List()
        for post_id in range(num_post):
            post2pre_list_nb.append(np.int64(post2pre_list[post_id]))
        post2pre_list = post2pre_list_nb
    return post2pre_list


def pre2syn(i, num_pre=None):
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
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = np.max(i)

    pre2syn_list = [[] for _ in range(num_pre)]
    for syn_id, pre_id in enumerate(i):
        pre2syn_list[pre_id].append(syn_id)

    if profile.is_numba_bk():
        pre2syn_list_nb = nb.typed.List()
        for pre_ids in pre2syn_list:
            pre2syn_list_nb.append(np.int64(pre_ids))
        pre2syn_list = pre2syn_list_nb
    return pre2syn_list


def post2syn(j, num_post=None):
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
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = np.max(j)

    post2syn_list = [[] for _ in range(num_post)]
    for syn_id, post_id in enumerate(j):
        post2syn_list[post_id].append(syn_id)

    if profile.is_numba_bk():
        post2syn_list_nb = nb.typed.List()
        for pre_ids in post2syn_list:
            post2syn_list_nb.append(np.int64(pre_ids))
        post2syn_list = post2syn_list_nb
    return post2syn_list


def pre_slice_syn(i, j, num_pre=None):
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
    # check
    try:
        assert len(i) == len(j)
    except AssertionError:
        raise ModelUseError('The length of "i" and "j" must be the same.')
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = np.max(i)

    # pre2post connection
    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)
    post_ids = np.asarray(np.concatenate(pre2post_list), dtype=np.int_)

    # pre2post slicing
    slicing = []
    start = 0
    for posts in pre2post_list:
        end = start + len(posts)
        slicing.append([start, end])
        start = end
    slicing = np.asarray(slicing, dtype=np.int_)

    # pre_ids
    pre_ids = np.repeat(np.arange(num_pre), slicing[:, 1] - slicing[:, 0])

    return pre_ids, post_ids, slicing


def post_slice_syn(i, j, num_post=None):
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
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = np.max(j)

    # post2pre connection
    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)
    pre_ids = np.asarray(np.concatenate(post2pre_list), dtype=np.int_)

    # post2pre slicing
    slicing = []
    start = 0
    for pres in post2pre_list:
        end = start + len(pres)
        slicing.append([start, end])
        start = end
    slicing = np.asarray(slicing, dtype=np.int_)

    # post_ids
    post_ids = np.repeat(np.arange(num_post), slicing[:, 1] - slicing[:, 0])

    return pre_ids, post_ids, slicing


class Connector(object):
    """Abstract connector class."""

    def __init__(self):
        # total size of the pre/post-synaptic neurons
        # useful for the construction of pre2post/pre2syn/etc.
        self.num_pre = None
        self.num_post = None
        # synaptic structures
        self.pre_ids = None
        self.post_ids = None
        self.conn_mat = None
        self.pre2post = None
        self.post2pre = None
        self.pre2syn = None
        self.post2syn = None
        self.pre_slice_syn = None
        self.post_slice_syn = None
        # synaptic weights
        self.weights = None
        # the required synaptic structures
        self.requires = ()

    def set_size(self, num_pre, num_post):
        try:
            assert isinstance(num_pre, int)
            assert 0 < num_pre
        except AssertionError:
            raise ModelUseError('"num_pre" must be integrator bigger than 0.')
        try:
            assert isinstance(num_post, int)
            assert 0 < num_post
        except AssertionError:
            raise ModelUseError('"num_post" must be integrator bigger than 0.')
        self.num_pre = num_pre
        self.num_post = num_post

    def set_requires(self, syn_requires):
        # get synaptic requires
        requires = set()
        for n in syn_requires:
            if n in ['pre_ids', 'post_ids', 'conn_mat',
                     'pre2post', 'post2pre',
                     'pre2syn', 'post2syn',
                     'pre_slice_syn', 'post_slice_syn']:
                requires.add(n)
        self.requires = list(requires)

        # synaptic structure to handle
        needs = []
        if 'pre_slice_syn' in self.requires and 'post_slice_syn' in self.requires:
            raise ModelUseError('Cannot use "pre_slice_syn" and "post_slice_syn" simultaneously. \n'
                                'We recommend you use "pre_slice_syn + post2syn" '
                                'or "post_slice_syn + pre2syn".')
        elif 'pre_slice_syn' in self.requires:
            needs.append('pre_slice_syn')
        elif 'post_slice_syn' in self.requires:
            needs.append('post_slice_syn')
        for n in self.requires:
            if n in ['pre_slice_syn', 'post_slice_syn', 'pre_ids', 'post_ids']:
                continue
            needs.append(n)

        # make synaptic data structure
        for n in needs:
            getattr(self, f'make_{n}')()

    def __call__(self, pre_indices, post_indices):
        raise NotImplementedError

    def make_conn_mat(self):
        self.conn_mat = ij2mat(self.pre_ids, self.post_ids, self.num_pre, self.num_post)

    def make_mat2ij(self):
        self.pre_ids, self.post_ids = mat2ij(self.conn_mat)

    def make_pre2post(self):
        self.pre2post = pre2post(self.pre_ids, self.post_ids, self.num_pre)

    def make_post2pre(self):
        self.post2pre = post2pre(self.pre_ids, self.post_ids, self.num_post)

    def make_pre2syn(self):
        self.pre2syn = pre2syn(self.pre_ids, self.num_pre)

    def make_post2syn(self):
        self.post2syn = post2syn(self.post_ids, self.num_post)

    def make_pre_slice_syn(self):
        self.pre_ids, self.post_ids, self.pre_slice_syn = \
            pre_slice_syn(self.pre_ids, self.post_ids, self.num_pre)

    def make_post_slice_syn(self):
        self.pre_ids, self.post_ids, self.post_slice_syn = \
            post_slice_syn(self.pre_ids, self.post_ids, self.num_post)
