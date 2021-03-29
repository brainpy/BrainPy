# -*- coding: utf-8 -*-


import abc

import numpy as np

from brainpy import backend
from brainpy import errors
from brainpy.simulation import utils

try:
    import numba as nb
except ModuleNotFoundError:
    nb = None

__all__ = [
    'ij2mat',
    'mat2ij',
    'pre2post',
    'post2pre',
    'pre2syn',
    'post2syn',
    'pre_slice_syn',
    'post_slice_syn',

    'AbstractConnector',
    'Connector',

    'One2One', 'one2one',
    'All2All', 'all2all',
    'GridFour', 'grid_four',
    'GridEight', 'grid_eight',
    'GridN',
    'FixedPostNum',
    'FixedPreNum',
    'FixedProb',
    'GaussianProb',
    'GaussianWeight',
    'DOG',
    'SmallWorld',
    'ScaleFree'
]


def _numba_backend():
    r = backend.get_backend().startswith('numba')
    if r and nb is None:
        raise errors.PackageMissingError('Please install numba for numba backend.')
    return r


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
    if len(i) != len(j):
        raise errors.ModelUseError('"i" and "j" must be the equal length.')
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = i.max()
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = j.max()
    conn_mat = backend.zeros((num_pre, num_post))
    conn_mat[i, j] = 1.
    return conn_mat


def mat2ij(conn_mat):
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
    if len(backend.shape(conn_mat)) != 2:
        raise errors.ModelUseError('Connectivity matrix must be in the '
                                   'shape of (num_pre, num_post).')
    pre_ids, post_ids = backend.where(conn_mat > 0)
    return backend.as_tensor(pre_ids), backend.as_tensor(post_ids)


def pre2post(i, j, num_pre=None):
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
        raise errors.ModelUseError('The length of "i" and "j" must be the same.')
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = i.max()

    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)
    pre2post_list = [backend.as_tensor(l) for l in pre2post_list]

    if _numba_backend:
        pre2post_list_nb = nb.typed.List()
        for pre_id in range(num_pre):
            pre2post_list_nb.append(pre2post_list[pre_id])
        pre2post_list = pre2post_list_nb
    return pre2post_list


def post2pre(i, j, num_post=None):
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
        raise errors.ModelUseError('The length of "i" and "j" must be the same.')
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = j.max()

    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)
    post2pre_list = [backend.as_tensor(l) for l in post2pre_list]

    if _numba_backend():
        post2pre_list_nb = nb.typed.List()
        for post_id in range(num_post):
            post2pre_list_nb.append(post2pre_list[post_id])
        post2pre_list = post2pre_list_nb
    return post2pre_list


def pre2syn(i, num_pre=None):
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
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = i.max()

    pre2syn_list = [[] for _ in range(num_pre)]
    for syn_id, pre_id in enumerate(i):
        pre2syn_list[pre_id].append(syn_id)
    pre2syn_list = [backend.as_tensor(l) for l in pre2syn_list]

    if _numba_backend():
        pre2syn_list_nb = nb.typed.List()
        for pre_ids in pre2syn_list:
            pre2syn_list_nb.append(pre_ids)
        pre2syn_list = pre2syn_list_nb

    return pre2syn_list


def post2syn(j, num_post=None):
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
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = j.max()

    post2syn_list = [[] for _ in range(num_post)]
    for syn_id, post_id in enumerate(j):
        post2syn_list[post_id].append(syn_id)
    post2syn_list = [backend.as_tensor(l) for l in post2syn_list]

    if _numba_backend():
        post2syn_list_nb = nb.typed.List()
        for pre_ids in post2syn_list:
            post2syn_list_nb.append(pre_ids)
        post2syn_list = post2syn_list_nb

    return post2syn_list


def pre_slice_syn(i, j, num_pre=None):
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
        raise errors.ModelUseError('The length of "i" and "j" must be the same.')
    if num_pre is None:
        print('WARNING: "num_pre" is not provided, the result may not be accurate.')
        num_pre = i.max()

    # pre2post connection
    pre2post_list = [[] for _ in range(num_pre)]
    for pre_id, post_id in zip(i, j):
        pre2post_list[pre_id].append(post_id)
    pre_ids, post_ids = [], []
    for pre_i, posts in enumerate(pre2post_list):
        post_ids.extend(posts)
        pre_ids.extend([pre_i] * len(posts))
    post_ids = backend.as_tensor(post_ids)
    pre_ids = backend.as_tensor(pre_ids)

    # pre2post slicing
    slicing = []
    start = 0
    for posts in pre2post_list:
        end = start + len(posts)
        slicing.append([start, end])
        start = end
    slicing = backend.as_tensor(slicing)

    return pre_ids, post_ids, slicing


def post_slice_syn(i, j, num_post=None):
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
        raise errors.ModelUseError('The length of "i" and "j" must be the same.')
    if num_post is None:
        print('WARNING: "num_post" is not provided, the result may not be accurate.')
        num_post = j.max()

    # post2pre connection
    post2pre_list = [[] for _ in range(num_post)]
    for pre_id, post_id in zip(i, j):
        post2pre_list[post_id].append(pre_id)
    pre_ids, post_ids = [], []
    for _post_id, _pre_ids in enumerate(post2pre_list):
        pre_ids.extend(_pre_ids)
        post_ids.extend([_post_id] * len(_pre_ids))
    post_ids = backend.as_tensor(post_ids)
    pre_ids = backend.as_tensor(pre_ids)

    # post2pre slicing
    slicing = []
    start = 0
    for pres in post2pre_list:
        end = start + len(pres)
        slicing.append([start, end])
        start = end
    slicing = backend.as_tensor(slicing)

    return pre_ids, post_ids, slicing


SUPPORTED_SYN_STRUCTURE = ['pre_ids', 'post_ids', 'conn_mat',
                           'pre2post', 'post2pre',
                           'pre2syn', 'post2syn',
                           'pre_slice_syn', 'post_slice_syn']


class AbstractConnector(abc.ABC):
    def __call__(self, *args, **kwargs):
        pass


class Connector(AbstractConnector):
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

    def requires(self, *syn_requires):
        # get synaptic requires
        requires = set()
        for n in syn_requires:
            if n in SUPPORTED_SYN_STRUCTURE:
                requires.add(n)
            else:
                raise ValueError(f'Unknown synapse structure {n}. We only support '
                                 f'{SUPPORTED_SYN_STRUCTURE}.')
        requires = list(requires)

        # synaptic structure to handle
        needs = []
        if 'pre_slice_syn' in requires and 'post_slice_syn' in requires:
            raise errors.ModelUseError('Cannot use "pre_slice_syn" and "post_slice_syn" '
                                       'simultaneously. \n'
                                       'We recommend you use "pre_slice_syn + '
                                       'post2syn" or "post_slice_syn + pre2syn".')
        elif 'pre_slice_syn' in requires:
            needs.append('pre_slice_syn')
        elif 'post_slice_syn' in requires:
            needs.append('post_slice_syn')
        for n in requires:
            if n in ['pre_slice_syn', 'post_slice_syn', 'pre_ids', 'post_ids']:
                continue
            needs.append(n)

        # make synaptic data structure
        for n in needs:
            getattr(self, f'make_{n}')()

        # returns
        if len(requires) == 1:
            return getattr(self, requires[0])
        else:
            return tuple([getattr(self, r) for r in requires])

    def make_conn_mat(self):
        if self.conn_mat is None:
            self.conn_mat = ij2mat(self.pre_ids, self.post_ids, self.num_pre, self.num_post)

    def make_mat2ij(self):
        if self.pre_ids is None or self.post_ids is None:
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


def _grid_four(height, width, row, include_self):
    conn_i = []
    conn_j = []

    for col in range(width):
        i_index = (row * width) + col
        if 0 <= row - 1 < height:
            j_index = ((row - 1) * width) + col
            conn_i.append(i_index)
            conn_j.append(j_index)
        if 0 <= row + 1 < height:
            j_index = ((row + 1) * width) + col
            conn_i.append(i_index)
            conn_j.append(j_index)
        if 0 <= col - 1 < width:
            j_index = (row * width) + col - 1
            conn_i.append(i_index)
            conn_j.append(j_index)
        if 0 <= col + 1 < width:
            j_index = (row * width) + col + 1
            conn_i.append(i_index)
            conn_j.append(j_index)
        if include_self:
            conn_i.append(i_index)
            conn_j.append(i_index)
    return conn_i, conn_j


def _grid_n(height, width, row, n, include_self):
    conn_i = []
    conn_j = []
    for col in range(width):
        i_index = (row * width) + col
        for row_diff in range(-n, n + 1):
            for col_diff in range(-n, n + 1):
                if (not include_self) and (row_diff == col_diff == 0):
                    continue
                if 0 <= row + row_diff < height and 0 <= col + col_diff < width:
                    j_index = ((row + row_diff) * width) + col + col_diff
                    conn_i.append(i_index)
                    conn_j.append(j_index)
    return conn_i, conn_j


def _gaussian_weight(pre_i, pre_width, pre_height,
                     num_post, post_width, post_height,
                     w_max, w_min, sigma, normalize, include_self):
    conn_i = []
    conn_j = []
    conn_w = []

    # get normalized coordination
    pre_coords = (pre_i // pre_width, pre_i % pre_width)
    if normalize:
        pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                      pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

    for post_i in range(num_post):
        if (pre_i == post_i) and (not include_self):
            continue

        # get normalized coordination
        post_coords = (post_i // post_width, post_i % post_width)
        if normalize:
            post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                           post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

        # Compute Euclidean distance between two coordinates
        distance = (pre_coords[0] - post_coords[0]) ** 2
        distance += (pre_coords[1] - post_coords[1]) ** 2
        # get weight and conn
        value = w_max * np.exp(-distance / (2.0 * sigma ** 2))
        if value > w_min:
            conn_i.append(pre_i)
            conn_j.append(post_i)
            conn_w.append(value)
    return conn_i, conn_j, conn_w


def _gaussian_prob(pre_i, pre_width, pre_height,
                   num_post, post_width, post_height,
                   p_min, sigma, normalize, include_self):
    conn_i = []
    conn_j = []
    conn_p = []

    # get normalized coordination
    pre_coords = (pre_i // pre_width, pre_i % pre_width)
    if normalize:
        pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                      pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

    for post_i in range(num_post):
        if (pre_i == post_i) and (not include_self):
            continue

        # get normalized coordination
        post_coords = (post_i // post_width, post_i % post_width)
        if normalize:
            post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                           post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

        # Compute Euclidean distance between two coordinates
        distance = (pre_coords[0] - post_coords[0]) ** 2
        distance += (pre_coords[1] - post_coords[1]) ** 2
        # get weight and conn
        value = np.exp(-distance / (2.0 * sigma ** 2))
        if value > p_min:
            conn_i.append(pre_i)
            conn_j.append(post_i)
            conn_p.append(value)
    return conn_i, conn_j, conn_p


def _dog(pre_i, pre_width, pre_height,
         num_post, post_width, post_height,
         w_max_p, w_max_n, w_min, sigma_p, sigma_n,
         normalize, include_self):
    conn_i = []
    conn_j = []
    conn_w = []

    # get normalized coordination
    pre_coords = (pre_i // pre_width, pre_i % pre_width)
    if normalize:
        pre_coords = (pre_coords[0] / (pre_height - 1) if pre_height > 1 else 1.,
                      pre_coords[1] / (pre_width - 1) if pre_width > 1 else 1.)

    for post_i in range(num_post):
        if (pre_i == post_i) and (not include_self):
            continue

        # get normalized coordination
        post_coords = (post_i // post_width, post_i % post_width)
        if normalize:
            post_coords = (post_coords[0] / (post_height - 1) if post_height > 1 else 1.,
                           post_coords[1] / (post_width - 1) if post_width > 1 else 1.)

        # Compute Euclidean distance between two coordinates
        distance = (pre_coords[0] - post_coords[0]) ** 2
        distance += (pre_coords[1] - post_coords[1]) ** 2
        # get weight and conn
        value = w_max_p * np.exp(-distance / (2.0 * sigma_p ** 2)) - \
                w_max_n * np.exp(-distance / (2.0 * sigma_n ** 2))
        if np.abs(value) > w_min:
            conn_i.append(pre_i)
            conn_j.append(post_i)
            conn_w.append(value)
    return conn_i, conn_j, conn_w


if nb is not None:
    _grid_four = nb.njit(_grid_four)
    _grid_n = nb.njit(_grid_n)
    _gaussian_weight = nb.njit(_gaussian_weight)
    _gaussian_prob = nb.njit(_gaussian_prob)
    _dog = nb.njit(_dog)


class One2One(Connector):
    """
    Connect two neuron groups one by one. This means
    The two neuron groups should have the same size.
    """

    def __init__(self):
        super(One2One, self).__init__()

    def __call__(self, pre_size, post_size):
        try:
            assert pre_size == post_size
        except AssertionError:
            raise errors.ModelUseError(f'One2One connection must be defined in two groups with the same size, '
                                       f'but we got {pre_size} != {post_size}.')

        length = utils.size2len(pre_size)
        self.num_pre = length
        self.num_post = length

        self.pre_ids = backend.arange(length)
        self.post_ids = backend.arange(length)
        return self


one2one = One2One()


class All2All(Connector):
    """Connect each neuron in first group to all neurons in the
    post-synaptic neuron groups. It means this kind of conn
    will create (num_pre x num_post) synapses.
    """

    def __init__(self, include_self=True):
        self.include_self = include_self
        super(All2All, self).__init__()

    def __call__(self, pre_size, post_size):
        pre_len = utils.size2len(pre_size)
        post_len = utils.size2len(post_size)
        self.num_pre = pre_len
        self.num_post = post_len

        mat = np.ones((pre_len, post_len))
        if not self.include_self:
            np.fill_diagonal(mat, 0)
        pre_ids, post_ids = np.where(mat > 0)
        self.pre_ids = backend.as_tensor(np.ascontiguousarray(pre_ids))
        self.post_ids = backend.as_tensor(np.ascontiguousarray(post_ids))
        self.conn_mat = backend.as_tensor(mat)
        return self


all2all = All2All(include_self=True)


class GridFour(Connector):
    """The nearest four neighbors conn method."""

    def __init__(self, include_self=False):
        super(GridFour, self).__init__()
        self.include_self = include_self

    def __call__(self, pre_size, post_size=None):
        self.num_pre = utils.size2len(pre_size)
        if post_size is not None:
            try:
                assert pre_size == post_size
            except AssertionError:
                raise errors.ModelUseError(f'The shape of pre-synaptic group should be the same with the '
                                           f'post group. But we got {pre_size} != {post_size}.')
            self.num_post = utils.size2len(post_size)
        else:
            self.num_post = self.num_pre

        if len(pre_size) == 1:
            height, width = pre_size[0], 1
        elif len(pre_size) == 2:
            height, width = pre_size
        else:
            raise errors.ModelUseError('Currently only support two-dimensional geometry.')
        conn_i = []
        conn_j = []
        for row in range(height):
            a = _grid_four(height, width, row, include_self=self.include_self)
            conn_i.extend(a[0])
            conn_j.extend(a[1])
        self.pre_ids = backend.as_tensor(conn_i)
        self.post_ids = backend.as_tensor(conn_j)
        return self


grid_four = GridFour()


class GridN(Connector):
    """The nearest (2*N+1) * (2*N+1) neighbors conn method.

    Parameters
    ----------
    N : int
        Extend of the conn scope. For example:
        When N=1,
            [x x x]
            [x I x]
            [x x x]
        When N=2,
            [x x x x x]
            [x x x x x]
            [x x I x x]
            [x x x x x]
            [x x x x x]
    include_self : bool
        Whether create (i, i) conn ?
    """

    def __init__(self, n=1, include_self=False):
        super(GridN, self).__init__()
        self.n = n
        self.include_self = include_self

    def __call__(self, pre_size, post_size=None):
        self.num_pre = utils.size2len(pre_size)
        if post_size is not None:
            try:
                assert pre_size == post_size
            except AssertionError:
                raise errors.ModelUseError(
                    f'The shape of pre-synaptic group should be the same with the post group. '
                    f'But we got {pre_size} != {post_size}.')
            self.num_post = utils.size2len(post_size)
        else:
            self.num_post = self.num_pre

        if len(pre_size) == 1:
            height, width = pre_size[0], 1
        elif len(pre_size) == 2:
            height, width = pre_size
        else:
            raise errors.ModelUseError('Currently only support two-dimensional geometry.')

        conn_i = []
        conn_j = []
        for row in range(height):
            res = _grid_n(height=height, width=width, row=row,
                          n=self.n, include_self=self.include_self)
            conn_i.extend(res[0])
            conn_j.extend(res[1])
        self.pre_ids = backend.as_tensor(conn_i)
        self.post_ids = backend.as_tensor(conn_j)
        return self


class GridEight(GridN):
    """The nearest eight neighbors conn method."""

    def __init__(self, include_self=False):
        super(GridEight, self).__init__(n=1, include_self=include_self)


grid_eight = GridEight()


class FixedProb(Connector):
    """Connect the post-synaptic neurons with fixed probability.

    Parameters
    ----------
    prob : float
        The conn probability.
    include_self : bool
        Whether create (i, i) conn ?
    seed : None, int
        Seed the random generator.
    """

    def __init__(self, prob, include_self=True, seed=None):
        super(FixedProb, self).__init__()
        self.prob = prob
        self.include_self = include_self
        self.seed = seed

    def __call__(self, pre_size, post_size):
        num_pre, num_post = utils.size2len(pre_size), utils.size2len(post_size)
        self.num_pre, self.num_post = num_pre, num_post

        prob_mat = np.random.random(size=(num_pre, num_post))
        if not self.include_self:
            diag_index = np.arange(min([num_pre, num_post]))
            prob_mat[diag_index, diag_index] = 1.
        conn_mat = np.array(prob_mat < self.prob, dtype=np.int_)
        pre_ids, post_ids = np.where(conn_mat)
        self.conn_mat = backend.as_tensor(conn_mat)
        self.pre_ids = backend.as_tensor(np.ascontiguousarray(pre_ids))
        self.post_ids = backend.as_tensor(np.ascontiguousarray(post_ids))
        return self


class FixedPreNum(Connector):
    """Connect the pre-synaptic neurons with fixed number for each
    post-synaptic neuron.

    Parameters
    ----------
    num : float, int
        The conn probability (if "num" is float) or the fixed number of
        connectivity (if "num" is int).
    include_self : bool
        Whether create (i, i) conn ?
    seed : None, int
        Seed the random generator.
    """

    def __init__(self, num, include_self=True, seed=None):
        super(FixedPreNum, self).__init__()
        if isinstance(num, int):
            assert num >= 0, '"num" must be bigger than 0.'
        elif isinstance(num, float):
            assert 0. <= num <= 1., '"num" must be in [0., 1.].'
        else:
            raise ValueError(f'Unknown type: {type(num)}')
        self.num = num
        self.include_self = include_self
        self.seed = seed

    def __call__(self, pre_size, post_size):
        num_pre, num_post = utils.size2len(pre_size), utils.size2len(post_size)
        self.num_pre, self.num_post = num_pre, num_post
        num = self.num if isinstance(self.num, int) else int(self.num * num_pre)
        assert num <= num_pre, f'"num" must be less than "num_pre", but got {num} > {num_pre}'
        prob_mat = np.random.random(size=(num_pre, num_post))
        if not self.include_self:
            diag_index = np.arange(min([num_pre, num_post]))
            prob_mat[diag_index, diag_index] = 1.1
        arg_sort = np.argsort(prob_mat, axis=0)[:num]
        pre_ids = np.asarray(np.concatenate(arg_sort), dtype=np.int_)
        post_ids = np.asarray(np.repeat(np.arange(num_post), num_pre), dtype=np.int_)
        self.pre_ids = backend.as_tensor(pre_ids)
        self.post_ids = backend.as_tensor(post_ids)
        return self


class FixedPostNum(Connector):
    """Connect the post-synaptic neurons with fixed number for each
    pre-synaptic neuron.

    Parameters
    ----------
    num : float, int
        The conn probability (if "num" is float) or the fixed number of
        connectivity (if "num" is int).
    include_self : bool
        Whether create (i, i) conn ?
    seed : None, int
        Seed the random generator.
    """

    def __init__(self, num, include_self=True, seed=None):
        if isinstance(num, int):
            assert num >= 0, '"num" must be bigger than 0.'
        elif isinstance(num, float):
            assert 0. <= num <= 1., '"num" must be in [0., 1.].'
        else:
            raise ValueError(f'Unknown type: {type(num)}')
        self.num = num
        self.include_self = include_self
        self.seed = seed
        super(FixedPostNum, self).__init__()

    def __call__(self, pre_size, post_size):
        num_pre = utils.size2len(pre_size)
        num_post = utils.size2len(post_size)
        self.num_pre = num_pre
        self.num_post = num_post
        num = self.num if isinstance(self.num, int) else int(self.num * num_post)
        assert num <= num_post, f'"num" must be less than "num_post", but got {num} > {num_post}'
        prob_mat = np.random.random(size=(num_pre, num_post))
        if not self.include_self:
            diag_index = np.arange(min([num_pre, num_post]))
            prob_mat[diag_index, diag_index] = 1.1
        arg_sort = np.argsort(prob_mat, axis=1)[:, num]
        post_ids = np.asarray(np.concatenate(arg_sort), dtype=np.int64)
        pre_ids = np.asarray(np.repeat(np.arange(num_pre), num_post), dtype=np.int64)
        self.pre_ids = backend.as_tensor(pre_ids)
        self.post_ids = backend.as_tensor(post_ids)
        return self


class GaussianWeight(Connector):
    """Builds a Gaussian conn pattern between the two populations, where
    the weights decay with gaussian function.

    Specifically,

    .. math::

        w(x, y) = w_{max} \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma^2})

    where :math:`(x, y)` is the position of the pre-synaptic neuron (normalized
    to [0,1]) and :math:`(x_c,y_c)` is the position of the post-synaptic neuron
    (normalized to [0,1]), :math:`w_{max}` is the maximum weight. In order to void
    creating useless synapses, :math:`w_{min}` can be set to restrict the creation
    of synapses to the cases where the value of the weight would be superior
    to :math:`w_{min}`. Default is :math:`0.01 w_{max}`.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian function.
    w_max : float
        The weight amplitude of the Gaussian function.
    w_min : float, None
        The minimum weight value below which synapses are not created (default: 0.01 * `w_max`).
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the conn at the same position.
    """

    def __init__(self, sigma, w_max, w_min=None, normalize=True, include_self=True):
        super(GaussianWeight, self).__init__()
        self.sigma = sigma
        self.w_max = w_max
        self.w_min = w_max * 0.01 if w_min is None else w_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, pre_size, post_size):
        num_pre = utils.size2len(pre_size)
        num_post = utils.size2len(post_size)
        self.num_pre = num_pre
        self.num_post = num_post
        assert len(pre_size) == 2
        assert len(post_size) == 2
        pre_height, pre_width = pre_size
        post_height, post_width = post_size

        # get the connections and weights
        i, j, w = [], [], []
        for pre_i in range(num_pre):
            a = _gaussian_weight(pre_i=pre_i,
                                 pre_width=pre_width,
                                 pre_height=pre_height,
                                 num_post=num_post,
                                 post_width=post_width,
                                 post_height=post_height,
                                 w_max=self.w_max,
                                 w_min=self.w_min,
                                 sigma=self.sigma,
                                 normalize=self.normalize,
                                 include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            w.extend(a[2])

        pre_ids = np.asarray(i, dtype=np.int_)
        post_ids = np.asarray(j, dtype=np.int_)
        w = np.asarray(w, dtype=np.float_)
        self.pre_ids = backend.as_tensor(pre_ids)
        self.post_ids = backend.as_tensor(post_ids)
        self.weights = backend.as_tensor(w)
        return self


class GaussianProb(Connector):
    """Builds a Gaussian conn pattern between the two populations, where
    the conn probability decay according to the gaussian function.

    Specifically,

    .. math::

        p=\\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma^2})

    where :math:`(x, y)` is the position of the pre-synaptic neuron
    and :math:`(x_c,y_c)` is the position of the post-synaptic neuron.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian function.
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the conn at the same position.
    seed : bool
        The random seed.
    """

    def __init__(self, sigma, p_min=0., normalize=True, include_self=True, seed=None):
        super(GaussianProb, self).__init__()
        self.sigma = sigma
        self.p_min = p_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, pre_size, post_size):
        self.num_pre = num_pre = utils.size2len(pre_size)
        self.num_post = num_post = utils.size2len(post_size)
        assert len(pre_size) == 2
        assert len(post_size) == 2
        pre_height, pre_width = pre_size
        post_height, post_width = post_size

        # get the connections
        i, j, p = [], [], []  # conn_i, conn_j, probabilities
        for pre_i in range(num_pre):
            a = _gaussian_prob(pre_i=pre_i,
                               pre_width=pre_width,
                               pre_height=pre_height,
                               num_post=num_post,
                               post_width=post_width,
                               post_height=post_height,
                               p_min=self.p_min,
                               sigma=self.sigma,
                               normalize=self.normalize,
                               include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            p.extend(a[2])
        p = np.asarray(p, dtype=np.float_)
        selected_idxs = np.where(np.random.random(len(p)) < p)[0]
        i = np.asarray(i, dtype=np.int_)[selected_idxs]
        j = np.asarray(j, dtype=np.int_)[selected_idxs]
        self.pre_ids = backend.as_tensor(i)
        self.post_ids = backend.as_tensor(j)
        return self


class DOG(Connector):
    """Builds a Difference-Of-Gaussian (dog) conn pattern between the two populations.

    Mathematically,

    .. math::

        w(x, y) = w_{max}^+ \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_+^2})
        -  w_{max}^- \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_-^2})

    where weights smaller than :math:`0.01 * abs(w_{max} - w_{min})` are not created and
    self-connections are avoided by default (parameter allow_self_connections).

    Parameters
    ----------
    sigmas : tuple
        Widths of the positive and negative Gaussian functions.
    ws_max : tuple
        The weight amplitudes of the positive and negative Gaussian functions.
    w_min : float, None
        The minimum weight value below which synapses are not created
        (default: :math:`0.01 * w_{max}^+ - w_{min}^-`).
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the conn at the same position.
    """

    def __init__(self, sigmas, ws_max, w_min=None, normalize=True, include_self=True):
        super(DOG, self).__init__()
        self.sigma_p, self.sigma_n = sigmas
        self.w_max_p, self.w_max_n = ws_max
        self.w_min = np.abs(ws_max[0] - ws_max[1]) * 0.01 if w_min is None else w_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, pre_size, post_size):
        self.num_pre = num_pre = utils.size2len(pre_size)
        self.num_post = num_post = utils.size2len(post_size)
        assert len(pre_size) == 2
        assert len(post_size) == 2
        pre_height, pre_width = pre_size
        post_height, post_width = post_size

        # get the connections and weights
        i, j, w = [], [], []  # conn_i, conn_j, weights
        for pre_i in range(num_pre):
            a = _dog(pre_i=pre_i,
                     pre_width=pre_width,
                     pre_height=pre_height,
                     num_post=num_post,
                     post_width=post_width,
                     post_height=post_height,
                     w_max_p=self.w_max_p,
                     w_max_n=self.w_max_n,
                     w_min=self.w_min,
                     sigma_p=self.sigma_p,
                     sigma_n=self.sigma_n,
                     normalize=self.normalize,
                     include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            w.extend(a[2])

        # format connections and weights
        i = np.asarray(i, dtype=np.int_)
        j = np.asarray(j, dtype=np.int_)
        w = np.asarray(w, dtype=np.float_)
        self.pre_ids = backend.as_tensor(i)
        self.post_ids = backend.as_tensor(j)
        self.weights = backend.as_tensor(w)
        return self


class ScaleFree(Connector):
    def __init__(self):
        raise NotImplementedError


class SmallWorld(Connector):
    def __init__(self):
        raise NotImplementedError
