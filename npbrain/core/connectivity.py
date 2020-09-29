# -*- coding: utf-8 -*-

"""
Connection toolkit.
"""

import numba as nb
import numpy as onp

from .. import _numpy as np
from .. import profile

__all__ = [
    # conn formatter
    'from_matrix',
    'from_ij',
    'pre2post',
    'post2pre',
    'pre2syn',
    'post2syn',

    # conn abstract
    'Connector',

    # conn methods
    'One2One', 'one2one',
    'All2All', 'all2all',
    'GridFour', 'grid_four',
    'GridEight', 'grid_eight',
    'GridN',
    'FixedProb', 'FixedPreNum', 'FixedPostNum',
    'GaussianProb', 'GaussianWeight', 'DOG',
    'ScaleFree', 'SmallWorld',
]


# -----------------------------------
# formatter of conn
# -----------------------------------


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
            l = nb.typed.List.empty_list(nb.types.int_)
            for idx in index:
                l.append(np.int_(idx))
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
            l = nb.typed.List.empty_list(nb.types.int_)
            for idx in index:
                l.append(np.int_(idx))
            post2syn_list.append(l)
    else:
        post2syn_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            post2syn_list.append(index)
    return post2syn_list


# -----------------------------------
# methods of conn
# -----------------------------------


def _product(a_list):
    p = 1
    for i in a_list:
        p *= i
    return p


class Connector(object):
    """Abstract connector class."""

    def __call__(self, geom_pre, geom_post):
        raise NotImplementedError


class One2One(Connector):
    """
    Connect two neuron groups one by one. This means
    The two neuron groups should have the same size.
    """

    def __int__(self):
        pass

    def __call__(self, geom_pre, geom_post):
        assert onp.array_equal(np.asarray(geom_pre), np.asarray(geom_post))
        id_list = [i for i in range(_product(geom_post))]
        pre_ids = np.asarray(id_list)
        post_ids = np.asarray(id_list)
        return {'i': pre_ids, 'j': post_ids}


one2one = One2One


class All2All(Connector):
    """Connect each neuron in first group to all neurons in the
    post-synaptic neuron groups. It means this kind of conn
    will create (num_pre x num_post) synapses.
    """

    def __init__(self, include_self=True):
        self.include_self = include_self

    def __call__(self, geom_pre, geom_post):
        num_pre = _product(geom_pre)
        num_post = _product(geom_post)

        pre_ids, post_ids = [], []
        for i_ in range(num_pre):
            for j_ in range(num_post):
                if (not self.include_self) and (i_ == j_):
                    continue
                else:
                    pre_ids.append(i_)
                    post_ids.append(j_)
        pre_ids = np.asarray(pre_ids)
        post_ids = np.asarray(post_ids)
        return {'i': pre_ids, 'j': post_ids}


all2all = All2All(include_self=True)


class GridFour(Connector):
    """The nearest four neighbors conn method."""

    def __init__(self, include_self=False):
        self.include_self = include_self

    def __call__(self, height_and_width, others=None):
        height = height_and_width[0]
        width = height_and_width[1]

        conn_i = []
        conn_j = []
        for row in range(height):
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
                if self.include_self:
                    conn_i.append(i_index)
                    conn_j.append(i_index)
        conn_i = np.asarray(conn_i)
        conn_j = np.asarray(conn_j)
        return {'i': conn_i, 'j': conn_j}


grid_four = GridFour()


class GridEight(Connector):
    """The nearest eight neighbors conn method."""

    def __init__(self, include_self=False):
        self.include_self = include_self

    def __call__(self, height_and_width, others=None):
        height = height_and_width[0]
        width = height_and_width[1]
        N = 1

        conn_i = []
        conn_j = []
        for row in range(height):
            for col in range(width):
                i_index = (row * width) + col
                for row_diff in [-1, 0, 1]:
                    for col_diff in [-1, 0, 1]:
                        if (not self.include_self) and (row_diff == col_diff == 0):
                            continue
                        if 0 <= row + row_diff < height and 0 <= col + col_diff < width:
                            j_index = ((row + row_diff) * width) + col + col_diff
                            conn_i.append(i_index)
                            conn_j.append(j_index)
        conn_i = np.asarray(conn_i)
        conn_j = np.asarray(conn_j)

        return {'i': conn_i, 'j': conn_j}


grid_eight = GridEight()


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
        self.n = n
        self.include_self = include_self

    def __call__(self, height_and_width, others=None):
        height = height_and_width[0]
        width = height_and_width[1]

        conn_i = []
        conn_j = []
        for row in range(height):
            for col in range(width):
                i_index = (row * width) + col
                for row_diff in range(-self.n, self.n + 1):
                    for col_diff in range(-self.n, self.n + 1):
                        if (not self.include_self) and (row_diff == col_diff == 0):
                            continue
                        if 0 <= row + row_diff < height and 0 <= col + col_diff < width:
                            j_index = ((row + row_diff) * width) + col + col_diff
                            conn_i.append(i_index)
                            conn_j.append(j_index)
        conn_i = np.asarray(conn_i)
        conn_j = np.asarray(conn_j)

        return {'i': conn_i, 'j': conn_j}


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
        self.prob = prob
        self.include_self = include_self
        self.seed = seed
        self.rng = np.random if seed is None else np.random.RandomState(seed)

    def __call__(self, geom_pre, geom_post):
        num_post = _product(geom_post)
        num_pre = _product(geom_pre)

        pre_ids = []
        post_ids = []
        for pre_idx in range(num_pre):
            random_vals = self.rng.random(num_post)
            idx_selected = list(np.where(random_vals < self.prob)[0])
            if (not self.include_self) and (pre_idx in idx_selected):
                idx_selected.remove(pre_idx)
            for post_idx in idx_selected:
                pre_ids.append(pre_idx)
                post_ids.append(post_idx)
        pre_ids = np.asarray(pre_ids)
        post_ids = np.asarray(post_ids)
        return {'i': pre_ids, 'j': post_ids}


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
        if isinstance(num, int):
            assert num >= 0, '"num" must be bigger than 0.'
        elif isinstance(num, float):
            assert 0. <= num <= 1., '"num" must be in [0., 1.].'
        else:
            raise ValueError(f'Unknown type: {type(num)}')
        self.num = num
        self.include_self = include_self
        self.seed = seed
        self.rng = np.random if seed is None else np.random.RandomState(seed)

    def __call__(self, geom_pre, geom_post):
        num_post = _product(geom_post)
        num_pre = _product(geom_pre)

        pre_ids, post_ids = [], []
        num = self.num if isinstance(self.num, int) else self.num * num_pre
        for j in range(num_post):
            idx_selected = self.rng.choice(num_pre, num, replace=False).tolist()
            if (not self.include_self) and (j in idx_selected):
                idx_selected.remove(j)
            size_pre = len(idx_selected)
            pre_ids.extend(idx_selected)
            post_ids.extend([j] * size_pre)
        pre_ids = np.asarray(pre_ids)
        post_ids = np.asarray(post_ids)
        return {'i': pre_ids, 'j': post_ids}


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
        self.rng = np.random if seed is None else np.random.RandomState(seed)

    def __call__(self, geom_pre, geom_post):
        num_post, num_pre = _product(geom_post), _product(geom_pre)
        pre_ids, post_ids = [], []
        num = self.num if isinstance(self.num, int) else self.num * num_post
        for i in range(num_pre):
            idx_selected = self.rng.choice(num_post, num, replace=False).tolist()
            if (not self.include_self) and (i in idx_selected):
                idx_selected.remove(i)
            size_pre = len(idx_selected)
            pre_ids.extend([i] * size_pre)
            post_ids.extend(idx_selected)
        pre_ids = np.asarray(pre_ids)
        post_ids = np.asarray(post_ids)
        return {'i': pre_ids, 'j': post_ids}


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
        self.sigma = sigma
        self.w_max = w_max
        self.w_min = w_max * 0.01 if w_min is None else w_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, geom_pre, geom_post):
        num_post, num_pre = _product(geom_post), _product(geom_pre)

        # get the connections and weights
        i, j, w = [], [], []  # conn_i, conn_j, weights
        for pre_i in range(num_pre):
            # get normalized coordination
            pre_coords = (pre_i // geom_pre[1], pre_i % geom_pre[1])
            if self.normalize:
                pre_coords = (pre_coords[0] / (geom_pre[0] - 1) if geom_pre[0] > 1 else 1.,
                              pre_coords[1] / (geom_pre[1] - 1) if geom_pre[1] > 1 else 1.)

            for post_i in range(num_post):
                if (pre_i == post_i) and (not self.include_self):
                    continue

                # get normalized coordination
                post_coords = (post_i // geom_post[1], post_i % geom_post[1])
                if self.normalize:
                    post_coords = (post_coords[0] / (geom_post[0] - 1) if geom_post[0] > 1 else 1.,
                                   post_coords[1] / (geom_post[1] - 1) if geom_post[1] > 1 else 1.)

                # Compute Euclidean distance between two coordinates
                distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])
                # get weight and conn
                value = self.w_max * np.exp(-distance / (2.0 * self.sigma ** 2))
                if value > self.w_min:
                    i.append(pre_i)
                    j.append(post_i)
                    w.append(value)

        pre_idxs = np.asarray(i, dtype=np.int_)
        post_idxs = np.asarray(j, dtype=np.int_)
        weights = np.asarray(w, dtype=np.int_)
        return {'i': pre_idxs, 'j': post_idxs, 'w': weights}


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

    def __init__(self, sigma, normalize=True, include_self=True, seed=None):
        self.sigma = sigma
        self.normalize = normalize
        self.include_self = include_self
        self.rng = np.random if seed is None else np.random.RandomState(seed=seed)

    def __call__(self, geom_pre, geom_post):
        num_post, num_pre = _product(geom_post), _product(geom_pre)

        # get the connections
        i, j, p = [], [], []  # conn_i, conn_j, probabilities
        for pre_i in range(num_pre):
            # get normalized coordination
            pre_coords = (pre_i // geom_pre[1], pre_i % geom_pre[1])
            if self.normalize:
                pre_coords = (pre_coords[0] / (geom_pre[0] - 1) if geom_pre[0] > 1 else 1.,
                              pre_coords[1] / (geom_pre[1] - 1) if geom_pre[1] > 1 else 1.)

            for post_i in range(num_post):
                if (pre_i == post_i) and (not self.include_self):
                    continue

                # get normalized coordination
                post_coords = (post_i // geom_post[1], post_i % geom_post[1])
                if self.normalize:
                    post_coords = (post_coords[0] / (geom_post[0] - 1) if geom_post[0] > 1 else 1.,
                                   post_coords[1] / (geom_post[1] - 1) if geom_post[1] > 1 else 1.)

                # Compute Euclidean distance between two coordinates
                distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])

                # get weight and probability
                i.append(pre_i)
                j.append(post_i)
                p.append(np.exp(-distance / (2.0 * self.sigma ** 2)))
        i, j, p = np.asarray(i), np.asarray(j), np.asarray(p)
        selected_idxs = np.where(self.rng.random(len(p)) < p)
        i, j = i[selected_idxs], j[selected_idxs]
        return {'i': i, 'j': j}


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
    ws_min : float, None
        The minimum weight value below which synapses are not created
        (default: :math:`0.01 * w_{max}^+ - w_{min}^-`).
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the conn at the same position.
    """

    def __init__(self, sigmas, ws_max, w_min=None, normalize=True, include_self=True):
        self.sigma_p, self.sigma_n = sigmas
        self.w_max_p, self.w_max_n = ws_max
        self.w_min = np.abs(ws_max[0] - ws_max[1]) * 0.01 if w_min is None else w_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, geom_pre, geom_post):
        num_post, num_pre = _product(geom_post), _product(geom_pre)

        # get the connections and weights
        i, j, w = [], [], []  # conn_i, conn_j, weights
        for pre_i in range(num_pre):
            # get normalized coordination
            pre_coords = (pre_i // geom_pre[1], pre_i % geom_pre[1])
            if self.normalize:
                pre_coords = (pre_coords[0] / (geom_pre[0] - 1) if geom_pre[0] > 1 else 1.,
                              pre_coords[1] / (geom_pre[1] - 1) if geom_pre[1] > 1 else 1.)

            for post_i in range(num_post):
                if (pre_i == post_i) and (not self.include_self):
                    continue

                # get normalized coordination
                post_coords = (post_i // geom_post[1], post_i % geom_post[1])
                if self.normalize:
                    post_coords = (post_coords[0] / (geom_post[0] - 1) if geom_post[0] > 1 else 1.,
                                   post_coords[1] / (geom_post[1] - 1) if geom_post[1] > 1 else 1.)

                # Compute Euclidean distance between two coordinates
                distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])
                # get weight and conn
                value = self.w_max_p * np.exp(-distance / (2.0 * self.sigma_p ** 2)) - \
                        self.w_max_n * np.exp(-distance / (2.0 * self.sigma_n ** 2))
                if np.abs(value) > self.w_min:
                    i.append(pre_i)
                    j.append(post_i)
                    w.append(value)

        # format connections and weights
        i = np.asarray(i, dtype=np.int_)
        j = np.asarray(j, dtype=np.int_)
        w = np.asarray(w, dtype=np.int_)
        return {'i': i, 'j': j, 'w': w}


class ScaleFree(Connector):
    def __init__(self):
        raise NotImplementedError


class SmallWorld(Connector):
    def __init__(self):
        raise NotImplementedError
