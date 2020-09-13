# -*- coding: utf-8 -*-

"""
Connection toolkit.
"""

import numba as nb
import numpy as np
from npbrain import profile

__all__ = [
    # connection formatter
    'from_matrix',
    'from_ij',
    'pre2post',
    'post2pre',
    'pre2syn',
    'post2syn',
    # connection methods
    'one2one',
    'all2all',
    'grid_four',
    'grid_eight',
    'grid_N',
    'fixed_prob',
    'fixed_prenum',
    'fixed_postnum',
    'gaussian_weight',
    'gaussian_prob',
    'dog',
    'scale_free',
    'small_world',
]


# -----------------------------------
# formatter of connection
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
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int, None
        The number of the pre-synaptic neurons.
    others : tuple, list, numpy.array
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
    assert isinstance(others, (list, tuple)), '"others" must be a list/tuple of arrays.'
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
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int, None
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The connection list of pre2post.
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
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int, None
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The connection list of post2pre.
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
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_pre : int
        The number of the pre-synaptic neurons.

    Returns
    -------
    conn : list
        The connection list of pre2syn.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        post2syn_list = nb.typed.List()
        for pre_i in range(num_pre):
            index = np.where(i == pre_i)[0]
            l = nb.typed.List.empty_list(nb.types.int32)
            for idx in index:
                l.append(np.int32(idx))
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
    i : list, numpy.ndarray
        The pre-synaptic neuron indexes.
    j : list, numpy.ndarray
        The post-synaptic neuron indexes.
    num_post : int
        The number of the post-synaptic neurons.

    Returns
    -------
    conn : list
        The connection list of post2syn.
    """
    i, j = np.asarray(i), np.asarray(j)

    if profile.is_numba_bk():
        post2syn_list = nb.typed.List()
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            l = nb.typed.List.empty_list(nb.types.int32)
            for idx in index:
                l.append(np.int32(idx))
            post2syn_list.append(l)
    else:
        post2syn_list = []
        for post_i in range(num_post):
            index = np.where(j == post_i)[0]
            post2syn_list.append(index)
    return post2syn_list


# -----------------------------------
# methods of connection
# -----------------------------------


class Connector(object):
    def __call__(self, *args, **kwargs):
        pass


def one2one(num_pre, num_post):
    """Connect two neuron groups one by one. This means
    The two neuron groups should have the same size.

    Parameters
    ----------
    num_pre : int
        Number of neurons in the pre-synaptic group.
    num_post : int
        Number of neurons in the post-synaptic group.

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    assert num_pre == num_post
    pre_ids = list(range(num_pre))
    post_ids = list(range(num_post))
    anchors = [[ii, ii + 1] for ii in range(num_post)]
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def all2all(num_pre, num_post, include_self=True):
    """Connect each neuron in first group to all neurons in the
    post-synaptic neuron groups. It means this kind of connection
    will create (num_pre x num_post) synapses.

    Parameters
    ----------
    num_pre : int
        Number of neurons in the pre-synaptic group.
    num_post : int
        Number of neurons in the post-synaptic group.
    include_self : bool
        Whether create (i, i) connection ?

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    pre_ids, post_ids, anchors = [], [], []
    ii = 0
    for i_ in range(num_pre):
        jj = 0
        for j_ in range(num_post):
            if (not include_self) and (i_ == j_):
                continue
            else:
                pre_ids.append(i_)
                post_ids.append(j_)
                jj += 1
        anchors.append([ii, ii + jj])
        ii += jj
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def grid_four(height, width, include_self=False):
    """The nearest four neighbors connection method.

    Parameters
    ----------
    height : int
        Number of rows.
    width : int
        Number of columns.
    include_self : bool
        Whether create (i, i) connection ?

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
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
            if include_self:
                conn_i.append(i_index)
                conn_j.append(i_index)
    conn_i = np.asarray(conn_i)
    conn_j = np.asarray(conn_j)

    pre_ids = []
    post_ids = []
    anchors = []
    num_pre = height * width
    ii = 0
    for i in range(num_pre):
        indexes = np.where(conn_i == i)[0]
        post_idx = conn_j[indexes]
        post_len = len(post_idx)
        pre_ids.extend([i] * post_len)
        post_ids.extend(post_idx)
        anchors.append([ii, ii + post_len])
        ii += post_len
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def grid_eight(height, width, include_self=False):
    """The nearest eight neighbors connection method.

    Parameters
    ----------
    height : int
        Number of rows.
    width : int
        Number of columns.
    include_self : bool
        Whether create (i, i) connection ?

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    return grid_N(height, width, 1, include_self)


def grid_N(height, width, N=1, include_self=False):
    """The nearest (2*N+1) * (2*N+1) neighbors connection method.

    Parameters
    ----------
    height : int
        Number of rows.
    width : int
        Number of columns.
    N : int
        Extend of the connection scope. For example:
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
        Whether create (i, i) connection ?

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    conn_i = []
    conn_j = []
    for row in range(height):
        for col in range(width):
            i_index = (row * width) + col
            for row_diff in [-N, 0, N]:
                for col_diff in [-N, 0, N]:
                    if (not include_self) and (row_diff == col_diff == 0):
                        continue
                    if 0 <= row + row_diff < height and 0 <= col + col_diff < width:
                        j_index = ((row + row_diff) * width) + col + col_diff
                        conn_i.append(i_index)
                        conn_j.append(j_index)
    conn_i = np.asarray(conn_i)
    conn_j = np.asarray(conn_j)

    pre_ids = []
    post_ids = []
    anchors = []
    num_pre = height * width
    ii = 0
    for i in range(num_pre):
        indexes = np.where(conn_i == i)[0]
        post_idx = conn_j[indexes]
        post_len = len(post_idx)
        pre_ids.extend([i] * post_len)
        post_ids.extend(post_idx)
        anchors.append([ii, ii + post_len])
        ii += post_len
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def fixed_prob(pre, post, prob, include_self=True, seed=None):
    """Connect the post-synaptic neurons with fixed probability.

    Parameters
    ----------
    pre : int, list
        Number of neurons in the pre-synaptic group.
    post : int, list
        Number of neurons in the post-synaptic group.
    prob : float
        The connection probability.
    include_self : bool
        Whether create (i, i) connection ?
    seed : None, int
        Seed the random generator.

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    if isinstance(pre, int):
        num_pre = pre
        all_pre = list(range(pre))
    elif isinstance(pre, (list, tuple)):
        all_pre = list(pre)
        num_pre = len(all_pre)
    else:
        raise ValueError
    if isinstance(post, int):
        num_post = post
        all_post = list(range(post))
    elif isinstance(post, (list, tuple)):
        all_post = list(post)
        num_post = len(all_post)
    else:
        raise ValueError
    assert isinstance(prob, (int, float)) and 0. <= prob <= 1.
    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_ids = []
    post_ids = []
    anchors = np.zeros((2, np.max(all_pre) + 1), dtype=np.int32)
    ii = 0
    for pre_idx in all_pre:
        random_vals = rng.random(num_post)
        idx_selected = list(np.where(random_vals < prob)[0])
        if (not include_self) and (pre_idx in idx_selected):
            idx_selected.remove(pre_idx)
        for post_idx in idx_selected:
            pre_ids.append(pre_idx)
            post_ids.append(all_post[post_idx])
        size_post = len(idx_selected)
        anchors[:, pre_idx] = [ii, ii + size_post]
        ii += size_post
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors)
    return pre_ids, post_ids, anchors


def fixed_prenum(num_pre, num_post, num, include_self=True, seed=None):
    """Connect the pre-synaptic neurons with fixed number for each
    post-synaptic neuron.

    Parameters
    ----------
    num_pre : int
        Number of neurons in the pre-synaptic group.
    num_post : int
        Number of neurons in the post-synaptic group.
    num : int
        The fixed connection number.
    include_self : bool
        Whether create (i, i) connection ?
    seed : None, int
        Seed the random generator.

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    assert isinstance(num_pre, int)
    assert isinstance(num_post, int)
    assert isinstance(num, int)
    rng = np.random if seed is None else np.random.RandomState(seed)

    conn_i = []
    conn_j = []
    for j in range(num_post):
        idx_selected = rng.choice(num_pre, num, replace=False).tolist()
        if (not include_self) and (j in idx_selected):
            idx_selected.remove(j)
        size_pre = len(idx_selected)
        conn_i.extend(idx_selected)
        conn_j.extend([j] * size_pre)
    conn_i = np.asarray(conn_i)
    conn_j = np.asarray(conn_j)

    pre_ids = []
    post_ids = []
    anchors = []
    ii = 0
    for i in range(num_pre):
        indexes = np.where(conn_i == i)[0]
        post_idx = conn_j[indexes]
        post_len = len(post_idx)
        pre_ids.extend([i] * post_len)
        post_ids.extend(post_idx)
        anchors.append([ii, ii + post_len])
        ii += post_len
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def fixed_postnum(num_pre, num_post, num, include_self=True, seed=None):
    """Connect the post-synaptic neurons with fixed number for each
    pre-synaptic neuron.

    Parameters
    ----------
    num_pre : int
        Number of neurons in the pre-synaptic group.
    num_post : int
        Number of neurons in the post-synaptic group.
    num : int
        The fixed connection number.
    include_self : bool
        Whether create (i, i) connection ?
    seed : None, int
        Seed the random generator.

    Returns
    -------
    connection : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron
         for each pre-synaptic neuron)
    """
    assert isinstance(num_pre, int)
    assert isinstance(num_post, int)
    assert isinstance(num, int)
    rng = np.random if seed is None else np.random.RandomState(seed)

    pre_ids = []
    post_ids = []
    anchors = []
    ii = 0
    for i in range(num_pre):
        idx_selected = rng.choice(num_post, num, replace=False).tolist()
        if (not include_self) and (i in idx_selected):
            idx_selected.remove(i)
        size_post = len(idx_selected)
        pre_ids.extend([i] * size_post)
        post_ids.extend(idx_selected)
        anchors.append([ii, ii + size_post])
        ii += size_post
    pre_ids = np.asarray(pre_ids)
    post_ids = np.asarray(post_ids)
    anchors = np.asarray(anchors).T
    return pre_ids, post_ids, anchors


def gaussian_weight(pre_geo, post_geo, sigma, w_max, w_min=None, normalize=True, include_self=True):
    """Builds a Gaussian connection pattern between the two populations, where
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
    pre_geo : tuple
        The geometry of pre-synaptic neuron group.
    post_geo : tuple
        The geometry of post-synaptic neuron group.
    sigma : float
        Width of the Gaussian function.
    w_max : float
        The weight amplitude of the Gaussian function.
    w_min : float, None
        The minimum weight value below which synapses are not created (default: 0.01 * `w_max`).
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the connection at the same position.

    Returns
    -------
    connection_and_weight : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron for each pre-synaptic neuron,
         weights)
    """

    # check parameters
    assert isinstance(pre_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert isinstance(post_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert len(post_geo) == len(pre_geo), 'The geometries of pre- and post-synaptic neuron group should be the same.'
    if len(pre_geo) == 1:
        pre_geo, post_geo = (1, pre_geo[0]), (1, post_geo[0])

    # get necessary data
    pre_num = int(np.prod(pre_geo))
    post_num = int(np.prod(post_geo))
    w_min = w_max * 0.01 if w_min is None else w_min

    # get the connections and weights
    i, j, w = [], [], []  # conn_i, conn_j, weights
    for pre_i in range(pre_num):
        # get normalized coordination
        pre_coords = (pre_i // pre_geo[1], pre_i % pre_geo[1])
        if normalize:
            pre_coords = (pre_coords[0] / (pre_geo[0] - 1) if pre_geo[0] > 1 else 1.,
                          pre_coords[1] / (pre_geo[1] - 1) if pre_geo[1] > 1 else 1.)

        for post_i in range(post_num):
            if (pre_i == post_i) and (not include_self):
                continue

            # get normalized coordination
            post_coords = (post_i // post_geo[1], post_i % post_geo[1])
            if normalize:
                post_coords = (post_coords[0] / (post_geo[0] - 1) if post_geo[0] > 1 else 1.,
                               post_coords[1] / (post_geo[1] - 1) if post_geo[1] > 1 else 1.)

            # Compute Euclidean distance between two coordinates
            distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])
            # get weight and connection
            value = w_max * np.exp(-distance / (2.0 * sigma ** 2))
            if value > w_min:
                i.append(pre_i)
                j.append(post_i)
                w.append(value)

    # format connections and weights
    pre_ids, post_ids, anchors, weights = from_ij(i, j, pre_num, others=(w,))
    return pre_ids, post_ids, anchors, weights


def gaussian_prob(pre_geo, post_geo, sigma, normalize=True, include_self=True, seed=None):
    """Builds a Gaussian connection pattern between the two populations, where
    the connection probability decay according to the gaussian function.

    Specifically,

    .. math::

        p=\\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma^2})

    where :math:`(x, y)` is the position of the pre-synaptic neuron
    and :math:`(x_c,y_c)` is the position of the post-synaptic neuron.

    Parameters
    ----------
    pre_geo : tuple
        The geometry of pre-synaptic neuron group.
    post_geo : tuple
        The geometry of post-synaptic neuron group.
    sigma : float
        Width of the Gaussian function.
    normalize : bool
        Whether normalize the coordination.
    include_self : bool
        Whether create the connection at the same position.
    seed : bool
        The random seed.

    Returns
    -------
    connection_and_weight : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron for each pre-synaptic neuron)
    """
    # check parameters
    assert isinstance(pre_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert isinstance(post_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert len(post_geo) == len(pre_geo), 'The geometries of pre- and post-synaptic neuron group should be the same.'
    if len(pre_geo) == 1:
        pre_geo, post_geo = (1, pre_geo[0]), (1, post_geo[0])

    # get necessary data
    pre_num = int(np.prod(pre_geo))
    post_num = int(np.prod(post_geo))
    rng = np.random if seed is None else np.random.RandomState(seed=seed)

    # get the connections
    i, j, p = [], [], []  # conn_i, conn_j, probabilities
    for pre_i in range(pre_num):
        # get normalized coordination
        pre_coords = (pre_i // pre_geo[1], pre_i % pre_geo[1])
        if normalize:
            pre_coords = (pre_coords[0] / (pre_geo[0] - 1) if pre_geo[0] > 1 else 1.,
                          pre_coords[1] / (pre_geo[1] - 1) if pre_geo[1] > 1 else 1.)

        for post_i in range(post_num):
            if (pre_i == post_i) and (not include_self):
                continue

            # get normalized coordination
            post_coords = (post_i // post_geo[1], post_i % post_geo[1])
            if normalize:
                post_coords = (post_coords[0] / (post_geo[0] - 1) if post_geo[0] > 1 else 1.,
                               post_coords[1] / (post_geo[1] - 1) if post_geo[1] > 1 else 1.)

            # Compute Euclidean distance between two coordinates
            distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])

            # get weight and probability
            i.append(pre_i)
            j.append(post_i)
            p.append(np.exp(-distance / (2.0 * sigma ** 2)))
    i, j, p = np.asarray(i), np.asarray(j), np.asarray(p)
    selected_idxs = np.where(rng.random(len(p)) < p)
    i, j = i[selected_idxs], j[selected_idxs]

    # format connections and weights
    pre_ids, post_ids, anchors = from_ij(i, j, pre_num)
    return pre_ids, post_ids, anchors


def dog(pre_geo, post_geo, sigmas, ws_max, w_min=None, normalize=True, include_self=True):
    """Builds a Difference-Of-Gaussian (dog) connection pattern between the two populations.

    Mathematically,

    .. math::

        w(x, y) = w_{max}^+ \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_+^2})
        -  w_{max}^- \\cdot \\exp(-\\frac{(x-x_c)^2+(y-y_c)^2}{2\\sigma_-^2})

    where weights smaller than :math:`0.01 * abs(w_{max} - w_{min})` are not created and
    self-connections are avoided by default (parameter allow_self_connections).

    Parameters
    ----------
    pre_geo : tuple
        The geometry of pre-synaptic neuron group.
    post_geo : tuple
        The geometry of post-synaptic neuron group.
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
        Whether create the connection at the same position.

    Returns
    -------
    connection_and_weight : tuple
        (pre-synaptic neuron indexes,
         post-synaptic neuron indexes,
         start and end positions of post-synaptic neuron for each pre-synaptic neuron,
         weights)
    """
    # check parameters
    assert isinstance(pre_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert isinstance(post_geo, (tuple, list)), 'Must provide a tuple geometry of the neuron group.'
    assert len(post_geo) == len(pre_geo), 'The geometries of pre- and post-synaptic neuron group should be the same.'
    if len(pre_geo) == 1:
        pre_geo, post_geo = (1, pre_geo[0]), (1, post_geo[0])
    assert len(sigmas) == 2, "Must provide two sigma."
    assert len(ws_max) == 2, 'Must provide two maximum weight (positive weight, negative weight).'

    # get necessary data
    pre_num = int(np.prod(pre_geo))
    post_num = int(np.prod(post_geo))
    sigma_p, sigma_n = sigmas
    w_max_p, w_max_n = ws_max
    w_min = np.abs(ws_max[0] - ws_max[1]) * 0.01 if w_min is None else w_min

    # get the connections and weights
    i, j, w = [], [], []  # conn_i, conn_j, weights
    for pre_i in range(pre_num):
        # get normalized coordination
        pre_coords = (pre_i // pre_geo[1], pre_i % pre_geo[1])
        if normalize:
            pre_coords = (pre_coords[0] / (pre_geo[0] - 1) if pre_geo[0] > 1 else 1.,
                          pre_coords[1] / (pre_geo[1] - 1) if pre_geo[1] > 1 else 1.)

        for post_i in range(post_num):
            if (pre_i == post_i) and (not include_self):
                continue

            # get normalized coordination
            post_coords = (post_i // post_geo[1], post_i % post_geo[1])
            if normalize:
                post_coords = (post_coords[0] / (post_geo[0] - 1) if post_geo[0] > 1 else 1.,
                               post_coords[1] / (post_geo[1] - 1) if post_geo[1] > 1 else 1.)

            # Compute Euclidean distance between two coordinates
            distance = sum([(pre_coords[i] - post_coords[i]) ** 2 for i in range(2)])
            # get weight and connection
            value = w_max_p * np.exp(-distance / (2.0 * sigma_p ** 2)) - \
                    w_max_n * np.exp(-distance / (2.0 * sigma_n ** 2))
            if np.abs(value) > w_min:
                i.append(pre_i)
                j.append(post_i)
                w.append(value)

    # format connections and weights
    pre_ids, post_ids, anchors, weights = from_ij(i, j, pre_num, others=(w,))
    return pre_ids, post_ids, anchors, weights


def scale_free(num_pre, num_post, **kwargs):
    raise NotImplementedError


def small_world(num_pre, num_post, **kwargs):
    raise NotImplementedError

