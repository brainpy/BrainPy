# -*- coding: utf-8 -*-

import numpy as onp

from .base import Connector
from .. import numpy as np
from ..errors import ModelUseError

try:
    import numba as nb
except ImportError:
    nb = None

__all__ = ['One2One', 'All2All',
           'GridFour', 'grid_four',
           'GridEight', 'grid_eight',
           'GridN',
           'FixedPostNum', 'FixedPreNum', 'FixedProb',
           'GaussianProb', 'GaussianWeight', 'DOG',
           'SmallWorld', 'ScaleFree']


def _product(a_list):
    p = 1
    for i in a_list:
        p *= i
    return p


class One2One(Connector):
    """
    Connect two neuron groups one by one. This means
    The two neuron groups should have the same size.
    """

    def __call__(self, pre_indices, post_indices):
        try:
            assert np.shape(pre_indices) == np.shape(post_indices)
        except AssertionError:
            raise ModelUseError('One2One connection must be defined in two groups with the same shape.')
        pre_ids = np.asarray(pre_indices.flatten(), dtype=np.int_)
        post_ids = np.asarray(post_indices.flatten(), dtype=np.int_)
        return pre_ids, post_ids


one2one = One2One()


class All2All(Connector):
    """Connect each neuron in first group to all neurons in the
    post-synaptic neuron groups. It means this kind of conn
    will create (num_pre x num_post) synapses.
    """

    def __init__(self, include_self=True):
        self.include_self = include_self

    def __call__(self, pre_indices, post_indices):
        pre_indices = pre_indices.flatten()
        post_indices = post_indices.flatten()
        num_pre = len(pre_indices)
        num_post = len(post_indices)
        mat = np.ones((num_pre, num_post))
        if not self.include_self:
            for i in range(min([num_post, num_pre])):
                mat[i, i] = 0
        pre_ids, post_ids = np.where(mat > 0)
        pre_ids = np.asarray(pre_ids, dtype=np.int_)
        post_ids = np.asarray(post_ids, dtype=np.int_)
        return pre_ids, post_ids


all2all = All2All(include_self=True)


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


class GridFour(Connector):
    """The nearest four neighbors conn method."""

    def __init__(self, include_self=False):
        self.include_self = include_self

    def __call__(self, pre_indices, post_indices=None):
        assert np.ndim(pre_indices) == 2
        if post_indices is not None:
            assert np.shape(pre_indices) == np.shape(post_indices)
        height, width = pre_indices.shape

        if nb is not None:
            f = nb.njit(_grid_four)
        else:
            f = _grid_four

        conn_i = []
        conn_j = []
        for row in range(height):
            a = f(height, width, row, include_self=self.include_self)
            conn_i.extend(a[0])
            conn_j.extend(a[1])
        conn_i = np.asarray(conn_i)
        conn_j = np.asarray(conn_j)

        if post_indices is None:
            return pre_indices.flatten()[conn_i], pre_indices.flatten()[conn_j]
        else:
            return pre_indices.flatten()[conn_i], post_indices.flatten()[conn_j]


grid_four = GridFour()



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

    def __call__(self, pre_indices, post_indices=None):
        assert np.ndim(pre_indices) == 2
        if post_indices is not None:
            assert np.shape(pre_indices) == np.shape(post_indices)

        height, width = pre_indices.shape

        if nb is not None:
            f = nb.njit(_grid_n)
        else:
            f = _grid_n

        conn_i = []
        conn_j = []
        for row in range(height):
            res = f(height=height, width=width, row=row,
                    n=self.n, include_self=self.include_self)
            conn_i.extend(res[0])
            conn_j.extend(res[1])
        conn_i = np.asarray(conn_i, dtype=np.int_)
        conn_j = np.asarray(conn_j, dtype=np.int_)
        if post_indices is None:
            return pre_indices.flatten()[conn_i], pre_indices.flatten()[conn_j]
        else:
            return pre_indices.flatten()[conn_i], post_indices.flatten()[conn_j]


class GridEight(Connector):
    """The nearest eight neighbors conn method."""

    def __init__(self, include_self=False):
        self.include_self = include_self
        self.connector = GridN(n=1)

    def __call__(self, pre_indices, post_indices=None):
        return self.connector(pre_indices, post_indices)


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
        self.prob = prob
        self.include_self = include_self
        self.seed = seed

    def __call__(self, pre_indices, post_indices):
        pre_indices = pre_indices.flatten()
        post_indices = post_indices.flatten()

        num_pre, num_post = len(pre_indices), len(post_indices)
        mat = np.random.random(size=(num_pre, num_post))
        if not self.include_self:
            for i in range(min([num_pre, num_post])):
                mat[i, i] = 1.
        pre_ids, post_ids = np.where(mat < self.prob)
        pre_ids = pre_indices[pre_ids]
        post_ids = post_indices[post_ids]
        return pre_ids, post_ids


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

    def __call__(self, pre_indices, post_indices):
        pre_indices = pre_indices.flatten()
        post_indices = post_indices.flatten()
        num_pre = len(pre_indices)
        num_post = len(post_indices)
        num = self.num if isinstance(self.num, int) else int(self.num * num_pre)
        assert num <= num_pre, f'"num" must be less than "num_pre", but got {num} > {num_pre}'

        pre_ids, post_ids = [], []
        for j in range(num_post):
            idx_selected = np.random.choice(num_pre, num, replace=False)
            pre_ids.extend(idx_selected)
            post_ids.extend(np.ones_like(idx_selected) * j)
        pre_ids = np.asarray(pre_ids, dtype=np.int64)
        post_ids = np.asarray(post_ids, dtype=np.int64)
        return pre_ids, post_ids


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
        self.rng = np.random if seed is None else onp.random.RandomState(seed)

    def __call__(self, pre_indices, post_indices):
        pre_indices = pre_indices.flatten()
        post_indices = post_indices.flatten()
        num_pre = len(pre_indices)
        num_post = len(post_indices)
        num = self.num if isinstance(self.num, int) else int(self.num * num_post)
        assert num <= num_post, f'"num" must be less than "num_post", but got {num} > {num_post}'

        pre_ids, post_ids = [], []
        for i in range(num_pre):
            idx_selected = self.rng.choice(num_post, num, replace=False).tolist()
            post_ids.extend(idx_selected)
            pre_ids.extend(np.ones_like(idx_selected) * i)
        pre_ids = np.asarray(pre_ids, dtype=np.int64)
        post_ids = np.asarray(post_ids, dtype=np.int64)
        return pre_ids, post_ids


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

    def __call__(self, pre_indices, post_indices):
        num_pre = np.size(pre_indices)
        num_post = np.size(post_indices)

        assert np.ndim(pre_indices) == 2
        assert np.ndim(post_indices) == 2
        pre_height, pre_width = pre_indices.shape
        post_height, post_width = post_indices.shape

        if nb is not None:
            f_gaussian_weight = nb.njit(_gaussian_weight)
        else:
            f_gaussian_weight = _gaussian_weight

        # get the connections and weights
        i, j, w = [], [], []
        for pre_i in range(num_pre):
            a = f_gaussian_weight(pre_i=pre_i, pre_width=pre_width, pre_height=pre_height,
                                  num_post=num_post, post_width=post_width, post_height=post_height,
                                  w_max=self.w_max, w_min=self.w_min, sigma=self.sigma,
                                  normalize=self.normalize, include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            w.extend(a[2])

        pre_ids = np.asarray(i, dtype=np.int_)
        pre_ids = pre_indices.flatten()[pre_ids]
        post_ids = np.asarray(j, dtype=np.int_)
        post_ids = post_indices.flatten()[post_ids]
        weights = np.asarray(w, dtype=np.float_)
        return pre_ids, post_ids, weights


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
        self.sigma = sigma
        self.p_min = p_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, pre_indices, post_indices):
        num_pre = np.size(pre_indices)
        num_post = np.size(post_indices)

        assert np.ndim(pre_indices) == 2
        assert np.ndim(post_indices) == 2
        pre_height, pre_width = pre_indices.shape
        post_height, post_width = post_indices.shape

        if nb is not None:
            f = nb.njit(_gaussian_prob)
        else:
            f = _gaussian_prob

        # get the connections
        i, j, p = [], [], []  # conn_i, conn_j, probabilities
        for pre_i in range(num_pre):
            a = f(pre_i=pre_i, pre_width=pre_width, pre_height=pre_height,
                  num_post=num_post, post_width=post_width, post_height=post_height,
                  p_min=self.p_min, sigma=self.sigma,
                  normalize=self.normalize, include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            p.extend(a[2])

        i = np.asarray(i, dtype=np.int_)
        j = np.asarray(j, dtype=np.int_)
        p = np.asarray(p, dtype=np.float_)

        selected_idxs = np.where(np.random.random(len(p)) < p)[0]
        i, j = i[selected_idxs], j[selected_idxs]
        return pre_indices.flatten()[i], post_indices.flatten()[j]


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
        self.sigma_p, self.sigma_n = sigmas
        self.w_max_p, self.w_max_n = ws_max
        self.w_min = np.abs(ws_max[0] - ws_max[1]) * 0.01 if w_min is None else w_min
        self.normalize = normalize
        self.include_self = include_self

    def __call__(self, pre_indices, post_indices):
        num_pre = np.size(pre_indices)
        num_post = np.size(post_indices)

        assert np.ndim(pre_indices) == 2
        assert np.ndim(post_indices) == 2
        pre_height, pre_width = pre_indices.shape
        post_height, post_width = post_indices.shape

        if nb is not None:
            f = nb.njit(_dog)
        else:
            f = _dog

        # get the connections and weights
        i, j, w = [], [], []  # conn_i, conn_j, weights
        for pre_i in range(num_pre):
            a = f(pre_i=pre_i, pre_width=pre_width, pre_height=pre_height,
                  num_post=num_post, post_width=post_width, post_height=post_height,
                  w_max_p=self.w_max_p, w_max_n=self.w_max_n,
                  w_min=self.w_min, sigma_p=self.sigma_p, sigma_n=self.sigma_n,
                  normalize=self.normalize, include_self=self.include_self)
            i.extend(a[0])
            j.extend(a[1])
            w.extend(a[2])

        # format connections and weights
        i = np.asarray(i, dtype=np.int_)
        j = np.asarray(j, dtype=np.int_)
        w = np.asarray(w, dtype=np.float_)
        return i, j, w


class ScaleFree(Connector):
    def __init__(self):
        raise NotImplementedError


class SmallWorld(Connector):
    def __init__(self):
        raise NotImplementedError
