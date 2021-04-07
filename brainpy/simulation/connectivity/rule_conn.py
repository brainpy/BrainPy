# -*- coding: utf-8 -*-


import numpy as np

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import Connector

__all__ = [
    'One2One', 'one2one',
    'All2All', 'all2all',
    'GridFour', 'grid_four',
    'GridEight', 'grid_eight',
    'GridN',
]


@tools.numba_jit
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


@tools.numba_jit
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
