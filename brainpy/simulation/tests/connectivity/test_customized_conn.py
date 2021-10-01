# -*- coding: utf-8 -*-

import brainpy as bp


class IndexConn(bp.connect.TwoEndConnector):
    def __init__(self, i, j):
        super(IndexConn, self).__init__()

        self.pre_ids = bp.ops.as_tensor(i)
        self.post_ids = bp.ops.as_tensor(j)

    def __call__(self, pre_size, post_size):
        self.num_pre = bp.size2len(pre_size)  # this is useful when create "pre2post" ,
        # "pre2syn"  etc. structures
        self.num_post = bp.size2len(post_size)  # this is useful when create "post2pre" ,
        # "post2syn"  etc. structures
        return self


def test():
    conn = IndexConn(i=[0, 1, 2], j=[0, 0, 0])
    conn = conn(pre_size=5, post_size=3)

    print(conn.requires('pre2post'))
