# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Union, Dict, Callable

import brainpy.math as bm
from brainpy.connect import TwoEndConnector
from brainpy.dyn.base import NeuDyn
from brainpy.dynold.synapses import TwoEndConn
from brainpy.initialize import Initializer, parameter
from brainpy.types import ArrayType

__all__ = [
    'GapJunction',
]


class GapJunction(TwoEndConn):
    def __init__(
        self,
        pre: NeuDyn,
        post: NeuDyn,
        conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
        comp_method: str = 'dense',
        g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
        name: str = None,
    ):
        super(GapJunction, self).__init__(pre=pre,
                                          post=post,
                                          conn=conn,
                                          name=name)
        # checking
        self.check_pre_attrs('V')
        self.check_post_attrs('V', 'input')

        # assert isinstance(self.output, _NullSynOut)
        # assert isinstance(self.stp, _NullSynSTP)

        # connections
        self.comp_method = comp_method
        if comp_method == 'dense':
            self.conn_mat = self.conn.require('conn_mat')
            self.weights = parameter(g_max, (pre.num, post.num), allow_none=False)
        elif comp_method == 'sparse':
            self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
            self.weights = parameter(g_max, self.pre_ids.shape, allow_none=False)
        else:
            raise ValueError

    def update(self):
        if self.comp_method == 'dense':
            # pre -> post
            diff = (self.pre.V.reshape((-1, 1)) - self.post.V) * self.conn_mat * self.weights
            self.post.input += bm.einsum('ij->j', diff)
            # post -> pre
            self.pre.input += bm.einsum('ij->i', -diff)
        else:
            diff = (self.pre.V[self.pre_ids] - self.post.V[self.post_ids]) * self.weights
            self.post.input += bm.syn2post_sum(diff, self.post_ids, self.post.num)
            self.pre.input += bm.syn2post_sum(-diff, self.pre_ids, self.pre.num)

    def reset_state(self, batch_size=None):
        pass
