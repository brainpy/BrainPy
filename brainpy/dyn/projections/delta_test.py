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
import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm


class NetForHalfProj(bp.DynamicalSystem):
    def __init__(self):
        super().__init__()

        self.pre = bp.dyn.PoissonGroup(10, 100.)
        self.post = bp.dyn.LifRef(1)
        self.syn = bp.dyn.HalfProjDelta(bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

    def update(self):
        self.syn(self.pre())
        self.post()
        return self.post.V.value


def test1():
    net = NetForHalfProj()
    indices = bm.arange(1000).to_numpy()
    vs = bm.for_loop(net.step_run, indices, progress_bar=True)
    bp.visualize.line_plot(indices, vs, show=False)
    plt.close('all')


class NetForFullProj(bp.DynamicalSystem):
    def __init__(self):
        super().__init__()

        self.pre = bp.dyn.PoissonGroup(10, 100.)
        self.post = bp.dyn.LifRef(1)
        self.syn = bp.dyn.FullProjDelta(self.pre, 0., bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

    def update(self):
        self.syn()
        self.pre()
        self.post()
        return self.post.V.value


def test2():
    net = NetForFullProj()
    indices = bm.arange(1000).to_numpy()
    vs = bm.for_loop(net.step_run, indices, progress_bar=True)
    bp.visualize.line_plot(indices, vs, show=False)
    plt.close('all')
