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
import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')
show = False


class Test_STDP(parameterized.TestCase):
    @parameterized.product(
        comm_method=['csr', 'dense', 'masked_linear', 'all2all', 'one2one'],
        delay=[None, 0., 2.],
        syn_model=['exp', 'dual_exp', 'ampa'],
        out_model=['cuba', 'coba', 'mg']
    )
    def test_STDP(self, comm_method, delay, syn_model, out_model):
        bm.random.seed()

        class STDPNet(bp.DynamicalSystem):
            def __init__(self, num_pre, num_post):
                super().__init__()
                self.pre = bp.dyn.LifRef(num_pre)
                self.post = bp.dyn.LifRef(num_post)

                if comm_method == 'all2all':
                    comm = bp.dnn.AllToAll(
                        self.pre.num, self.post.num, weight=bp.init.Uniform(.1, 0.1),
                        mode=bm.TrainingMode()
                    )
                elif comm_method == 'csr':
                    if syn_model == 'exp':
                        comm = bp.dnn.EventCSRLinear(
                            bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                            weight=bp.init.Uniform(0., 0.1),
                            mode=bm.TrainingMode()
                        )
                    else:
                        comm = bp.dnn.CSRLinear(
                            bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                            weight=bp.init.Uniform(0., 0.1),
                            mode=bm.TrainingMode()
                        )
                elif comm_method == 'masked_linear':
                    comm = bp.dnn.MaskedLinear(
                        bp.conn.FixedProb(1, pre=self.pre.num, post=self.post.num),
                        weight=bp.init.Uniform(0., 0.1),
                        mode=bm.TrainingMode()
                    )
                elif comm_method == 'dense':
                    comm = bp.dnn.Dense(
                        self.pre.num, self.post.num, W_initializer=bp.init.Uniform(.1, 0.1),
                        mode=bm.TrainingMode()
                    )
                elif comm_method == 'one2one':
                    comm = bp.dnn.OneToOne(self.pre.num, weight=bp.init.Uniform(.1, 0.1), mode=bm.TrainingMode())
                else:
                    raise ValueError

                if syn_model == 'exp':
                    syn = bp.dyn.Expon.desc(self.post.varshape, tau=5.)
                elif syn_model == 'dual_exp':
                    syn = bp.dyn.DualExpon.desc(self.post.varshape)
                elif syn_model == 'dual_exp_v2':
                    syn = bp.dyn.DualExponV2.desc(self.post.varshape)
                elif syn_model == 'ampa':
                    syn = bp.dyn.AMPA.desc(self.post.varshape)
                else:
                    raise ValueError

                if out_model == 'cuba':
                    out = bp.dyn.CUBA.desc()
                elif out_model == 'coba':
                    out = bp.dyn.COBA.desc(E=0.)
                elif out_model == 'mg':
                    out = bp.dyn.MgBlock.desc(E=0.)
                else:
                    raise ValueError

                self.syn = bp.dyn.STDP_Song2000(
                    pre=self.pre,
                    delay=delay,
                    comm=comm,
                    syn=syn,
                    out=out,
                    post=self.post,
                    tau_s=16.8,
                    tau_t=33.7,
                    A1=0.96,
                    A2=0.53,
                    W_min=0.,
                    W_max=1.
                )

            def update(self, I_pre, I_post):
                self.syn()
                self.pre(I_pre)
                self.post(I_post)
                conductance = self.syn.refs['syn'].g
                Apre = self.syn.refs['pre_trace'].g
                Apost = self.syn.refs['post_trace'].g
                current = self.post.sum_current_inputs(self.post.V)
                if comm_method == 'dense':
                    w = self.syn.comm.W.flatten()
                else:
                    w = self.syn.comm.weight.flatten()
                return self.pre.spike, self.post.spike, conductance, Apre, Apost, current, w

        duration = 300.
        I_pre = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                        [5, 15, 15, 15, 15, 15, 100, 15, 15, 15, 15, 15,
                                         duration - 255])
        I_post = bp.inputs.section_input([0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0],
                                         [10, 15, 15, 15, 15, 15, 90, 15, 15, 15, 15, 15,
                                          duration - 250])

        net = STDPNet(1, 1)

        def run(i, I_pre, I_post):
            pre_spike, post_spike, g, Apre, Apost, current, W = net.step_run(i, I_pre, I_post)
            return pre_spike, post_spike, g, Apre, Apost, current, W

        indices = np.arange(int(duration / bm.dt))
        pre_spike, post_spike, g, Apre, Apost, current, W = bm.for_loop(run, [indices, I_pre, I_post])

        # import matplotlib.pyplot as plt
        # fig, gs = bp.visualize.get_figure(4, 1, 3, 10)
        # bp.visualize.line_plot(indices, g, ax=fig.add_subplot(gs[0, 0]))
        # bp.visualize.line_plot(indices, Apre, ax=fig.add_subplot(gs[1, 0]))
        # bp.visualize.line_plot(indices, Apost, ax=fig.add_subplot(gs[2, 0]))
        # bp.visualize.line_plot(indices, W, ax=fig.add_subplot(gs[3, 0]))
        # plt.show()
