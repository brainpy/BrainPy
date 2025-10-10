# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import brainpy as bp


class EINet3(bp.DynSysGroup):
    def __init__(self):
        super().__init__()
        self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                               V_initializer=bp.init.Normal(-55., 2.))
        self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
        self.E = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                            syn=bp.dyn.Expon.desc(size=4000, tau=5.),
                                            out=bp.dyn.COBA.desc(E=0.),
                                            post=self.N)
        self.I = bp.dyn.HalfProjAlignPostMg(comm=bp.dnn.EventJitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                            syn=bp.dyn.Expon.desc(size=4000, tau=10.),
                                            out=bp.dyn.COBA.desc(E=-80.),
                                            post=self.N)

    def update(self, input):
        spk = self.delay.at('I')
        self.E(spk[:3200])
        self.I(spk[3200:])
        self.delay(self.N(input))
        return self.N.spike.value


def test1():
    model = EINet3()

    bp.integrators.compile_integrators(model.step_run, 0, 0.)
    for intg in model.nodes().subset(bp.Integrator).values():
        print(intg.to_math_expr())
