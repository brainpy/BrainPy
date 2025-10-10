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
import numpy as np

import brainpy as bp
import brainpy.math as bm

show = False

neu_pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))


def test_ProjAlignPreMg1():
    class EICOBA_PreAlign(bp.DynamicalSystem):
        def __init__(self, scale=1., inp=20., delay=None):
            super().__init__()

            self.inp = inp
            self.E = bp.dyn.LifRefLTC(int(3200 * scale), **neu_pars)
            self.I = bp.dyn.LifRefLTC(int(800 * scale), **neu_pars)

            prob = 80 / (4000 * scale)

            self.E2I = bp.dyn.FullProjAlignPreSDMg(
                pre=self.E,
                syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
                delay=delay,
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=self.E.num, post=self.I.num), 0.6),
                out=bp.dyn.COBA(E=0.),
                post=self.I,
            )
            self.E2E = bp.dyn.FullProjAlignPreSDMg(
                pre=self.E,
                syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
                delay=delay,
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=self.E.num, post=self.E.num), 0.6),
                out=bp.dyn.COBA(E=0.),
                post=self.E,
            )
            self.I2E = bp.dyn.FullProjAlignPreSDMg(
                pre=self.I,
                syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
                delay=delay,
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=self.I.num, post=self.E.num), 6.7),
                out=bp.dyn.COBA(E=-80.),
                post=self.E,
            )
            self.I2I = bp.dyn.FullProjAlignPreSDMg(
                pre=self.I,
                syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
                delay=delay,
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=self.I.num, post=self.I.num), 6.7),
                out=bp.dyn.COBA(E=-80.),
                post=self.I,
            )

        def update(self):
            self.E2I()
            self.I2I()
            self.I2E()
            self.E2E()
            self.E(self.inp)
            self.I(self.inp)
            return self.E.spike.value

    net = EICOBA_PreAlign(0.5)
    indices = np.arange(400)
    spks = bm.for_loop(net.step_run, indices)
    bp.visualize.raster_plot(indices * bm.dt, spks, show=show)

    net = EICOBA_PreAlign(0.5, delay=1.)
    indices = np.arange(400)
    spks = bm.for_loop(net.step_run, indices)
    bp.visualize.raster_plot(indices * bm.dt, spks, show=show)

    plt.close()


def test_ProjAlignPostMg2():
    class EICOBA_PostAlign(bp.DynamicalSystem):
        def __init__(self, scale, inp=20., ltc=True, delay=None):
            super().__init__()
            self.inp = inp

            if ltc:
                self.E = bp.dyn.LifRefLTC(int(3200 * scale), **neu_pars)
                self.I = bp.dyn.LifRefLTC(int(800 * scale), **neu_pars)
            else:
                self.E = bp.dyn.LifRef(int(3200 * scale), **neu_pars)
                self.I = bp.dyn.LifRef(int(800 * scale), **neu_pars)

            prob = 80 / (4000 * scale)

            self.E2E = bp.dyn.FullProjAlignPostMg(
                pre=self.E,
                delay=delay,
                comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=self.E.num, post=self.E.num), 0.6),
                syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
                out=bp.dyn.COBA.desc(E=0.),
                post=self.E,
            )
            self.E2I = bp.dyn.FullProjAlignPostMg(
                pre=self.E,
                delay=delay,
                comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=self.E.num, post=self.I.num), 0.6),
                syn=bp.dyn.Expon.desc(self.I.varshape, tau=5.),
                out=bp.dyn.COBA.desc(E=0.),
                post=self.I,
            )
            self.I2E = bp.dyn.FullProjAlignPostMg(
                pre=self.I,
                delay=delay,
                comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=self.I.num, post=self.E.num), 6.7),
                syn=bp.dyn.Expon.desc(self.E.varshape, tau=10.),
                out=bp.dyn.COBA.desc(E=-80.),
                post=self.E,
            )
            self.I2I = bp.dyn.FullProjAlignPostMg(
                pre=self.I,
                delay=delay,
                comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=self.I.num, post=self.I.num), 6.7),
                syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
                out=bp.dyn.COBA.desc(E=-80.),
                post=self.I,
            )

        def update(self):
            self.E2I()
            self.I2I()
            self.I2E()
            self.E2E()
            self.E(self.inp)
            self.I(self.inp)
            return self.E.spike.value

    net = EICOBA_PostAlign(0.5)
    indices = np.arange(400)
    spks = bm.for_loop(net.step_run, indices)
    bp.visualize.raster_plot(indices * bm.dt, spks, show=show)

    net = EICOBA_PostAlign(0.5, delay=1.)
    indices = np.arange(400)
    spks = bm.for_loop(net.step_run, indices)
    bp.visualize.raster_plot(indices * bm.dt, spks, show=show)

    net = EICOBA_PostAlign(0.5, ltc=False)
    indices = np.arange(400)
    spks = bm.for_loop(net.step_run, indices)
    bp.visualize.raster_plot(indices * bm.dt, spks, show=show)

    plt.close()


def test_ProjAlignPost1():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale=1.):
            super().__init__()
            num = int(4000 * scale)
            self.num_exc = int(3200 * scale)
            self.num_inh = num - self.num_exc
            prob = 80 / num

            self.N = bp.dyn.LifRefLTC(num, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
            self.E = bp.dyn.HalfProjAlignPost(
                comm=bp.dnn.EventJitFPHomoLinear(self.num_exc, num, prob=prob, weight=0.6),
                syn=bp.dyn.Expon(size=num, tau=5.),
                out=bp.dyn.COBA(E=0.),
                post=self.N)
            self.I = bp.dyn.HalfProjAlignPost(
                comm=bp.dnn.EventJitFPHomoLinear(self.num_inh, num, prob=prob, weight=6.7),
                syn=bp.dyn.Expon(size=num, tau=10.),
                out=bp.dyn.COBA(E=-80.),
                post=self.N)

        def update(self, input):
            spk = self.delay.at('I')
            self.E(spk[:self.num_exc])
            self.I(spk[self.num_exc:])
            self.delay(self.N(input))
            return self.N.spike.value

    model = EINet(0.5)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    plt.close()


def test_ProjAlignPost2():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale, delay=None):
            super().__init__()
            ne, ni = int(3200 * scale), int(800 * scale)
            p = 80 / (ne + ni)

            self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.E2E = bp.dyn.FullProjAlignPost(pre=self.E,
                                                delay=delay,
                                                comm=bp.dnn.EventJitFPHomoLinear(ne, ne, prob=p, weight=0.6),
                                                syn=bp.dyn.Expon(size=ne, tau=5.),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.E)
            self.E2I = bp.dyn.FullProjAlignPost(pre=self.E,
                                                delay=delay,
                                                comm=bp.dnn.EventJitFPHomoLinear(ne, ni, prob=p, weight=0.6),
                                                syn=bp.dyn.Expon(size=ni, tau=5.),
                                                out=bp.dyn.COBA(E=0.),
                                                post=self.I)
            self.I2E = bp.dyn.FullProjAlignPost(pre=self.I,
                                                delay=delay,
                                                comm=bp.dnn.EventJitFPHomoLinear(ni, ne, prob=p, weight=6.7),
                                                syn=bp.dyn.Expon(size=ne, tau=10.),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.E)
            self.I2I = bp.dyn.FullProjAlignPost(pre=self.I,
                                                delay=delay,
                                                comm=bp.dnn.EventJitFPHomoLinear(ni, ni, prob=p, weight=6.7),
                                                syn=bp.dyn.Expon(size=ni, tau=10.),
                                                out=bp.dyn.COBA(E=-80.),
                                                post=self.I)

        def update(self, inp):
            self.E2E()
            self.E2I()
            self.I2E()
            self.I2I()
            self.E(inp)
            self.I(inp)
            return self.E.spike

    model = EINet(0.5, delay=1.)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    model = EINet(0.5, delay=None)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    plt.close()


def test_VanillaProj():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale=0.5):
            super().__init__()
            num = int(4000 * scale)
            self.ne = int(3200 * scale)
            self.ni = num - self.ne
            p = 80 / num

            self.N = bp.dyn.LifRefLTC(num, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
            self.syn1 = bp.dyn.Expon(size=self.ne, tau=5.)
            self.syn2 = bp.dyn.Expon(size=self.ni, tau=10.)
            self.E = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(self.ne, num, prob=p, weight=0.6),
                                        out=bp.dyn.COBA(E=0.),
                                        post=self.N)
            self.I = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(self.ni, num, prob=p, weight=6.7),
                                        out=bp.dyn.COBA(E=-80.),
                                        post=self.N)

        def update(self, input):
            spk = self.delay.at('I')
            self.E(self.syn1(spk[:self.ne]))
            self.I(self.syn2(spk[self.ne:]))
            self.delay(self.N(input))
            return self.N.spike.value

    model = EINet()
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    plt.close()


def test_ProjAlignPreMg1_v2():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale=1., delay=None):
            super().__init__()
            ne, ni = int(3200 * scale), int(800 * scale)
            p = 80 / (4000 * scale)
            self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.E2E = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                   syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                   delay=delay,
                                                   comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=p, weight=0.6),
                                                   out=bp.dyn.COBA(E=0.),
                                                   post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreSDMg(pre=self.E,
                                                   syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                   delay=delay,
                                                   comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=p, weight=0.6),
                                                   out=bp.dyn.COBA(E=0.),
                                                   post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                                   syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                   delay=delay,
                                                   comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=p, weight=6.7),
                                                   out=bp.dyn.COBA(E=-80.),
                                                   post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreSDMg(pre=self.I,
                                                   syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                   delay=delay,
                                                   comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=p, weight=6.7),
                                                   out=bp.dyn.COBA(E=-80.),
                                                   post=self.I)

        def update(self, inp):
            self.E2E()
            self.E2I()
            self.I2E()
            self.I2I()
            self.E(inp)
            self.I(inp)
            return self.E.spike

    model = EINet()
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    model = EINet(delay=1.)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    plt.close()


def test_ProjAlignPreMg2():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale=1., delay=None):
            super().__init__()
            ne, ni = int(3200 * scale), int(800 * scale)
            p = 80 / (4000 * scale)
            self.E = bp.dyn.LifRefLTC(ne, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.I = bp.dyn.LifRefLTC(ni, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 2.))
            self.E2E = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                   delay=delay,
                                                   syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                   comm=bp.dnn.JitFPHomoLinear(ne, ne, prob=p, weight=0.6),
                                                   out=bp.dyn.COBA(E=0.),
                                                   post=self.E)
            self.E2I = bp.dyn.FullProjAlignPreDSMg(pre=self.E,
                                                   delay=delay,
                                                   syn=bp.dyn.Expon.desc(size=ne, tau=5.),
                                                   comm=bp.dnn.JitFPHomoLinear(ne, ni, prob=p, weight=0.6),
                                                   out=bp.dyn.COBA(E=0.),
                                                   post=self.I)
            self.I2E = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                                   delay=delay,
                                                   syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                   comm=bp.dnn.JitFPHomoLinear(ni, ne, prob=p, weight=6.7),
                                                   out=bp.dyn.COBA(E=-80.),
                                                   post=self.E)
            self.I2I = bp.dyn.FullProjAlignPreDSMg(pre=self.I,
                                                   delay=delay,
                                                   syn=bp.dyn.Expon.desc(size=ni, tau=10.),
                                                   comm=bp.dnn.JitFPHomoLinear(ni, ni, prob=p, weight=6.7),
                                                   out=bp.dyn.COBA(E=-80.),
                                                   post=self.I)

        def update(self, inp):
            self.E2E()
            self.E2I()
            self.I2E()
            self.I2I()
            self.E(inp)
            self.I(inp)
            return self.E.spike

    model = EINet(scale=0.2, delay=None)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    model = EINet(scale=0.2, delay=1.)
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
    bp.visualize.raster_plot(indices, spks, show=show)

    plt.close()


def test_vanalla_proj_v2():
    class EINet(bp.DynSysGroup):
        def __init__(self, scale=1.):
            super().__init__()
            num = int(4000 * scale)
            self.ne = int(3200 * scale)
            self.ni = num - self.ne
            p = 80 / num

            self.N = bp.dyn.LifRefLTC(num, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                      V_initializer=bp.init.Normal(-55., 1.))
            self.delay = bp.VarDelay(self.N.spike, entries={'delay': 2})
            self.syn1 = bp.dyn.Expon(size=self.ne, tau=5.)
            self.syn2 = bp.dyn.Expon(size=self.ni, tau=10.)
            self.E = bp.dyn.VanillaProj(
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(p, pre=self.ne, post=num), weight=0.6),
                out=bp.dyn.COBA(E=0.),
                post=self.N
            )
            self.I = bp.dyn.VanillaProj(
                comm=bp.dnn.CSRLinear(bp.conn.FixedProb(p, pre=self.ni, post=num), weight=6.7),
                out=bp.dyn.COBA(E=-80.),
                post=self.N
            )

        def update(self, input):
            spk = self.delay.at('delay')
            self.E(self.syn1(spk[:self.ne]))
            self.I(self.syn2(spk[self.ne:]))
            self.delay(self.N(input))
            return self.N.spike.value

    model = EINet()
    indices = bm.arange(400)
    spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices, progress_bar=True)
    bp.visualize.raster_plot(indices, spks, show=show)
    plt.close()
