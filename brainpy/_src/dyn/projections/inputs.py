from typing import Optional, Any

from brainpy import math as bm
from brainpy._src.dynsys import Dynamic
from brainpy._src.mixin import SupportAutoDelay
from brainpy.types import Shape

__all__ = [
    'InputVar',
]


class InputVar(Dynamic, SupportAutoDelay):
    """Define an input variable.

    Example::

        import brainpy as bp


        class Exponential(bp.Projection):
            def __init__(self, pre, post, prob, g_max, tau, E=0.):
                super().__init__()
                self.proj = bp.dyn.ProjAlignPostMg2(
                    pre=pre,
                    delay=None,
                    comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    syn=bp.dyn.Expon.desc(post.num, tau=tau),
                    out=bp.dyn.COBA.desc(E=E),
                    post=post,
                )


        class EINet(bp.DynSysGroup):
            def __init__(self, num_exc, num_inh, method='exp_auto'):
                super(EINet, self).__init__()

                # neurons
                pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                            V_initializer=bp.init.Normal(-55., 2.), method=method)
                self.E = bp.dyn.LifRef(num_exc, **pars)
                self.I = bp.dyn.LifRef(num_inh, **pars)

                # synapses
                w_e = 0.6  # excitatory synaptic weight
                w_i = 6.7  # inhibitory synaptic weight

                # Neurons connect to each other randomly with a connection probability of 2%
                self.E2E = Exponential(self.E, self.E, 0.02, g_max=w_e, tau=5., E=0.)
                self.E2I = Exponential(self.E, self.I, 0.02, g_max=w_e, tau=5., E=0.)
                self.I2E = Exponential(self.I, self.E, 0.02, g_max=w_i, tau=10., E=-80.)
                self.I2I = Exponential(self.I, self.I, 0.02, g_max=w_i, tau=10., E=-80.)

                # define input variables given to E/I populations
                self.Ein = bp.dyn.InputVar(self.E.varshape)
                self.Iin = bp.dyn.InputVar(self.I.varshape)
                self.E.add_inp_fun('', self.Ein)
                self.I.add_inp_fun('', self.Iin)


        net = EINet(3200, 800, method='exp_auto')  # "method": the numerical integrator method
        runner = bp.DSRunner(net, monitors=['E.spike', 'I.spike'], inputs=[('Ein.input', 20.), ('Iin.input', 20.)])
        runner.run(100.)

        # visualization
        bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'],
                                 title='Spikes of Excitatory Neurons', show=True)
        bp.visualize.raster_plot(runner.mon.ts, runner.mon['I.spike'],
                                 title='Spikes of Inhibitory Neurons', show=True)


    """
    def __init__(
            self,
            size: Shape,
            keep_size: bool = False,
            sharding: Optional[Any] = None,
            name: Optional[str] = None,
            mode: Optional[bm.Mode] = None,
            method: str = 'exp_auto'
    ):
        super().__init__(size=size, keep_size=keep_size, sharding=sharding, name=name, mode=mode, method=method)

        self.reset_state(self.mode)

    def reset_state(self, batch_or_mode=None):
        self.input = self.init_variable(bm.zeros, batch_or_mode)

    def update(self, *args, **kwargs):
        return self.input.value

    def return_info(self):
        return self.input

    def clear_input(self, *args, **kwargs):
        self.reset_state(self.mode)
