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
import numbers
from typing import Any
from typing import Union, Optional

from brainpy import check, math as bm
from brainpy.context import share
from brainpy.dynsys import Dynamic
from brainpy.dynsys import Projection
from brainpy.mixin import SupportAutoDelay
from brainpy.types import Shape

__all__ = [
    'InputVar',
    'PoissonInput',
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

    def reset_state(self, batch_or_mode=None, **kwargs):
        self.input = self.init_variable(bm.zeros, batch_or_mode)

    def update(self, *args, **kwargs):
        return self.input.value

    def return_info(self):
        return self.input

    def clear_input(self, *args, **kwargs):
        self.reset_state(self.mode)


class PoissonInput(Projection):
    """Poisson Input to the given :py:class:`~.Variable`.

    Adds independent Poisson input to a target variable. For large
    numbers of inputs, this is much more efficient than creating a
    `PoissonGroup`. The synaptic events are generated randomly during the
    simulation and are not preloaded and stored in memory. All the inputs must
    target the same variable, have the same frequency and same synaptic weight.
    All neurons in the target variable receive independent realizations of
    Poisson spike trains.

    Args:
      target_var: The variable that is targeted by this input. Should be an instance of :py:class:`~.Variable`.
      num_input: The number of inputs.
      freq: The frequency of each of the inputs. Must be a scalar.
      weight: The synaptic weight. Must be a scalar.
      name: The target name.
      mode: The computing mode.
    """

    def __init__(
        self,
        target_var: bm.Variable,
        num_input: int,
        freq: Union[int, float],
        weight: Union[int, float],
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, mode=mode)

        if not isinstance(target_var, bm.Variable):
            raise TypeError(f'"target_var" must be an instance of Variable. '
                            f'But we got {type(target_var)}: {target_var}')
        self.target_var = target_var
        self.num_input = check.is_integer(num_input, min_bound=1)
        self.freq = check.is_float(freq, min_bound=0., allow_int=True)
        self.weight = check.is_float(weight, allow_int=True)

    def reset_state(self, *args, **kwargs):
        pass

    def update(self):
        p = self.freq * share['dt'] / 1e3
        a = self.num_input * p
        b = self.num_input * (1 - p)

        if isinstance(share['dt'], numbers.Number):  # dt is not traced
            if (a > 5) and (b > 5):
                inp = bm.random.normal(a, b * p, self.target_var.shape)
            else:
                inp = bm.random.binomial(self.num_input, p, self.target_var.shape)

        else:  # dt is traced
            inp = bm.cond((a > 5) * (b > 5),
                          lambda: bm.random.normal(a, b * p, self.target_var.shape),
                          lambda: bm.random.binomial(self.num_input, p, self.target_var.shape))

        # inp = bm.sharding.partition(inp, self.target_var.sharding)
        self.target_var += inp * self.weight

    def __repr__(self):
        return f'{self.name}(num_input={self.num_input}, freq={self.freq}, weight={self.weight})'
