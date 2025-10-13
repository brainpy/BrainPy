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
from functools import partial
from typing import Union, Callable, Optional, Any, Sequence

from jax.lax import stop_gradient

import brainpy.math as bm
from brainpy.check import is_initializer
from brainpy.context import share
from brainpy.dyn._docs import ref_doc, lif_doc, pneu_doc, dpneu_doc, ltc_doc, if_doc
from brainpy.dyn.neurons.base import GradNeuDyn
from brainpy.initialize import ZeroInit, OneInit, noise as init_noise
from brainpy.integrators import odeint, sdeint, JointEq
from brainpy.types import Shape, ArrayType, Sharding

__all__ = [
    'IF',
    'IFLTC',
    'Lif',
    'LifLTC',
    'LifRef',
    'LifRefLTC',
    'ExpIF',
    'ExpIFLTC',
    'ExpIFRef',
    'ExpIFRefLTC',
    'AdExIF',
    'AdExIFLTC',
    'AdExIFRef',
    'AdExIFRefLTC',
    'QuaIF',
    'QuaIFLTC',
    'QuaIFRef',
    'QuaIFRefLTC',
    'AdQuaIF',
    'AdQuaIFLTC',
    'AdQuaIFRef',
    'AdQuaIFRefLTC',
    'Gif',
    'GifLTC',
    'GifRef',
    'GifRefLTC',
    'Izhikevich',
    'IzhikevichLTC',
    'IzhikevichRef',
    'IzhikevichRefLTC',
]


class IFLTC(GradNeuDyn):
    r"""Leaky Integrator Model %s.

    **Model Descriptions**

    This class implements a leaky integrator model, in which its dynamics is
    given by:

    .. math::

       \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t)

    where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
    membrane potential, :math:`\tau` is the time constant, and :math:`R` is the
    resistance.


    Args:
      %s
      %s
      %s
    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = 0.,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)

        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.tau = self.init_param(tau)
        self.R = self.init_param(R)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # integral
        self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def derivative(self, V, t, I):
        I = self.sum_current_inputs(V, init=I)
        return (-V + self.V_rest + self.R * I) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        self.V.value = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        return self.V.value

    def return_info(self):
        return self.V


class IF(IFLTC):
    def derivative(self, V, t, I):
        return (-V + self.V_rest + self.R * I) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


IF.__doc__ = IFLTC.__doc__ % ('', if_doc, pneu_doc, dpneu_doc)
IFLTC.__doc__ = IFLTC.__doc__ % (ltc_doc, if_doc, pneu_doc, dpneu_doc)


class LifLTC(GradNeuDyn):
    r"""Leaky integrate-and-fire neuron model with liquid time-constant.

    The formal equations of a LIF model [1]_ is given by:

    .. math::

        \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
        \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset}

    where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
    membrane potential, :math:`V_{reset}` is the reset membrane potential,
    :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
    and :math:`I` is the time-variant synaptic inputs.


    .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
           neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

    **Examples**

    There is an example usage: mustang u r lvd by the blonde boy

    .. code-block:: python

        import brainpy as bp

        lif = bp.dyn.LifLTC(1)

        # raise input current from 4 mA to 40 mA
        inputs = bp.inputs.ramp_input(4, 40, 700, 100, 600,)

        runner = bp.DSRunner(lif, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)

    Args:
      %s
      %s
      %s

    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = 0.,
        V_reset: Union[float, ArrayType, Callable] = -5.,
        V_th: Union[float, ArrayType, Callable] = 20.,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # noise
        noise: Optional[Union[float, ArrayType, Callable]] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)

        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.tau = self.init_param(tau)
        self.R = self.init_param(R)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # noise
        self.noise = init_noise(noise, self.varshape)

        # integral
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def derivative(self, V, t, I):
        I = self.sum_current_inputs(V, init=I)
        return (-V + self.V_rest + self.R * I) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)

        self.V.value = V
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class Lif(LifLTC):
    r"""Leaky integrate-and-fire neuron model.

    The formal equations of a LIF model [1]_ is given by:

    .. math::

        \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
        \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset}

    where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
    membrane potential, :math:`V_{reset}` is the reset membrane potential,
    :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
    and :math:`I` is the time-variant synaptic inputs.

    .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
           neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        lif = bp.dyn.Lif(1)

        # raise input current from 4 mA to 40 mA
        inputs = bp.inputs.ramp_input(4, 40, 700, 100, 600,)
        runner = bp.DSRunner(lif, monitors=['V'])
        runner.run(inputs=inputs)
        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)


    Args:
      %s
      %s
      %s

    """

    def derivative(self, V, t, I):
        return (-V + self.V_rest + self.R * I) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


Lif.__doc__ = Lif.__doc__ % (lif_doc, pneu_doc, dpneu_doc)
LifLTC.__doc__ = LifLTC.__doc__ % (lif_doc, pneu_doc, dpneu_doc)


class LifRefLTC(LifLTC):
    r"""Leaky integrate-and-fire neuron model with liquid time-constant which has refractory periods .

    The formal equations of a LIF model [1]_ is given by:

    .. math::

        \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
        \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset} \quad
        \text{last} \quad \tau_{ref} \quad  \text{ms}

    where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
    membrane potential, :math:`V_{reset}` is the reset membrane potential,
    :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
    :math:`\tau_{ref}` is the refractory time period,
    and :math:`I` is the time-variant synaptic inputs.

    .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
           neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.LifRefLTC(1, )


        # example for section input
        inputs = bp.inputs.section_input([0., 21., 0.], [100., 300., 100.])

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)

    Args:
      %s
      %s
      %s
      %s

    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        detach_spk: bool = False,
        spk_reset: str = 'soft',
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = 0.,
        V_reset: Union[float, ArrayType, Callable] = -5.,
        V_th: Union[float, ArrayType, Callable] = 20.,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Optional[Union[float, ArrayType, Callable]] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th=V_th,
            R=R,
            tau=tau,
            V_initializer=V_initializer,

            noise=noise,
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e7)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class LifRef(LifRefLTC):
    r"""Leaky integrate-and-fire neuron model %s which has refractory periods.

    The formal equations of a LIF model [1]_ is given by:

    .. math::

        \tau \frac{dV}{dt} = - (V(t) - V_{rest}) + RI(t) \\
        \text{after} \quad V(t) \gt V_{th}, V(t) = V_{reset} \quad
        \text{last} \quad \tau_{ref} \quad  \text{ms}

    where :math:`V` is the membrane potential, :math:`V_{rest}` is the resting
    membrane potential, :math:`V_{reset}` is the reset membrane potential,
    :math:`V_{th}` is the spike threshold, :math:`\tau` is the time constant,
    :math:`\tau_{ref}` is the refractory time period,
    and :math:`I` is the time-variant synaptic inputs.

    .. [1] Abbott, Larry F. "Lapicque’s introduction of the integrate-and-fire model
           neuron (1907)." Brain research bulletin 50, no. 5-6 (1999): 303-304.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.LifRef(1, )


        # example for section input
        inputs = bp.inputs.section_input([0., 21., 0.], [100., 300., 100.])

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)

    Args:
      %s
      %s
      %s
      %s

    """

    def derivative(self, V, t, I):
        return (-V + self.V_rest + self.R * I) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


LifRef.__doc__ = LifRefLTC.__doc__ % (lif_doc, pneu_doc, dpneu_doc, ref_doc)
LifRefLTC.__doc__ = LifRefLTC.__doc__ % (lif_doc, pneu_doc, dpneu_doc, ref_doc)


class ExpIFLTC(GradNeuDyn):
    r"""Exponential integrate-and-fire neuron model with liquid time-constant.

      **Model Descriptions**

      In the exponential integrate-and-fire model [1]_, the differential
      equation for the membrane potential is given by

      .. math::

          \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
          \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

      This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
      and "threshold" :math:`\vartheta_{rh}`.

      The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
      defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
      :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
      where :math:`\tau_{\rm ref}` is an absolute refractory time.
      If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
      its exact value does not play any role. The reason is that the upswing of the action
      potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
      an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
      convenience. For a formal mathematical analysis of the model, the threshold can be pushed
      to infinity.

      The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
      and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
      It is one of the prominent examples of a precise theoretical prediction in computational
      neuroscience that was later confirmed by experimental neuroscience.

      Two important remarks:

      - (i) The right-hand side of the above equation contains a nonlinearity
        that can be directly extracted from experimental data [3]_. In this sense the exponential
        nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
      - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
        rate for constant input, and the linear response to fluctuations, even in the presence
        of input noise [4]_.

      **References**

      .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
             mechanisms determine the neuronal response to fluctuating
             inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
      .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
             Neuronal dynamics: From single neurons to networks and models
             of cognition. Cambridge University Press.
      .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
             Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
             are reliable predictors of naturalistic pyramidal-neuron voltage
             traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
      .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
             integrate-and-fire neurons to modulated current-based and
             conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
      .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire

      **Examples**

      There is a simple usage example::

          import brainpy as bp

          neu = bp.dyn.ExpIFLTC(1, )

          # example for section input
          inputs = bp.inputs.section_input([0., 5., 0.], [100., 300., 100.])

          runner = bp.DSRunner(neu, monitors=['V'])
          runner.run(inputs=inputs)

          bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)


      **Model Parameters**

      ============= ============== ======== ===================================================
      **Parameter** **Init Value** **Unit** **Explanation**
      ------------- -------------- -------- ---------------------------------------------------
      V_rest        -65            mV       Resting potential.
      V_reset       -68            mV       Reset potential after spike.
      V_th          -30            mV       Threshold potential of spike.
      V_T           -59.9          mV       Threshold potential of generating action potential.
      delta_T       3.48           \        Spike slope factor.
      R             1              \        Membrane resistance.
      tau           10             \        Membrane time constant. Compute by R * C.
      tau_ref       1.7            \        Refractory period length.
      ============= ============== ======== ===================================================

      **Model Variables**

      ================== ================= =========================================================
      **Variables name** **Initial Value** **Explanation**
      ------------------ ----------------- ---------------------------------------------------------
      V                  0                 Membrane potential.
      input              0                 External and synaptic input current.
      spike              False             Flag to mark whether the neuron is spiking.
      refractory         False             Flag to mark whether the neuron is in refractory period.
      t_last_spike       -1e7              Last spike time stamp.
      ================== ================= =========================================================


      """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -55.,
        V_T: Union[float, ArrayType, Callable] = -59.9,
        delta_T: Union[float, ArrayType, Callable] = 3.48,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)

        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.V_T = self.offset_scaling(self.init_param(V_T))
        self.delta_T = self.std_scaling(self.init_param(delta_T))
        self.tau = self.init_param(tau)
        self.R = self.init_param(R)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # noise
        self.noise = init_noise(noise, self.varshape)
        # integral
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def derivative(self, V, t, I):
        I = self.sum_current_inputs(V, init=I)
        exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dvdt = (- (V - self.V_rest) + exp_v + self.R * I) / self.tau
        return dvdt

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)

        self.V.value = V
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class ExpIF(ExpIFLTC):
    r"""Exponential integrate-and-fire neuron model.

    **Model Descriptions**

    In the exponential integrate-and-fire model [1]_, the differential
    equation for the membrane potential is given by

    .. math::

        \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
        \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

    This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
    and "threshold" :math:`\vartheta_{rh}`.

    The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
    defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
    :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
    where :math:`\tau_{\rm ref}` is an absolute refractory time.
    If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
    its exact value does not play any role. The reason is that the upswing of the action
    potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
    an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
    convenience. For a formal mathematical analysis of the model, the threshold can be pushed
    to infinity.

    The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
    and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
    It is one of the prominent examples of a precise theoretical prediction in computational
    neuroscience that was later confirmed by experimental neuroscience.

    Two important remarks:

    - (i) The right-hand side of the above equation contains a nonlinearity
      that can be directly extracted from experimental data [3]_. In this sense the exponential
      nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
    - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
      rate for constant input, and the linear response to fluctuations, even in the presence
      of input noise [4]_.

    **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models
           of cognition. Cambridge University Press.
    .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
           Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
           are reliable predictors of naturalistic pyramidal-neuron voltage
           traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
    .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
           integrate-and-fire neurons to modulated current-based and
           conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
    .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire

    .. seealso::

       :class:`brainpy.state.ExpIF` provides the state-based formulation of this neuron.

    **Examples**

    There is a simple usage example::

        import brainpy as bp

        neu = bp.dyn.ExpIF(1, )

        # example for section input
        inputs = bp.inputs.section_input([0., 5., 0.], [100., 300., 100.])

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)


    **Model Parameters**

    ============= ============== ======== ===================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    R             1              \        Membrane resistance.
    tau           10             \        Membrane time constant. Compute by R * C.
    tau_ref       1.7            \        Refractory period length.
    ============= ============== ======== ===================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  0                 Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    refractory         False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================




    Args:
      %s
      %s
      """

    def derivative(self, V, t, I):
        exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dvdt = (- (V - self.V_rest) + exp_v + self.R * I) / self.tau
        return dvdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class ExpIFRefLTC(ExpIFLTC):
    r"""Exponential integrate-and-fire neuron model with liquid time-constant.

    **Model Descriptions**

    In the exponential integrate-and-fire model [1]_, the differential
    equation for the membrane potential is given by

    .. math::

        \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
        \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

    This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
    and "threshold" :math:`\vartheta_{rh}`.

    The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
    defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
    :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
    where :math:`\tau_{\rm ref}` is an absolute refractory time.
    If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
    its exact value does not play any role. The reason is that the upswing of the action
    potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
    an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
    convenience. For a formal mathematical analysis of the model, the threshold can be pushed
    to infinity.

    The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
    and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
    It is one of the prominent examples of a precise theoretical prediction in computational
    neuroscience that was later confirmed by experimental neuroscience.

    Two important remarks:

    - (i) The right-hand side of the above equation contains a nonlinearity
      that can be directly extracted from experimental data [3]_. In this sense the exponential
      nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
    - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
      rate for constant input, and the linear response to fluctuations, even in the presence
      of input noise [4]_.

    **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models
           of cognition. Cambridge University Press.
    .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
           Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
           are reliable predictors of naturalistic pyramidal-neuron voltage
           traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
    .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
           integrate-and-fire neurons to modulated current-based and
           conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
    .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire

    .. seealso::

       :class:`brainpy.state.ExpIFRef` provides the state-based formulation of this neuron.

    **Examples**

    There is a simple usage example::

        import brainpy as bp

        neu = bp.dyn.ExpIFRefLTC(1, )

        # example for section input
        inputs = bp.inputs.section_input([0., 5., 0.], [100., 300., 100.])

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)


    **Model Parameters**

    ============= ============== ======== ===================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    R             1              \        Membrane resistance.
    tau           10             \        Membrane time constant. Compute by R * C.
    tau_ref       1.7            \        Refractory period length.
    ============= ============== ======== ===================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  0                 Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    refractory         False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s

    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        detach_spk: bool = False,
        spk_reset: str = 'soft',
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -55.,
        V_T: Union[float, ArrayType, Callable] = -59.9,
        delta_T: Union[float, ArrayType, Callable] = 3.48,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th=V_th,
            V_T=V_T,
            delta_T=delta_T,
            R=R,
            tau=tau,
            V_initializer=V_initializer,
            noise=noise,
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # integral
        self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e7)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class ExpIFRef(ExpIFRefLTC):
    r"""Exponential integrate-and-fire neuron model .

    **Model Descriptions**

    In the exponential integrate-and-fire model [1]_, the differential
    equation for the membrane potential is given by

    .. math::

        \tau\frac{d V}{d t}= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} + RI(t), \\
        \text{after} \, V(t) \gt V_{th}, V(t) = V_{reset} \, \text{last} \, \tau_{ref} \, \text{ms}

    This equation has an exponential nonlinearity with "sharpness" parameter :math:`\Delta_{T}`
    and "threshold" :math:`\vartheta_{rh}`.

    The moment when the membrane potential reaches the numerical threshold :math:`V_{th}`
    defines the firing time :math:`t^{(f)}`. After firing, the membrane potential is reset to
    :math:`V_{rest}` and integration restarts at time :math:`t^{(f)}+\tau_{\rm ref}`,
    where :math:`\tau_{\rm ref}` is an absolute refractory time.
    If the numerical threshold is chosen sufficiently high, :math:`V_{th}\gg v+\Delta_T`,
    its exact value does not play any role. The reason is that the upswing of the action
    potential for :math:`v\gg v +\Delta_{T}` is so rapid, that it goes to infinity in
    an incredibly short time. The threshold :math:`V_{th}` is introduced mainly for numerical
    convenience. For a formal mathematical analysis of the model, the threshold can be pushed
    to infinity.

    The model was first introduced by Nicolas Fourcaud-Trocmé, David Hansel, Carl van Vreeswijk
    and Nicolas Brunel [1]_. The exponential nonlinearity was later confirmed by Badel et al. [3]_.
    It is one of the prominent examples of a precise theoretical prediction in computational
    neuroscience that was later confirmed by experimental neuroscience.

    Two important remarks:

    - (i) The right-hand side of the above equation contains a nonlinearity
      that can be directly extracted from experimental data [3]_. In this sense the exponential
      nonlinearity is not an arbitrary choice but directly supported by experimental evidence.
    - (ii) Even though it is a nonlinear model, it is simple enough to calculate the firing
      rate for constant input, and the linear response to fluctuations, even in the presence
      of input noise [4]_.

    **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
           Neuronal dynamics: From single neurons to networks and models
           of cognition. Cambridge University Press.
    .. [3] Badel, Laurent, Sandrine Lefort, Romain Brette, Carl CH Petersen,
           Wulfram Gerstner, and Magnus JE Richardson. "Dynamic IV curves
           are reliable predictors of naturalistic pyramidal-neuron voltage
           traces." Journal of Neurophysiology 99, no. 2 (2008): 656-666.
    .. [4] Richardson, Magnus JE. "Firing-rate response of linear and nonlinear
           integrate-and-fire neurons to modulated current-based and
           conductance-based synaptic drive." Physical Review E 76, no. 2 (2007): 021919.
    .. [5] https://en.wikipedia.org/wiki/Exponential_integrate-and-fire

    **Examples**

    There is a simple usage example::

        import brainpy as bp

        neu = bp.dyn.ExpIFRef(1, )

        # example for section input
        inputs = bp.inputs.section_input([0., 5., 0.], [100., 300., 100.])

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)


    **Model Parameters**

    ============= ============== ======== ===================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ---------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    R             1              \        Membrane resistance.
    tau           10             \        Membrane time constant. Compute by R * C.
    tau_ref       1.7            \        Refractory period length.
    ============= ============== ======== ===================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  0                 Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    refractory         False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s
    """

    def derivative(self, V, t, I):
        exp_v = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dvdt = (- (V - self.V_rest) + exp_v + self.R * I) / self.tau
        return dvdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


ExpIF.__doc__ = ExpIF.__doc__ % (pneu_doc, dpneu_doc)
ExpIFRefLTC.__doc__ = ExpIFRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
ExpIFRef.__doc__ = ExpIFRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
ExpIFLTC.__doc__ = ExpIFLTC.__doc__ % ()


class AdExIFLTC(GradNeuDyn):
    r"""Adaptive exponential integrate-and-fire neuron model with liquid time-constant.

    **Model Descriptions**

    The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
    spiking neuron model with two variables [1]_ [2]_.

    .. math::

        \begin{aligned}
        \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
        \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    The first equation describes the dynamics of the membrane potential and includes
    an activation term with an exponential voltage dependence. Voltage is coupled to
    a second equation which describes adaptation. Both variables are reset if an action
    potential has been triggered. The combination of adaptation and exponential voltage
    dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

    The adaptive exponential integrate-and-fire model is capable of describing known
    neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
    initial bursting, fast spiking, and regular spiking.

      **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    .. seealso::

       :class:`brainpy.state.AdExIF` provides the state-based formulation of this model.

    **Examples**

    An example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdExIFLTC(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)

    **Model Examples**

    - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Gerstner_2005_AdExIF_model.html>`_

    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
    b             1              \        The increment of :math:`w` produced by a spike.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_w         30             ms       Time constant of the adaptation current.
    tau_ref       0.             ms       Refractory time.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= =========================================================



    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -55.,
        V_T: Union[float, ArrayType, Callable] = -59.9,
        delta_T: Union[float, ArrayType, Callable] = 3.48,
        a: Union[float, ArrayType, Callable] = 1.,
        b: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        tau_w: Union[float, ArrayType, Callable] = 30.,
        R: Union[float, ArrayType, Callable] = 1.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
        w_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)
        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.V_T = self.offset_scaling(self.init_param(V_T))
        self.a = self.init_param(a)
        self.b = self.std_scaling(self.init_param(b))
        self.R = self.init_param(R)
        self.delta_T = self.std_scaling(self.init_param(delta_T))
        self.tau = self.init_param(tau)
        self.tau_w = self.init_param(tau_w)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._w_initializer = is_initializer(w_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def dV(self, V, t, w, I):
        I = self.sum_current_inputs(V, init=I)
        exp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dVdt = (- V + self.V_rest + exp - self.R * w + self.R * I) / self.tau
        return dVdt

    def dw(self, w, t, V):
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dwdt

    @property
    def derivative(self):
        return JointEq([self.dV, self.dw])

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.w = self.std_scaling(self.init_variable(self._w_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, w = self.integral(self.V.value, self.w.value, t, x, dt)
        V += self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError
            w += self.b * spike

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            w = bm.where(spike, w + self.b, w)

        self.V.value = V
        self.w.value = w
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class AdExIF(AdExIFLTC):
    r"""Adaptive exponential integrate-and-fire neuron model.

    **Model Descriptions**

    The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
    spiking neuron model with two variables [1]_ [2]_.

    .. math::

        \begin{aligned}
        \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
        \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    The first equation describes the dynamics of the membrane potential and includes
    an activation term with an exponential voltage dependence. Voltage is coupled to
    a second equation which describes adaptation. Both variables are reset if an action
    potential has been triggered. The combination of adaptation and exponential voltage
    dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

    The adaptive exponential integrate-and-fire model is capable of describing known
    neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
    initial bursting, fast spiking, and regular spiking.

    **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    **Examples**

    An example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdExIF(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)

    **Model Examples**

    - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Gerstner_2005_AdExIF_model.html>`_

    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
    b             1              \        The increment of :math:`w` produced by a spike.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_w         30             ms       Time constant of the adaptation current.
    tau_ref       0.             ms       Refractory time.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
    """

    def dV(self, V, t, w, I):
        exp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dVdt = (- V + self.V_rest + exp - self.R * w + self.R * I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class AdExIFRefLTC(AdExIFLTC):
    r"""Adaptive exponential integrate-and-fire neuron model with liquid time-constant.

    **Model Descriptions**

    The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
    spiking neuron model with two variables [1]_ [2]_.

    .. math::

        \begin{aligned}
        \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
        \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    The first equation describes the dynamics of the membrane potential and includes
    an activation term with an exponential voltage dependence. Voltage is coupled to
    a second equation which describes adaptation. Both variables are reset if an action
    potential has been triggered. The combination of adaptation and exponential voltage
    dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

    The adaptive exponential integrate-and-fire model is capable of describing known
    neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
    initial bursting, fast spiking, and regular spiking.

      **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    **Examples**

    An example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdExIFRefLTC(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)

    **Model Examples**

    - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Gerstner_2005_AdExIF_model.html>`_

    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
    b             1              \        The increment of :math:`w` produced by a spike.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_w         30             ms       Time constant of the adaptation current.
    tau_ref       0.             ms       Refractory time.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= =========================================================




    Args:
      %s
      %s
      %s
    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -55.,
        V_T: Union[float, ArrayType, Callable] = -59.9,
        delta_T: Union[float, ArrayType, Callable] = 3.48,
        a: Union[float, ArrayType, Callable] = 1.,
        b: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        tau_w: Union[float, ArrayType, Callable] = 30.,
        R: Union[float, ArrayType, Callable] = 1.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
        w_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th=V_th,
            V_T=V_T,
            delta_T=delta_T,
            a=a,
            b=b,
            R=R,
            tau=tau,
            tau_w=tau_w,
            V_initializer=V_initializer,
            w_initializer=w_initializer
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._w_initializer = is_initializer(w_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e8)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, w = self.integral(self.V.value, self.w.value, t, x, dt)
        V += self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            w += self.b * spike_no_grad
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            w = bm.where(spike, w + self.b, w)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.w.value = w
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class AdExIFRef(AdExIFRefLTC):
    r"""Adaptive exponential integrate-and-fire neuron model.

    **Model Descriptions**

    The **adaptive exponential integrate-and-fire model**, also called AdEx, is a
    spiking neuron model with two variables [1]_ [2]_.

    .. math::

        \begin{aligned}
        \tau_m\frac{d V}{d t} &= - (V-V_{rest}) + \Delta_T e^{\frac{V-V_T}{\Delta_T}} - Rw + RI(t), \\
        \tau_w \frac{d w}{d t} &=a(V-V_{rest}) - w
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    The first equation describes the dynamics of the membrane potential and includes
    an activation term with an exponential voltage dependence. Voltage is coupled to
    a second equation which describes adaptation. Both variables are reset if an action
    potential has been triggered. The combination of adaptation and exponential voltage
    dependence gives rise to the name Adaptive Exponential Integrate-and-Fire model.

    The adaptive exponential integrate-and-fire model is capable of describing known
    neuronal firing patterns, e.g., adapting, bursting, delayed spike initiation,
    initial bursting, fast spiking, and regular spiking.

      **References**

    .. [1] Fourcaud-Trocmé, Nicolas, et al. "How spike generation
           mechanisms determine the neuronal response to fluctuating
           inputs." Journal of Neuroscience 23.37 (2003): 11628-11640.
    .. [2] http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model

    **Examples**

    Here is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdExIFRef(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)

    **Model Examples**

    - `Examples for different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Gerstner_2005_AdExIF_model.html>`_

    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_T           -59.9          mV       Threshold potential of generating action potential.
    delta_T       3.48           \        Spike slope factor.
    a             1              \        The sensitivity of the recovery variable :math:`u` to the sub-threshold fluctuations of the membrane potential :math:`v`
    b             1              \        The increment of :math:`w` produced by a spike.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_w         30             ms       Time constant of the adaptation current.
    tau_ref       0.             ms       Refractory time.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= =========================================================

    Args:
      %s
      %s
      %s
    """

    def dV(self, V, t, w, I):
        exp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        dVdt = (- V + self.V_rest + exp - self.R * w + self.R * I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


AdExIF.__doc__ = AdExIF.__doc__ % (pneu_doc, dpneu_doc)
AdExIFRefLTC.__doc__ = AdExIFRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
AdExIFRef.__doc__ = AdExIFRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
AdExIFLTC.__doc__ = AdExIFLTC.__doc__ % ()


class QuaIFLTC(GradNeuDyn):
    r"""Quadratic Integrate-and-Fire neuron model with liquid time-constant.

    **Model Descriptions**

    In contrast to physiologically accurate but computationally expensive
    neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
    to produce **action potential-like patterns** and ignores subtleties
    like gating variables, which play an important role in generating action
    potentials in a real neuron. However, the QIF model is incredibly easy
    to implement and compute, and relatively straightforward to study and
    understand, thus has found ubiquitous use in computational neuroscience.

    .. math::

        \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

    where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

    **References**

    .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
            (2000) Intrinsic dynamics in neuronal networks. I. Theory.
            J. Neurophysiology 83, pp. 808–827.

    **Examples**

    Here is an example usage:

    .. code-block:: python

      import brainpy as bp

      neu = bp.dyn.QuaIFLTC(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)

    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
    c             .07            \        Coefficient describes membrane potential update. Larger than 0.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_ref       0              ms       Refractory period length.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7               Last spike time stamp.
    ================== ================= =========================================================
    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -30.,
        V_c: Union[float, ArrayType, Callable] = -50.0,
        c: Union[float, ArrayType, Callable] = 0.07,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)
        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.V_c = self.offset_scaling(self.init_param(V_c))
        self.c = self.inv_scaling(self.init_param(c))
        self.R = self.init_param(R)
        self.tau = self.init_param(tau)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=1)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def derivative(self, V, t, I):
        I = self.sum_current_inputs(V, init=I)
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * I) / self.tau
        return dVdt

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)

        self.V.value = V
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class QuaIF(QuaIFLTC):
    r"""Quadratic Integrate-and-Fire neuron model.

    **Model Descriptions**

    In contrast to physiologically accurate but computationally expensive
    neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
    to produce **action potential-like patterns** and ignores subtleties
    like gating variables, which play an important role in generating action
    potentials in a real neuron. However, the QIF model is incredibly easy
    to implement and compute, and relatively straightforward to study and
    understand, thus has found ubiquitous use in computational neuroscience.

    .. math::

        \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

    where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

    **References**

    .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
            (2000) Intrinsic dynamics in neuronal networks. I. Theory.
            J. Neurophysiology 83, pp. 808–827.


    **Examples**

    There is an example usage:

    .. code-block:: python

      import brainpy as bp

      neu = bp.dyn.QuaIF(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
    c             .07            \        Coefficient describes membrane potential update. Larger than 0.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_ref       0              ms       Refractory period length.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7               Last spike time stamp.
    ================== ================= =========================================================




    Args:
      %s
      %s
    """

    def derivative(self, V, t, I):
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class QuaIFRefLTC(QuaIFLTC):
    r"""Quadratic Integrate-and-Fire neuron model with liquid time-constant.

    **Model Descriptions**

    In contrast to physiologically accurate but computationally expensive
    neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
    to produce **action potential-like patterns** and ignores subtleties
    like gating variables, which play an important role in generating action
    potentials in a real neuron. However, the QIF model is incredibly easy
    to implement and compute, and relatively straightforward to study and
    understand, thus has found ubiquitous use in computational neuroscience.

    .. math::

        \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

    where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

    **References**

    .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
            (2000) Intrinsic dynamics in neuronal networks. I. Theory.
            J. Neurophysiology 83, pp. 808–827.

    **Examples**

    There is an example usage:

    .. code-block:: python

      import brainpy as bp

      neu = bp.dyn.QuaIFRefLTC(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
    c             .07            \        Coefficient describes membrane potential update. Larger than 0.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_ref       0              ms       Refractory period length.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7               Last spike time stamp.
    ================== ================= =========================================================

    Args:
      %s
      %s
      %s
    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -30.,
        V_c: Union[float, ArrayType, Callable] = -50.0,
        c: Union[float, ArrayType, Callable] = 0.07,
        R: Union[float, ArrayType, Callable] = 1.,
        tau: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th=V_th,
            V_c=V_c,
            c=c,
            R=R,
            tau=tau,
            V_initializer=V_initializer,
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=1)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e7)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V = self.integral(self.V.value, t, x, dt) + self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class QuaIFRef(QuaIFRefLTC):
    r"""Quadratic Integrate-and-Fire neuron model.

    **Model Descriptions**

    In contrast to physiologically accurate but computationally expensive
    neuron models like the Hodgkin–Huxley model, the QIF model [1]_ seeks only
    to produce **action potential-like patterns** and ignores subtleties
    like gating variables, which play an important role in generating action
    potentials in a real neuron. However, the QIF model is incredibly easy
    to implement and compute, and relatively straightforward to study and
    understand, thus has found ubiquitous use in computational neuroscience.

    .. math::

        \tau \frac{d V}{d t}=c(V-V_{rest})(V-V_c) + RI(t)

    where the parameters are taken to be :math:`c` =0.07, and :math:`V_c = -50 mV` (Latham et al., 2000).

    **References**

    .. [1]  P. E. Latham, B.J. Richmond, P. Nelson and S. Nirenberg
            (2000) Intrinsic dynamics in neuronal networks. I. Theory.
            J. Neurophysiology 83, pp. 808–827.

    **Examples**

    There is an example usage:

    .. code-block:: python

      import brainpy as bp

      neu = bp.dyn.QuaIFRef(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Parameters**

    ============= ============== ======== ========================================================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- ------------------------------------------------------------------------------------------------------------------------
    V_rest        -65            mV       Resting potential.
    V_reset       -68            mV       Reset potential after spike.
    V_th          -30            mV       Threshold potential of spike and reset.
    V_c           -50            mV       Critical voltage for spike initiation. Must be larger than V_rest.
    c             .07            \        Coefficient describes membrane potential update. Larger than 0.
    R             1              \        Membrane resistance.
    tau           10             ms       Membrane time constant. Compute by R * C.
    tau_ref       0              ms       Refractory period length.
    ============= ============== ======== ========================================================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                   0                 Membrane potential.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    refractory          False             Flag to mark whether the neuron is in refractory period.
    t_last_spike       -1e7               Last spike time stamp.
    ================== ================= =========================================================

    Args:
      %s
      %s
      %s
    """

    def derivative(self, V, t, I):
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


QuaIF.__doc__ = QuaIF.__doc__ % (pneu_doc, dpneu_doc)
QuaIFRefLTC.__doc__ = QuaIFRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
QuaIFRef.__doc__ = QuaIFRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
QuaIFLTC.__doc__ = QuaIFLTC.__doc__ % ()


class AdQuaIFLTC(GradNeuDyn):
    r"""Adaptive quadratic integrate-and-fire neuron model with liquid time-constant.

    **Model Descriptions**

    The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

    .. math::

        \begin{aligned}
        \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
        \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    **References**

    .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
           neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
    .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
           nonlinear integrate-and-fire neurons." SIAM Journal on Applied
           Mathematics 68, no. 4 (2008): 1045-1079.

    **Examples**

    Here is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdQuaIFLTC(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)



    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_rest         -65            mV       Resting potential.
    V_reset        -68            mV       Reset potential after spike.
    V_th           -30            mV       Threshold potential of spike and reset.
    V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                           than :math:`V_{rest}`.
    a               1              \       The sensitivity of the recovery variable :math:`u` to
                                           the sub-threshold fluctuations of the membrane
                                           potential :math:`v`
    b              .1             \        The increment of :math:`w` produced by a spike.
    c              .07             \       Coefficient describes membrane potential update.
                                           Larger than 0.
    tau            10             ms       Membrane time constant.
    tau_w          10             ms       Time constant of the adaptation current.
    ============= ============== ======== =======================================================

    **Model Variables**

    ================== ================= ==========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ----------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= ==========================================================


    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -30.,
        V_c: Union[float, ArrayType, Callable] = -50.0,
        a: Union[float, ArrayType, Callable] = 1.,
        b: Union[float, ArrayType, Callable] = .1,
        c: Union[float, ArrayType, Callable] = .07,
        tau: Union[float, ArrayType, Callable] = 10.,
        tau_w: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
        w_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)
        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.V_c = self.offset_scaling(self.init_param(V_c))
        self.a = self.init_param(a)
        self.b = self.std_scaling(self.init_param(b))
        self.c = self.inv_scaling(self.init_param(c))
        self.tau = self.init_param(tau)
        self.tau_w = self.init_param(tau_w)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._w_initializer = is_initializer(w_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def dV(self, V, t, w, I):
        I = self.sum_current_inputs(V, init=I)
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + I) / self.tau
        return dVdt

    def dw(self, w, t, V):
        dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
        return dwdt

    @property
    def derivative(self):
        return JointEq([self.dV, self.dw])

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.w = self.std_scaling(self.init_variable(self._w_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, w = self.integral(self.V.value, self.w.value, t, x, dt)
        V += self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError
            w += self.b * spike

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            w = bm.where(spike, w + self.b, w)

        self.V.value = V
        self.w.value = w
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class AdQuaIF(AdQuaIFLTC):
    r"""Adaptive quadratic integrate-and-fire neuron model.

    **Model Descriptions**

    The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

    .. math::

        \begin{aligned}
        \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
        \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    **References**

    .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
           neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
    .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
           nonlinear integrate-and-fire neurons." SIAM Journal on Applied
           Mathematics 68, no. 4 (2008): 1045-1079.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdQuaIF(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)



    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_rest         -65            mV       Resting potential.
    V_reset        -68            mV       Reset potential after spike.
    V_th           -30            mV       Threshold potential of spike and reset.
    V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                           than :math:`V_{rest}`.
    a               1              \       The sensitivity of the recovery variable :math:`u` to
                                           the sub-threshold fluctuations of the membrane
                                           potential :math:`v`
    b              .1             \        The increment of :math:`w` produced by a spike.
    c              .07             \       Coefficient describes membrane potential update.
                                           Larger than 0.
    tau            10             ms       Membrane time constant.
    tau_w          10             ms       Time constant of the adaptation current.
    ============= ============== ======== =======================================================

    **Model Variables**

    ================== ================= ==========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ----------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= ==========================================================




    Args:
      %s
      %s
    """

    def dV(self, V, t, w, I):
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class AdQuaIFRefLTC(AdQuaIFLTC):
    r"""Adaptive quadratic integrate-and-fire neuron model with liquid time-constant.

    **Model Descriptions**

    The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

    .. math::

        \begin{aligned}
        \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
        \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    **References**

    .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
           neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
    .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
           nonlinear integrate-and-fire neurons." SIAM Journal on Applied
           Mathematics 68, no. 4 (2008): 1045-1079.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdQuaIFRefLTC(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)




    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_rest         -65            mV       Resting potential.
    V_reset        -68            mV       Reset potential after spike.
    V_th           -30            mV       Threshold potential of spike and reset.
    V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                           than :math:`V_{rest}`.
    a               1              \       The sensitivity of the recovery variable :math:`u` to
                                           the sub-threshold fluctuations of the membrane
                                           potential :math:`v`
    b              .1             \        The increment of :math:`w` produced by a spike.
    c              .07             \       Coefficient describes membrane potential update.
                                           Larger than 0.
    tau            10             ms       Membrane time constant.
    tau_w          10             ms       Time constant of the adaptation current.
    ============= ============== ======== =======================================================

    **Model Variables**

    ================== ================= ==========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ----------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= ==========================================================

    Args:
      %s
      %s
      %s
    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = -65.,
        V_reset: Union[float, ArrayType, Callable] = -68.,
        V_th: Union[float, ArrayType, Callable] = -30.,
        V_c: Union[float, ArrayType, Callable] = -50.0,
        a: Union[float, ArrayType, Callable] = 1.,
        b: Union[float, ArrayType, Callable] = .1,
        c: Union[float, ArrayType, Callable] = .07,
        tau: Union[float, ArrayType, Callable] = 10.,
        tau_w: Union[float, ArrayType, Callable] = 10.,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
        w_initializer: Union[Callable, ArrayType] = ZeroInit(),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th=V_th,
            V_c=V_c,
            a=a,
            b=b,
            c=c,
            tau=tau,
            tau_w=tau_w,
            V_initializer=V_initializer,
            w_initializer=w_initializer
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._w_initializer = is_initializer(w_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e8)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, w = self.integral(self.V.value, self.w.value, t, x, dt)
        V += self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            w += self.b * spike_no_grad
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            w = bm.where(spike, w + self.b, w)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.w.value = w
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class AdQuaIFRef(AdQuaIFRefLTC):
    r"""Adaptive quadratic integrate-and-fire neuron model.

    **Model Descriptions**

    The adaptive quadratic integrate-and-fire neuron model [1]_ is given by:

    .. math::

        \begin{aligned}
        \tau_m \frac{d V}{d t}&=c(V-V_{rest})(V-V_c) - w + I(t), \\
        \tau_w \frac{d w}{d t}&=a(V-V_{rest}) - w,
        \end{aligned}

    once the membrane potential reaches the spike threshold,

    .. math::

        V \rightarrow V_{reset}, \\
        w \rightarrow w+b.

    **References**

    .. [1] Izhikevich, E. M. (2004). Which model to use for cortical spiking
           neurons?. IEEE transactions on neural networks, 15(5), 1063-1070.
    .. [2] Touboul, Jonathan. "Bifurcation analysis of a general class of
           nonlinear integrate-and-fire neurons." SIAM Journal on Applied
           Mathematics 68, no. 4 (2008): 1045-1079.

    **Examples**

    There is an example usage:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.AdQuaIFRef(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_rest         -65            mV       Resting potential.
    V_reset        -68            mV       Reset potential after spike.
    V_th           -30            mV       Threshold potential of spike and reset.
    V_c            -50            mV       Critical voltage for spike initiation. Must be larger
                                           than :math:`V_{rest}`.
    a               1              \       The sensitivity of the recovery variable :math:`u` to
                                           the sub-threshold fluctuations of the membrane
                                           potential :math:`v`
    b              .1             \        The increment of :math:`w` produced by a spike.
    c              .07             \       Coefficient describes membrane potential update.
                                           Larger than 0.
    tau            10             ms       Membrane time constant.
    tau_w          10             ms       Time constant of the adaptation current.
    ============= ============== ======== =======================================================

    **Model Variables**

    ================== ================= ==========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ----------------------------------------------------------
    V                   0                 Membrane potential.
    w                   0                 Adaptation current.
    input               0                 External and synaptic input current.
    spike               False             Flag to mark whether the neuron is spiking.
    t_last_spike        -1e7              Last spike time stamp.
    ================== ================= ==========================================================



    Args:
      %s
      %s
      %s
    """

    def dV(self, V, t, w, I):
        dVdt = (self.c * (V - self.V_rest) * (V - self.V_c) - w + I) / self.tau
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


AdQuaIF.__doc__ = AdQuaIF.__doc__ % (pneu_doc, dpneu_doc)
AdQuaIFRefLTC.__doc__ = AdQuaIFRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
AdQuaIFRef.__doc__ = AdQuaIFRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
AdQuaIFLTC.__doc__ = AdQuaIFLTC.__doc__ % ()


class GifLTC(GradNeuDyn):
    r"""Generalized Integrate-and-Fire model with liquid time-constant.

    **Model Descriptions**

    The generalized integrate-and-fire model [1]_ is given by

    .. math::

        &\frac{d I_j}{d t} = - k_j I_j

        &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

        &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

    When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

    .. math::

        &I_j \leftarrow R_j I_j + A_j

        &V \leftarrow V_{reset}

        &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

    Note that :math:`I_j` refers to arbitrary number of internal currents.


    **References**

    .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
           integrate-and-fire neural model produces diverse spiking
           behaviors." Neural computation 21.3 (2009): 704-718.
    .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
           Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
           leaky integrate-and-fire models classify multiple neuron types."
           Nature communications 9, no. 1 (2018): 1-15.

    **Examples**

    There is a simple usage: you r bound to be together, roy and edward

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        # Tonic Spiking
        neu = bp.dyn.Gif(1)
        inputs = bp.inputs.ramp_input(.2, 2, 400, 0, 400)

        runner = bp.DSRunner(neu, monitors=['V', 'V_th'])
        runner.run(inputs=inputs)

        ts = runner.mon.ts

        fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(ts, runner.mon.V[:, 0], label='V')
        ax1.plot(ts, runner.mon.V_th[:, 0], label='V_th')

        plt.show()

    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

    **Model Parameters**

    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70            mV       Resting potential.
    V_reset       -70            mV       Reset potential after spike.
    V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
    V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
    R             20             \        Membrane resistance.
    tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
    a             0              \        Coefficient describes the dependence of
                                          :math:`V_{th}` on membrane potential.
    b             0.01           \        Coefficient describes :math:`V_{th}` update.
    k1            0.2            \        Constant pf :math:`I1`.
    k2            0.02           \        Constant of :math:`I2`.
    R1            0              \        Free parameter.
                                          Describes dependence of :math:`I_1` reset value on
                                          :math:`I_1` value before spiking.
    R2            1              \        Free parameter.
                                          Describes dependence of :math:`I_2` reset value on
                                          :math:`I_2` value before spiking.
    A1            0              \        Free parameter.
    A2            0              \        Free parameter.
    ============= ============== ======== ====================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  -70               Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    V_th               -50               Spiking threshold potential.
    I1                 0                 Internal current 1.
    I2                 0                 Internal current 2.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================


  """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_rest: Union[float, ArrayType, Callable] = -70.,
        V_reset: Union[float, ArrayType, Callable] = -70.,
        V_th_inf: Union[float, ArrayType, Callable] = -50.,
        V_th_reset: Union[float, ArrayType, Callable] = -60.,
        R: Union[float, ArrayType, Callable] = 20.,
        tau: Union[float, ArrayType, Callable] = 20.,
        a: Union[float, ArrayType, Callable] = 0.,
        b: Union[float, ArrayType, Callable] = 0.01,
        k1: Union[float, ArrayType, Callable] = 0.2,
        k2: Union[float, ArrayType, Callable] = 0.02,
        R1: Union[float, ArrayType, Callable] = 0.,
        R2: Union[float, ArrayType, Callable] = 1.,
        A1: Union[float, ArrayType, Callable] = 0.,
        A2: Union[float, ArrayType, Callable] = 0.,
        V_initializer: Union[Callable, ArrayType] = OneInit(-70.),
        I1_initializer: Union[Callable, ArrayType] = ZeroInit(),
        I2_initializer: Union[Callable, ArrayType] = ZeroInit(),
        Vth_initializer: Union[Callable, ArrayType] = OneInit(-50.),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)
        # parameters
        self.V_rest = self.offset_scaling(self.init_param(V_rest))
        self.V_reset = self.offset_scaling(self.init_param(V_reset))
        self.V_th_inf = self.offset_scaling(self.init_param(V_th_inf))
        self.V_th_reset = self.offset_scaling(self.init_param(V_th_reset))
        self.R = self.init_param(R)
        self.a = self.init_param(a)
        self.b = self.init_param(b)
        self.k1 = self.init_param(k1)
        self.k2 = self.init_param(k2)
        self.R1 = self.init_param(R1)
        self.R2 = self.init_param(R2)
        self.A1 = self.std_scaling(self.init_param(A1))
        self.A2 = self.std_scaling(self.init_param(A2))
        self.tau = self.init_param(tau)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._I1_initializer = is_initializer(I1_initializer)
        self._I2_initializer = is_initializer(I2_initializer)
        self._Vth_initializer = is_initializer(Vth_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=4)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def dI1(self, I1, t):
        return - self.k1 * I1

    def dI2(self, I2, t):
        return - self.k2 * I2

    def dVth(self, V_th, t, V):
        return self.a * (V - self.V_rest) - self.b * (V_th - self.V_th_inf)

    def dV(self, V, t, I1, I2, I):
        I = self.sum_current_inputs(V, init=I)
        return (- (V - self.V_rest) + self.R * (I + I1 + I2)) / self.tau

    @property
    def derivative(self):
        return JointEq(self.dI1, self.dI2, self.dVth, self.dV)

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.offset_scaling(self.init_variable(self._V_initializer, batch_size))
        self.V_th = self.offset_scaling(self.init_variable(self._Vth_initializer, batch_size))
        self.I1 = self.std_scaling(self.init_variable(self._I1_initializer, batch_size))
        self.I2 = self.std_scaling(self.init_variable(self._I2_initializer, batch_size))
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        I1, I2, V_th, V = self.integral(self.I1.value, self.I2.value, self.V_th.value, self.V.value, t, x, dt)
        V += self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike
            else:
                raise ValueError
            I1 += spike * (self.R1 * I1 + self.A1 - I1)
            I2 += spike * (self.R2 * I2 + self.A2 - I2)
            V_th += (bm.maximum(self.V_th_reset, V_th) - V_th) * spike

        else:
            spike = self.V_th <= V
            V = bm.where(spike, self.V_reset, V)
            I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)
            I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)
            V_th = bm.where(spike, bm.maximum(self.V_th_reset, V_th), V_th)
        self.spike.value = spike
        self.I1.value = I1
        self.I2.value = I2
        self.V_th.value = V_th
        self.V.value = V
        return spike

    def return_info(self):
        return self.spike


class Gif(GifLTC):
    r"""Generalized Integrate-and-Fire model.

    **Model Descriptions**

    The generalized integrate-and-fire model [1]_ is given by

    .. math::

        &\frac{d I_j}{d t} = - k_j I_j

        &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

        &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

    When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

    .. math::

        &I_j \leftarrow R_j I_j + A_j

        &V \leftarrow V_{reset}

        &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

    Note that :math:`I_j` refers to arbitrary number of internal currents.


    **References**

    .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
           integrate-and-fire neural model produces diverse spiking
           behaviors." Neural computation 21.3 (2009): 704-718.
    .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
           Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
           leaky integrate-and-fire models classify multiple neuron types."
           Nature communications 9, no. 1 (2018): 1-15.

    **Examples**

    There is a simple usage:

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        # Phasic Spiking
        neu = bp.dyn.Gif(1, a=0.005)
        inputs = bp.inputs.section_input((0, 1.5), (50, 500))

        runner = bp.DSRunner(neu, monitors=['V', 'V_th'])
        runner.run(inputs=inputs)

        ts = runner.mon.ts

        fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(ts, runner.mon.V[:, 0], label='V')
        ax1.plot(ts, runner.mon.V_th[:, 0], label='V_th')

        plt.show()

    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

    **Model Parameters**

    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70            mV       Resting potential.
    V_reset       -70            mV       Reset potential after spike.
    V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
    V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
    R             20             \        Membrane resistance.
    tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
    a             0              \        Coefficient describes the dependence of
                                          :math:`V_{th}` on membrane potential.
    b             0.01           \        Coefficient describes :math:`V_{th}` update.
    k1            0.2            \        Constant pf :math:`I1`.
    k2            0.02           \        Constant of :math:`I2`.
    R1            0              \        Free parameter.
                                          Describes dependence of :math:`I_1` reset value on
                                          :math:`I_1` value before spiking.
    R2            1              \        Free parameter.
                                          Describes dependence of :math:`I_2` reset value on
                                          :math:`I_2` value before spiking.
    A1            0              \        Free parameter.
    A2            0              \        Free parameter.
    ============= ============== ======== ====================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  -70               Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    V_th               -50               Spiking threshold potential.
    I1                 0                 Internal current 1.
    I2                 0                 Internal current 2.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
    """

    def dV(self, V, t, I1, I2, I):
        return (- (V - self.V_rest) + self.R * (I + I1 + I2)) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class GifRefLTC(GifLTC):
    r"""Generalized Integrate-and-Fire model with liquid time-constant.

    **Model Descriptions**

    The generalized integrate-and-fire model [1]_ is given by

    .. math::

        &\frac{d I_j}{d t} = - k_j I_j

        &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

        &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

    When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

    .. math::

        &I_j \leftarrow R_j I_j + A_j

        &V \leftarrow V_{reset}

        &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

    Note that :math:`I_j` refers to arbitrary number of internal currents.


    **References**

    .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
           integrate-and-fire neural model produces diverse spiking
           behaviors." Neural computation 21.3 (2009): 704-718.
    .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
           Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
           leaky integrate-and-fire models classify multiple neuron types."
           Nature communications 9, no. 1 (2018): 1-15.

    **Examples**

    There is a simple usage: mustang i love u

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        # Hyperpolarization-induced Spiking
        neu = bp.dyn.GifRefLTC(1, a=0.005)
        neu.V_th[:] = -50.
        inputs = bp.inputs.section_input((1.5, 1.7, 1.5, 1.7), (100, 400, 100, 400))

        runner = bp.DSRunner(neu, monitors=['V', 'V_th'])
        runner.run(inputs=inputs)

        ts = runner.mon.ts

        fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(ts, runner.mon.V[:, 0], label='V')
        ax1.plot(ts, runner.mon.V_th[:, 0], label='V_th')

        plt.show()

    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

    **Model Parameters**

    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70            mV       Resting potential.
    V_reset       -70            mV       Reset potential after spike.
    V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
    V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
    R             20             \        Membrane resistance.
    tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
    a             0              \        Coefficient describes the dependence of
                                          :math:`V_{th}` on membrane potential.
    b             0.01           \        Coefficient describes :math:`V_{th}` update.
    k1            0.2            \        Constant pf :math:`I1`.
    k2            0.02           \        Constant of :math:`I2`.
    R1            0              \        Free parameter.
                                          Describes dependence of :math:`I_1` reset value on
                                          :math:`I_1` value before spiking.
    R2            1              \        Free parameter.
                                          Describes dependence of :math:`I_2` reset value on
                                          :math:`I_2` value before spiking.
    A1            0              \        Free parameter.
    A2            0              \        Free parameter.
    ============= ============== ======== ====================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  -70               Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    V_th               -50               Spiking threshold potential.
    I1                 0                 Internal current 1.
    I2                 0                 Internal current 2.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s
  """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_rest: Union[float, ArrayType, Callable] = -70.,
        V_reset: Union[float, ArrayType, Callable] = -70.,
        V_th_inf: Union[float, ArrayType, Callable] = -50.,
        V_th_reset: Union[float, ArrayType, Callable] = -60.,
        R: Union[float, ArrayType, Callable] = 20.,
        tau: Union[float, ArrayType, Callable] = 20.,
        a: Union[float, ArrayType, Callable] = 0.,
        b: Union[float, ArrayType, Callable] = 0.01,
        k1: Union[float, ArrayType, Callable] = 0.2,
        k2: Union[float, ArrayType, Callable] = 0.02,
        R1: Union[float, ArrayType, Callable] = 0.,
        R2: Union[float, ArrayType, Callable] = 1.,
        A1: Union[float, ArrayType, Callable] = 0.,
        A2: Union[float, ArrayType, Callable] = 0.,
        V_initializer: Union[Callable, ArrayType] = OneInit(-70.),
        I1_initializer: Union[Callable, ArrayType] = ZeroInit(),
        I2_initializer: Union[Callable, ArrayType] = ZeroInit(),
        Vth_initializer: Union[Callable, ArrayType] = OneInit(-50.),

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_rest=V_rest,
            V_reset=V_reset,
            V_th_inf=V_th_inf,
            V_th_reset=V_th_reset,
            R=R,
            a=a,
            b=b,
            k1=k1,
            k2=k2,
            R1=R1,
            R2=R2,
            A1=A1,
            A2=A2,
            tau=tau,
            V_initializer=V_initializer,
            I1_initializer=I1_initializer,
            I2_initializer=I2_initializer,
            Vth_initializer=Vth_initializer,
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._I1_initializer = is_initializer(I1_initializer)
        self._I2_initializer = is_initializer(I2_initializer)
        self._Vth_initializer = is_initializer(Vth_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=4)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e8)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        I1, I2, V_th, V = self.integral(self.I1.value, self.I2.value, self.V_th.value, self.V.value, t, x, dt)
        V += self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            if self.spk_reset == 'soft':
                V -= (self.V_th - self.V_reset) * spike_no_grad
            elif self.spk_reset == 'hard':
                V += (self.V_reset - V) * spike_no_grad
            else:
                raise ValueError
            I1 += spike * (self.R1 * I1 + self.A1 - I1)
            I2 += spike * (self.R2 * I2 + self.A2 - I2)
            V_th += (bm.maximum(self.V_th_reset, V_th) - V_th) * spike_no_grad
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_reset, V)
            I1 = bm.where(spike, self.R1 * I1 + self.A1, I1)
            I2 = bm.where(spike, self.R2 * I2 + self.A2, I2)
            V_th = bm.where(spike, bm.maximum(self.V_th_reset, V_th), V_th)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.I1.value = I1
        self.I2.value = I2
        self.V_th.value = V_th
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class GifRef(GifRefLTC):
    r"""Generalized Integrate-and-Fire model.

    **Model Descriptions**

    The generalized integrate-and-fire model [1]_ is given by

    .. math::

        &\frac{d I_j}{d t} = - k_j I_j

        &\frac{d V}{d t} = ( - (V - V_{rest}) + R\sum_{j}I_j + RI) / \tau

        &\frac{d V_{th}}{d t} = a(V - V_{rest}) - b(V_{th} - V_{th\infty})

    When :math:`V` meet :math:`V_{th}`, Generalized IF neuron fires:

    .. math::

        &I_j \leftarrow R_j I_j + A_j

        &V \leftarrow V_{reset}

        &V_{th} \leftarrow max(V_{th_{reset}}, V_{th})

    Note that :math:`I_j` refers to arbitrary number of internal currents.


    **References**

    .. [1] Mihalaş, Ştefan, and Ernst Niebur. "A generalized linear
           integrate-and-fire neural model produces diverse spiking
           behaviors." Neural computation 21.3 (2009): 704-718.
    .. [2] Teeter, Corinne, Ramakrishnan Iyer, Vilas Menon, Nathan
           Gouwens, David Feng, Jim Berg, Aaron Szafer et al. "Generalized
           leaky integrate-and-fire models classify multiple neuron types."
           Nature communications 9, no. 1 (2018): 1-15.

    **Examples**

    There is a simple usage:

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        # Tonic Bursting
        neu = bp.dyn.GifRef(1, a=0.005, A1=10., A2=-0.6)
        neu.V_th[:] = -50.
        inputs = bp.inputs.section_input((1.5, 1.7,), (100, 400))

        runner = bp.DSRunner(neu, monitors=['V', 'V_th'])
        runner.run(inputs=inputs)

        ts = runner.mon.ts

        fig, gs = bp.visualize.get_figure(1, 1, 4, 8)
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(ts, runner.mon.V[:, 0], label='V')
        ax1.plot(ts, runner.mon.V_th[:, 0], label='V_th')

        plt.show()
    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Niebur_2009_GIF.html>`_

    **Model Parameters**

    ============= ============== ======== ====================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------
    V_rest        -70            mV       Resting potential.
    V_reset       -70            mV       Reset potential after spike.
    V_th_inf      -50            mV       Target value of threshold potential :math:`V_{th}` updating.
    V_th_reset    -60            mV       Free parameter, should be larger than :math:`V_{reset}`.
    R             20             \        Membrane resistance.
    tau           20             ms       Membrane time constant. Compute by :math:`R * C`.
    a             0              \        Coefficient describes the dependence of
                                          :math:`V_{th}` on membrane potential.
    b             0.01           \        Coefficient describes :math:`V_{th}` update.
    k1            0.2            \        Constant pf :math:`I1`.
    k2            0.02           \        Constant of :math:`I2`.
    R1            0              \        Free parameter.
                                          Describes dependence of :math:`I_1` reset value on
                                          :math:`I_1` value before spiking.
    R2            1              \        Free parameter.
                                          Describes dependence of :math:`I_2` reset value on
                                          :math:`I_2` value before spiking.
    A1            0              \        Free parameter.
    A2            0              \        Free parameter.
    ============= ============== ======== ====================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                  -70               Membrane potential.
    input              0                 External and synaptic input current.
    spike              False             Flag to mark whether the neuron is spiking.
    V_th               -50               Spiking threshold potential.
    I1                 0                 Internal current 1.
    I2                 0                 Internal current 2.
    t_last_spike       -1e7              Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s
  """

    def dV(self, V, t, I1, I2, I):
        return (- (V - self.V_rest) + self.R * (I + I1 + I2)) / self.tau

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


Gif.__doc__ = Gif.__doc__ % (pneu_doc, dpneu_doc)
GifRefLTC.__doc__ = GifRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
GifRef.__doc__ = GifRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
GifLTC.__doc__ = GifLTC.__doc__ % ()


class IzhikevichLTC(GradNeuDyn):
    r"""The Izhikevich neuron model with liquid time-constant.

      **Model Descriptions**

      The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

      .. math ::

          \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

          \frac{d u}{d t} &=a(b V-u)

      .. math ::

          \text{if}  v \geq 30  \text{mV}, \text{then}
          \begin{cases} v \leftarrow c \\
          u \leftarrow u+d \end{cases}


      **References**

      .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
             Transactions on neural networks 14.6 (2003): 1569-1572.

      .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
             IEEE transactions on neural networks 15.5 (2004): 1063-1070.

      **Examples**

      There is a simple usage example::

        import brainpy as bp

        neu = bp.dyn.IzhikevichLTC(2)

        # section input with wiener process
        inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
        inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)



      **Model Examples**

      - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

      **Model Parameters**

      ============= ============== ======== ================================================================================
      **Parameter** **Init Value** **Unit** **Explanation**
      ------------- -------------- -------- --------------------------------------------------------------------------------
      a             0.02           \        It determines the time scaling of
                                            the recovery variable :math:`u`.
      b             0.2            \        It describes the sensitivity of the
                                            recovery variable :math:`u` to
                                            the sub-threshold fluctuations of the
                                            membrane potential :math:`v`.
      c             -65            \        It describes the after-spike reset value
                                            of the membrane potential :math:`v` caused by
                                            the fast high-threshold :math:`K^{+}`
                                            conductance.
      d             8              \        It describes after-spike reset of the
                                            recovery variable :math:`u`
                                            caused by slow high-threshold
                                            :math:`Na^{+}` and :math:`K^{+}` conductance.
      tau_ref       0              ms       Refractory period length. [ms]
      V_th          30             mV       The membrane potential threshold.
      ============= ============== ======== ================================================================================

      **Model Variables**

      ================== ================= =========================================================
      **Variables name** **Initial Value** **Explanation**
      ------------------ ----------------- ---------------------------------------------------------
      V                          -65        Membrane potential.
      u                          1          Recovery variable.
      input                      0          External and synaptic input current.
      spike                      False      Flag to mark whether the neuron is spiking.
      refractory                False       Flag to mark whether the neuron is in refractory period.
      t_last_spike               -1e7       Last spike time stamp.
      ================== ================= =========================================================
      """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # neuron parameters
        V_th: Union[float, ArrayType, Callable] = 30.,
        p1: Union[float, ArrayType, Callable] = 0.04,
        p2: Union[float, ArrayType, Callable] = 5.,
        p3: Union[float, ArrayType, Callable] = 140.,
        a: Union[float, ArrayType, Callable] = 0.02,
        b: Union[float, ArrayType, Callable] = 0.20,
        c: Union[float, ArrayType, Callable] = -65.,
        d: Union[float, ArrayType, Callable] = 8.,
        tau: Union[float, ArrayType, Callable] = 10.,
        R: Union[float, ArrayType, Callable] = 1.,
        V_initializer: Union[Callable, ArrayType] = OneInit(-70.),
        u_initializer: Union[Callable, ArrayType] = None,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         name=name,
                         keep_size=keep_size,
                         mode=mode,
                         sharding=sharding,
                         spk_fun=spk_fun,
                         detach_spk=detach_spk,
                         method=method,
                         spk_dtype=spk_dtype,
                         spk_reset=spk_reset,
                         scaling=scaling)
        # parameters
        self.V_th = self.offset_scaling(self.init_param(V_th))
        self.p1 = self.inv_scaling(self.init_param(p1))
        p2_scaling = self.scaling.clone(bias=-p1 * 2 * self.scaling.bias, scale=1.)
        self.p2 = p2_scaling.offset_scaling(self.init_param(p2))
        p3_bias = p1 * self.scaling.bias ** 2 + b * self.scaling.bias - p2 * self.scaling.bias
        p3_scaling = self.scaling.clone(bias=p3_bias, scale=self.scaling.scale)
        self.p3 = p3_scaling.offset_scaling(self.init_param(p3))
        self.a = self.init_param(a)
        self.b = self.init_param(b)
        self.c = self.offset_scaling(self.init_param(c))
        self.d = self.std_scaling(self.init_param(d))
        self.R = self.init_param(R)
        self.tau = self.init_param(tau)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._u_initializer = is_initializer(u_initializer, allow_none=True)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def dV(self, V, t, u, I):
        I = self.sum_current_inputs(V, init=I)
        dVdt = self.p1 * V * V + self.p2 * V + self.p3 - u + I
        return dVdt

    def du(self, u, t, V):
        dudt = self.a * (self.b * V - u)
        return dudt

    @property
    def derivative(self):
        return JointEq([self.dV, self.du])

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.init_variable(self._V_initializer, batch_size)
        u_initializer = OneInit(self.b * self.V) if self._u_initializer is None else self._u_initializer
        self._u_initializer = is_initializer(u_initializer)
        self.V = self.offset_scaling(self.V)
        self.u = self.offset_scaling(self.init_variable(self._u_initializer, batch_size),
                                     bias=self.b * self.scaling.bias,
                                     scale=self.scaling.scale)
        self.spike = self.init_variable(partial(bm.zeros, dtype=self.spk_dtype), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, u = self.integral(self.V.value, self.u.value, t, x, dt)
        V += self.sum_delta_inputs()

        # spike, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike = stop_gradient(spike) if self.detach_spk else spike
            V += spike * (self.c - V)
            u += spike * self.d

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.c, V)
            u = bm.where(spike, u + self.d, u)

        self.V.value = V
        self.u.value = u
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class Izhikevich(IzhikevichLTC):
    r"""The Izhikevich neuron model.

    **Model Descriptions**

    The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

    .. math ::

        \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

        \frac{d u}{d t} &=a(b V-u)

    .. math ::

        \text{if}  v \geq 30  \text{mV}, \text{then}
        \begin{cases} v \leftarrow c \\
        u \leftarrow u+d \end{cases}


    **References**

    .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
           Transactions on neural networks 14.6 (2003): 1569-1572.

    .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
           IEEE transactions on neural networks 15.5 (2004): 1063-1070.

    **Examples**

    There is a simple usage example::

      import brainpy as bp

      neu = bp.dyn.Izhikevich(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

    **Model Parameters**

    ============= ============== ======== ================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------------------
    a             0.02           \        It determines the time scaling of
                                          the recovery variable :math:`u`.
    b             0.2            \        It describes the sensitivity of the
                                          recovery variable :math:`u` to
                                          the sub-threshold fluctuations of the
                                          membrane potential :math:`v`.
    c             -65            \        It describes the after-spike reset value
                                          of the membrane potential :math:`v` caused by
                                          the fast high-threshold :math:`K^{+}`
                                          conductance.
    d             8              \        It describes after-spike reset of the
                                          recovery variable :math:`u`
                                          caused by slow high-threshold
                                          :math:`Na^{+}` and :math:`K^{+}` conductance.
    tau_ref       0              ms       Refractory period length. [ms]
    V_th          30             mV       The membrane potential threshold.
    ============= ============== ======== ================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                          -65        Membrane potential.
    u                          1          Recovery variable.
    input                      0          External and synaptic input current.
    spike                      False      Flag to mark whether the neuron is spiking.
    refractory                False       Flag to mark whether the neuron is in refractory period.
    t_last_spike               -1e7       Last spike time stamp.
    ================== ================= =========================================================


    Args:
      %s
      %s

    """

    def dV(self, V, t, u, I):
        dVdt = self.p1 * V * V + self.p2 * V + self.p3 - u + I
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class IzhikevichRefLTC(IzhikevichLTC):
    r"""The Izhikevich neuron model with liquid time-constant.

    **Model Descriptions**

    The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

    .. math ::

        \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

        \frac{d u}{d t} &=a(b V-u)

    .. math ::

        \text{if}  v \geq 30  \text{mV}, \text{then}
        \begin{cases} v \leftarrow c \\
        u \leftarrow u+d \end{cases}

      **References**

    .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
           Transactions on neural networks 14.6 (2003): 1569-1572.

    .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
           IEEE transactions on neural networks 15.5 (2004): 1063-1070.

    **Examples**

    There is a simple usage example::

      import brainpy as bp

      neu = bp.dyn.IzhikevichRefLTC(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

    **Model Parameters**

    ============= ============== ======== ================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------------------
    a             0.02           \        It determines the time scaling of
                                          the recovery variable :math:`u`.
    b             0.2            \        It describes the sensitivity of the
                                          recovery variable :math:`u` to
                                          the sub-threshold fluctuations of the
                                          membrane potential :math:`v`.
    c             -65            \        It describes the after-spike reset value
                                          of the membrane potential :math:`v` caused by
                                          the fast high-threshold :math:`K^{+}`
                                          conductance.
    d             8              \        It describes after-spike reset of the
                                          recovery variable :math:`u`
                                          caused by slow high-threshold
                                          :math:`Na^{+}` and :math:`K^{+}` conductance.
    tau_ref       0              ms       Refractory period length. [ms]
    V_th          30             mV       The membrane potential threshold.
    ============= ============== ======== ================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                          -65        Membrane potential.
    u                          1          Recovery variable.
    input                      0          External and synaptic input current.
    spike                      False      Flag to mark whether the neuron is spiking.
    refractory                False       Flag to mark whether the neuron is in refractory period.
    t_last_spike               -1e7       Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s

    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sharding] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = 'soft',
        detach_spk: bool = False,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,

        # old neuron parameter
        V_th: Union[float, ArrayType, Callable] = 30.,
        p1: Union[float, ArrayType, Callable] = 0.04,
        p2: Union[float, ArrayType, Callable] = 5.,
        p3: Union[float, ArrayType, Callable] = 140.,
        a: Union[float, ArrayType, Callable] = 0.02,
        b: Union[float, ArrayType, Callable] = 0.20,
        c: Union[float, ArrayType, Callable] = -65.,
        d: Union[float, ArrayType, Callable] = 8.,
        tau: Union[float, ArrayType, Callable] = 10.,
        R: Union[float, ArrayType, Callable] = 1.,
        V_initializer: Union[Callable, ArrayType] = OneInit(-70.),
        u_initializer: Union[Callable, ArrayType] = None,

        # new neuron parameter
        tau_ref: Union[float, ArrayType, Callable] = 0.,
        ref_var: bool = False,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            method=method,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,

            init_var=False,
            scaling=scaling,

            V_th=V_th,
            p1=p1,
            p2=p2,
            p3=p3,
            a=a,
            b=b,
            c=c,
            d=d,
            R=R,
            tau=tau,
            V_initializer=V_initializer,
            u_initializer=u_initializer
        )

        # parameters
        self.ref_var = ref_var
        self.tau_ref = self.init_param(tau_ref)

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._u_initializer = is_initializer(u_initializer, allow_none=True)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        super().reset_state(batch_size, **kwargs)
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e7)
        if self.ref_var:
            self.refractory = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        # integrate membrane potential
        V, u = self.integral(self.V.value, self.u.value, t, x, dt)
        V += self.sum_delta_inputs()

        # refractory
        refractory = (t - self.t_last_spike) <= self.tau_ref
        if isinstance(self.mode, bm.TrainingMode):
            refractory = stop_gradient(refractory)
        V = bm.where(refractory, self.V.value, V)

        # spike, refractory, spiking time, and membrane potential reset
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike
            V += spike * (self.c - V)
            u += spike * self.d
            spike_ = spike_no_grad > 0.
            # will be used in other place, like Delta Synapse, so stop its gradient
            if self.ref_var:
                self.refractory.value = stop_gradient(bm.logical_or(refractory, spike_).value)
            t_last_spike = stop_gradient(bm.where(spike_, t, self.t_last_spike.value))

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.c, V)
            u = bm.where(spike, u + self.d, u)
            if self.ref_var:
                self.refractory.value = bm.logical_or(refractory, spike)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)
        self.V.value = V
        self.u.value = u
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike
        return spike


class IzhikevichRef(IzhikevichRefLTC):
    r"""The Izhikevich neuron model.

    **Model Descriptions**

    The dynamics of the Izhikevich neuron model [1]_ [2]_ is given by:

    .. math ::

        \frac{d V}{d t} &= 0.04 V^{2}+5 V+140-u+I

        \frac{d u}{d t} &=a(b V-u)

    .. math ::

        \text{if}  v \geq 30  \text{mV}, \text{then}
        \begin{cases} v \leftarrow c \\
        u \leftarrow u+d \end{cases}



    **References**

    .. [1] Izhikevich, Eugene M. "Simple model of spiking neurons." IEEE
           Transactions on neural networks 14.6 (2003): 1569-1572.

    .. [2] Izhikevich, Eugene M. "Which model to use for cortical spiking neurons?."
           IEEE transactions on neural networks 15.5 (2004): 1063-1070.

    **Examples**

    There is a simple usage example::

      import brainpy as bp

      neu = bp.dyn.IzhikevichRef(2)

      # section input with wiener process
      inp1 = bp.inputs.wiener_process(500., n=1, t_start=100., t_end=400.).flatten()
      inputs = bp.inputs.section_input([0., 22., 0.], [100., 300., 100.]) + inp1

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs=inputs)

      bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], plot_ids=(0, 1), show=True)


    **Model Examples**

    - `Detailed examples to reproduce different firing patterns <https://brainpy-examples.readthedocs.io/en/latest/neurons/Izhikevich_2003_Izhikevich_model.html>`_

    **Model Parameters**

    ============= ============== ======== ================================================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- --------------------------------------------------------------------------------
    a             0.02           \        It determines the time scaling of
                                          the recovery variable :math:`u`.
    b             0.2            \        It describes the sensitivity of the
                                          recovery variable :math:`u` to
                                          the sub-threshold fluctuations of the
                                          membrane potential :math:`v`.
    c             -65            \        It describes the after-spike reset value
                                          of the membrane potential :math:`v` caused by
                                          the fast high-threshold :math:`K^{+}`
                                          conductance.
    d             8              \        It describes after-spike reset of the
                                          recovery variable :math:`u`
                                          caused by slow high-threshold
                                          :math:`Na^{+}` and :math:`K^{+}` conductance.
    tau_ref       0              ms       Refractory period length. [ms]
    V_th          30             mV       The membrane potential threshold.
    ============= ============== ======== ================================================================================

    **Model Variables**

    ================== ================= =========================================================
    **Variables name** **Initial Value** **Explanation**
    ------------------ ----------------- ---------------------------------------------------------
    V                          -65        Membrane potential.
    u                          1          Recovery variable.
    input                      0          External and synaptic input current.
    spike                      False      Flag to mark whether the neuron is spiking.
    refractory                False       Flag to mark whether the neuron is in refractory period.
    t_last_spike               -1e7       Last spike time stamp.
    ================== ================= =========================================================



    Args:
      %s
      %s
      %s
   """

    def dV(self, V, t, u, I):
        dVdt = self.p1 * V * V + self.p2 * V + self.p3 - u + I
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


Izhikevich.__doc__ = Izhikevich.__doc__ % (pneu_doc, dpneu_doc)
IzhikevichRefLTC.__doc__ = IzhikevichRefLTC.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
IzhikevichRef.__doc__ = IzhikevichRef.__doc__ % (pneu_doc, dpneu_doc, ref_doc)
IzhikevichLTC.__doc__ = IzhikevichLTC.__doc__ % ()
