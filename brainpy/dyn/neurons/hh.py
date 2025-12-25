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
from typing import Any, Sequence
from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy.check import is_initializer
from brainpy.context import share
from brainpy.dyn.base import NeuDyn, IonChaDyn
from brainpy.initialize import OneInit
from brainpy.initialize import Uniform, variable_, noise as init_noise
from brainpy.integrators import JointEq
from brainpy.integrators import odeint, sdeint
from brainpy.mixin import Container, TreeNode
from brainpy.types import ArrayType
from brainpy.types import Shape

__all__ = [
    'HHTypedNeuron',
    'CondNeuGroupLTC',
    'CondNeuGroup',
    'HHLTC',
    'HH',
    'MorrisLecarLTC',
    'MorrisLecar',
    'WangBuzsakiHHLTC',
    'WangBuzsakiHH'
]


class HHTypedNeuron(NeuDyn):
    pass


class CondNeuGroupLTC(HHTypedNeuron, Container, TreeNode):
    r"""Base class to model conductance-based neuron group.

    The standard formulation for a conductance-based model is given as

    .. math::

        C_m {dV \over dt} = \sum_jg_j(E - V) + I_{ext}

    where :math:`g_j=\bar{g}_{j} M^x N^y` is the channel conductance, :math:`E` is the
    reversal potential, :math:`M` is the activation variable, and :math:`N` is the
    inactivation variable.

    :math:`M` and :math:`N` have the dynamics of

    .. math::

        {dx \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}

    where :math:`x \in [M, N]`, :math:`\phi_x` is a temperature-dependent factor,
    :math:`x_\infty` is the steady state, and :math:`\tau_x` is the time constant.
    Equivalently, the above equation can be written as:

    .. math::

        \frac{d x}{d t}=\phi_{x}\left(\alpha_{x}(1-x)-\beta_{x} x\right)

    where :math:`\alpha_{x}` and :math:`\beta_{x}` are rate constants.

    .. versionadded:: 2.1.9
       Modeling the conductance-based neuron model.

    Parameters::

    size : int, sequence of int
      The network size of this neuron group.
    method: str
      The numerical integration method.
    name : optional, str
      The neuron group name.

    """

    def __init__(
        self,
        size: Shape,
        keep_size: bool = False,
        C: Union[float, ArrayType, Callable] = 1.,
        A: Union[float, ArrayType, Callable] = 1e-3,
        V_th: Union[float, ArrayType, Callable] = 0.,
        V_initializer: Union[Callable, ArrayType] = Uniform(-70, -60.),
        noise: Optional[Union[float, ArrayType, Callable]] = None,
        method: str = 'exp_auto',
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        init_var: bool = True,
        input_var: bool = True,
        spk_type: Optional[type] = None,
        **channels
    ):
        super().__init__(size, keep_size=keep_size, mode=mode, name=name, )

        # attribute for ``Container``
        self.children = bm.node_dict(self.format_elements(IonChaDyn, **channels))

        # parameters for neurons
        self.input_var = input_var
        self.C = C
        self.A = A
        self.V_th = V_th
        self.noise = init_noise(noise, self.varshape, num_vars=1)
        self._V_initializer = V_initializer
        self.spk_type = ((bm.float_ if isinstance(self.mode, bm.TrainingMode) else bm.bool)
                         if (spk_type is None) else spk_type)

        # function
        if self.noise is None:
            self.integral = odeint(f=self.derivative, method=method)
        else:
            self.integral = sdeint(f=self.derivative, g=self.noise, method=method)

        if init_var:
            self.reset_state(self.mode)

    def derivative(self, V, t, I):
        # synapses
        I = self.sum_current_inputs(V, init=I)
        # channels
        for ch in self.nodes(level=1, include_self=False).subset(IonChaDyn).unique().values():
            I = I + ch.current(V)
        return I / self.C

    def reset_state(self, batch_size=None):
        self.V = variable_(self._V_initializer, self.varshape, batch_size)
        self.spike = variable_(partial(bm.zeros, dtype=self.spk_type), self.varshape, batch_size)
        if self.input_var:
            self.input = variable_(bm.zeros, self.varshape, batch_size)
        for channel in self.nodes(level=1, include_self=False).subset(IonChaDyn).unique().values():
            channel.reset_state(self.V.value, batch_size=batch_size)

    def update(self, x=None):
        # inputs
        x = 0. if x is None else x
        if self.input_var:
            self.input += x
            x = self.input.value
        x = x * (1e-3 / self.A)

        # integral
        V = self.integral(self.V.value, share['t'], x, share['dt']) + self.sum_delta_inputs()

        # check whether the children channels have the correct parents.
        channels = self.nodes(level=1, include_self=False).subset(IonChaDyn).unique()
        self.check_hierarchies(self.__class__, **channels)

        # update channels
        for node in channels.values():
            node(self.V.value)

        # update variables
        if self.spike.dtype == bool:
            self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th)
        else:
            self.spike.value = bm.logical_and(V >= self.V_th, self.V < self.V_th).astype(self.spike.dtype)
        self.V.value = V
        return self.spike.value

    def clear_input(self):
        """Useful for monitoring inputs. """
        if self.input_var:
            self.input.value = bm.zeros_like(self.input)

    def return_info(self):
        return self.spike


class CondNeuGroup(CondNeuGroupLTC):
    def derivative(self, V, t, I):
        for ch in self.nodes(level=1, include_self=False).subset(IonChaDyn).unique().values():
            I = I + ch.current(V)
        return I / self.C

    def update(self, x=None):
        # inputs
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class HHLTC(NeuDyn):
    r"""Hodgkin–Huxley neuron model with liquid time constant.

    **Model Descriptions**

    The Hodgkin-Huxley (HH; Hodgkin & Huxley, 1952) model [1]_ for the generation of
    the nerve action potential is one of the most successful mathematical models of
    a complex biological process that has ever been formulated. The basic concepts
    expressed in the model have proved a valid approach to the study of bio-electrical
    activity from the most primitive single-celled organisms such as *Paramecium*,
    right through to the neurons within our own brains.

    Mathematically, the model is given by,

    .. math::

        C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V &-E_{Na})
        + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

        \frac {dx} {dt} &= \alpha_x (1-x)  - \beta_x, \quad x\in {\rm{\{m, h, n\}}}

        &\alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}

        &\beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})

        &\alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})

        &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}

        &\alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}

        &\beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})

    The illustrated example of HH neuron model please see `this notebook <../neurons/HH_model.ipynb>`_.

    The Hodgkin–Huxley model can be thought of as a differential equation system with
    four state variables, :math:`V_{m}(t),n(t),m(t)`, and :math:`h(t)`, that change
    with respect to time :math:`t`. The system is difficult to study because it is a
    nonlinear system and cannot be solved analytically. However, there are many numeric
    methods available to analyze the system. Certain properties and general behaviors,
    such as limit cycles, can be proven to exist.


    References::

    .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
           of membrane current and its application to conduction and excitation
           in nerve." The Journal of physiology 117.4 (1952): 500.
    .. [2] https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
    .. [3] Ashwin, Peter, Stephen Coombes, and Rachel Nicks. "Mathematical
           frameworks for oscillatory network dynamics in neuroscience."
           The Journal of Mathematical Neuroscience 6, no. 1 (2016): 1-92.

    **Examples**

    Here is a simple usage example:

    .. code-block:: python

        import brainpy as bp

        neu = bp.dyn.HHLTC(1)

        # raise input current from 4 mA to 40 mA
        inputs = bp.inputs.ramp_input(4, 40, 700, 100, 600,)

        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)

        bp.visualize.line_plot(runner.mon['ts'], runner.mon['V'], show=True)



    Parameters::

    size: sequence of int, int
      The size of the neuron group.
    ENa: float, ArrayType, Initializer, callable
      The reversal potential of sodium. Default is 50 mV.
    gNa: float, ArrayType, Initializer, callable
      The maximum conductance of sodium channel. Default is 120 msiemens.
    EK: float, ArrayType, Initializer, callable
      The reversal potential of potassium. Default is -77 mV.
    gK: float, ArrayType, Initializer, callable
      The maximum conductance of potassium channel. Default is 36 msiemens.
    EL: float, ArrayType, Initializer, callable
      The reversal potential of learky channel. Default is -54.387 mV.
    gL: float, ArrayType, Initializer, callable
      The conductance of learky channel. Default is 0.03 msiemens.
    V_th: float, ArrayType, Initializer, callable
      The threshold of the membrane spike. Default is 20 mV.
    C: float, ArrayType, Initializer, callable
      The membrane capacitance. Default is 1 ufarad.
    V_initializer: ArrayType, Initializer, callable
      The initializer of membrane potential.
    m_initializer: ArrayType, Initializer, callable
      The initializer of m channel.
    h_initializer: ArrayType, Initializer, callable
      The initializer of h channel.
    n_initializer: ArrayType, Initializer, callable
      The initializer of n channel.
    method: str
      The numerical integration method.
    name: str
      The group name.




    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        sharding: Any = None,
        keep_size: bool = False,
        mode: bm.Mode = None,
        name: str = None,
        method: str = 'exp_auto',
        init_var: bool = True,

        # neuron parameters
        ENa: Union[float, ArrayType, Callable] = 50.,
        gNa: Union[float, ArrayType, Callable] = 120.,
        EK: Union[float, ArrayType, Callable] = -77.,
        gK: Union[float, ArrayType, Callable] = 36.,
        EL: Union[float, ArrayType, Callable] = -54.387,
        gL: Union[float, ArrayType, Callable] = 0.03,
        V_th: Union[float, ArrayType, Callable] = 20.,
        C: Union[float, ArrayType, Callable] = 1.0,
        V_initializer: Union[Callable, ArrayType] = Uniform(-70, -60.),
        m_initializer: Optional[Union[Callable, ArrayType]] = None,
        h_initializer: Optional[Union[Callable, ArrayType]] = None,
        n_initializer: Optional[Union[Callable, ArrayType]] = None,

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         sharding=sharding,
                         keep_size=keep_size,
                         mode=mode,
                         name=name,
                         method=method)

        # parameters
        self.ENa = self.init_param(ENa)
        self.EK = self.init_param(EK)
        self.EL = self.init_param(EL)
        self.gNa = self.init_param(gNa)
        self.gK = self.init_param(gK)
        self.gL = self.init_param(gL)
        self.C = self.init_param(C)
        self.V_th = self.init_param(V_th)

        # initializers
        self._m_initializer = is_initializer(m_initializer, allow_none=True)
        self._h_initializer = is_initializer(h_initializer, allow_none=True)
        self._n_initializer = is_initializer(n_initializer, allow_none=True)
        self._V_initializer = is_initializer(V_initializer)

        # noise
        self.noise = init_noise(noise, self.varshape, num_vars=4)

        # integral
        if self.noise is None:
            self.integral = odeint(method=method, f=self.derivative)
        else:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)

        # model
        if init_var:
            self.reset_state(self.mode)

    # m channel
    # m_alpha = lambda self, V: 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    m_alpha = lambda self, V: 1. / bm.exprel(-(V + 40) / 10)
    m_beta = lambda self, V: 4.0 * bm.exp(-(V + 65) / 18)
    m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))
    dm = lambda self, m, t, V: self.m_alpha(V) * (1 - m) - self.m_beta(V) * m

    # h channel
    h_alpha = lambda self, V: 0.07 * bm.exp(-(V + 65) / 20.)
    h_beta = lambda self, V: 1 / (1 + bm.exp(-(V + 35) / 10))
    h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))
    dh = lambda self, h, t, V: self.h_alpha(V) * (1 - h) - self.h_beta(V) * h

    # n channel
    # n_alpha = lambda self, V: 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    n_alpha = lambda self, V: 0.1 / bm.exprel(-(V + 55) / 10)
    n_beta = lambda self, V: 0.125 * bm.exp(-(V + 65) / 80)
    n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))
    dn = lambda self, n, t, V: self.n_alpha(V) * (1 - n) - self.n_beta(V) * n

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.init_variable(self._V_initializer, batch_size)
        if self._m_initializer is None:
            self.m = bm.Variable(self.m_inf(self.V.value), batch_axis=self.V.batch_axis)
        else:
            self.m = self.init_variable(self._m_initializer, batch_size)
        if self._h_initializer is None:
            self.h = bm.Variable(self.h_inf(self.V.value), batch_axis=self.V.batch_axis)
        else:
            self.h = self.init_variable(self._h_initializer, batch_size)
        if self._n_initializer is None:
            self.n = bm.Variable(self.n_inf(self.V.value), batch_axis=self.V.batch_axis)
        else:
            self.n = self.init_variable(self._n_initializer, batch_size)
        self.spike = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def dV(self, V, t, m, h, n, I):
        I = self.sum_current_inputs(V, init=I)
        I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
        n2 = n * n
        I_K = (self.gK * n2 * n2) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (- I_Na - I_K - I_leak + I) / self.C
        return dVdt

    @property
    def derivative(self):
        return JointEq(self.dV, self.dm, self.dh, self.dn)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        V, m, h, n = self.integral(self.V.value, self.m.value, self.h.value, self.n.value, t, x, dt)
        V += self.sum_delta_inputs()
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n
        return self.spike.value

    def return_info(self):
        return self.spike


class HH(HHLTC):
    r"""Hodgkin–Huxley neuron model.

    **Model Descriptions**

    The Hodgkin-Huxley (HH; Hodgkin & Huxley, 1952) model [1]_ for the generation of
    the nerve action potential is one of the most successful mathematical models of
    a complex biological process that has ever been formulated. The basic concepts
    expressed in the model have proved a valid approach to the study of bio-electrical
    activity from the most primitive single-celled organisms such as *Paramecium*,
    right through to the neurons within our own brains.

    Mathematically, the model is given by,

    .. math::

        C \frac {dV} {dt} = -(\bar{g}_{Na} m^3 h (V &-E_{Na})
        + \bar{g}_K n^4 (V-E_K) + g_{leak} (V - E_{leak})) + I(t)

        \frac {dx} {dt} &= \alpha_x (1-x)  - \beta_x, \quad x\in {\rm{\{m, h, n\}}}

        &\alpha_m(V) = \frac {0.1(V+40)}{1-\exp(\frac{-(V + 40)} {10})}

        &\beta_m(V) = 4.0 \exp(\frac{-(V + 65)} {18})

        &\alpha_h(V) = 0.07 \exp(\frac{-(V+65)}{20})

        &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V + 35)} {10})}

        &\alpha_n(V) = \frac {0.01(V+55)}{1-\exp(-(V+55)/10)}

        &\beta_n(V) = 0.125 \exp(\frac{-(V + 65)} {80})

    References::

    .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description
           of membrane current and its application to conduction and excitation
           in nerve." The Journal of physiology 117.4 (1952): 500.
    .. [2] https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
    .. [3] Ashwin, Peter, Stephen Coombes, and Rachel Nicks. "Mathematical
           frameworks for oscillatory network dynamics in neuroscience."
           The Journal of Mathematical Neuroscience 6, no. 1 (2016): 1-92.

    **Examples**

    Here is a simple usage example:

    .. code-block:: python

      import brainpy as bp
      import matplotlib.pyplot as plt

      neu = bp.dyn.HH(1,)

      inputs = bp.inputs.ramp_input(4, 40, 700, 100, 600, )

      runner = bp.DSRunner(neu, monitors=['V'])
      runner.run(inputs = inputs)

      plt.plot(runner.mon['ts'], runner.mon['V'])
      plt.plot(runner.mon.ts, inputs.value)       # show input current
      plt.legend(['Membrane potential/mA', 'Input current/mA'], loc='upper right')

      plt.tight_layout()
      plt.show()

    The illustrated example of HH neuron model please see `this notebook <../neurons/HH_model.ipynb>`_.


    Parameters::

    size: sequence of int, int
      The size of the neuron group.
    ENa: float, ArrayType, Initializer, callable
      The reversal potential of sodium. Default is 50 mV.
    gNa: float, ArrayType, Initializer, callable
      The maximum conductance of sodium channel. Default is 120 msiemens.
    EK: float, ArrayType, Initializer, callable
      The reversal potential of potassium. Default is -77 mV.
    gK: float, ArrayType, Initializer, callable
      The maximum conductance of potassium channel. Default is 36 msiemens.
    EL: float, ArrayType, Initializer, callable
      The reversal potential of learky channel. Default is -54.387 mV.
    gL: float, ArrayType, Initializer, callable
      The conductance of learky channel. Default is 0.03 msiemens.
    V_th: float, ArrayType, Initializer, callable
      The threshold of the membrane spike. Default is 20 mV.
    C: float, ArrayType, Initializer, callable
      The membrane capacitance. Default is 1 ufarad.
    V_initializer: ArrayType, Initializer, callable
      The initializer of membrane potential.
    m_initializer: ArrayType, Initializer, callable
      The initializer of m channel.
    h_initializer: ArrayType, Initializer, callable
      The initializer of h channel.
    n_initializer: ArrayType, Initializer, callable
      The initializer of n channel.
    method: str
      The numerical integration method.
    name: str
      The group name.

    """

    def dV(self, V, t, m, h, n, I):
        I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
        n2 = n * n
        I_K = (self.gK * n2 * n2) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (- I_Na - I_K - I_leak + I) / self.C
        return dVdt

    @property
    def derivative(self):
        return JointEq(self.dV, self.dm, self.dh, self.dn)

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class MorrisLecarLTC(NeuDyn):
    r"""The Morris-Lecar neuron model with liquid time constant.

    **Model Descriptions**

    The Morris-Lecar model [4]_ (Also known as :math:`I_{Ca}+I_K`-model)
    is a two-dimensional "reduced" excitation model applicable to
    systems having two non-inactivating voltage-sensitive conductances.
    This model was named after Cathy Morris and Harold Lecar, who
    derived it in 1981. Because it is two-dimensional, the Morris-Lecar
    model is one of the favorite conductance-based models in computational neuroscience.

    The original form of the model employed an instantaneously
    responding voltage-sensitive Ca2+ conductance for excitation and a delayed
    voltage-dependent K+ conductance for recovery. The equations of the model are:

    .. math::

        \begin{aligned}
        C\frac{dV}{dt} =& -  g_{Ca} M_{\infty} (V - V_{Ca})- g_{K} W(V - V_{K}) -
                          g_{Leak} (V - V_{Leak}) + I_{ext} \\
        \frac{dW}{dt} =& \frac{W_{\infty}(V) - W}{ \tau_W(V)}
        \end{aligned}

    Here, :math:`V` is the membrane potential, :math:`W` is the "recovery variable",
    which is almost invariably the normalized :math:`K^+`-ion conductance, and
    :math:`I_{ext}` is the applied current stimulus.


    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_Ca          130            mV       Equilibrium potentials of Ca+.(mV)
    g_Ca          4.4            \        Maximum conductance of corresponding Ca+.(mS/cm2)
    V_K           -84            mV       Equilibrium potentials of K+.(mV)
    g_K           8              \        Maximum conductance of corresponding K+.(mS/cm2)
    V_Leak        -60            mV       Equilibrium potentials of leak current.(mV)
    g_Leak        2              \        Maximum conductance of leak current.(mS/cm2)
    C             20             \        Membrane capacitance.(uF/cm2)
    V1            -1.2           \        Potential at which M_inf = 0.5.(mV)
    V2            18             \        Reciprocal of slope of voltage dependence of M_inf.(mV)
    V3            2              \        Potential at which W_inf = 0.5.(mV)
    V4            30             \        Reciprocal of slope of voltage dependence of W_inf.(mV)
    phi           0.04           \        A temperature factor. (1/s)
    V_th          10             mV       The spike threshold.
    ============= ============== ======== =======================================================

    References::

    .. [4] Lecar, Harold. "Morris-lecar model." Scholarpedia 2.10 (2007): 1333.
    .. [5] http://www.scholarpedia.org/article/Morris-Lecar_model
    .. [6] https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model
    """

    supported_modes = (bm.NonBatchingMode, bm.BatchingMode)

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        sharding: Any = None,
        keep_size: bool = False,
        mode: bm.Mode = None,
        name: str = None,
        method: str = 'exp_auto',
        init_var: bool = True,

        # neuron parameters
        V_Ca: Union[float, ArrayType, Callable] = 130.,
        g_Ca: Union[float, ArrayType, Callable] = 4.4,
        V_K: Union[float, ArrayType, Callable] = -84.,
        g_K: Union[float, ArrayType, Callable] = 8.,
        V_leak: Union[float, ArrayType, Callable] = -60.,
        g_leak: Union[float, ArrayType, Callable] = 2.,
        C: Union[float, ArrayType, Callable] = 20.,
        V1: Union[float, ArrayType, Callable] = -1.2,
        V2: Union[float, ArrayType, Callable] = 18.,
        V3: Union[float, ArrayType, Callable] = 2.,
        V4: Union[float, ArrayType, Callable] = 30.,
        phi: Union[float, ArrayType, Callable] = 0.04,
        V_th: Union[float, ArrayType, Callable] = 10.,
        W_initializer: Union[Callable, ArrayType] = OneInit(0.02),
        V_initializer: Union[Callable, ArrayType] = Uniform(-70., -60.),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         sharding=sharding,
                         keep_size=keep_size,
                         mode=mode,
                         name=name,
                         method=method)

        # parameters
        self.V_Ca = self.init_param(V_Ca)
        self.g_Ca = self.init_param(g_Ca)
        self.V_K = self.init_param(V_K)
        self.g_K = self.init_param(g_K)
        self.V_leak = self.init_param(V_leak)
        self.g_leak = self.init_param(g_leak)
        self.C = self.init_param(C)
        self.V1 = self.init_param(V1)
        self.V2 = self.init_param(V2)
        self.V3 = self.init_param(V3)
        self.V4 = self.init_param(V4)
        self.phi = self.init_param(phi)
        self.V_th = self.init_param(V_th)

        # initializers
        self._W_initializer = is_initializer(W_initializer)
        self._V_initializer = is_initializer(V_initializer)

        # noise
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        # integral
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # model
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_or_mode=None, **kwargs):
        self.V = self.init_variable(self._V_initializer, batch_or_mode)
        self.W = self.init_variable(self._W_initializer, batch_or_mode)
        self.spike = self.init_variable(partial(bm.zeros, dtype=bool), batch_or_mode)

    def dV(self, V, t, W, I):
        I = self.sum_current_inputs(V, init=I)
        M_inf = (1 / 2) * (1 + bm.tanh((V - self.V1) / self.V2))
        I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
        I_K = self.g_K * W * (V - self.V_K)
        I_Leak = self.g_leak * (V - self.V_leak)
        dVdt = (- I_Ca - I_K - I_Leak + I) / self.C
        return dVdt

    def dW(self, W, t, V):
        tau_W = 1 / (self.phi * bm.cosh((V - self.V3) / (2 * self.V4)))
        W_inf = (1 / 2) * (1 + bm.tanh((V - self.V3) / self.V4))
        dWdt = (W_inf - W) / tau_W
        return dWdt

    @property
    def derivative(self):
        return JointEq(self.dV, self.dW)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x
        V, W = self.integral(self.V.value, self.W.value, t, x, dt)
        V += self.sum_delta_inputs()
        spike = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V.value = V
        self.W.value = W
        self.spike.value = spike
        return spike

    def return_info(self):
        return self.spike


class MorrisLecar(MorrisLecarLTC):
    r"""The Morris-Lecar neuron model.

    **Model Descriptions**

    The Morris-Lecar model [4]_ (Also known as :math:`I_{Ca}+I_K`-model)
    is a two-dimensional "reduced" excitation model applicable to
    systems having two non-inactivating voltage-sensitive conductances.
    This model was named after Cathy Morris and Harold Lecar, who
    derived it in 1981. Because it is two-dimensional, the Morris-Lecar
    model is one of the favorite conductance-based models in computational neuroscience.

    The original form of the model employed an instantaneously
    responding voltage-sensitive Ca2+ conductance for excitation and a delayed
    voltage-dependent K+ conductance for recovery. The equations of the model are:

    .. math::

        \begin{aligned}
        C\frac{dV}{dt} =& -  g_{Ca} M_{\infty} (V - V_{Ca})- g_{K} W(V - V_{K}) -
                          g_{Leak} (V - V_{Leak}) + I_{ext} \\
        \frac{dW}{dt} =& \frac{W_{\infty}(V) - W}{ \tau_W(V)}
        \end{aligned}

    Here, :math:`V` is the membrane potential, :math:`W` is the "recovery variable",
    which is almost invariably the normalized :math:`K^+`-ion conductance, and
    :math:`I_{ext}` is the applied current stimulus.

    **Model Parameters**

    ============= ============== ======== =======================================================
    **Parameter** **Init Value** **Unit** **Explanation**
    ------------- -------------- -------- -------------------------------------------------------
    V_Ca          130            mV       Equilibrium potentials of Ca+.(mV)
    g_Ca          4.4            \        Maximum conductance of corresponding Ca+.(mS/cm2)
    V_K           -84            mV       Equilibrium potentials of K+.(mV)
    g_K           8              \        Maximum conductance of corresponding K+.(mS/cm2)
    V_Leak        -60            mV       Equilibrium potentials of leak current.(mV)
    g_Leak        2              \        Maximum conductance of leak current.(mS/cm2)
    C             20             \        Membrane capacitance.(uF/cm2)
    V1            -1.2           \        Potential at which M_inf = 0.5.(mV)
    V2            18             \        Reciprocal of slope of voltage dependence of M_inf.(mV)
    V3            2              \        Potential at which W_inf = 0.5.(mV)
    V4            30             \        Reciprocal of slope of voltage dependence of W_inf.(mV)
    phi           0.04           \        A temperature factor. (1/s)
    V_th          10             mV       The spike threshold.
    ============= ============== ======== =======================================================

    References::

    .. [4] Lecar, Harold. "Morris-lecar model." Scholarpedia 2.10 (2007): 1333.
    .. [5] http://www.scholarpedia.org/article/Morris-Lecar_model
    .. [6] https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model
    """

    def dV(self, V, t, W, I):
        M_inf = (1 / 2) * (1 + bm.tanh((V - self.V1) / self.V2))
        I_Ca = self.g_Ca * M_inf * (V - self.V_Ca)
        I_K = self.g_K * W * (V - self.V_K)
        I_Leak = self.g_leak * (V - self.V_leak)
        dVdt = (- I_Ca - I_K - I_Leak + I) / self.C
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)


class WangBuzsakiHHLTC(NeuDyn):
    r"""Wang-Buzsaki model [9]_, an implementation of a modified Hodgkin-Huxley model with liquid time constant.

    Each model is described by a single compartment and obeys the current balance equation:

    .. math::

        C_{m} \frac{d V}{d t}=-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{syn}}+I_{\mathrm{app}}

    where :math:`C_{m}=1 \mu \mathrm{F} / \mathrm{cm}^{2}` and :math:`I_{\mathrm{app}}` is the
    injected current (in :math:`\mu \mathrm{A} / \mathrm{cm}^{2}` ). The leak current
    :math:`I_{\mathrm{L}}=g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)` has a conductance
    :math:`g_{\mathrm{L}}=0.1 \mathrm{mS} / \mathrm{cm}^{2}`, so that the passive time constant
    :math:`\tau_{0}=C_{m} / g_{\mathrm{L}}=10 \mathrm{msec} ; E_{\mathrm{L}}=-65 \mathrm{mV}`.

    The spike-generating :math:`\mathrm{Na}^{+}` and :math:`\mathrm{K}^{+}` voltage-dependent ion
    currents :math:`\left(I_{\mathrm{Na}}\right.` and :math:`I_{\mathrm{K}}` ) are of the
    Hodgkin-Huxley type (Hodgkin and Huxley, 1952). The transient sodium current
    :math:`I_{\mathrm{Na}}=g_{\mathrm{Na}} m_{\infty}^{3} h\left(V-E_{\mathrm{Na}}\right)`,
    where the activation variable :math:`m` is assumed fast and substituted by its steady-state
    function :math:`m_{\infty}=\alpha_{m} /\left(\alpha_{m}+\beta_{m}\right)` ;
    :math:`\alpha_{m}(V)=-0.1(V+35) /(\exp (-0.1(V+35))-1), \beta_{m}(V)=4 \exp (-(V+60) / 18)`.
    The inactivation variable :math:`h` obeys a first-order kinetics:

    .. math::

        \frac{d h}{d t}=\phi\left(\alpha_{h}(1-h)-\beta_{h} h\right)

    where :math:`\alpha_{h}(V)=0.07 \exp (-(V+58) / 20)` and
    :math:`\beta_{h}(V)=1 /(\exp (-0.1(V+28)) +1) \cdot g_{\mathrm{Na}}=35 \mathrm{mS} / \mathrm{cm}^{2}` ;
    :math:`E_{\mathrm{Na}}=55 \mathrm{mV}, \phi=5 .`

    The delayed rectifier :math:`I_{\mathrm{K}}=g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)`,
    where the activation variable :math:`n` obeys the following equation:

    .. math::

       \frac{d n}{d t}=\phi\left(\alpha_{n}(1-n)-\beta_{n} n\right)

    with :math:`\alpha_{n}(V)=-0.01(V+34) /(\exp (-0.1(V+34))-1)` and
    :math:`\beta_{n}(V)=0.125\exp (-(V+44) / 80)` ; :math:`g_{\mathrm{K}}=9 \mathrm{mS} / \mathrm{cm}^{2}`, and
    :math:`E_{\mathrm{K}}=-90 \mathrm{mV}`.


    References::

    .. [9] Wang, X.J. and Buzsaki, G., (1996) Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model. Journal of
           neuroscience, 16(20), pp.6402-6413.

    **Examples**

    Here is a simple usage example:

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        neu = bp.dyn.WangBuzsakiHHLTC(1, )

        inputs = bp.inputs.ramp_input(.1, 1, 700, 100, 600, )
        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)
        plt.plot(runner.mon['ts'], runner.mon['V'])
        plt.legend(['Membrane potential/mA', loc='upper right')
        plt.tight_layout()
        plt.show()

    Parameters::

    size: sequence of int, int
      The size of the neuron group.
    ENa: float, ArrayType, Initializer, callable
      The reversal potential of sodium. Default is 50 mV.
    gNa: float, ArrayType, Initializer, callable
      The maximum conductance of sodium channel. Default is 120 msiemens.
    EK: float, ArrayType, Initializer, callable
      The reversal potential of potassium. Default is -77 mV.
    gK: float, ArrayType, Initializer, callable
      The maximum conductance of potassium channel. Default is 36 msiemens.
    EL: float, ArrayType, Initializer, callable
      The reversal potential of learky channel. Default is -54.387 mV.
    gL: float, ArrayType, Initializer, callable
      The conductance of learky channel. Default is 0.03 msiemens.
    V_th: float, ArrayType, Initializer, callable
      The threshold of the membrane spike. Default is 20 mV.
    C: float, ArrayType, Initializer, callable
      The membrane capacitance. Default is 1 ufarad.
    phi: float, ArrayType, Initializer, callable
      The temperature regulator constant.
    V_initializer: ArrayType, Initializer, callable
      The initializer of membrane potential.
    h_initializer: ArrayType, Initializer, callable
      The initializer of h channel.
    n_initializer: ArrayType, Initializer, callable
      The initializer of n channel.
    method: str
      The numerical integration method.
    name: str
      The group name.


    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        sharding: Any = None,
        keep_size: bool = False,
        mode: bm.Mode = None,
        name: str = None,
        method: str = 'exp_auto',
        init_var: bool = True,

        # neuron parameters
        ENa: Union[float, ArrayType, Callable] = 55.,
        gNa: Union[float, ArrayType, Callable] = 35.,
        EK: Union[float, ArrayType, Callable] = -90.,
        gK: Union[float, ArrayType, Callable] = 9.,
        EL: Union[float, ArrayType, Callable] = -65,
        gL: Union[float, ArrayType, Callable] = 0.1,
        V_th: Union[float, ArrayType, Callable] = 20.,
        phi: Union[float, ArrayType, Callable] = 5.0,
        C: Union[float, ArrayType, Callable] = 1.0,
        V_initializer: Union[Callable, ArrayType] = OneInit(-65.),
        h_initializer: Union[Callable, ArrayType] = OneInit(0.6),
        n_initializer: Union[Callable, ArrayType] = OneInit(0.32),

        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        # initialization
        super().__init__(size=size,
                         sharding=sharding,
                         keep_size=keep_size,
                         mode=mode,
                         name=name,
                         method=method)

        # parameters
        self.ENa = self.init_param(ENa)
        self.EK = self.init_param(EK)
        self.EL = self.init_param(EL)
        self.gNa = self.init_param(gNa)
        self.gK = self.init_param(gK)
        self.gL = self.init_param(gL)
        self.phi = self.init_param(phi)
        self.C = self.init_param(C)
        self.V_th = self.init_param(V_th)

        # initializers
        self._h_initializer = is_initializer(h_initializer)
        self._n_initializer = is_initializer(n_initializer)
        self._V_initializer = is_initializer(V_initializer)

        # noise
        self.noise = init_noise(noise, self.varshape, num_vars=3)
        # integral
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # model
        if init_var:
            self.reset_state(self.mode)

    def reset_state(self, batch_size=None):
        self.V = self.init_variable(self._V_initializer, batch_size)
        self.h = self.init_variable(self._h_initializer, batch_size)
        self.n = self.init_variable(self._n_initializer, batch_size)
        self.spike = self.init_variable(partial(bm.zeros, dtype=bool), batch_size)

    def m_inf(self, V):
        # alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
        alpha = 1. / bm.exprel(-0.1 * (V + 35))
        beta = 4. * bm.exp(-(V + 60.) / 18.)
        return alpha / (alpha + beta)

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 58) / 20)
        beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
        dhdt = alpha * (1 - h) - beta * h
        return self.phi * dhdt

    def dn(self, n, t, V):
        # alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
        alpha = 1. / bm.exprel(-0.1 * (V + 34))
        beta = 0.125 * bm.exp(-(V + 44) / 80)
        dndt = alpha * (1 - n) - beta * n
        return self.phi * dndt

    def dV(self, V, t, h, n, I):
        I = self.sum_current_inputs(V, init=I)
        INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + I) / self.C
        return dVdt

    @property
    def derivative(self):
        return JointEq(self.dV, self.dh, self.dn)

    def update(self, x=None):
        t = share.load('t')
        dt = share.load('dt')
        x = 0. if x is None else x

        V, h, n = self.integral(self.V.value, self.h.value, self.n.value, t, x, dt)
        V += self.sum_delta_inputs()
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V.value = V
        self.h.value = h
        self.n.value = n
        return self.spike.value

    def return_info(self):
        return self.spike


class WangBuzsakiHH(WangBuzsakiHHLTC):
    r"""Wang-Buzsaki model [9]_, an implementation of a modified Hodgkin-Huxley model.

    Each model is described by a single compartment and obeys the current balance equation:

    .. math::

        C_{m} \frac{d V}{d t}=-I_{\mathrm{Na}}-I_{\mathrm{K}}-I_{\mathrm{L}}-I_{\mathrm{syn}}+I_{\mathrm{app}}

    where :math:`C_{m}=1 \mu \mathrm{F} / \mathrm{cm}^{2}` and :math:`I_{\mathrm{app}}` is the
    injected current (in :math:`\mu \mathrm{A} / \mathrm{cm}^{2}` ). The leak current
    :math:`I_{\mathrm{L}}=g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)` has a conductance
    :math:`g_{\mathrm{L}}=0.1 \mathrm{mS} / \mathrm{cm}^{2}`, so that the passive time constant
    :math:`\tau_{0}=C_{m} / g_{\mathrm{L}}=10 \mathrm{msec} ; E_{\mathrm{L}}=-65 \mathrm{mV}`.

    The spike-generating :math:`\mathrm{Na}^{+}` and :math:`\mathrm{K}^{+}` voltage-dependent ion
    currents :math:`\left(I_{\mathrm{Na}}\right.` and :math:`I_{\mathrm{K}}` ) are of the
    Hodgkin-Huxley type (Hodgkin and Huxley, 1952). The transient sodium current
    :math:`I_{\mathrm{Na}}=g_{\mathrm{Na}} m_{\infty}^{3} h\left(V-E_{\mathrm{Na}}\right)`,
    where the activation variable :math:`m` is assumed fast and substituted by its steady-state
    function :math:`m_{\infty}=\alpha_{m} /\left(\alpha_{m}+\beta_{m}\right)` ;
    :math:`\alpha_{m}(V)=-0.1(V+35) /(\exp (-0.1(V+35))-1), \beta_{m}(V)=4 \exp (-(V+60) / 18)`.
    The inactivation variable :math:`h` obeys a first-order kinetics:

    .. math::

        \frac{d h}{d t}=\phi\left(\alpha_{h}(1-h)-\beta_{h} h\right)

    where :math:`\alpha_{h}(V)=0.07 \exp (-(V+58) / 20)` and
    :math:`\beta_{h}(V)=1 /(\exp (-0.1(V+28)) +1) \cdot g_{\mathrm{Na}}=35 \mathrm{mS} / \mathrm{cm}^{2}` ;
    :math:`E_{\mathrm{Na}}=55 \mathrm{mV}, \phi=5 .`

    The delayed rectifier :math:`I_{\mathrm{K}}=g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)`,
    where the activation variable :math:`n` obeys the following equation:

    .. math::

       \frac{d n}{d t}=\phi\left(\alpha_{n}(1-n)-\beta_{n} n\right)

    with :math:`\alpha_{n}(V)=-0.01(V+34) /(\exp (-0.1(V+34))-1)` and
    :math:`\beta_{n}(V)=0.125\exp (-(V+44) / 80)` ; :math:`g_{\mathrm{K}}=9 \mathrm{mS} / \mathrm{cm}^{2}`, and
    :math:`E_{\mathrm{K}}=-90 \mathrm{mV}`.


    References::

    .. [9] Wang, X.J. and Buzsaki, G., (1996) Gamma oscillation by synaptic
           inhibition in a hippocampal interneuronal network model. Journal of
           neuroscience, 16(20), pp.6402-6413.

    **Examples**

    Here is an example:

    .. code-block:: python

        import brainpy as bp
        import matplotlib.pyplot as plt

        neu = bp.dyn.WangBuzsakiHH(1, )

        inputs = bp.inputs.ramp_input(.1, 1, 700, 100, 600, )
        runner = bp.DSRunner(neu, monitors=['V'])
        runner.run(inputs=inputs)
        plt.plot(runner.mon['ts'], runner.mon['V'])
        plt.legend(['Membrane potential/mA', loc='upper right')
        plt.tight_layout()
        plt.show()

    Parameters::

    size: sequence of int, int
      The size of the neuron group.
    ENa: float, ArrayType, Initializer, callable
      The reversal potential of sodium. Default is 50 mV.
    gNa: float, ArrayType, Initializer, callable
      The maximum conductance of sodium channel. Default is 120 msiemens.
    EK: float, ArrayType, Initializer, callable
      The reversal potential of potassium. Default is -77 mV.
    gK: float, ArrayType, Initializer, callable
      The maximum conductance of potassium channel. Default is 36 msiemens.
    EL: float, ArrayType, Initializer, callable
      The reversal potential of learky channel. Default is -54.387 mV.
    gL: float, ArrayType, Initializer, callable
      The conductance of learky channel. Default is 0.03 msiemens.
    V_th: float, ArrayType, Initializer, callable
      The threshold of the membrane spike. Default is 20 mV.
    C: float, ArrayType, Initializer, callable
      The membrane capacitance. Default is 1 ufarad.
    phi: float, ArrayType, Initializer, callable
      The temperature regulator constant.
    V_initializer: ArrayType, Initializer, callable
      The initializer of membrane potential.
    h_initializer: ArrayType, Initializer, callable
      The initializer of h channel.
    n_initializer: ArrayType, Initializer, callable
      The initializer of n channel.
    method: str
      The numerical integration method.
    name: str
      The group name.

    """

    def dV(self, V, t, h, n, I):
        INa = self.gNa * self.m_inf(V) ** 3 * h * (V - self.ENa)
        IK = self.gK * n ** 4 * (V - self.EK)
        IL = self.gL * (V - self.EL)
        dVdt = (- INa - IK - IL + I) / self.C
        return dVdt

    def update(self, x=None):
        x = 0. if x is None else x
        x = self.sum_current_inputs(self.V.value, init=x)
        return super().update(x)
