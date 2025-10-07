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
pneu_doc = '''    
    size: int, or sequence of int. The neuronal population size.
    sharding: The sharding strategy. 
    keep_size: bool. Keep the neuron group size.
    mode: Mode. The computing mode.
    name: str. The group name.
'''.strip()

dpneu_doc = '''
    spk_fun: callable. The spike activation function.
    detach_spk: bool.
    method: str. The numerical integration method.
    spk_type: The spike data type.
    spk_reset: The way to reset the membrane potential when the neuron generates spikes.
        This parameter only works when the computing mode is ``TrainingMode``.
        It can be ``soft`` and ``hard``. Default is ``soft``.
'''.strip()

ref_doc = '''
    tau_ref: float, ArrayType, callable. Refractory period length (ms).
    has_ref_var: bool. Whether has the refractory variable. Default is ``False``.
'''.strip()

if_doc = '''
    V_rest: float, ArrayType, callable. Resting membrane potential.
    R: float, ArrayType, callable. Membrane resistance.
    tau: float, ArrayType, callable. Membrane time constant.
    V_initializer: ArrayType, callable. The initializer of membrane potential.
'''.strip()

lif_doc = '''
    V_rest: float, ArrayType, callable. Resting membrane potential.
    V_reset: float, ArrayType, callable. Reset potential after spike.
    V_th: float, ArrayType, callable. Threshold potential of spike.
    R: float, ArrayType, callable. Membrane resistance.
    tau: float, ArrayType, callable. Membrane time constant.
    V_initializer: ArrayType, callable. The initializer of membrane potential.
'''.strip()

ltc_doc = 'with liquid time-constant'

dual_exp_syn_doc = r'''

  **Model Descriptions**

  The dual exponential synapse model [1]_, also named as *difference of two exponentials* model,
  is given by:

  .. math::

    g_{\mathrm{syn}}(t)=g_{\mathrm{max}} A \left(\exp \left(-\frac{t-t_{0}}{\tau_{1}}\right)
        -\exp \left(-\frac{t-t_{0}}{\tau_{2}}\right)\right)

  where :math:`\tau_1` is the time constant of the decay phase, :math:`\tau_2`
  is the time constant of the rise phase, :math:`t_0` is the time of the pre-synaptic
  spike, :math:`g_{\mathrm{max}}` is the maximal conductance.

  However, in practice, this formula is hard to implement. The equivalent solution is
  two coupled linear differential equations [2]_:

  .. math::

      \begin{aligned}
      &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ (\frac{1}{\tau_{\text{rise}}} - \frac{1}{\tau_{\text{decay}}}) A \delta\left(t_{0}-t\right),
      \end{aligned}

  By default, :math:`A` has the following value:

  .. math::

     A = \frac{{\tau }_{decay}}{{\tau }_{decay}-{\tau }_{rise}}{\left(\frac{{\tau }_{rise}}{{\tau }_{decay}}\right)}^{\frac{{\tau }_{rise}}{{\tau }_{rise}-{\tau }_{decay}}}

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.
  .. [2] Roth, A., & Van Rossum, M. C. W. (2009). Modeling Synapses. Computational
         Modeling Methods for Neuroscientists.
  
'''

dual_exp_args = '''
    tau_decay: float, ArrayArray, Callable. The time constant of the synaptic decay phase. [ms]
    tau_rise: float, ArrayArray, Callable. The time constant of the synaptic rise phase. [ms]
    A: float. The normalization factor. Default None.

'''

alpha_syn_doc = r'''

  **Model Descriptions**

  The analytical expression of alpha synapse is given by:

  .. math::

      g_{syn}(t)= g_{max} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right).

  While, this equation is hard to implement. So, let's try to convert it into the
  differential forms:

  .. math::

      \begin{aligned}
      &\frac{d g}{d t}=-\frac{g}{\tau}+\frac{h}{\tau} \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}
  
  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
        "The Synapse." Principles of Computational Modelling in Neuroscience.
        Cambridge: Cambridge UP, 2011. 172-95. Print.


'''

exp_syn_doc = r'''

  **Model Descriptions**

  The single exponential decay synapse model assumes the release of neurotransmitter,
  its diffusion across the cleft, the receptor binding, and channel opening all happen
  very quickly, so that the channels instantaneously jump from the closed to the open state.
  Therefore, its expression is given by

  .. math::

      g_{\mathrm{syn}}(t)=g_{\mathrm{max}} e^{-\left(t-t_{0}\right) / \tau}

  where :math:`\tau_{delay}` is the time constant of the synaptic state decay,
  :math:`t_0` is the time of the pre-synaptic spike,
  :math:`g_{\mathrm{max}}` is the maximal conductance.

  Accordingly, the differential form of the exponential synapse is given by

  .. math::

      \begin{aligned}
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

'''

std_doc = r'''

  This model filters the synaptic current by the following equation:

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x

  where :math:`x` is the normalized variable between 0 and 1, and
  :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STD filtering.

  Moreover, :math:`x` is updated according to the dynamics of:

  .. math::

     \frac{dx}{dt} = \frac{1-x}{\tau} - U * x * \delta(t-t_{spike})

  where :math:`U` is the fraction of resources used per action potential,
  :math:`\tau` is the time constant of recovery of the synaptic vesicles.

'''

stp_doc = r'''

  This model filters the synaptic currents according to two variables: :math:`u` and :math:`x`.

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x * u

  where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STP filtering, :math:`x` denotes the fraction of resources that remain available
  after neurotransmitter depletion, and :math:`u` represents the fraction of available
  resources ready for use (release probability).

  The dynamics of :math:`u` and :math:`x` are governed by

  .. math::

     \begin{aligned}
     \frac{du}{dt} & = & -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}), \\
     \frac{dx}{dt} & = & \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
     \end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike. :math:`u^-, x^-` are the corresponding
  variables just before the arrival of the spike, and :math:`u^+`
  refers to the moment just after the spike.


'''
