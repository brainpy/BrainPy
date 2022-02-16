# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.building.brainobjects import TwoEndConn
from brainpy.building.brainobjects.delays import ConstantDelay
from brainpy.integrators.ode import odeint

__all__ = [
  'AMPA',
  'GABAa',
]


class AMPA(TwoEndConn):
  r"""AMPA conductance-based synapse model.

  **Model Descriptions**

  AMPA receptor is an ionotropic receptor, which is an ion channel.
  When it is bound by neurotransmitters, it will immediately open the
  ion channel, causing the change of membrane potential of postsynaptic neurons.

  A classical model is to use the Markov process to model ion channel switch.
  Here :math:`g` represents the probability of channel opening, :math:`1-g`
  represents the probability of ion channel closing, and :math:`\alpha` and
  :math:`\beta` are the transition probability. Because neurotransmitters can
  open ion channels, the transfer probability from :math:`1-g` to :math:`g`
  is affected by the concentration of neurotransmitters. We denote the concentration
  of neurotransmitters as :math:`[T]` and get the following Markov process.

  .. image:: ../../images/synapse_markov.png
      :align: center

  We obtained the following formula when describing the process by a differential equation.

  .. math::

      \frac{ds}{dt} =\alpha[T](1-g)-\beta g

  where :math:`\alpha [T]` denotes the transition probability from state :math:`(1-g)`
  to state :math:`(g)`; and :math:`\beta` represents the transition probability of
  the other direction. :math:`\alpha` is the binding constant. :math:`\beta` is the
  unbinding constant. :math:`[T]` is the neurotransmitter concentration, and
  has the duration of 0.5 ms.

  Moreover, the post-synaptic current on the post-synaptic neuron is formulated as

  .. math::

      I_{syn} = g_{max} g (V-E)

  where :math:`g_{max}` is the maximum conductance, and `E` is the reverse potential.

  **Model Examples**

  - `Simple illustrated example <../synapses/ampa.ipynb>`_


  **Model Parameters**

  ============= ============== ======== ================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  g_max         .42            µmho(µS) Maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  alpha         .98            \        Binding constant.
  beta          .18            \        Unbinding constant.
  T             .5             mM       Neurotransmitter concentration.
  T_duration    .5             ms       Duration of the neurotransmitter concentration.
  ============= ============== ======== ================================================


  **Model Variables**

  ================== ================== ==================================================
  **Member name**    **Initial values** **Explanation**
  ------------------ ------------------ --------------------------------------------------
  g                  0                  Synapse gating variable.
  pre_spike          False              The history of pre-synaptic neuron spikes.
  spike_arrival_time -1e7               The arrival time of the pre-synaptic neuron spike.
  ================== ================== ==================================================

  **References**

  .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
         and implications for stimulus processing[J]. Proceedings of the
         National Academy of Sciences, 2012, 109(45): 18553-18558.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0., alpha=0.98,
               beta=0.18, T=0.5, T_duration=0.5, method='exp_auto', **kwargs):
    super(AMPA, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.delay = delay
    self.g_max = g_max
    self.E = E
    self.alpha = alpha
    self.beta = beta
    self.T = T
    self.T_duration = T_duration

    # connection
    assert self.conn is not None
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # variables
    self.g = bm.Variable(bm.zeros(self.num))
    self.pre_spike = ConstantDelay(self.pre.num, delay, dtype=pre.spike.dtype)
    self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)

    # functions
    self.integral = odeint(method=method, f=self.derivative)

  def derivative(self, g, t, TT):
    dg = self.alpha * TT * (1 - g) - self.beta * g
    return dg

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    self.spike_arrival_time.value = bm.where(self.pre_spike.pull(), _t, self.spike_arrival_time)
    syn_sp_times = bm.pre2syn(self.spike_arrival_time, self.pre_ids)
    TT = ((_t - syn_sp_times) < self.T_duration) * self.T
    self.g.value = self.integral(self.g, _t, TT, dt=_dt)
    g_post = bm.syn2post(self.g, self.post_ids, self.post.num)
    self.post.input -= self.g_max * g_post * (self.post.V - self.E)


class GABAa(AMPA):
  r"""GABAa conductance-based synapse model.

  **Model Descriptions**

  GABAa synapse model has the same equation with the `AMPA synapse <./brainmodels.synapses.AMPA.rst>`_,

  .. math::

      \frac{d g}{d t}&=\alpha[T](1-g) - \beta g \\
      I_{syn}&= - g_{max} g (V - E)

  but with the difference of:

  - Reversal potential of synapse :math:`E` is usually low, typically -80. mV
  - Activating rate constant :math:`\alpha=0.53`
  - De-activating rate constant :math:`\beta=0.18`
  - Transmitter concentration :math:`[T]=1\,\mu ho(\mu S)` when synapse is
    triggered by a pre-synaptic spike, with the duration of 1. ms.

  **Model Examples**

  - `Gamma oscillation network model <../../examples/oscillation_synchronization/Wang_1996_gamma_oscillation.ipynb>`_

  **Model Parameters**

  ============= ============== ======== =======================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ---------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  g_max         0.04           µmho(µS) Maximum synapse conductance.
  E             -80            mV       Reversal potential of synapse.
  alpha         0.53           \        Activating rate constant of G protein catalyzed by activated GABAb receptor.
  beta          0.18           \        De-activating rate constant of G protein.
  T             1              mM       Transmitter concentration when synapse is triggered by a pre-synaptic spike.
  T_duration    1              ms       Transmitter concentration duration time after being triggered.
  ============= ============== ======== =======================================

  **Model Variables**

  ================== ================== ==================================================
  **Member name**    **Initial values** **Explanation**
  ------------------ ------------------ --------------------------------------------------
  g                  0                  Synapse gating variable.
  pre_spike          False              The history of pre-synaptic neuron spikes.
  spike_arrival_time -1e7               The arrival time of the pre-synaptic neuron spike.
  ================== ================== ==================================================

  **References**

  .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
         on the integrative properties of neocortical pyramidal neurons
         in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=0.04, E=-80., alpha=0.53,
               beta=0.18, T=1., T_duration=1., method='exp_auto', **kwargs):
    super(GABAa, self).__init__(pre, post, conn,
                                delay=delay,
                                g_max=g_max,
                                E=E,
                                alpha=alpha,
                                beta=beta,
                                T=T,
                                T_duration=T_duration,
                                method=method,
                                **kwargs)
