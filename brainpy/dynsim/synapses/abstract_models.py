# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.dynsim.base import TwoEndConn, ConstantDelay

__all__ = [
  'DeltaSynapse',
  'ExpCUBA',
  'ExpCOBA',
  'DualExpCUBA',
  'DualExpCOBA',
  'AlphaCUBA',
  'AlphaCOBA',
]


class DeltaSynapse(TwoEndConn):
  """Voltage Jump Synapse Model, or alias of Delta Synapse Model.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} w \delta(t-t_j-D)

  where :math:`w` denotes the chemical synaptic strength, :math:`t_j` the spiking
  moment of the presynaptic neuron :math:`j`, :math:`C` the set of neurons connected
  to the post-synaptic neuron, and :math:`D` the transmission delay of chemical
  synapses. For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.

  **Model Examples**

  - `Simple illustrated example <../synapses/voltage_jump.ipynb>`_
  - `(Bouchacourt & Buschman, 2019) Flexible Working Memory Model <../../examples/working_memory/Bouchacourt_2019_Flexible_working_memory.ipynb>`_

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.synapses.DeltaSynapse(neu1, neu2, bp.connect.All2All(), w=5.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.), ('post.input', 10.)], monitors=['pre.V', 'post.V', 'pre.spike'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(1, 1, 3, 8)
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.xlim(40, 150)
    >>> plt.legend()
    >>> plt.show()

  **Model Parameters**

  ============= ============== ======== ===========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------
  w             1              mV       The synaptic strength.
  ============= ============== ======== ===========================================

  """

  def __init__(self, pre, post, conn, delay=0., post_has_ref=False, w=1.,
               post_key='V', name=None):
    super(DeltaSynapse, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs(post_key)

    # parameters
    self.delay = delay
    self.post_key = post_key
    self.post_has_ref = post_has_ref
    if post_has_ref:  # checking
      self.check_post_attrs('refractory')

    # connections
    assert self.conn is not None
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.w = w
    # assert bm.size(w) == 1 or bm.size(w) ==
    self.pre_spike = ConstantDelay(self.pre.num, delay, dtype=pre.spike.dtype)

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    post_vs = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.w)
    target = getattr(self.post, self.post_key)
    if self.post_has_ref:
      target += post_vs * (1. - self.post.refractory)
    else:
      target += post_vs


class ExpCUBA(TwoEndConn):
  r"""Current-based exponential decay synapse model.

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
       & g_{\mathrm{syn}}(t) = g_{max} g \\
       & \frac{d g}{d t} = -\frac{g}{\tau_{decay}}+\sum_{k} \delta(t-t_{j}^{k}).
       \end{aligned}

  For the current output onto the post-synaptic neuron, its expression is given by

  .. math::

      I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)


  **Model Examples**

  - `(Brunel & Hakim, 1999) Fast Global Oscillation <https://brainpy-examples.readthedocs.io/en/latest/oscillation_synchronization/Brunel_Hakim_1999_fast_oscillation.html>`_
  - `(Vreeswijk & Sompolinsky, 1996) E/I balanced network <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Vreeswijk_1996_EI_net.html>`_
  - `(Brette, et, al., 2007) CUBA <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Brette_2007_CUBA.html>`_
  - `(Tian, et al., 2020) E/I Net for fast response <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Tian_2020_EI_net_for_fast_response.html>`_

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.synapses.ExpCUBA(neu1, neu2, bp.conn.All2All(), g_max=5.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.legend()
    >>> plt.show()


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     8              ms       The time constant of decay.
  g_max         1              µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                 0                 Gating variable.
  pre_spike         False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0,
               method='exp_auto', name=None):
    super(ExpCUBA, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.tau = tau
    self.delay = delay
    self.g_max = g_max

    # connection
    assert self.conn is not None
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))
    self.pre_spike = ConstantDelay(self.pre.num, delay=delay, dtype=pre.spike.dtype)

    # function
    self.integral = odeint(lambda g, t: -g / self.tau, method=method)

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spike = self.pre_spike.pull()
    post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = self.integral(self.g.value, _t, dt=_dt) + post_sp
    self.post.input += self.g


class ExpCOBA(ExpCUBA):
  """Conductance-based exponential decay synapse model.

  **Model Descriptions**

  The conductance-based exponential decay synapse model is similar with the
  `current-based exponential decay synapse model <./brainmodels.synapses.ExpCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.

  **Model Examples**

  - `(Brette, et, al., 2007) COBA <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Brette_2007_COBA.html>`_
  - `(Brette, et, al., 2007) COBAHH <https://brainpy-examples.readthedocs.io/en/latest/ei_nets/Brette_2007_COBAHH.html>`_


  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.HH(1)
    >>> neu2 = bp.neurons.HH(1)
    >>> syn1 = bp.synapses.ExpCOBA(neu1, neu2, bp.connect.All2All(), E=0.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.legend()
    >>> plt.show()

  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     8              ms       The time constant of decay.
  g_max         1              µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                 0                 Gating variable.
  pre_spike         False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto', name=None):
    super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                  g_max=g_max, delay=delay, tau=tau,
                                  method=method, name=name)

    # parameter
    self.E = E

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_spike = self.pre_spike.pull()
    post_sp = bm.pre2post_event_sum(delayed_spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = self.integral(self.g.value, _t, dt=_dt) + post_sp
    self.post.input += self.g * (self.E - self.post.V)


class DualExpCUBA(TwoEndConn):
  r"""Current-based dual exponential synapse model.

  **Model Descriptions**

  The dual exponential synapse model [1]_, also named as *difference of two exponentials* model,
  is given by:

  .. math::

    g_{\mathrm{syn}}(t)=g_{\mathrm{max}} \frac{\tau_{1} \tau_{2}}{
        \tau_{1}-\tau_{2}}\left(\exp \left(-\frac{t-t_{0}}{\tau_{1}}\right)
        -\exp \left(-\frac{t-t_{0}}{\tau_{2}}\right)\right)

  where :math:`\tau_1` is the time constant of the decay phase, :math:`\tau_2`
  is the time constant of the rise phase, :math:`t_0` is the time of the pre-synaptic
  spike, :math:`g_{\mathrm{max}}` is the maximal conductance.

  However, in practice, this formula is hard to implement. The equivalent solution is
  two coupled linear differential equations [2]_:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)=g_{\mathrm{max}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau_{\mathrm{decay}}}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau_{\text {rise }}}+ \delta\left(t_{0}-t\right),
      \end{aligned}

  The current onto the post-synaptic neuron is given by

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t).


  **Model Examples**


  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.synapses.DualExpCUBA(neu1, neu2, bp.connect.All2All())
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
    >>> plt.legend()
    >>> plt.show()


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     10             ms       The time constant of the synaptic decay phase.
  tau_rise      1              ms       The time constant of the synaptic rise phase.
  g_max         1              µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                  0                 Synapse conductance on the post-synaptic neuron.
  s                  0                 Gating variable.
  pre_spike          False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.
  .. [2] Roth, A., & Van Rossum, M. C. W. (2009). Modeling Synapses. Computational
         Modeling Methods for Neuroscientists.

  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0, tau_rise=1.,
               method='exp_auto', name=None):
    super(DualExpCUBA, self).__init__(pre=pre, post=post, conn=conn, name=name)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input')

    # parameters
    self.tau_rise = tau_rise
    self.tau_decay = tau_decay
    self.delay = delay
    self.g_max = g_max

    # connections
    self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')

    # variables
    self.g = bm.Variable(bm.zeros(len(self.pre_ids)))
    self.h = bm.Variable(bm.zeros(len(self.pre_ids)))
    self.pre_spike = ConstantDelay(self.pre.num, delay, dtype=pre.spike.dtype)

    # integral
    self.integral = odeint(method=method, f=self.derivative)

  @property
  def derivative(self):
    dg = lambda g, t, h: -g / self.tau_decay + h
    dh = lambda h, t: -h / self.tau_rise
    return JointEq([dg, dh])

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spikes = self.pre_spike.pull()
    self.g.value, self.h.value = self.integral(self.g, self.h, _t, dt=_dt)
    self.h.value += bm.pre2syn(delayed_pre_spikes, self.pre_ids)
    self.post.input += self.g_max * bm.syn2post(self.g, self.post_ids, self.post.num)


class DualExpCOBA(DualExpCUBA):
  """Conductance-based dual exponential synapse model.

  **Model Descriptions**

  The conductance-based dual exponential synapse model is similar with the
  `current-based dual exponential synapse model <./brainmodels.synapses.DualExpCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.

  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.HH(1)
    >>> neu2 = bp.neurons.HH(1)
    >>> syn1 = bp.synapses.DualExpCOBA(neu1, neu2, bp.connect.All2All(), E=0.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
    >>> plt.legend()
    >>> plt.show()


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     10             ms       The time constant of the synaptic decay phase.
  tau_rise      1              ms       The time constant of the synaptic rise phase.
  g_max         1              µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================ ================== =========================================================
  **Member name**  **Initial values** **Explanation**
  ---------------- ------------------ ---------------------------------------------------------
  g                  0                 Synapse conductance on the post-synaptic neuron.
  s                  0                 Gating variable.
  pre_spike          False             The history spiking states of the pre-synaptic neurons.
  ================ ================== =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
         "The Synapse." Principles of Computational Modelling in Neuroscience.
         Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0, tau_rise=1.,
               E=0., method='exp_auto', name=None):
    super(DualExpCOBA, self).__init__(pre, post, conn, delay=delay, g_max=g_max,
                                      tau_decay=tau_decay, tau_rise=tau_rise,
                                      method=method, name=name)
    self.check_post_attrs('V')

    # parameters
    self.E = E

  def update(self, _t, _dt):
    self.pre_spike.push(self.pre.spike)
    delayed_pre_spikes = self.pre_spike.pull()
    self.g.value, self.h.value = self.integral(self.g, self.h, _t, dt=_dt)
    self.h.value += bm.pre2syn(delayed_pre_spikes, self.pre_ids)
    post_g = bm.syn2post(self.g, self.post_ids, self.post.num)
    self.post.input += self.g_max * post_g * (self.E - self.post.V)


class AlphaCUBA(DualExpCUBA):
  r"""Current-based alpha synapse model.

  **Model Descriptions**

  The analytical expression of alpha synapse is given by:

  .. math::

      g_{syn}(t)= g_{max} \frac{t-t_{s}}{\tau} \exp \left(-\frac{t-t_{s}}{\tau}\right).

  While, this equation is hard to implement. So, let's try to convert it into the
  differential forms:

  .. math::

      \begin{aligned}
      &g_{\mathrm{syn}}(t)= g_{\mathrm{max}} g \\
      &\frac{d g}{d t}=-\frac{g}{\tau}+h \\
      &\frac{d h}{d t}=-\frac{h}{\tau}+\delta\left(t_{0}-t\right)
      \end{aligned}

  The current onto the post-synaptic neuron is given by

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t).


  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.synapses.AlphaCUBA(neu1, neu2, bp.connect.All2All())
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 25.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
    >>> plt.legend()
    >>> plt.show()

  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     2              ms       The decay time constant of the synaptic state.
  g_max         .2             µmho(µS) The maximum conductance.
  ============= ============== ======== ===================================================================================

  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  g                   0                Synapse conductance on the post-synaptic neuron.
  h                   0                Gating variable.
  pre_spike           False            The history spiking states of the pre-synaptic neurons.
  ================== ================= =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.
  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               method='exp_auto', name=None):
    super(AlphaCUBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay,
                                    g_max=g_max,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    method=method,
                                    name=name)


class AlphaCOBA(DualExpCOBA):
  """Conductance-based alpha synapse model.

  **Model Descriptions**

  The conductance-based alpha synapse model is similar with the
  `current-based alpha synapse model <./brainmodels.synapses.AlphaCUBA.rst>`_,
  except the expression which output onto the post-synaptic neurons:

  .. math::

      I_{syn}(t) = g_{\mathrm{syn}}(t) (V(t)-E)

  where :math:`V(t)` is the membrane potential of the post-synaptic neuron,
  :math:`E` is the reversal potential.


  **Model Examples**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.HH(1)
    >>> neu2 = bp.neurons.HH(1)
    >>> syn1 = bp.synapses.AlphaCOBA(neu1, neu2, bp.connect.All2All(), E=0.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 5.)], monitors=['pre.V', 'post.V', 'syn.g', 'syn.h'])
    >>> runner.run(150.)
    >>>
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['pre.V'], label='pre-V')
    >>> plt.plot(runner.mon.ts, runner.mon['post.V'], label='post-V')
    >>> plt.legend()
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.h'], label='h')
    >>> plt.legend()
    >>> plt.show()


  **Model Parameters**

  ============= ============== ======== ===================================================================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -----------------------------------------------------------------------------------
  delay         0              ms       The decay length of the pre-synaptic spikes.
  tau_decay     2              ms       The decay time constant of the synaptic state.
  g_max         .2             µmho(µS) The maximum conductance.
  E             0              mV       The reversal potential for the synaptic current.
  ============= ============== ======== ===================================================================================


  **Model Variables**

  ================== ================= =========================================================
  **Variables name** **Initial Value** **Explanation**
  ------------------ ----------------- ---------------------------------------------------------
  g                   0                Synapse conductance on the post-synaptic neuron.
  h                   0                Gating variable.
  pre_spike           False            The history spiking states of the pre-synaptic neurons.
  ================== ================= =========================================================

  **References**

  .. [1] Sterratt, David, Bruce Graham, Andrew Gillies, and David Willshaw.
          "The Synapse." Principles of Computational Modelling in Neuroscience.
          Cambridge: Cambridge UP, 2011. 172-95. Print.

  """

  def __init__(self, pre, post, conn, delay=0., g_max=1., tau_decay=10.0,
               E=0., method='exp_auto', name=None):
    super(AlphaCOBA, self).__init__(pre=pre, post=post, conn=conn,
                                    delay=delay,
                                    g_max=g_max,
                                    E=E,
                                    tau_decay=tau_decay,
                                    tau_rise=tau_decay,
                                    method=method,
                                    name=name)
