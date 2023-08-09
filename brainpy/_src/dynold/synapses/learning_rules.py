# -*- coding: utf-8 -*-

from typing import Union, Dict, Callable, Optional

from brainpy._src.connect import TwoEndConnector
from brainpy._src.dyn import synapses
from brainpy._src.dynold.synouts import CUBA
from brainpy._src.dynold.synapses import _TwoEndConnAlignPre
from brainpy._src.dynsys import Sequential
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.initialize import Initializer
from brainpy._src.mixin import ParamDesc
from brainpy.types import ArrayType

__all__ = [
  'STP'
]


class _STPModel(Sequential, ParamDesc):
  def __init__(self, size, keep_size, tau, U, tau_f, tau_d, mode=None, method='exp_euler'):
    stp = synapses.STP(size, keep_size, U=U, tau_f=tau_f, tau_d=tau_d, method=method, mode=mode)
    exp = synapses.Expon(size, keep_size, tau=tau, method=method, mode=mode)
    super().__init__(stp, exp)


class STP(_TwoEndConnAlignPre):
  r"""Short-term plasticity model.

  **Model Descriptions**

  Short-term plasticity (STP) [1]_ [2]_ [3]_, also called dynamical synapses,
  refers to the changes of synaptic strengths over time in a way that reflects
  the history of presynaptic activity. Two types of STP, with opposite effects
  on synaptic efficacy, have been observed in experiments. They are known as
  Short-Term Depression (STD) and Short-Term Facilitation (STF).

  In the model proposed by Tsodyks and Markram [4]_ [5]_, the STD effect is
  modeled by a normalized variable :math:`x (0 \le x \le 1)`, denoting the fraction
  of resources that remain available after neurotransmitter depletion.
  The STF effect is modeled by a utilization parameter :math:`u`, representing
  the fraction of available resources ready for use (release probability).
  Following a spike,

  - (i) :math:`u` increases due to spike-induced calcium influx to the presynaptic
    terminal, after which
  - (ii) a fraction :math:`u` of available resources is consumed to produce the
    post-synaptic current.

  Between spikes, :math:`u` decays back to zero with time constant :math:`\tau_f`
  and :math:`x` recovers to 1 with time constant :math:`\tau_d`.

  In summary, the dynamics of STP is given by

  .. math::

      \begin{aligned}
      \frac{du}{dt} & =  -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}),\nonumber \\
      \frac{dx}{dt} & =  \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
      \frac{dI}{dt} & =  -\frac{I}{\tau_s} + Au^+x^-\delta(t-t_{sp}),
      \end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike. :math:`u^-, x^-` are the corresponding
  variables just before the arrival of the spike, and :math:`u^+`
  refers to the moment just after the spike. The synaptic current generated
  at the synapse by the spike arriving at :math:`t_{sp}` is then given by

  .. math::

      \Delta I(t_{spike}) = Au^+x^-

  where :math:`A` denotes the response amplitude that would be produced
  by total release of all the neurotransmitter (:math:`u=x=1`), called
  absolute synaptic efficacy of the connections.

  **Model Examples**

  - `STP for Working Memory Capacity <https://brainpy-examples.readthedocs.io/en/latest/working_memory/Mi_2017_working_memory_capacity.html>`_

  **STD**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.synapses.STP(neu1, neu2, bp.connect.All2All(), U=0.2, tau_d=150., tau_f=2.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    >>> runner.run(150.)
    >>>
    >>>
    >>> # plot
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 7)
    >>>
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.I'][:, 0], label='I')
    >>> plt.legend()
    >>>
    >>> plt.xlabel('Time (ms)')
    >>> plt.show()

  **STF**

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> neu1 = bp.neurons.LIF(1)
    >>> neu2 = bp.neurons.LIF(1)
    >>> syn1 = bp.neurons.STP(neu1, neu2, bp.connect.All2All(), U=0.1, tau_d=10, tau_f=100.)
    >>> net = bp.Network(pre=neu1, syn=syn1, post=neu2)
    >>>
    >>> runner = bp.DSRunner(net, inputs=[('pre.input', 28.)], monitors=['syn.I', 'syn.u', 'syn.x'])
    >>> runner.run(150.)
    >>>
    >>>
    >>> # plot
    >>> fig, gs = bp.visualize.get_figure(2, 1, 3, 7)
    >>>
    >>> fig.add_subplot(gs[0, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.u'][:, 0], label='u')
    >>> plt.plot(runner.mon.ts, runner.mon['syn.x'][:, 0], label='x')
    >>> plt.legend()
    >>>
    >>> fig.add_subplot(gs[1, 0])
    >>> plt.plot(runner.mon.ts, runner.mon['syn.I'][:, 0], label='I')
    >>> plt.legend()
    >>>
    >>> plt.xlabel('Time (ms)')
    >>> plt.show()



  **Model Parameters**

  ============= ============== ======== ===========================================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- -------------------------------------------
  tau_d         200            ms       Time constant of short-term depression.
  tau_f         1500           ms       Time constant of short-term facilitation.
  U             .15            \        The increment of :math:`u` produced by a spike.
  A             1              \        The response amplitude that would be produced by total release of all the neurotransmitter
  delay         0              ms       The decay time of the current :math:`I` output onto the post-synaptic neuron groups.
  ============= ============== ======== ===========================================


  **Model Variables**

  =============== ================== =====================================================================
  **Member name** **Initial values** **Explanation**
  --------------- ------------------ ---------------------------------------------------------------------
  u                 0                 Release probability of the neurotransmitters.
  x                 1                 A Normalized variable denoting the fraction of remain neurotransmitters.
  I                 0                 Synapse current output onto the post-synaptic neurons.
  =============== ================== =====================================================================

  **References**

  .. [1] Stevens, Charles F., and Yanyan Wang. "Facilitation and depression
         at single central synapses." Neuron 14, no. 4 (1995): 795-802.
  .. [2] Abbott, Larry F., J. A. Varela, Kamal Sen, and S. B. Nelson. "Synaptic
         depression and cortical gain control." Science 275, no. 5297 (1997): 221-224.
  .. [3] Abbott, L. F., and Wade G. Regehr. "Synaptic computation."
         Nature 431, no. 7010 (2004): 796-803.
  .. [4] Tsodyks, Misha, Klaus Pawelzik, and Henry Markram. "Neural networks
         with dynamic synapses." Neural computation 10.4 (1998): 821-835.
  .. [5] Tsodyks, Misha, and Si Wu. "Short-term synaptic plasticity."
         Scholarpedia 8, no. 10 (2013): 3153.

  """

  def __init__(
      self,
      pre: NeuDyn,
      post: NeuDyn,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]],
      U: Union[float, ArrayType] = 0.15,
      tau_f: Union[float, ArrayType] = 1500.,
      tau_d: Union[float, ArrayType] = 200.,
      tau: Union[float, ArrayType] = 8.,
      A: Union[float, ArrayType] = 1.,
      delay_step: Union[int, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None
  ):
    # parameters
    self.tau_d = tau_d
    self.tau_f = tau_f
    self.tau = tau
    self.U = U
    self.A = A

    syn = _STPModel(pre.size,
                    pre.keep_size,
                    tau,
                    U,
                    tau_f,
                    tau_d,
                    method=method)

    super().__init__(pre=pre,
                     post=post,
                     syn=syn,
                     conn=conn,
                     g_max=A,
                     output=CUBA(),
                     comp_method='sparse',
                     delay_step=delay_step,
                     name=name)

    # variables
    self.x = self.syn[0].x
    self.u = self.syn[0].u
    self.I = self.syn[1].g
