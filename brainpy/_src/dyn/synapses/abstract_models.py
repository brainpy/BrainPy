from typing import Union, Sequence, Callable, Optional

from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.initialize import parameter
from brainpy._src.dyn import _docs
from brainpy._src.dyn.base import SynDyn
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy._src.mixin import AlignPost, ReturnInfo
from brainpy.types import ArrayType

__all__ = [
  'Expon',
  'DualExpon',
  'DualExponV2',
  'Alpha',
  'NMDA',
  'STD',
  'STP',
]


class Expon(SynDyn, AlignPost):
  r"""Exponential decay synapse model.

  %s

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. code-block:: python

        import numpy as np
        import brainpy as bp
        import brainpy.math as bm

        import matplotlib.pyplot as plt


        class ExponSparseCOBA(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPreMg2(
                    pre=pre,
                    delay=delay,
                    syn=bp.dyn.Expon.desc(pre.num, tau=tau),
                    comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    out=bp.dyn.COBA(E=E),
                    post=post,
                )


        class SimpleNet(bp.DynSysGroup):
            def __init__(self, syn_cls, E=0.):
                super().__init__()
                self.pre = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
                self.post = bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                          V_initializer=bp.init.Constant(-60.))
                self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1., tau=5., E=E)

            def update(self):
                self.pre()
                self.syn()
                self.post()
                # monitor the following variables
                conductance = self.syn.proj.refs['syn'].g
                current = self.post.sum_inputs(self.post.V)
                return conductance, current, self.post.V

  Moreover, it can also be used with interface ``ProjAlignPostMg2``:

  .. code-block:: python

        class ExponSparseCOBAPost(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPostMg2(
                    pre=pre,
                    delay=delay,
                    comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    syn=bp.dyn.Expon.desc(post.num, tau=tau),
                    out=bp.dyn.COBA.desc(E=E),
                    post=post,
                )


  Args:
    tau: float. The time constant of decay. [ms]
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau: Union[float, ArrayType, Callable] = 8.0,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau = self.init_param(tau)

    # function
    self.integral = odeint(self.derivative, method=method)
    self._current = None

    self.reset_state(self.mode)

  def derivative(self, g, t):
    return -g / self.tau

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.g = self.init_variable(bm.zeros, batch_or_mode)

  def update(self, x=None):
    self.g.value = self.integral(self.g.value, share['t'], share['dt'])
    if x is not None:
      self.add_current(x)
    return self.g.value

  def add_current(self, x):
    self.g.value += x

  def return_info(self):
    return self.g


Expon.__doc__ = Expon.__doc__ % (_docs.exp_syn_doc, _docs.pneu_doc,)


def _format_dual_exp_A(self, A):
  A = parameter(A, sizes=self.varshape, allow_none=True, sharding=self.sharding)
  if A is None:
    A = (self.tau_decay / (self.tau_decay - self.tau_rise) *
         bm.float_power(self.tau_rise / self.tau_decay, self.tau_rise / (self.tau_rise - self.tau_decay)))
  return A


class DualExpon(SynDyn):
  r"""Dual exponential synapse model.

  %s

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. code-block:: python

        import numpy as np
        import brainpy as bp
        import brainpy.math as bm

        import matplotlib.pyplot as plt

        class DualExpSparseCOBA(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau_decay, tau_rise, E):
                super().__init__()
                self.proj = bp.dyn.ProjAlignPreMg2(
                    pre=pre,
                    delay=delay,
                    syn=bp.dyn.DualExpon.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise),
                    comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    out=bp.dyn.COBA(E=E),
                    post=post,
                )

        class SimpleNet(bp.DynSysGroup):
            def __init__(self, syn_cls, E=0.):
                super().__init__()
                self.pre = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
                self.post = bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                          V_initializer=bp.init.Constant(-60.))
                self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1.,
                                   tau_decay=5., tau_rise=1., E=E)

            def update(self):
                self.pre()
                self.syn()
                self.post()
                # monitor the following variables
                conductance = self.syn.proj.refs['syn'].g
                current = self.post.sum_inputs(self.post.V)
                return conductance, current, self.post.V


        indices = np.arange(1000)  # 100 ms, dt= 0.1 ms
        net = SimpleNet(DualExpSparseCOBA, E=0.)
        conductances, currents, potentials = bm.for_loop(net.step_run, indices, progress_bar=True)
        ts = indices * bm.get_dt()
        fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
        fig.add_subplot(gs[0, 0])
        plt.plot(ts, conductances)
        plt.title('Syn conductance')
        fig.add_subplot(gs[0, 1])
        plt.plot(ts, currents)
        plt.title('Syn current')
        fig.add_subplot(gs[0, 2])
        plt.plot(ts, potentials)
        plt.title('Post V')
        plt.show()

  See Also:
    DualExponV2

  .. note::

     The implementation of this model can only be used in ``AlignPre`` projections.
     One the contrary, to seek the ``AlignPost`` projection, please use ``DualExponV2``.

  Args:
    %s
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
      tau_rise: Union[float, ArrayType, Callable] = 1.,
      A: Optional[Union[float, ArrayType, Callable]] = None,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_rise = self.init_param(tau_rise)
    self.tau_decay = self.init_param(tau_decay)
    A = _format_dual_exp_A(self, A)
    self.a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A

    # integrator
    self.integral = odeint(JointEq(self.dg, self.dh), method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.h = self.init_variable(bm.zeros, batch_or_mode)
    self.g = self.init_variable(bm.zeros, batch_or_mode)

  def dh(self, h, t):
    return -h / self.tau_rise

  def dg(self, g, t, h):
    return -g / self.tau_decay + h

  def update(self, x):
    # x: the pre-synaptic spikes

    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, share['t'], dt=share['dt'])
    self.h += self.a * x
    return self.g.value

  def return_info(self):
    return self.g


DualExpon.__doc__ = DualExpon.__doc__ % (_docs.dual_exp_syn_doc, _docs.pneu_doc, _docs.dual_exp_args)


class DualExponV2(SynDyn, AlignPost):
  r"""Dual exponential synapse model.

  %s

  .. note::

     Different from ``DualExpon``, this model can be used in both modes of ``AlignPre`` and ``AlignPost`` projections.

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. code-block:: python

        import numpy as np
        import brainpy as bp
        import brainpy.math as bm

        import matplotlib.pyplot as plt


        class DualExponV2SparseCOBA(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau_decay, tau_rise, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPreMg2(
                    pre=pre,
                    delay=delay,
                    syn=bp.dyn.DualExponV2.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise),
                    comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    out=bp.dyn.COBA(E=E),
                    post=post,
                )


        class SimpleNet(bp.DynSysGroup):
            def __init__(self, syn_cls, E=0.):
                super().__init__()
                self.pre = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
                self.post = bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                          V_initializer=bp.init.Constant(-60.))
                self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1., tau_decay=5., tau_rise=1., E=E)

            def update(self):
                self.pre()
                self.syn()
                self.post()
                # monitor the following variables
                conductance = self.syn.proj.refs['syn'].g_rise
                current = self.post.sum_inputs(self.post.V)
                return conductance, current, self.post.V

        indices = np.arange(1000)  # 100 ms, dt= 0.1 ms
        net = SimpleNet(DualExponV2SparseCOBAPost, E=0.)
        conductances, currents, potentials = bm.for_loop(net.step_run, indices, progress_bar=True)
        ts = indices * bm.get_dt()
        fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
        fig.add_subplot(gs[0, 0])
        plt.plot(ts, conductances)
        plt.title('Syn conductance')
        fig.add_subplot(gs[0, 1])
        plt.plot(ts, currents)
        plt.title('Syn current')
        fig.add_subplot(gs[0, 2])
        plt.plot(ts, potentials)
        plt.title('Post V')
        plt.show()

  Moreover, it can also be used with interface ``ProjAlignPostMg2``:

  .. code-block:: python

        class DualExponV2SparseCOBAPost(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau_decay, tau_rise, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPostMg2(
                    pre=pre,
                    delay=delay,
                    comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    syn=bp.dyn.DualExponV2.desc(post.num, tau_decay=tau_decay, tau_rise=tau_rise),
                    out=bp.dyn.COBA.desc(E=E),
                    post=post,
                )

  See Also:
    DualExpon

  Args:
    %s
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
      tau_rise: Union[float, ArrayType, Callable] = 1.,
      A: Optional[Union[float, ArrayType, Callable]] = None,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_rise = self.init_param(tau_rise)
    self.tau_decay = self.init_param(tau_decay)
    self.a = _format_dual_exp_A(self, A)

    # integrator
    self.integral = odeint(lambda g, t, tau: -g / tau, method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.g_rise = self.init_variable(bm.zeros, batch_or_mode)
    self.g_decay = self.init_variable(bm.zeros, batch_or_mode)

  def update(self, x=None):
    self.g_rise.value = self.integral(self.g_rise.value, share['t'], self.tau_rise, share['dt'])
    self.g_decay.value = self.integral(self.g_decay.value, share['t'], self.tau_decay, share['dt'])
    if x is not None:
      self.add_current(x)
    return self.a * (self.g_decay - self.g_rise)

  def add_current(self, inp):
    self.g_rise += inp
    self.g_decay += inp

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode,
                      lambda shape: self.a * (self.g_decay - self.g_rise))


DualExponV2.__doc__ = DualExponV2.__doc__ % (_docs.dual_exp_syn_doc, _docs.pneu_doc, _docs.dual_exp_args,)


class Alpha(SynDyn):
  r"""Alpha synapse model.

  %s

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. code-block:: python

        import numpy as np
        import brainpy as bp
        import brainpy.math as bm

        import matplotlib.pyplot as plt


        class AlphaSparseCOBA(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau_decay, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPreMg2(
                    pre=pre,
                    delay=delay,
                    syn=bp.dyn.Alpha.desc(pre.num, tau_decay=tau_decay),
                    comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    out=bp.dyn.COBA(E=E),
                    post=post,
                )


        class SimpleNet(bp.DynSysGroup):
            def __init__(self, syn_cls, E=0.):
                super().__init__()
                self.pre = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
                self.post = bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                          V_initializer=bp.init.Constant(-60.))
                self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1.,
                                   tau_decay=5., E=E)

            def update(self):
                self.pre()
                self.syn()
                self.post()
                # monitor the following variables
                conductance = self.syn.proj.refs['syn'].g
                current = self.post.sum_inputs(self.post.V)
                return conductance, current, self.post.V


        indices = np.arange(1000)  # 100 ms, dt= 0.1 ms
        net = SimpleNet(AlphaSparseCOBA, E=0.)
        conductances, currents, potentials = bm.for_loop(net.step_run, indices, progress_bar=True)
        ts = indices * bm.get_dt()
        fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
        fig.add_subplot(gs[0, 0])
        plt.plot(ts, conductances)
        plt.title('Syn conductance')
        fig.add_subplot(gs[0, 1])
        plt.plot(ts, currents)
        plt.title('Syn current')
        fig.add_subplot(gs[0, 2])
        plt.plot(ts, potentials)
        plt.title('Post V')
        plt.show()


  Args:
    %s
    tau_decay: float, ArrayType, Callable. The time constant [ms] of the synaptic decay phase.
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau_decay: Union[float, ArrayType, Callable] = 10.0,
  ):
    super().__init__(
      name=name,
      mode=mode,
      size=size,
      keep_size=keep_size,
      sharding=sharding
    )

    # parameters
    self.tau_decay = self.init_param(tau_decay)

    # integrator
    self.integral = odeint(JointEq(self.dg, self.dh), method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.h = self.init_variable(bm.zeros, batch_or_mode)
    self.g = self.init_variable(bm.zeros, batch_or_mode)

  def dh(self, h, t):
    return -h / self.tau_decay

  def dg(self, g, t, h):
    return -g / self.tau_decay + h / self.tau_decay

  def update(self, x):
    # update synaptic variables
    self.g.value, self.h.value = self.integral(self.g.value, self.h.value, share['t'], dt=share['dt'])
    self.h += x
    return self.g.value

  def return_info(self):
    return self.g


Alpha.__doc__ = Alpha.__doc__ % (_docs.alpha_syn_doc, _docs.pneu_doc,)


class NMDA(SynDyn):
  r"""NMDA synapse model.

  **Model Descriptions**

  The NMDA receptor is a glutamate receptor and ion channel found in neurons.
  The NMDA receptor is one of three types of ionotropic glutamate receptors,
  the other two being AMPA and kainate receptors.

  The NMDA receptor mediated conductance depends on the postsynaptic voltage.
  The voltage dependence is due to the blocking of the pore of the NMDA receptor
  from the outside by a positively charged magnesium ion. The channel is
  nearly completely blocked at resting potential, but the magnesium block is
  relieved if the cell is depolarized. The fraction of channels :math:`g_{\infty}`
  that are not blocked by magnesium can be fitted to

  .. math::

      g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V}
      \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

  Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration,
  usually 1 mM. Thus, the channel acts as a
  "coincidence detector" and only once both of these conditions are met, the
  channel opens and it allows positively charged ions (cations) to flow through
  the cell membrane [2]_.

  If we make the approximation that the magnesium block changes
  instantaneously with voltage and is independent of the gating of the channel,
  the net NMDA receptor-mediated synaptic current is given by

  .. math::

      I_{syn} = g_\mathrm{NMDA}(t) (V(t)-E) \cdot g_{\infty}

  where :math:`V(t)` is the post-synaptic neuron potential, :math:`E` is the
  reversal potential.

  Simultaneously, the kinetics of synaptic state :math:`g` is given by

  .. math::

      & g_\mathrm{NMDA} (t) = g_{max} g \\
      & \frac{d g}{dt} = -\frac{g} {\tau_{decay}}+a x(1-g) \\
      & \frac{d x}{dt} = -\frac{x}{\tau_{rise}}+ \sum_{k} \delta(t-t_{j}^{k})

  where the decay time of NMDA currents is usually taken to be
  :math:`\tau_{decay}` =100 ms, :math:`a= 0.5 ms^{-1}`, and :math:`\tau_{rise}` =2 ms.

  The NMDA receptor has been thought to be very important for controlling
  synaptic plasticity and mediating learning and memory functions [3]_.

  This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

  .. code-block:: python

        import numpy as np
        import brainpy as bp
        import brainpy.math as bm

        import matplotlib.pyplot as plt

        class NMDASparseCOBA(bp.Projection):
            def __init__(self, pre, post, delay, prob, g_max, tau_decay, tau_rise, E):
                super().__init__()

                self.proj = bp.dyn.ProjAlignPreMg2(
                    pre=pre,
                    delay=delay,
                    syn=bp.dyn.NMDA.desc(pre.num, tau_decay=tau_decay, tau_rise=tau_rise),
                    comm=bp.dnn.CSRLinear(bp.conn.FixedProb(prob, pre=pre.num, post=post.num), g_max),
                    out=bp.dyn.COBA(E=E),
                    post=post,
                )


        class SimpleNet(bp.DynSysGroup):
            def __init__(self, syn_cls, E=0.):
                super().__init__()
                self.pre = bp.dyn.SpikeTimeGroup(1, indices=(0, 0, 0, 0), times=(10., 30., 50., 70.))
                self.post = bp.dyn.LifRef(1, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                          V_initializer=bp.init.Constant(-60.))
                self.syn = syn_cls(self.pre, self.post, delay=None, prob=1., g_max=1.,
                                   tau_decay=5., tau_rise=1., E=E)

            def update(self):
                self.pre()
                self.syn()
                self.post()
                # monitor the following variables
                conductance = self.syn.proj.refs['syn'].g
                current = self.post.sum_inputs(self.post.V)
                return conductance, current, self.post.V


        indices = np.arange(1000)  # 100 ms, dt= 0.1 ms
        net = SimpleNet(NMDASparseCOBA, E=0.)
        conductances, currents, potentials = bm.for_loop(net.step_run, indices, progress_bar=True)
        ts = indices * bm.get_dt()
        fig, gs = bp.visualize.get_figure(1, 3, 3.5, 4)
        fig.add_subplot(gs[0, 0])
        plt.plot(ts, conductances)
        plt.title('Syn conductance')
        fig.add_subplot(gs[0, 1])
        plt.plot(ts, currents)
        plt.title('Syn current')
        fig.add_subplot(gs[0, 2])
        plt.plot(ts, potentials)
        plt.title('Post V')
        plt.show()



  .. [1] Brunel N, Wang X J. Effects of neuromodulation in a
         cortical network model of object working memory dominated
         by recurrent inhibition[J].
         Journal of computational neuroscience, 2001, 11(1): 63-85.
  .. [2] Furukawa, Hiroyasu, Satinder K. Singh, Romina Mancusso, and
         Eric Gouaux. "Subunit arrangement and function in NMDA receptors."
         Nature 438, no. 7065 (2005): 185-192.
  .. [3] Li, F. and Tsien, J.Z., 2009. Memory and the NMDA receptors. The New
         England journal of medicine, 361(3), p.302.
  .. [4] https://en.wikipedia.org/wiki/NMDA_receptor

  Args:
    tau_decay: float, ArrayType, Callable. The time constant of the synaptic decay phase. Default 100 [ms]
    tau_rise: float, ArrayType, Callable. The time constant of the synaptic rise phase. Default 2 [ms]
    a: float, ArrayType, Callable. Default 0.5 ms^-1.
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      a: Union[float, ArrayType, Callable] = 0.5,
      tau_decay: Union[float, ArrayType, Callable] = 100.,
      tau_rise: Union[float, ArrayType, Callable] = 2.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_decay = self.init_param(tau_decay)
    self.tau_rise = self.init_param(tau_rise)
    self.a = self.init_param(a)

    # integral
    self.integral = odeint(method=method, f=JointEq(self.dg, self.dx))

    self.reset_state(self.mode)

  def dg(self, g, t, x):
    return -g / self.tau_decay + self.a * x * (1 - g)

  def dx(self, x, t):
    return -x / self.tau_rise

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.g = self.init_variable(bm.zeros, batch_or_mode)
    self.x = self.init_variable(bm.zeros, batch_or_mode)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    self.g.value, self.x.value = self.integral(self.g.value, self.x.value, t, dt=dt)
    self.x += pre_spike
    return self.g.value

  def return_info(self):
    return self.g


NMDA.__doc__ = NMDA.__doc__ % (_docs.pneu_doc,)


class STD(SynDyn):
  r"""Synaptic output with short-term depression.

  %s

  Args:
    tau: float, ArrayType, Callable. The time constant of recovery of the synaptic vesicles.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      tau: Union[float, ArrayType, Callable] = 200.,
      U: Union[float, ArrayType, Callable] = 0.07,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau = self.init_param(tau)
    self.U = self.init_param(U)

    # integral function
    self.integral = odeint(lambda x, t: (1 - x) / self.tau, method=method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.x = self.init_variable(bm.ones, batch_or_mode)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    x = self.integral(self.x.value, t, dt)

    # --- original code:
    # self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

    # --- simplified code:
    self.x.value = x - pre_spike * self.U * self.x

    return self.x.value

  def return_info(self):
    return self.x


STD.__doc__ = STD.__doc__ % (_docs.std_doc, _docs.pneu_doc,)


class STP(SynDyn):
  r"""Synaptic output with short-term plasticity.

  %s

  Args:
    tau_f: float, ArrayType, Callable. The time constant of short-term facilitation.
    tau_d: float, ArrayType, Callable. The time constant of short-term depression.
    U: float, ArrayType, Callable. The fraction of resources used per action potential.
    %s
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,

      # synapse parameters
      U: Union[float, ArrayType, Callable] = 0.15,
      tau_f: Union[float, ArrayType, Callable] = 1500.,
      tau_d: Union[float, ArrayType, Callable] = 200.,
  ):
    super().__init__(name=name,
                     mode=mode,
                     size=size,
                     keep_size=keep_size,
                     sharding=sharding)

    # parameters
    self.tau_f = self.init_param(tau_f)
    self.tau_d = self.init_param(tau_d)
    self.U = self.init_param(U)
    self.method = method

    # integral function
    self.integral = odeint(self.derivative, method=self.method)

    self.reset_state(self.mode)

  def reset_state(self, batch_or_mode=None, **kwargs):
    self.x = self.init_variable(bm.ones, batch_or_mode)
    self.u = self.init_variable(bm.ones, batch_or_mode)
    self.u.fill_(self.U)

  @property
  def derivative(self):
    du = lambda u, t: self.U - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    return JointEq(du, dx)

  def update(self, pre_spike):
    t = share.load('t')
    dt = share.load('dt')
    u, x = self.integral(self.u.value, self.x.value, t, dt)

    # --- original code:
    #   if pre_spike.dtype == jax.numpy.bool_:
    #     u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    #     x = bm.where(pre_spike, x - u * self.x, x)
    #   else:
    #     u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
    #     x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x

    # --- simplified code:
    u = pre_spike * self.U * (1 - self.u) + u
    x = pre_spike * -u * self.x + x

    self.x.value = x
    self.u.value = u
    return u * x

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode,
                      lambda shape: self.u * self.x)


STP.__doc__ = STP.__doc__ % (_docs.stp_doc, _docs.pneu_doc,)
