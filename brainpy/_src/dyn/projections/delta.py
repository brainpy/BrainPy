from typing import Optional, Union

from brainpy import math as bm, check
from brainpy._src.delay import (delay_identifier, register_delay_by_return)
from brainpy._src.dynsys import DynamicalSystem, Projection
from brainpy._src.mixin import (JointType, SupportAutoDelay)

__all__ = [
  'HalfProjDelta', 'FullProjDelta',
]


class _Delta:
  def __init__(self):
    self._cond = None

  def bind_cond(self, cond):
    self._cond = cond

  def __call__(self, *args, **kwargs):
    r = self._cond
    return r


class HalfProjDelta(Projection):
  """Delta synaptic projection.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

  where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
  :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
  :math:`C` the set of neurons connected to the post-synaptic neuron,
  and :math:`D` the transmission delay of chemical synapses.
  For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.


  **Code Examples**

  .. code-block::

      import brainpy as bp
      import brainpy.math as bm

      class Net(bp.DynamicalSystem):
        def __init__(self):
          super().__init__()

          self.pre = bp.dyn.PoissonGroup(10, 100.)
          self.post = bp.dyn.LifRef(1)
          self.syn = bp.dyn.HalfProjDelta(bp.dnn.Linear(10, 1, bp.init.OneInit(2.)), self.post)

        def update(self):
          self.syn(self.pre())
          self.post()
          return self.post.V.value

      net = Net()
      indices = bm.arange(1000).to_numpy()
      vs = bm.for_loop(net.step_run, indices, progress_bar=True)
      bp.visualize.line_plot(indices, vs, show=True)

  Args:
    comm: DynamicalSystem. The synaptic communication.
    post: DynamicalSystem. The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  def __init__(
      self,
      comm: DynamicalSystem,
      post: DynamicalSystem,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # output initialization
    out = _Delta()
    post.add_inp_fun(self.name, out, category='delta')

    # references
    self.refs = dict(post=post, out=out)  # invisible to ``self.nodes()``
    self.refs['comm'] = comm  # unify the access

  def update(self, x):
    # call the communication
    current = self.comm(x)
    # bind the output
    self.refs['out'].bind_cond(current)
    # return the current, if needed
    return current


class FullProjDelta(Projection):
  """Delta synaptic projection.

  **Model Descriptions**

  .. math::

      I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

  where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
  :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
  :math:`C` the set of neurons connected to the post-synaptic neuron,
  and :math:`D` the transmission delay of chemical synapses.
  For simplicity, the rise and decay phases of post-synaptic currents are
  omitted in this model.


  **Code Examples**

  To simulate an E/I balanced network model:

  .. code-block::

      class EINet(bp.DynSysGroup):
        def __init__(self):
          super().__init__()
          self.N = bp.dyn.LifRef(4000, V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                                 V_initializer=bp.init.Normal(-55., 2.))
          self.delay = bp.VarDelay(self.N.spike, entries={'I': None})
          self.syn1 = bp.dyn.Expon(size=3200, tau=5.)
          self.syn2 = bp.dyn.Expon(size=800, tau=10.)
          self.E = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(3200, 4000, prob=0.02, weight=0.6),
                                      out=bp.dyn.COBA(E=0.),
                                      post=self.N)
          self.I = bp.dyn.VanillaProj(comm=bp.dnn.JitFPHomoLinear(800, 4000, prob=0.02, weight=6.7),
                                      out=bp.dyn.COBA(E=-80.),
                                      post=self.N)

        def update(self, input):
          spk = self.delay.at('I')
          self.E(self.syn1(spk[:3200]))
          self.I(self.syn2(spk[3200:]))
          self.delay(self.N(input))
          return self.N.spike.value

      model = EINet()
      indices = bm.arange(1000)
      spks = bm.for_loop(lambda i: model.step_run(i, 20.), indices)
      bp.visualize.raster_plot(indices, spks, show=True)


  Args:
    pre: The pre-synaptic neuron group.
    delay: The synaptic delay.
    comm: DynamicalSystem. The synaptic communication.
    post: DynamicalSystem. The post-synaptic neuron group.
    name: str. The projection name.
    mode: Mode. The computing mode.
  """

  def __init__(
      self,
      pre: JointType[DynamicalSystem, SupportAutoDelay],
      delay: Union[None, int, float],
      comm: DynamicalSystem,
      post: DynamicalSystem,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # synaptic models
    check.is_instance(pre, JointType[DynamicalSystem, SupportAutoDelay])
    check.is_instance(comm, DynamicalSystem)
    check.is_instance(post, DynamicalSystem)
    self.comm = comm

    # delay initialization
    delay_cls = register_delay_by_return(pre)
    delay_cls.register_entry(self.name, delay)

    # output initialization
    out = _Delta()
    post.add_inp_fun(self.name, out, category='delta')

    # references
    self.refs = dict(pre=pre, post=post, out=out)  # invisible to ``self.nodes()``
    self.refs['comm'] = comm  # unify the access
    self.refs['delay'] = pre.get_aft_update(delay_identifier)

  def update(self):
    # get delay
    x = self.refs['pre'].get_aft_update(delay_identifier).at(self.name)
    # call the communication
    current = self.comm(x)
    # bind the output
    self.refs['out'].bind_cond(current)
    # return the current, if needed
    return current
