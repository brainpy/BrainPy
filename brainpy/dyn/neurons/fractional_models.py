# -*- coding: utf-8 -*-

from typing import Union, Sequence, Callable

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.initialize import ZeroInit, OneInit, Initializer, init_param
from brainpy.integrators.fde import CaputoL1Schema
from brainpy.integrators.fde import GLShortMemory
from brainpy.integrators.joint_eq import JointEq
from brainpy.tools.checking import check_float, check_integer
from brainpy.tools.checking import check_initializer
from brainpy.types import Shape, Tensor

__all__ = [
  'FractionalNeuron',
  'FractionalFHR',
  'FractionalIzhikevich',
]


class FractionalNeuron(NeuGroup):
  """Fractional-order neuron model."""
  pass


class FractionalFHR(FractionalNeuron):
  r"""The fractional-order FH-R model [1]_.

  FitzHugh and Rinzel introduced FH-R model (1976, in an unpublished article),
  which is the modification of the classical FHN neuron model. The fractional-order
  FH-R model is described as

  .. math::

     \begin{array}{rcl}
     \frac{{d}^{\alpha }v}{d{t}^{\alpha }} & = & v-{v}^{3}/3-w+y+I={f}_{1}(v,w,y),\\
     \frac{{d}^{\alpha }w}{d{t}^{\alpha }} & = & \delta (a+v-bw)={f}_{2}(v,w,y),\\
     \frac{{d}^{\alpha }y}{d{t}^{\alpha }} & = & \mu (c-v-dy)={f}_{3}(v,w,y),
     \end{array}

  where :math:`v, w` and :math:`y` represent the membrane voltage, recovery variable
  and slow modulation of the current respectively.
  :math:`I` measures the constant magnitude of external stimulus current, and :math:`\alpha`
  is the fractional exponent which ranges in the interval :math:`(0 < \alpha \le 1)`.
  :math:`a, b, c, d, \delta` and :math:`\mu` are the system parameters.

  The system reduces to the original classical order system when :math:`\alpha=1`.

  :math:`\mu` indicates a small parameter that determines the pace of the slow system
  variable :math:`y`. The fast subsystem (:math:`v-w`) presents a relaxation oscillator
  in the phase plane where :math:`\delta`  is a small parameter.
  :math:`v` is expressed in mV (millivolt) scale. Time :math:`t` is in ms (millisecond) scale.
  It exhibits tonic spiking or quiescent state depending on the parameter sets for a fixed
  value of :math:`I`. The parameter :math:`a` in the 2D FHN model corresponds to the
  parameter :math:`c` of the FH-R neuron model. If we decrease the value of :math:`a`,
  it causes longer intervals between two burstings, however there exists :math:`a`
  relatively fixed time of bursting duration. With the increasing of :math:`a`, the
  interburst intervals become shorter and periodic bursting changes to tonic spiking.

  Examples
  --------

  - [(Mondal, et, al., 2019): Fractional-order FitzHugh-Rinzel bursting neuron model](https://brainpy-examples.readthedocs.io/en/latest/neurons/2019_Fractional_order_FHR_model.html)


  Parameters
  ----------
  size: int, sequence of int
    The size of the neuron group.
  alpha: float, tensor
    The fractional order.
  num_memory: int
    The total number of the short memory.

  References
  ----------
  .. [1] Mondal, A., Sharma, S.K., Upadhyay, R.K. *et al.* Firing activities of a fractional-order FitzHugh-Rinzel bursting neuron model and its coupled dynamics. *Sci Rep* **9,** 15721 (2019). https://doi.org/10.1038/s41598-019-52061-4
  """

  def __init__(
      self,
      size: Shape,
      alpha: Union[float, Sequence[float]],
      num_memory: int = 1000,
      a: Union[float, Tensor, Initializer, Callable] = 0.7,
      b: Union[float, Tensor, Initializer, Callable] = 0.8,
      c: Union[float, Tensor, Initializer, Callable] = -0.775,
      d: Union[float, Tensor, Initializer, Callable] = 1.,
      delta: Union[float, Tensor, Initializer, Callable] = 0.08,
      mu: Union[float, Tensor, Initializer, Callable] = 0.0001,
      Vth: Union[float, Tensor, Initializer, Callable] = 1.8,
      V_initializer: Union[Initializer, Callable, Tensor] = OneInit(2.5),
      w_initializer: Union[Initializer, Callable, Tensor] = ZeroInit(),
      y_initializer: Union[Initializer, Callable, Tensor] = ZeroInit(),
      name: str = None
  ):
    super(FractionalFHR, self).__init__(size, name=name)

    # fractional order
    self.alpha = alpha
    check_integer(num_memory, 'num_memory', allow_none=False)

    # parameters
    self.a = init_param(a, self.num, allow_none=False)
    self.b = init_param(b, self.num, allow_none=False)
    self.c = init_param(c, self.num, allow_none=False)
    self.d = init_param(d, self.num, allow_none=False)
    self.mu = init_param(mu, self.num, allow_none=False)
    self.Vth = init_param(Vth, self.num, allow_none=False)
    self.delta = init_param(delta, self.num, allow_none=False)

    # initializers
    check_initializer(V_initializer, 'V_initializer', allow_none=False)
    check_initializer(w_initializer, 'w_initializer', allow_none=False)
    check_initializer(y_initializer, 'y_initializer', allow_none=False)
    self._V_initializer = V_initializer
    self._w_initializer = w_initializer
    self._y_initializer = y_initializer

    # variables
    self.V = bm.Variable(init_param(V_initializer, (self.num,)))
    self.w = bm.Variable(init_param(w_initializer, (self.num,)))
    self.y = bm.Variable(init_param(y_initializer, (self.num,)))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # integral function
    self.integral = GLShortMemory(self.derivative,
                                  alpha=alpha,
                                  num_memory=num_memory,
                                  inits=[self.V, self.w, self.y])

  def reset(self):
    self.V.value = init_param(self._V_initializer, (self.num,))
    self.w.value = init_param(self._w_initializer, (self.num,))
    self.y.value = init_param(self._y_initializer, (self.num,))
    self.input[:] = 0
    self.spike[:] = False
    # integral function reset
    self.integral.reset([self.V, self.w, self.y])

  def dV(self, V, t, w, y):
    return V - V ** 3 / 3 - w + y + self.input

  def dw(self, w, t, V):
    return self.delta * (self.a + V - self.b * w)

  def dy(self, y, t, V):
    return self.mu * (self.c - V - self.d * y)

  @property
  def derivative(self):
    return JointEq([self.dV, self.dw, self.dy])

  def update(self, t, dt):
    V, w, y = self.integral(self.V, self.w, self.y, t, dt)
    self.spike.value = bm.logical_and(V >= self.Vth, self.V < self.Vth)
    self.V.value = V
    self.w.value = w
    self.y.value = y
    self.input[:] = 0.


class FractionalIzhikevich(FractionalNeuron):
  r"""Fractional-order Izhikevich model [10]_.

  The fractional-order Izhikevich model is given by

  .. math::

     \begin{aligned}
      &\tau \frac{d^{\alpha} v}{d t^{\alpha}}=\mathrm{f} v^{2}+g v+h-u+R I \\
      &\tau \frac{d^{\alpha} u}{d t^{\alpha}}=a(b v-u)
      \end{aligned}

  where :math:`\alpha` is the fractional order (exponent) such that :math:`0<\alpha\le1`.
  It is a commensurate system that reduces to classical Izhikevich model at :math:`\alpha=1`.

  The time :math:`t` is in ms; and the system variable :math:`v` expressed in mV
  corresponds to membrane voltage. Moreover, :math:`u` expressed in mV is the
  recovery variable that corresponds to the activation of K+ ionic current and
  inactivation of Na+ ionic current.

  The parameters :math:`f, g, h` are fixed constants (should not be changed) such
  that :math:`f=0.04` (mV)−1, :math:`g=5, h=140` mV; and :math:`a` and :math:`b` are
  dimensionless parameters. The time constant :math:`\tau=1` ms; the resistance
  :math:`R=1` Ω; and :math:`I` expressed in mA measures the injected (applied)
  dc stimulus current to the system.

  When the membrane voltage reaches the spike peak :math:`v_{peak}`, the two variables
  are rest as follow:

  .. math::

     \text { if } v \geq v_{\text {peak }} \text { then }\left\{\begin{array}{l}
      v \leftarrow c \\
      u \leftarrow u+d
      \end{array}\right.

  we used :math:`v_{peak}=30` mV, and :math:`c` and :math:`d` are parameters expressed
  in mV. When the spike reaches its peak value, the membrane voltage :math:`v` and the
  recovery variable :math:`u` are reset according to the above condition.

  Examples
  --------

  - [(Teka, et. al, 2018): Fractional-order Izhikevich neuron model](https://brainpy-examples.readthedocs.io/en/latest/neurons/2018_Fractional_Izhikevich_model.html)


  References
  ----------
  .. [10] Teka, Wondimu W., Ranjit Kumar Upadhyay, and Argha Mondal. "Spiking and
          bursting patterns of fractional-order Izhikevich model." Communications
          in Nonlinear Science and Numerical Simulation 56 (2018): 161-176.

  """

  def __init__(
      self,
      size: Shape,
      alpha: Union[float, Sequence[float]],
      num_step: int,
      a: Union[float, Tensor, Initializer, Callable] = 0.02,
      b: Union[float, Tensor, Initializer, Callable] = 0.20,
      c: Union[float, Tensor, Initializer, Callable] = -65.,
      d: Union[float, Tensor, Initializer, Callable] = 8.,
      f: Union[float, Tensor, Initializer, Callable] = 0.04,
      g: Union[float, Tensor, Initializer, Callable] = 5.,
      h: Union[float, Tensor, Initializer, Callable] = 140.,
      R: Union[float, Tensor, Initializer, Callable] = 1.,
      tau: Union[float, Tensor, Initializer, Callable] = 1.,
      V_th: Union[float, Tensor, Initializer, Callable] = 30.,
      V_initializer: Union[Initializer, Callable, Tensor] = OneInit(-65.),
      u_initializer: Union[Initializer, Callable, Tensor] = OneInit(0.20 * -65.),
      name: str = None
  ):
    # initialization
    super(FractionalIzhikevich, self).__init__(size=size, name=name)

    # params
    self.alpha = alpha
    check_float(alpha, 'alpha', min_bound=0., max_bound=1., allow_none=False, allow_int=True)
    self.a = init_param(a, self.num, allow_none=False)
    self.b = init_param(b, self.num, allow_none=False)
    self.c = init_param(c, self.num, allow_none=False)
    self.d = init_param(d, self.num, allow_none=False)
    self.f = init_param(f, self.num, allow_none=False)
    self.g = init_param(g, self.num, allow_none=False)
    self.h = init_param(h, self.num, allow_none=False)
    self.tau = init_param(tau, self.num, allow_none=False)
    self.R = init_param(R, self.num, allow_none=False)
    self.V_th = init_param(V_th, self.num, allow_none=False)

    # initializers
    check_initializer(V_initializer, 'V_initializer', allow_none=False)
    check_initializer(u_initializer, 'u_initializer', allow_none=False)
    self._V_initializer = V_initializer
    self._u_initializer = u_initializer

    # variables
    self.V = bm.Variable(init_param(V_initializer, (self.num,)))
    self.u = bm.Variable(init_param(u_initializer, (self.num,)))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # functions
    check_integer(num_step, 'num_step', allow_none=False)
    self.integral = CaputoL1Schema(f=self.derivative,
                                   alpha=alpha,
                                   num_step=num_step,
                                   inits=[self.V, self.u])

  def reset(self):
    self.V.value = init_param(self._V_initializer, (self.num,))
    self.u.value = init_param(self._u_initializer, (self.num,))
    self.input[:] = 0
    self.spike[:] = False
    # integral function reset
    self.integral.reset([self.V, self.u])

  def dV(self, V, t, u, I_ext):
    dVdt = self.f * V * V + self.g * V + self.h - u + self.R * I_ext
    return dVdt / self.tau

  def du(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.du])

  def update(self, t, dt):
    V, u = self.integral(self.V, self.u, t=t, I_ext=self.input, dt=dt)
    spikes = V >= self.V_th
    self.V.value = bm.where(spikes, self.c, V)
    self.u.value = bm.where(spikes, u + self.d, u)
    self.spike.value = spikes
    self.input[:] = 0.
