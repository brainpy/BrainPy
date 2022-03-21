# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators import odeint, JointEq
from brainpy.types import Parameter, Shape
from .base import RateModel

__all__ = [
  'MeanFieldQIF'
]


class MeanFieldQIF(RateModel):
  r"""A mean-field model of a quadratic integrate-and-fire neuron population.

  **Model Descriptions**

  The QIF population mean-field model, which has been derived from a
  population of all-to-all coupled QIF neurons in [5]_.
  The model equations are given by:

  .. math::

     \begin{aligned}
     \tau \dot{r} &=\frac{\Delta}{\pi \tau}+2 r v \\
     \tau \dot{v} &=v^{2}+\bar{\eta}+I(t)+J r \tau-(\pi r \tau)^{2}
     \end{aligned}

  where :math:`r` is the average firing rate and :math:`v` is the
  average membrane potential of the QIF population [5]_.

  This mean-field model is an exact representation of the macroscopic
  firing rate and membrane potential dynamics of a spiking neural network
  consisting of QIF neurons with Lorentzian distributed background
  excitabilities. While the mean-field derivation is mathematically
  only valid for all-to-all coupled populations of infinite size, it
  has been shown that there is a close correspondence between the
  mean-field model and neural populations with sparse coupling and
  population sizes of a few thousand neurons [6]_.

  **Model Parameters**

  ============= ============== ======== ========================
  **Parameter** **Init Value** **Unit** **Explanation**
  ------------- -------------- -------- ------------------------
  tau           1              ms       the population time constant
  eta           -5.            \        the mean of a Lorenzian distribution over the neural excitability in the population
  delta         1.0            \        the half-width at half maximum of the Lorenzian distribution over the neural excitability
  J             15             \        the strength of the recurrent coupling inside the population
  ============= ============== ======== ========================


  References
  ----------
  .. [5] E. Montbrió, D. Pazó, A. Roxin (2015) Macroscopic description for
         networks of spiking neurons. Physical Review X, 5:021028,
         https://doi.org/10.1103/PhysRevX.5.021028.
  .. [6] R. Gast, H. Schmidt, T.R. Knösche (2020) A Mean-Field Description
         of Bursting Dynamics in Spiking Neural Networks with Short-Term
         Adaptation. Neural Computation 32.9 (2020): 1615-1634.

  """

  def __init__(self,
               size: Shape,
               tau: Parameter = 1.,
               eta: Parameter = -5.0,
               delta: Parameter = 1.0,
               J: Parameter = 15.,
               method: str = 'exp_auto',
               name: str = None):
    super(MeanFieldQIF, self).__init__(size=size, name=name)

    # parameters
    self.tau = tau  #
    self.eta = eta  # the mean of a Lorenzian distribution over the neural excitability in the population
    self.delta = delta  # the half-width at half maximum of the Lorenzian distribution over the neural excitability
    self.J = J  # the strength of the recurrent coupling inside the population

    # variables
    self.r = bm.Variable(bm.ones(1))
    self.V = bm.Variable(bm.ones(1))
    self.input = bm.Variable(bm.zeros(1))

    # functions
    self.integral = odeint(self.derivative, method=method)

  def dr(self, r, t, v):
    return (self.delta / (bm.pi * self.tau) + 2. * r * v) / self.tau

  def dV(self, v, t, r, I_ext):
    return (v ** 2 + self.eta + I_ext + self.J * r * self.tau -
            (bm.pi * r * self.tau) ** 2) / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.dr])

  def update(self, _t, _dt):
    v, r = self.integral(self.V, self.r, t=_t, I_ext=self.input, dt=_dt)
    self.V.value = v
    self.r.value = r
    self.input[:] = 0.


class ThetaNeuron(RateModel):
  pass


class MeanFieldQIFWithSFA(RateModel):
  pass
