# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.dyn.base import IonChaDyn
from brainpy._src.initialize import OneInit, Initializer, parameter, variable
from brainpy._src.integrators.ode.generic import odeint
from brainpy.types import Shape, ArrayType
from .base import Ion

__all__ = [
  'Calcium',
  'CalciumFixed',
  'CalciumDetailed',
  'CalciumFirstOrder',
]


class Calcium(Ion):
  pass


class CalciumFixed(Calcium):
  """Fixed Calcium dynamics.

  This calcium model has no dynamics. It holds fixed reversal
  potential :math:`E` and concentration :math:`C`.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, ArrayType, Initializer, Callable] = 120.,
      C: Union[float, ArrayType, Initializer, Callable] = 2.4e-4,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      **channels
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     method=method,
                     name=name,
                     mode=mode,
                     **channels)
    self.E = parameter(E, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)

  def reset_state(self, V, C_Ca=None, E_Ca=None, batch_size=None):
    C_Ca = self.C if C_Ca is None else C_Ca
    E_Ca = self.E if E_Ca is None else E_Ca
    for node in self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values():
      node.reset_state(V, C_Ca, E_Ca, batch_size=batch_size)


class CalciumDyna(Calcium):
  """Calcium ion flow with dynamics.

  Parameters
  ----------
  size: int, tuple of int
    The ion size.
  keep_size: bool
    Keep the geometry size.
  C0: float, ArrayType, Initializer, Callable
    The Calcium concentration outside of membrane.
  T: float, ArrayType, Initializer, Callable
    The temperature.
  C_initializer: Initializer, Callable, ArrayType
    The initializer for Calcium concentration.
  method: str
    The numerical method.
  name: str
    The ion name.
  """
  R = 8.31441  # gas constant, J*mol-1*K-1
  F = 96.489  # the Faraday constant

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      C0: Union[float, ArrayType, Initializer, Callable] = 2.,
      T: Union[float, ArrayType, Initializer, Callable] = 36.,
      C_initializer: Union[Initializer, Callable, ArrayType] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      **channels
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     method=method,
                     name=name,
                     mode=mode,
                     **channels)

    # parameters
    self.C0 = parameter(C0, self.varshape, allow_none=False)
    self.T = parameter(T, self.varshape, allow_none=False)  # temperature
    self._C_initializer = C_initializer
    self._constant = self.R / (2 * self.F) * (273.15 + self.T)

    # variables
    self.C = variable(C_initializer, self.mode, self.varshape)  # Calcium concentration
    self.E = bm.Variable(self._reversal_potential(self.C),
                         batch_axis=0 if isinstance(self.mode, bm.BatchingMode) else None)  # Reversal potential

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, C, t, V):
    raise NotImplementedError

  def reset_state(self, V, C_Ca=None, E_Ca=None, batch_size=None):
    self.C.value = variable(self._C_initializer, batch_size, self.varshape) if (C_Ca is None) else C_Ca
    self.E.value = self._reversal_potential(self.C)
    for node in self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values():
      node.reset(V, self.C, self.E, batch_size=batch_size)

  def update(self, V):
    for node in self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values():
      node.update(V, self.C.value, self.E.value)
    self.C.value = self.integral(self.C.value, share['t'], V, share['dt'])
    self.E.value = self._reversal_potential(self.C.value)

  def _reversal_potential(self, C):
    return self._constant * bm.log(self.C0 / C)


class CalciumDetailed(CalciumDyna):
  r"""Dynamical Calcium model proposed.

  **1. The dynamics of intracellular** :math:`Ca^{2+}`

  The dynamics of intracellular :math:`Ca^{2+}` were determined by two contributions [1]_ :

  *(i) Influx of* :math:`Ca^{2+}` *due to Calcium currents*

  :math:`Ca^{2+}` ions enter through :math:`Ca^{2+}` channels and diffuse into the
  interior of the cell. Only the :math:`Ca^{2+}` concentration in a thin shell beneath
  the membrane was modeled. The influx of :math:`Ca^{2+}` into such a thin shell followed:

  .. math::

      [Ca]_{i}=-\frac{k}{2 F d} I_{Ca}

  where :math:`F=96489\, \mathrm{C\, mol^{-1}}` is the Faraday constant,
  :math:`d=1\, \mathrm{\mu m}` is the depth of the shell beneath the membrane,
  the unit conversion constant is :math:`k=0.1` for :math:`I_T` in
  :math:`\mathrm{\mu A/cm^{2}}` and :math:`[Ca]_{i}` in millimolar,
  and :math:`I_{Ca}` is the summation of all :math:`Ca^{2+}` currents.

  *(ii) Efflux of* :math:`Ca^{2+}` *due to an active pump*

  In a thin shell beneath the membrane, :math:`Ca^{2+}` retrieval usually consists of a
  combination of several processes, such as binding to :math:`Ca^{2+}` buffers, calcium
  efflux due to :math:`Ca^{2+}` ATPase pump activity and diffusion to neighboring shells.
  Only the :math:`Ca^{2+}` pump was modeled here. We adopted the following kinetic scheme:

  .. math::

      Ca _{i}^{2+}+ P \overset{c_1}{\underset{c_2}{\rightleftharpoons}} CaP \xrightarrow{c_3} P+ Ca _{0}^{2+}

  where P represents the :math:`Ca^{2+}` pump, CaP is an intermediate state,
  :math:`Ca _{ o }^{2+}` is the extracellular :math:`Ca^{2+}` concentration,
  and :math:`c_{1}, c_{2}` and :math:`c_{3}` are rate constants. :math:`Ca^{2+}`
  ions have a high affinity for the pump :math:`P`, whereas extrusion of
  :math:`Ca^{2+}` follows a slower process (Blaustein, 1988 ). Therefore,
  :math:`c_{3}` is low compared to :math:`c_{1}` and :math:`c_{2}` and the
  Michaelis-Menten approximation can be used for describing the kinetics of the pump.
  According to such a scheme, the kinetic equation for the :math:`Ca^{2+}` pump is:

  .. math::

      \frac{[Ca^{2+}]_{i}}{dt}=-\frac{K_{T}[Ca]_{i}}{[Ca]_{i}+K_{d}}

  where :math:`K_{T}=10^{-4}\, \mathrm{mM\, ms^{-1}}` is the product of :math:`c_{3}`
  with the total concentration of :math:`P` and :math:`K_{d}=c_{2} / c_{1}=10^{-4}\, \mathrm{mM}`
  is the dissociation constant, which can be interpreted here as the value of
  :math:`[Ca]_{i}` at which the pump is half activated (if :math:`[Ca]_{i} \ll K_{d}`
  then the efflux is negligible).

  **2.A simple first-order model**

  While, in (Bazhenov, et al., 1998) [2]_, the :math:`Ca^{2+}` dynamics is
  described by a simple first-order model,

  .. math::

      \frac{d\left[Ca^{2+}\right]_{i}}{d t}=-\frac{I_{Ca}}{z F d}+\frac{\left[Ca^{2+}\right]_{rest}-\left[C a^{2+}\right]_{i}}{\tau_{Ca}}

  where :math:`I_{Ca}` is the summation of all :math:`Ca ^{2+}` currents, :math:`d`
  is the thickness of the perimembrane "shell" in which calcium is able to affect
  membrane properties :math:`(1.\, \mathrm{\mu M})`, :math:`z=2` is the valence of the
  :math:`Ca ^{2+}` ion, :math:`F` is the Faraday constant, and :math:`\tau_{C a}` is
  the :math:`Ca ^{2+}` removal rate. The resting :math:`Ca ^{2+}` concentration was
  set to be :math:`\left[ Ca ^{2+}\right]_{\text {rest}}=.05\, \mathrm{\mu M}` .

  **3. The reversal potential**

  The reversal potential of calcium :math:`Ca ^{2+}` is calculated according to the
  Nernst equation:

  .. math::

      E = k'{RT \over 2F} log{[Ca^{2+}]_0 \over [Ca^{2+}]_i}

  where :math:`R=8.31441 \, \mathrm{J} /(\mathrm{mol}^{\circ} \mathrm{K})`,
  :math:`T=309.15^{\circ} \mathrm{K}`,
  :math:`F=96,489 \mathrm{C} / \mathrm{mol}`,
  and :math:`\left[\mathrm{Ca}^{2+}\right]_{0}=2 \mathrm{mM}`.

  Parameters
  ----------
  d : float
    The thickness of the peri-membrane "shell".
  F : float
    The Faraday constant. (:math:`C*mmol^{-1}`)
  tau : float
    The time constant of the :math:`Ca ^{2+}` removal rate. (ms)
  C_rest : float
    The resting :math:`Ca ^{2+}` concentration.
  C0 : float
    The :math:`Ca ^{2+}` concentration outside of the membrane.
  R : float
    The gas constant. (:math:` J*mol^{-1}*K^{-1}`)

  References
  ----------

  .. [1] Destexhe, Alain, Agnessa Babloyantz, and Terrence J. Sejnowski.
         "Ionic mechanisms for intrinsic slow oscillations in thalamic
         relay neurons." Biophysical journal 65, no. 4 (1993): 1538-1552.
  .. [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and Terrence J.
         Sejnowski. "Cellular and network models for intrathalamic augmenting
         responses during 10-Hz stimulation." Journal of neurophysiology 79,
         no. 5 (1998): 2730-2748.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, ArrayType, Initializer, Callable] = 36.,
      d: Union[float, ArrayType, Initializer, Callable] = 1.,
      C_rest: Union[float, ArrayType, Initializer, Callable] = 2.4e-4,
      tau: Union[float, ArrayType, Initializer, Callable] = 5.,
      C0: Union[float, ArrayType, Initializer, Callable] = 2.,
      C_initializer: Union[Initializer, Callable, ArrayType] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      **channels
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     method=method,
                     name=name,
                     T=T,
                     C0=C0,
                     C_initializer=C_initializer,
                     mode=mode,
                     **channels)

    # parameters
    self.d = parameter(d, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.C_rest = parameter(C_rest, self.varshape, allow_none=False)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E, external=True)
    drive = bm.maximum(- ICa / (2 * self.F * self.d), 0.)
    return drive + (self.C_rest - C) / self.tau


class CalciumFirstOrder(CalciumDyna):
  r"""The first-order calcium concentration model.

  .. math::

     Ca' = -\alpha I_{Ca} + -\beta Ca

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, ArrayType, Initializer, Callable] = 36.,
      alpha: Union[float, ArrayType, Initializer, Callable] = 0.13,
      beta: Union[float, ArrayType, Initializer, Callable] = 0.075,
      C0: Union[float, ArrayType, Initializer, Callable] = 2.,
      C_initializer: Union[Initializer, Callable, ArrayType] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      **channels
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     method=method,
                     name=name,
                     T=T,
                     C0=C0,
                     C_initializer=C_initializer,
                     mode=mode,
                     **channels)

    # parameters
    self.alpha = parameter(alpha, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E, external=True)
    drive = bm.maximum(- self.alpha * ICa, 0.)
    return drive - self.beta * C
