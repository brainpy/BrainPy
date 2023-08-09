# -*- coding: utf-8 -*-


"""
This module implements calcium-dependent potassium channels.

"""

from typing import Union, Callable

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.dyn.ions.calcium import Calcium
from brainpy._src.initialize import Initializer, parameter, variable
from brainpy._src.integrators.ode.generic import odeint
from brainpy.types import Shape, ArrayType
from .base import IonChannel

__all__ = [
  'IAHP_De1994',
]


class IAHP_De1994(IonChannel):
  r"""The calcium-dependent potassium current model proposed by (Destexhe, et al., 1994) [1]_.

  Both in vivo (Contreras et al. 1993; Mulle et al. 1986) and in
  vitro recordings (Avanzini et al. 1989) show the presence of a
  marked after-hyper-polarization (AHP) after each burst of the RE
  cell. This slow AHP is mediated by a slow :math:`Ca^{2+}`-dependent K+
  current (Bal and McCormick 1993). (Destexhe, et al., 1994) adopted a
  modified version of a model of :math:`I_{KCa}` introduced previously (Yamada et al.
  1989) that requires the binding of :math:`nCa^{2+}` to open the channel

  .. math::

      (\text { closed })+n \mathrm{Ca}_{i}^{2+} \underset{\beta}{\stackrel{\alpha}{\rightleftharpoons}(\text { open })

  where :math:`Ca_i^{2+}` is the intracellular calcium and :math:`\alpha` and
  :math:`\beta` are rate constants. The ionic current is then given by

  .. math::

      \begin{aligned}
      I_{AHP} &= g_{\mathrm{max}} p^2 (V - E_K) \\
      {dp \over dt} &= \phi {p_{\infty}(V, [Ca^{2+}]_i) - p \over \tau_p(V, [Ca^{2+}]_i)} \\
      p_{\infty} &=\frac{\alpha[Ca^{2+}]_i^n}{\left(\alpha[Ca^{2+}]_i^n + \beta\right)} \\
      \tau_p &=\frac{1}{\left(\alpha[Ca^{2+}]_i +\beta\right)}
      \end{aligned}

  where :math:`E` is the reversal potential, :math:`g_{max}` is the maximum conductance,
  :math:`[Ca^{2+}]_i` is the intracellular Calcium concentration.
  The values :math:`n=2, \alpha=48 \mathrm{~ms}^{-1} \mathrm{mM}^{-2}` and
  :math:`\beta=0.03 \mathrm{~ms}^{-1}` yielded AHPs very similar to those RE cells
  recorded in vivo and in vitro.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).

  References
  ----------

  .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.

  """

  '''The type of the master object.'''
  master_type = Calcium

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, ArrayType, Initializer, Callable] = -95.,
      n: Union[float, ArrayType, Initializer, Callable] = 2,
      g_max: Union[float, ArrayType, Initializer, Callable] = 10.,
      alpha: Union[float, ArrayType, Initializer, Callable] = 48.,
      beta: Union[float, ArrayType, Initializer, Callable] = 0.09,
      phi: Union[float, ArrayType, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(size=size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.n = parameter(n, self.varshape, allow_none=False)
    self.alpha = parameter(alpha, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, self.mode, self.varshape)

    # function
    self.integral = odeint(self.dp, method=method)

  def dp(self, p, t, C_Ca):
    C2 = self.alpha * bm.power(C_Ca, self.n)
    C3 = C2 + self.beta
    return self.phi * (C2 / C3 - p) * C3

  def update(self, V, C_Ca, E_Ca):
    self.p.value = self.integral(self.p.value, share['t'], C_Ca=C_Ca, dt=share['dt'])

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * (self.E - V)

  def reset_state(self, V, C_Ca, E_Ca, batch_size=None):
    C2 = self.alpha * bm.power(C_Ca, self.n)
    C3 = C2 + self.beta
    self.p[:] = C2 / C3
    if isinstance(batch_size, int):
      batch_size = batch_size
      size = (batch_size,) + self.varshape
    elif isinstance(batch_size, bm.Mode):
      if isinstance(batch_size, bm.BatchingMode):
        size = (batch_size.batch_size,) + self.varshape
      else:
        batch_size = None
        size = self.varshape
    else:
      size = self.varshape
    self.p.value = bm.broadcast_to(C2 / C3, size)

