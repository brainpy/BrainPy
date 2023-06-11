
from typing import Union, Optional, Sequence

import numpy as np
from brainpy import math as bm, initialize as init
from brainpy._src.dynsys import DynamicalSystemNS
from brainpy.types import ArrayType

from brainpy._src.pnn.mixin import ParamDesc
from brainpy._src.pnn.utils import NEU_AXIS

__all__ = [
  'PSynOut',
  'COBA',
  'CUBA',
  'MgBlock'
]


class PSynOut(DynamicalSystemNS, ParamDesc):
  def __init__(
      self,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._conductance = None

  def bind_cond(self, conductance):
    self._conductance = conductance

  def unbind_cond(self):
    self._conductance = None

  def __call__(self, *args, **kwargs):
    if self._conductance is None:
      raise ValueError(f'Please first pack data at the current step using '
                       f'".bind_cond(data)". {self}')
    ret = super().__call__(self._conductance, *args, **kwargs)
    return ret


class COBA(PSynOut):
  r"""Conductance-based synaptic output.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

  Parameters
  ----------
  E: float, ArrayType, ndarray
    The reversal potential.
  name: str
    The model name.

  See Also
  --------
  CUBA
  """

  def __init__(
      self,
      E: Union[float, ArrayType] = 0.,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS, ),
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.axis_names = axis_names
    self.E = init.parameter(E, np.shape(E), axis_names=axis_names)

  def update(self, conductance, potential):
    return conductance * (self.E - potential)


class CUBA(PSynOut):
  r"""Current-based synaptic output.

  Given the conductance, this model outputs the post-synaptic current with a identity function:

  .. math::

     I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)

  Parameters
  ----------
  name: str
    The model name.


  See Also
  --------
  COBA
  """

  def __init__(
      self,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

  def update(self, conductance, potential=None):
    return conductance


class MgBlock(PSynOut):
  r"""Synaptic output based on Magnesium blocking.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t)) g_{\infty}(V,[{Mg}^{2+}]_{o})

  where The fraction of channels :math:`g_{\infty}` that are not blocked by magnesium can be fitted to

  .. math::

     g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V} \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

  Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration.

  Parameters
  ----------
  E: float, ArrayType
    The reversal potential for the synaptic current. [mV]
  alpha: float, ArrayType
    Binding constant. Default 0.062
  beta: float, ArrayType
    Unbinding constant. Default 3.57
  cc_Mg: float, ArrayType
    Concentration of Magnesium ion. Default 1.2 [mM].
  name: str
    The model name.
  """
  def __init__(
      self,
      E: Union[float, ArrayType] = 0.,
      cc_Mg: Union[float, ArrayType] = 1.2,
      alpha: Union[float, ArrayType] = 0.062,
      beta: Union[float, ArrayType] = 3.57,
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.axis_names = axis_names
    self.E = init.parameter(E, np.shape(E), axis_names=axis_names)
    self.cc_Mg = init.parameter(cc_Mg, np.shape(cc_Mg), axis_names=axis_names)
    self.alpha = init.parameter(alpha, np.shape(alpha), axis_names=axis_names)
    self.beta = init.parameter(alpha, np.shape(beta), axis_names=axis_names)

  def update(self, conductance, potential):
    return conductance * (self.E - potential) / (1 + self.cc_Mg / self.beta * bm.exp(-self.alpha * potential))



