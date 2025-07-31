# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

from brainpy._src.dynold.synapses.base import _SynOut
from brainpy._src.initialize import parameter, Initializer
from brainpy.math import Variable
from brainpy.types import ArrayType

__all__ = [
  'COBA',
  'CUBA',
]


class CUBA(_SynOut):
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
      target_var: Optional[Union[str, Variable]] = 'input',
      name: str = None,
  ):
    self._target_var = target_var
    super().__init__(name=name, target_var=target_var)

  def clone(self):
    return CUBA(target_var=self._target_var)


class COBA(_SynOut):
  r"""Conductance-based synaptic output.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

  Parameters
  ----------
  E: float, ArrayType, ndarray, callable, Initializer
    The reversal potential.
  name: str
    The model name.

  See Also
  --------
  CUBA
  """

  def __init__(
      self,
      E: Union[float, ArrayType, Callable, Initializer] = 0.,
      target_var: Optional[Union[str, Variable]] = 'input',
      membrane_var: Union[str, Variable] = 'V',
      name: str = None,
  ):
    super().__init__(name=name, target_var=target_var)
    self._E = E
    self._target_var = target_var
    self._membrane_var = membrane_var

  def clone(self):
    return COBA(E=self._E,
                target_var=self._target_var,
                membrane_var=self._membrane_var)

  def register_master(self, master):
    super().register_master(master)

    # reversal potential
    self.E = parameter(self._E, self.master.post.num, allow_none=False)

    # membrane potential
    if isinstance(self._membrane_var, str):
      if not hasattr(self.master.post, self._membrane_var):
        raise KeyError(f'Post-synaptic group does not have membrane variable: {self._membrane_var}')
      self.membrane_var = getattr(self.master.post, self._membrane_var)
    elif isinstance(self._membrane_var, Variable):
      self.membrane_var = self._membrane_var
    else:
      raise TypeError('"membrane_var" must be instance of string or Variable. '
                      f'But we got {type(self._membrane_var)}')

  def filter(self, g):
    V = self.membrane_var.value
    I = g * (self.E - V)
    return super(COBA, self).filter(I)

