# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

from brainpy.math import Variable
from brainpy.dyn.base import SynOutput
from brainpy.initialize import parameter, Initializer
from brainpy.types import Tensor

__all__ = [
  'COBA',
  'CUBA',
]


class CUBA(SynOutput):
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
    super(CUBA, self).__init__(name=name, target_var=target_var)


class COBA(SynOutput):
  r"""Conductance-based synaptic output.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

  Parameters
  ----------
  E: float, JaxArray, ndarray, callable, Initializer
    The reversal potential.
  name: str
    The model name.

  See Also
  --------
  CUBA
  """

  def __init__(
      self,
      E: Union[float, Tensor, Callable, Initializer] = 0.,
      target_var: Optional[Union[str, Variable]] = 'input',
      membrane_var: Union[str, Variable] = 'V',
      name: str = None,
  ):
    super(COBA, self).__init__(name=name, target_var=target_var)
    self.E = E
    self.membrane_var = membrane_var

  def register_master(self, master):
    super(COBA, self).register_master(master)
    self.E = parameter(self.E, self.master.post.num, allow_none=False)

    if isinstance(self.membrane_var, str):
      if not hasattr(self.master.post, self.membrane_var):
        raise KeyError(f'Post-synaptic group does not have membrane variable: {self.membrane_var}')
      self.membrane_var = getattr(self.master.post, self.membrane_var)
    elif isinstance(self.membrane_var, Variable):
      self.membrane_var = self.membrane_var
    else:
      raise TypeError('"membrane_var" must be instance of string or Variable. '
                      f'But we got {type(self.membrane_var)}')

  def filter(self, g):
    V = self.membrane_var.value
    I = g * (self.E - V)
    return super(COBA, self).filter(I)
