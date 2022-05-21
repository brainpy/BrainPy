# -*- coding: utf-8 -*-

from typing import Union, Callable

from brainpy.initialize import init_param, Initializer
from brainpy.dyn.base import SynapseOutput, NeuGroup
from brainpy.types import Tensor

__all__ = [
  'COBA',
  'CUBA',
]


class CUBA(SynapseOutput):
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

  def __init__(self, name: str = None):
    super(CUBA, self).__init__(name=name)

  def filter(self, g):
    return g


class COBA(SynapseOutput):
  r"""Conductance-based synaptic output.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

  Parameters
  ----------
  post: NeuGroup
    The post-synaptic neuron group.
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
      post: NeuGroup,
      E: Union[float, Tensor, Callable, Initializer] = 0.,
      name: str = None
  ):
    super(COBA, self).__init__(name=name)
    if not isinstance(post, NeuGroup):
      raise ValueError(f'post should be instance of {NeuGroup.__name__}, but we got {type(post)}')
    if not hasattr(post, 'V'):
      raise ValueError('post should has the attribute of "V".')
    self.E = init_param(E, post.num, allow_none=False)
    self.post = post

  def filter(self, g):
    return g * (self.E - self.post.V)
