from typing import Sequence, Union, Callable, Any, Optional, Dict

import brainpy.math as bm
from brainpy._src.dynsys import NeuGroupNS
from brainpy._src.initialize import (parameter,
                                     variable_)
from brainpy.check import is_callable

from brainpy._src.pnn.utils import NEU_AXIS
from brainpy._src.pnn.synapses.syn_output import PSynOut
from ._docs import pneu_doc, dpneu_doc


__all__ = [
  'PNeuGroup',
  'DPNeuGroup',
]


class PNeuGroup(NeuGroupNS):
  """Parallelizable Neuron Group.

  Args:
    {pneu}
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name)

    # axis names for parallelization
    self.axis_names = axis_names
    if axis_names is not None:
      if len(axis_names) != len(self.varshape):
        raise ValueError(f'Except len(varshape) == len(axis_names), '
                         f'but got {len(self.varshape)} != {len(axis_names)}.')

    # the post updates used for computing
    self.pre_updates: Dict[str, Callable] = bm.node_dict()
    self.post_updates: Dict[str, Callable] = bm.node_dict()

    # outputs
    self.cur_outputs: Dict[str, PSynOut] = bm.node_dict()

  def sharding_param(self, param, shape=None, axis_names=None):
    """Sharding parameters across the default given devices. """
    if shape is None:
      shape = self.varshape
    if axis_names is None:
      axis_names = self.axis_names
    return parameter(param, sizes=shape, allow_none=False, axis_names=axis_names)

  def sharding_variable(self, var, batch_or_mode, shape=None, axis_names=None):
    """Sharding variables across the given devices."""
    if shape is None:
      shape = self.varshape
    if axis_names is None:
      axis_names = self.axis_names
    return variable_(var, sizes=shape, batch_or_mode=batch_or_mode,
                     axis_names=axis_names, batch_axis_name='batch')

  def __call__(self, *args, **kwargs):
    for model in tuple(self.pre_updates.values()):
      model()
    ret = super().__call__(*args, **kwargs)
    for model in tuple(self.post_updates.values()):
      model(ret)
    return ret


PNeuGroup.__doc__ = PNeuGroup.__doc__.format(pneu=pneu_doc)


class DPNeuGroup(PNeuGroup):
  """Differentiable and Parallelizable Neuron Group.

  Args:
    {pneu}
    {dpneu}
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      axis_names: Optional[Sequence[str]] = (NEU_AXIS,),
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,

      spk_fun: Callable = bm.surrogate.InvSquareGrad(),
      detach_spk: bool = False,
      method: str = 'exp_auto',
      spk_type: Any = None,
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name,
                     axis_names=axis_names)

    self.spk_fun = is_callable(spk_fun)
    self.detach_spk = detach_spk
    self.method = method
    self._spk_type = spk_type

  @property
  def spk_type(self):
    if self._spk_type is None:
      return bm.float_ if isinstance(self.mode, bm.TrainingMode) else bm.bool_
    else:
      return self._spk_type


DPNeuGroup.__doc__ = DPNeuGroup.__doc__.format(pneu=pneu_doc, dpneu=dpneu_doc)
