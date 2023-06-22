from typing import Sequence, Union, Callable, Any, Optional, Dict

import brainpy.math as bm
from brainpy._src.dyn._docs import pneu_doc, dpneu_doc
from brainpy._src.dynsys import NeuGroupNS, DynamicalSystemNS
from brainpy._src.initialize.generic import parameter, variable_
from brainpy._src.mixin import ParamDesc, SupportProjection
from brainpy.check import is_callable

__all__ = [
  'NeuDyn',
  'SynDyn',
  'SynOut',
]


class NeuDyn(NeuGroupNS, SupportProjection):
  """Parallelizable Neuron Group.

  Args:
    {pneu}
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      sharding: Any = None,
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name)

    # axis names for parallelization
    self.sharding = sharding

    # the pre- / post-updates used for computing
    self.pre_updates: Dict[str, Callable] = bm.node_dict()
    self.post_updates: Dict[str, Callable] = bm.node_dict()

    # outputs
    self.cur_outputs: Dict[str, SynOut] = bm.node_dict()

  def init_param(self, param, shape=None, sharding=None):
    """Initialize parameters.

    If ``sharding`` is provided and ``param`` is array, this function will
    partition the parameter across the default device mesh.

    See :py:func:`~.brainpy.math.sharding.device_mesh` for the mesh setting.
    """
    shape = self.varshape if shape is None else shape
    sharding = self.sharding if sharding is None else sharding
    return parameter(param,
                     sizes=shape,
                     allow_none=False,
                     sharding=sharding)

  def init_variable(self, var_data, batch_or_mode, shape=None, sharding=None):
    """Initialize variables.

    If ``sharding`` is provided and ``var_data`` is array, this function will
    partition the variable across the default device mesh.

    See :py:func:`~.brainpy.math.sharding.device_mesh` for the mesh setting.
    """
    shape = self.varshape if shape is None else shape
    sharding = self.sharding if sharding is None else sharding
    return variable_(var_data,
                     sizes=shape,
                     batch_or_mode=batch_or_mode,
                     axis_names=sharding,
                     batch_axis_name=bm.sharding.BATCH_AXIS)

  def __call__(self, *args, **kwargs):
    # update ``pre_updates``
    for model in tuple(self.pre_updates.values()):
      model()

    # update the model self
    ret = super().__call__(*args, **kwargs)

    # update ``post_updates``
    for model in tuple(self.post_updates.values()):
      model(ret)
    return ret


NeuDyn.__doc__ = NeuDyn.__doc__.format(pneu=pneu_doc)


class DiffNeuDyn(NeuDyn):
  """Differentiable and Parallelizable Neuron Group.

  Args:
    {pneu}
    {dpneu}
  """

  supported_modes = (bm.TrainingMode, bm.NonBatchingMode)

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      sharding: Any = None,
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,

      spk_fun: Callable = bm.surrogate.InvSquareGrad(),
      detach_spk: bool = False,
      method: str = 'exp_auto',
      spk_type: Any = None,
  ):
    super().__init__(size=size,
                     mode=mode,
                     keep_size=keep_size,
                     name=name,
                     sharding=sharding)

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


DiffNeuDyn.__doc__ = DiffNeuDyn.__doc__.format(pneu=pneu_doc, dpneu=dpneu_doc)


class SynDyn(NeuDyn, ParamDesc):
  """Parallelizable synaptic dynamics.

  :py:class:`~.PSynDyn` is a subclass of :py:class:`~.ParamDesc`, because it uses
  the parameter description to describe the uniqueness of the synapse model.
  """
  pass


class SynOut(DynamicalSystemNS, ParamDesc):
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
    ret = self.update(self._conductance, *args, **kwargs)
    return ret


