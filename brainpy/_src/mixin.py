from typing import Optional, Sequence, Union, Tuple, Callable
from dataclasses import dataclass
from brainpy import tools, math as bm

__all__ = [
  'MixIn',
  'ParamDesc',
  'AlignPost',
  'ProjAutoDelay',
]


class MixIn(object):
  pass


class DelayedInit(object):
  """Delayed initialization.
  """

  def __init__(
      self,
      cls: type,
      identifier,
      *args,
      **kwargs
  ):
    self.cls = cls
    self.args = args
    self.kwargs = kwargs
    self._identifier = identifier

  def __call__(self, *args, **kwargs):
    return self.cls(*self.args, *args, **self.kwargs, **kwargs)

  def init(self, *args, **kwargs):
    return self.__call__(*args, **kwargs)

  @classmethod
  def __class_getitem__(cls, item):
    return cls


class ParamDesc(MixIn):
  """Parameter description MixIn.

  This mixin enables the subclass has a classmethod ``desc``, which
  produces an instance of :py:class:`~.DelayedInit`.
  """

  not_desc_params: Optional[Sequence[str]] = None

  @classmethod
  def desc(cls, *args, **kwargs) -> DelayedInit:
    # cls_args = list(inspect.signature(cls.__init__).parameters.values())[1:]
    # names = [arg.name for arg in cls_args]
    # defaults = [arg.default for arg in cls_args]
    if cls.not_desc_params is not None:
      repr_kwargs = {k: v for k, v in kwargs.items() if k not in cls.not_desc_params}
    else:
      repr_kwargs = {k: v for k, v in kwargs.items()}
    for k in tuple(repr_kwargs.keys()):
      if isinstance(repr_kwargs[k], bm.Variable):
        repr_kwargs[k] = id(repr_kwargs[k])
    repr_args = tools.repr_dict(repr_kwargs)
    if len(args):
      repr_args = f"{', '.join([repr(arg) for arg in args])}, {repr_args}"
    return DelayedInit(cls, f'{cls.__name__}({repr_args})', *args, **kwargs)


class AlignPost(MixIn):
  """Align post MixIn.

  This class provides a ``add_current()`` function for
  add external currents.
  """

  def add_current(self, *args, **kwargs):
    raise NotImplementedError


@dataclass
class ReturnInfo:
  size: Sequence[int]
  axis_names: Sequence[str]
  batch_or_mode: Optional[Union[int, bm.Mode]]
  init: Callable


class ProjAutoDelay(MixIn):
  """Support for automatic delay in synaptic projection :py:class:`~.SynProj`."""

  def return_info(self) -> Union[bm.Variable, ReturnInfo]:
    raise NotImplementedError

