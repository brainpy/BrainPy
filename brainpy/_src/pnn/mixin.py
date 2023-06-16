from typing import Optional, Sequence

from .utils import DelayedInit
from brainpy._src import tools, math as bm


__all__ = [
  'MixIn',
  'ParamDesc',
  'AlignPost',
]


class MixIn(object):
  pass


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


