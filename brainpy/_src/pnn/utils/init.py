__all__ = [
  'DelayedInit',
]


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
