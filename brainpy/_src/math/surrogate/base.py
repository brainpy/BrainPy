

__all__ = [
  'Surrogate'
]


class Surrogate(object):
  """The base surrograte gradient function."""
  def __call__(self, *args, **kwargs):
    raise NotImplementedError




