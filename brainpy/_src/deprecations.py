import warnings
import functools

__all__ = [
  'deprecated',
  'deprecation_getattr',
  'deprecation_getattr2',
]


_update_deprecate_msg = '''
From brainpy>=2.4.3, update() function no longer needs to receive a global shared argument.

Instead of using:

  def update(self, tdi, *args, **kwagrs):
     t = tdi['t']
     ...

Please use:

  def update(self, *args, **kwagrs):
     t = bp.share['t']
     ...
'''


_input_deprecate_msg = '''
From brainpy>=2.4.3, input() function no longer needs to receive a global shared argument.

Instead of using:

  def input(tdi):
     ...

Please use:

  def input():
     t = bp.share['t']
     ...
'''



def _deprecate(msg):
  warnings.simplefilter('always', DeprecationWarning)  # turn off filter
  warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
  warnings.simplefilter('default', DeprecationWarning)  # reset filter


def deprecated(func):
  """This is a decorator which can be used to mark functions
  as deprecated. It will result in a warning being emitted
  when the function is used."""

  @functools.wraps(func)
  def new_func(*args, **kwargs):
    _deprecate("Call to deprecated function {}.".format(func.__name__))
    return func(*args, **kwargs)

  return new_func


def deprecation_getattr(module, deprecations):
  def getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:
        raise AttributeError(message)
      _deprecate(message)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr


def deprecation_getattr2(module, deprecations):
  def getattr(name):
    if name in deprecations:
      old_name, new_name, fn = deprecations[name]
      message = f"{old_name} is deprecated. "
      if new_name is not None:
        message += f'Use {new_name} instead.'
      if fn is None:
        raise AttributeError(message)
      _deprecate(message)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr
