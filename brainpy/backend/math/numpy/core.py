# -*- coding: utf-8 -*-


from brainpy import errors
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.tools.collector import Collector

__all__ = [
  'Function',
  'JIT',
  'Vectorize',
  'Parallel',
]

_NUMPY_FUNC_NO = 0
_NUMPY_JIT_NO = 0


class Function(DynamicSystem):
  """Turn a function into a DynamicSystem."""
  target_backend = 'numpy'

  @staticmethod
  def with_VIN(VIN, name=None):
    """Turn a function into a DynamicSystem.

    Parameters
    ----------
    VIN : tuple of Collector, list of Collector
      The collector of variables, integrators and nodes used by the function.
    name : str
      The function name.
    """
    return lambda f: Function(f, VIN, name=name)

  def __init__(self, f, VIN, name=None, monitors=None):
    """Function constructor.

    Parameters
    ----------
    f : function
      The function or the module to represent.
    VIN : list of Collector, tuple of Collector
      The collection of variables, integrators, and nodes.
    """
    V, I, N = VIN
    if hasattr(f, '__name__'):
      self.all_vars = Collector((f'{{{f.__name__}}}{k}', v) for k, v in V.items())
      self.all_ints = Collector((f'{{{f.__name__}}}{k}', v) for k, v in I.items())
      self.all_nodes = Collector((f'{{{f.__name__}}}{k}', v) for k, v in N.items())
    else:
      self.all_vars = Collector(V)
      self.all_ints = Collector(I)
      self.all_nodes = Collector(N)
    self.raw = f

    # monitors
    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    # name
    if name is None:
      global _NUMPY_FUNC_NO
      name = f'NPFunc{_NUMPY_FUNC_NO}'
      _NUMPY_FUNC_NO += 1

    super(Function, self).__init__(steps={'call': self.__call__},
                                   name=name,
                                   monitors=None)

  def __call__(self, *args, **kwargs):
    """Call the the function."""
    return self.raw(*args, **kwargs)

  def vars(self, prefix=''):
    """Return the Collection of the variables used by the function."""
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_vars.items())
    else:
      return Collector(self.all_vars)

  def ints(self, prefix=''):
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_ints.items())
    else:
      return Collector(self.all_ints)

  def nodes(self, prefix=''):
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_nodes.items())
    else:
      return Collector(self.all_nodes)

  def __repr__(self):
    return f'{self.name}(f={str(self.raw)})'


class JIT(DynamicSystem):
  """Jit constructor.

  Parameters
  ----------
  ds : function, DynamicSystem
    The function or the DynamicSystem to compile.
  VIN : None, list of Collector, tuple of Collector
    The collections of variables, integrators and nodes.
    This argument is required for functions.
  static_argnums: None, int,
    Tuple of indexes of f's input arguments to treat as static (constants)).
    A new graph is compiled for each different combination of values for such inputs.
  """
  target_backend = 'numpy'

  def __init__(self, ds, VIN=None, static_argnums=None, name=None, monitors=None):
    self.static_argnums = static_argnums

    if not isinstance(ds, DynamicSystem):
      if VIN is None:
        raise ValueError('You must supply the VIN used by the function f.')
      ds = Function(ds, VIN, name=name)

    self.raw = ds
    self.all_vars = ds.vars() if VIN is None else VIN[0]
    self.all_ints = ds.ints() if VIN is None else VIN[1]
    self.all_nodes = ds.nodes() if VIN is None else VIN[2]

    # monitors
    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    # name
    if name is None:
      global _NUMPY_JIT_NO
      name = f'NPJIT{_NUMPY_JIT_NO}'
      _NUMPY_JIT_NO += 1

    super(JIT, self).__init__(steps=ds.steps, name=name, monitors=monitors)

  def __call__(self, *args, **kwargs):
    if not isinstance(self.raw, DynamicSystem):
      return self.raw(*args, **kwargs)
    else:
      raise ValueError

  def vars(self, prefix=''):
    """Return the Collection of the variables used by the function."""
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_vars.items())
    else:
      return Collector(self.all_vars)

  def ints(self, prefix=''):
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_ints.items())
    else:
      return Collector(self.all_ints)

  def nodes(self, prefix=''):
    if prefix:
      return Collector((prefix + k, v) for k, v in self.all_nodes.items())
    else:
      return Collector(self.all_nodes)

  def __repr__(self):
    return f'{self.__class__.__name__}(f={self.raw}, static_argnums={self.static_argnums or None})'



class Vectorize(DynamicSystem):
  target_backend = 'numpy'


class Parallel(DynamicSystem):
  target_backend = 'numpy'
