# -*- coding: utf-8 -*-

import warnings
from brainpy.base.base import Base
from brainpy.base.collector import Collector
from brainpy.errors import ModelBuildError

ConstantDelay = None

__all__ = [
  'DynamicalSystem',
  'Container',
]

_error_msg = 'Unknown type of the update function: {} ({}). ' \
             'Currently, BrainPy only supports: \n' \
             '1. function \n' \
             '2. function name (str) \n' \
             '3. tuple/dict of functions \n' \
             '4. tuple of function names \n'


class DynamicalSystem(Base):
  """Base Dynamical System class.

  Any object has step functions will be a dynamical system.
  That is to say, in BrainPy, the essence of the dynamical system
  is the "step functions".

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, name=None):
    super(DynamicalSystem, self).__init__(name=name)

  @property
  def steps(self):
    warnings.warn('.steps has been deprecated since version 2.0.3.', DeprecationWarning)
    return {}

  def child_ds(self, method='absolute', include_self=False):
    """Return the children instance of dynamical systems.

    This is a shortcut function to get all children dynamical system
    in this object. For example:

    >>> import brainpy as bp
    >>>
    >>> class Net(bp.DynamicalSystem):
    >>>   def __init__(self, **kwargs):
    >>>     super(Net, self).__init__(**kwargs)
    >>>     self.A = bp.NeuGroup(10)
    >>>     self.B = bp.NeuGroup(20)
    >>>
    >>>   def update(self, _t, _dt):
    >>>     for node in self.child_ds().values():
    >>>        node.update(_t, _dt)
    >>>
    >>> net = Net()
    >>> net.child_ds()
    {'NeuGroup0': <brainpy.simulation.brainobjects.neuron.NeuGroup object at 0x000001ABD4FF02B0>,
    'NeuGroup1': <brainpy.simulation.brainobjects.neuron.NeuGroup object at 0x000001ABD74E5670>}

    Parameters
    ----------
    method : str
      The method to access the children nodes.
    include_self : bool
      Whether include the self dynamical system.

    Returns
    -------
    collector: Collector
      A Collector includes all children systems.
    """
    nodes = self.nodes(method=method).subset(DynamicalSystem).unique()
    if not include_self:
      if method == 'absolute':
        nodes.pop(self.name)
      elif method == 'relative':
        nodes.pop('')
      else:
        raise ValueError(f'Unknown access method: {method}')
    return nodes

  def register_constant_delay(self, key, size, delay, dtype=None):
    """Register a constant delay, whose update method will be appended into
    the ``self.steps`` in this host class.

    Parameters
    ----------
    key : str
      The delay name.
    size : int, list of int, tuple of int
      The delay data size.
    delay : int, float, ndarray
      The delay time, with the unit same with `brainpy.math.get_dt()`.
    dtype : optional
      The data type.

    Returns
    -------
    delay : ConstantDelay
        An instance of ConstantDelay.
    """
    global ConstantDelay
    if ConstantDelay is None: from brainpy.building.brainobjects.delays import ConstantDelay

    if not hasattr(self, 'steps'):
      raise ModelBuildError('Please initialize the super class first before '
                            'registering constant_delay. \n\n'
                            'super(YourClassName, self).__init__(**kwargs)')
    if not key.isidentifier(): raise ValueError(f'{key} is not a valid identifier.')
    cdelay = ConstantDelay(size=size,
                           delay=delay,
                           name=f'{self.name}_delay_{key}',
                           dtype=dtype)
    self.steps[f'{key}_update'] = cdelay.update
    return cdelay

  def __call__(self, *args, **kwargs):
    """The shortcut to call ``update`` methods."""
    return self.update(*args, **kwargs)

  def update(self, _t, _dt):
    """The function to specify the updating rule.
    Assume any dynamical system depends on the time variable ``t`` and
    the time step ``dt``.
    """
    raise NotImplementedError('Must implement "update" function by user self.')

  def run(self, duration, start_t=None, monitors=None, inputs=(),
          dt=None, jit=False, dyn_vars=None, numpy_mon_after_run=False):
    pass


class Container(DynamicalSystem):
  """Container object which is designed to add other instances of DynamicalSystem.

  Parameters
  ----------
  steps : tuple of function, tuple of str, dict of (str, function), optional
      The step functions.
  monitors : tuple, list, Monitor, optional
      The monitor object.
  name : str, optional
      The object name.
  show_code : bool
      Whether show the formatted code.
  ds_dict : dict of (str, )
      The instance of DynamicalSystem with the format of "key=dynamic_system".
  """

  def __init__(self, *ds_tuple, name=None, **ds_dict):
    super(Container, self).__init__(name=name)

    # children dynamical systems
    self.implicit_nodes = Collector()
    for ds in ds_tuple:
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if ds.name in self.implicit_nodes:
        raise ValueError(f'{ds.name} has been paired with {ds}. Please change a unique name.')
    self.register_implicit_nodes({node.name: node for node in ds_tuple})
    for key, ds in ds_dict.items():
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if key in self.implicit_nodes:
        raise ValueError(f'{key} has been paired with {ds}. Please change a unique name.')
    self.register_implicit_nodes(ds_dict)

  def update(self, _t, _dt):
    """Step function of a network.

    In this update function, the update functions in children systems are
    iteratively called.
    """
    for node in self.child_ds().values():
      node.update(_t, _dt)

  def __getattr__(self, item):
    child_ds = super(Container, self).__getattribute__('implicit_nodes')
    if item in child_ds:
      return child_ds[item]
    else:
      return super(Container, self).__getattribute__(item)
