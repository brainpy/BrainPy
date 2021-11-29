# -*- coding: utf-8 -*-

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
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, steps=None, name=None):
    super(DynamicalSystem, self).__init__(name=name)

    # step functions
    if steps is None:
      steps = ('update',)
    self.steps = Collector()
    if isinstance(steps, tuple):
      for step in steps:
        if isinstance(step, str):
          self.steps[step] = getattr(self, step)
        elif callable(step):
          self.steps[step.__name__] = step
        else:
          raise ModelBuildError(_error_msg.format(steps[0].__class__, str(steps[0])))
    elif isinstance(steps, dict):
      for key, step in steps.items():
        if callable(step):
          self.steps[key] = step
        else:
          raise ModelBuildError(_error_msg.format(steps.__class__, str(steps)))
    else:
      raise ModelBuildError(_error_msg.format(steps.__class__, str(steps)))

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

    >>> import brainpy as bp
    >>> group = bp.NeuGroup(10)
    >>> group.steps
    {'update': <bound method NeuGroup.update of <brainpy.simulation.brainobjects.neuron.NeuGroup object at 0xxxx>>}
    >>> delay1 = group.register_constant_delay('delay1', size=(10,), delay=2)
    >>> delay1
    <brainpy.simulation.brainobjects.delays.ConstantDelay at 0x219d5188280>
    >>> group.steps
    {'update': <bound method NeuGroup.update of <brainpy.simulation.brainobjects.neuron.NeuGroup object at 0xxxx>>,
     'delay1_update': <bound method ConstantDelay.update of <brainpy.simulation.brainobjects.delays.ConstantDelay object at 0xxxx>>}
    >>> delay1.data.shape
    (20, 10)

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
    if ConstantDelay is None: from brainpy.simulation.brainobjects.delays import ConstantDelay

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

  def update(self, *args, **kwargs):
    """The function to specify the updating rule."""
    raise NotImplementedError('Must implement "update" function by user self.')


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

  def __init__(self, *ds_tuple, steps=None, name=None, **ds_dict):
    # children dynamical systems
    self.implicit_nodes = Collector()
    for ds in ds_tuple:
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if ds.name in self.implicit_nodes:
        raise ValueError(f'{ds.name} has been paired with {ds}. Please change a unique name.')
      self.implicit_nodes[ds.name] = ds
    for key, ds in ds_dict.items():
      if not isinstance(ds, DynamicalSystem):
        raise ModelBuildError(f'{self.__class__.__name__} receives instances of '
                              f'DynamicalSystem, however, we got {type(ds)}.')
      if key in self.implicit_nodes:
        raise ValueError(f'{key} has been paired with {ds}. Please change a unique name.')
      self.implicit_nodes[key] = ds

    # integrative step function
    if steps is None:
      steps = ('update',)
    super(Container, self).__init__(steps=steps, name=name)

  def update(self, _t, _dt):
    """Step function of a network.

    In this update function, the step functions in children systems are
    iteratively called.
    """
    for node in self.child_ds().values():
      for step in node.steps.values():
        step(_t, _dt)

  def __getattr__(self, item):
    child_ds = super(Container, self).__getattribute__('implicit_nodes')
    if item in child_ds:
      return child_ds[item]
    else:
      return super(Container, self).__getattribute__(item)
