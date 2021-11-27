# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.base.base import Base
from brainpy.base.collector import Collector
from brainpy.errors import RunningError, MonitorError
from brainpy.simulation import utils
from brainpy.simulation.monitor import Monitor

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

  def __init__(self, steps=None, monitors=None, name=None):
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
          raise RunningError(_error_msg.format(steps[0].__class__, str(steps[0])))
    elif isinstance(steps, dict):
      for key, step in steps.items():
        if callable(step):
          self.steps[key] = step
        else:
          raise RunningError(_error_msg.format(steps.__class__, str(steps)))
    else:
      raise RunningError(_error_msg.format(steps.__class__, str(steps)))

    # monitors
    if monitors is None:
      self.mon = Monitor(target=self, variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(target=self, variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise MonitorError(f'"monitors" only supports list/tuple/dict/ '
                         f'instance of Monitor, not {type(monitors)}.')

    # runner and run function
    self._input_step = None
    self._monitor_step = None
    self._step = None
    self._post = None
    self._former_run = {}

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
    """Register a constant delay.

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
      raise RunningError('Please initialize the super class first before '
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

  def _build_inputs(self, inputs=(), show_code=False):
    """Build input function."""
    inputs = utils.check_and_format_inputs(host=self, inputs=inputs)
    self._input_step = utils.build_input_func(inputs, show_code=show_code)

  def _build_monitors(self, show_code=False, method=None):
    """Build monitor function."""
    if self._monitor_step is None:
      monitors = utils.check_and_format_monitors(host=self)
      self._monitor_step, assigns = utils.build_monitor_func(monitors, show_code=show_code, method=method)
      return assigns
    return []

  def build(self, inputs=(), method=utils.STRUCT_RUN, show_code=False,
            dyn_vars=None, jit=False  # arguments only works for STRUCT_RUN
            ):
    # Build the inputs:
    #   All the inputs are wrapped into a single function.
    self._build_inputs(inputs=inputs, show_code=show_code)
    # Build the monitors:
    #   All the monitors are wrapped in a single function.
    assigns = self._build_monitors(method=method, show_code=show_code)

    if method == utils.STRUCT_RUN:

      if dyn_vars is None:
        dyn_vars = self.vars().unique()
      if isinstance(dyn_vars, (list, tuple)):
        dyn_vars = {f'_v{i}': v for i, v in enumerate(dyn_vars)}
      assert isinstance(dyn_vars, dict)

      def step(t_and_dt):  # the step function
        self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
        for step in self.steps.values():
          step(_t=t_and_dt[0], _dt=t_and_dt[1])
        return self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

      def post(x):  # monitor
        times, returns = x
        nodes = self.nodes()
        for i, (n, k) in enumerate(assigns):
          nodes[n].mon.item_contents[k] = math.asarray(returns[i])
          nodes[n].mon.ts = times

      self._step = math.make_loop(step, dyn_vars=dyn_vars, has_return=True)
      if jit: self._step = math.jit(self._step, dyn_vars=dyn_vars)
      self._post = post

    elif method == utils.NORMAL_RUN:

      def step(t_and_dt):  # the step function
        self._input_step(_t=t_and_dt[0], _dt=t_and_dt[1])
        for _step in self.steps.values():
          _step(_t=t_and_dt[0], _dt=t_and_dt[1])
        self._monitor_step(_t=t_and_dt[0], _dt=t_and_dt[1])

      def post(x):  # monitor
        times = x
        for node in self.nodes().values():
          if hasattr(node, 'mon'):
            if node.mon.num_item > 0:
              node.mon.ts = times
              for key, val in node.mon.item_contents.items():
                node.mon.item_contents[key] = math.asarray(val)

      self._step = step
      self._post = post

    else:
      raise ValueError

  def run(self, duration, dt=None, report=0., inputs=()):
    """The running function.

    Parameters
    ----------
    inputs : list, tuple
      The inputs for this instance of DynamicalSystem. It should the format
      of `[(target, value, [type, operation])]`, where `target` is the
      input target, `value` is the input value, `type` is the input type
      (such as "fix" or "iter"), `operation` is the operation for inputs
      (such as "+", "-", "*", "/", "=").

      - ``target``: should be a string. Can be specified by the *absolute access* or *relative access*.
      - ``value``: should be a scalar, vector, matrix, iterable function or objects.
      - ``type``: should be a string. "fix" means the input `value` is a constant. "iter" means the
        input `value` can be changed over time.
      - ``operation``: should be a string, support `+`, `-`, `*`, `/`, `=`.
      - Also, if you want to specify multiple inputs, just give multiple ``(target, value, [type, operation])``,
        for example ``[(target1, value1), (target2, value2)]``.

    duration : float, int, tuple, list
      The running duration.

    report : float
      The percent of progress to report. [0, 1]. If zero, the model
      will not output report progress.

    dt : float, optional
      The numerical integration step size.

    Returns
    -------
    running_time : float
      The total running time.
    """

    # time step
    if dt is None: dt = math.get_dt()
    assert isinstance(dt, (int, float))

    # times
    start, end = utils.check_duration(duration)
    times = math.arange(start, end, dt)

    # build inputs and monitors
    self.build(inputs=inputs, method=utils.NORMAL_RUN)
    for node in self.nodes().values():
      if hasattr(node, 'mon'):
        for key in node.mon.item_contents.keys():
          node.mon.item_contents[key] = []  # reshape the monitor items

    # run the model
    running_time = utils.run_model(run_func=self._step,
                                   times=times,
                                   report=report,
                                   dt=dt)
    self._post(times)

    return running_time


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

  def __init__(self, *ds_tuple, steps=None, monitors=None, name=None, **ds_dict):
    # children dynamical systems
    self.implicit_nodes = Collector()
    for ds in ds_tuple:
      if not isinstance(ds, DynamicalSystem):
        raise RunningError(f'{self.__class__.__name__} receives instances of '
                           f'DynamicalSystem, however, we got {type(ds)}.')
      if ds.name in self.implicit_nodes:
        raise ValueError(f'{ds.name} has been paired with {ds}. Please change a unique name.')
      self.implicit_nodes[ds.name] = ds
    for key, ds in ds_dict.items():
      if not isinstance(ds, DynamicalSystem):
        raise RunningError(f'{self.__class__.__name__} receives instances of '
                           f'DynamicalSystem, however, we got {type(ds)}.')
      if key in self.implicit_nodes:
        raise ValueError(f'{key} has been paired with {ds}. Please change a unique name.')
      self.implicit_nodes[key] = ds

    # integrative step function
    if steps is None:
      steps = ('update',)
    super(Container, self).__init__(steps=steps, monitors=monitors, name=name)

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
