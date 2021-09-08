# -*- coding: utf-8 -*-

from brainpy import math, errors
from brainpy.base import collector
from brainpy.base.base import Base
from brainpy.simulation import utils
from brainpy.simulation.monitor import Monitor

__all__ = [
  'DynamicalSystem',
  'Container',
]

_error_msg = 'Unknown model type: {type}. ' \
             'Currently, BrainPy only supports: ' \
             'function, tuple/dict of functions, ' \
             'tuple of function names.'


class DynamicalSystem(Base):
  """Base Dynamic System class.

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
  target_backend = None

  def __init__(self, steps=(), monitors=None, name=None):
    super(DynamicalSystem, self).__init__(name=name)

    # step functions
    self.steps = collector.Collector()
    steps = tuple() if steps is None else steps
    if isinstance(steps, tuple):
      for step in steps:
        if isinstance(step, str):
          self.steps[step] = getattr(self, step)
        elif callable(step):
          self.steps[step.__name__] = step
        else:
          raise errors.BrainPyError(_error_msg.format(type(steps[0])))
    elif isinstance(steps, dict):
      for key, step in steps.items():
        if callable(step):
          self.steps[key] = step
        else:
          raise errors.BrainPyError(_error_msg.format(type(step)))
    else:
      raise errors.BrainPyError(_error_msg.format(type(steps)))

    # monitors
    if monitors is None:
      self.mon = Monitor(target=self, variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(target=self, variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise errors.BrainPyError(f'"monitors" only supports list/tuple/dict/ '
                                f'instance of Monitor, not {type(monitors)}.')

    # target backend
    if self.target_backend is None:
      self._target_backend = ('general',)
    elif isinstance(self.target_backend, str):
      self._target_backend = (self.target_backend,)
    elif isinstance(self.target_backend, (tuple, list)):
      if not isinstance(self.target_backend[0], str):
        raise errors.BrainPyError('"target_backend" must be a list/tuple of string.')
      self._target_backend = tuple(self.target_backend)
    else:
      raise errors.BrainPyError(f'Unknown setting of "target_backend": {self.target_backend}')

    # runner and run function
    self.driver = None
    self._input_step = lambda _t, _dt: None
    self._monitor_step = lambda _t, _dt: None

  def update(self, _t, _dt):
    raise NotImplementedError

  def _step_run(self, _t, _dt):
    self._monitor_step(_t, _dt)
    self._input_step(_t, _dt)
    for step in self.steps.values():
      step(_t, _dt)

  def run(self, duration, report=0., inputs=()):
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
      - ``operation``: should be a string.
      - Also, if you want to specify multiple inputs, just give multiple ``(target, value, [type, operation])``.

    duration : float, int, tuple, list
      The running duration.

    report : float
      The percent of progress to report. [0, 1]. If zero, the model
      will not output report progress.

    Returns
    -------
    running_time : float
      The total running time.
    """

    # 1. Backend checking
    for node in self.nodes().values():
      check1 = node._target_backend[0] != 'general'
      check2 = math.get_backend_name() not in node._target_backend
      if check1 and check2:
        raise errors.BrainPyError(f'The model {node.name} is target to run on '
                                  f'{node._target_backend}, but currently the '
                                  f'selected backend is {math.get_backend_name()}')

    # 2. Build the inputs.
    #    All the inputs are wrapped into a single function.
    self._input_step = utils.build_input_func(utils.check_and_format_inputs(host=self, inputs=inputs),
                                              show_code=False)

    # 3. Build the monitors.
    #    All the monitors are wrapped in a single function.
    self._monitor_step = utils.build_monitor_func(utils.check_and_format_monitors(host=self),
                                                  show_code=False)

    # 4. times
    start, end = utils.check_duration(duration)
    times = math.arange(start, end, math.get_dt())

    # 5. run the model
    # ----
    # 5.1 iteratively run the step function.
    # 5.2 report the running progress.
    # 5.3 return the overall running time.
    running_time = utils.run_model(run_func=self._step_run, times=times, report=report)

    # 6. monitor
    # --
    # 6.1 add 'ts' variable to every monitor
    # 6.2 wrap the monitor iterm with the 'list' type into the 'ndarray' type
    for node in [self] + list(self.nodes().unique().values()):
      if node.mon.num_item > 0:
        node.mon.ts = times
        for key, val in list(node.mon.item_contents.items()):
          val = math.asarray(node.mon.item_contents[key])
          node.mon.item_contents[key] = val

    return running_time


class Container(DynamicalSystem):
  """Container object which is designed to add other instances of DynamicalSystem.

  What's different from the other objects of DynamicalSystem is that Container has
  one more useful function :py:func:`add`. It can be used to add the children
  objects.

  Parameters
  ----------
  steps : function, list of function, tuple of function, dict of (str, function), optional
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
    self.child_ds = dict()
    for ds in ds_tuple:
      if not isinstance(ds, DynamicalSystem):
        raise errors.BrainPyError(f'{self.__class__.__name__} receives instances of '
                                  f'DynamicalSystem, however, we got {type(ds)}.')
      if ds.name in self.child_ds:
        raise ValueError(f'{ds.name} has been paired with {ds}. Please change a unique name.')
      self.child_ds[ds.name] = ds
    for key, ds in ds_dict.items():
      if not isinstance(ds, DynamicalSystem):
        raise errors.BrainPyError(f'{self.__class__.__name__} receives instances of '
                                  f'DynamicalSystem, however, we got {type(ds)}.')
      if key in self.child_ds:
        raise ValueError(f'{key} has been paired with {ds}. Please change a unique name.')
      self.child_ds[key] = ds
    # step functions in children dynamical systems
    self.child_steps = dict()
    for ds_key, ds in self.child_ds.items():
      for step_key, step in ds.steps.items():
        self.child_steps[f'{ds_key}_{step_key}'] = step

    # integrative step function
    if steps is None:
      steps = ('update',)
    super(Container, self).__init__(steps=steps, monitors=monitors, name=name)

  def update(self, _t, _dt):
    """Step function of a network.

    In this update function, the step functions in children systems are
    iteratively called.
    """
    for step in self.child_steps.values():
      step(_t, _dt)

  def vars(self, method='absolute'):
    """Collect all the variables (and their names) contained
    in the list and its children instance of DynamicalSystem.

    Parameters
    ----------
    method : str
      string to prefix to the variable names.

    Returns
    -------
    gather : collector.ArrayCollector
        A collection of all the variables.
    """
    gather = self._vars_in_container(self.child_ds, method=method)
    gather.update(super(Container, self).vars(method=method))
    return gather

  def nodes(self, method='absolute', _paths=None):
    if _paths is None:
      _paths = set()
    gather = self._nodes_in_container(self.child_ds, method=method, _paths=_paths)
    gather.update(super(Container, self).nodes(method=method, _paths=_paths))
    return gather

  def __getattr__(self, item):
    children_ds = super(Container, self).__getattribute__('child_ds')
    if item in children_ds:
      return children_ds[item]
    else:
      return super(Container, self).__getattribute__(item)


