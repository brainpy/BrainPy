# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

from brainpy import math, errors
from brainpy.integrators.constants import DE_PREFIX
from brainpy.primary import Primary, collector
from brainpy.simulation import utils
from brainpy.simulation.driver import get_driver
from brainpy.simulation.monitor import Monitor

__all__ = [
  'DynamicSystem',
  'Container',
]

_error_msg = 'Unknown model type: {type}. ' \
             'Currently, BrainPy only supports: ' \
             'function, tuple/dict of functions, ' \
             'tuple of function names.'


def _find_integrator(obj):
  for key in dir(obj):
    value = getattr(obj, key)
    if callable(value) and hasattr(value, '__name__') and value.__name__.startswith(DE_PREFIX):
      yield value


class DynamicSystem(Primary):
  """Base Dynamic System Class.

  Any object has iterable step functions will be a dynamical system.
  That is to say, in BrainPy, the essence of the dynamical system
  is the "steps".

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
    super(DynamicSystem, self).__init__(name=name)

    # runner and run function
    self.driver = None
    self.run_func = None
    self.input_step = lambda _t, _i: None
    self.monitor_step = lambda _t, _i: None

    # step functions
    self.steps = OrderedDict()
    steps = tuple() if steps is None else steps
    if isinstance(steps, tuple):
      for step in steps:
        if isinstance(step, str):
          self.steps[step] = getattr(self, step)
        elif callable(step):
          self.steps[step.__name__] = step
        else:
          raise errors.ModelDefError(_error_msg.format(type(steps[0])))
    elif isinstance(steps, dict):
      for key, step in steps.items():
        if callable(step):
          self.steps[key] = step
        else:
          raise errors.ModelDefError(_error_msg.format(type(step)))
    else:
      raise errors.ModelDefError(_error_msg.format(type(steps)))

    # monitors
    if monitors is None:
      self.mon = Monitor(target=self, variables=[])
    elif isinstance(monitors, (list, tuple, dict)):
      self.mon = Monitor(target=self, variables=monitors)
    elif isinstance(monitors, Monitor):
      self.mon = monitors
      self.mon.target = self
    else:
      raise errors.ModelUseError(f'"monitors" only supports '
                                 f'list/tuple/dict/ instance '
                                 f'of Monitor, not {type(monitors)}.')

    # target backend
    if self.target_backend is None:
      self._target_backend = ('general',)
    elif isinstance(self.target_backend, str):
      self._target_backend = (self.target_backend,)
    elif isinstance(self.target_backend, (tuple, list)):
      if not isinstance(self.target_backend[0], str):
        raise errors.ModelDefError('"target_backend" must be a '
                                   'list/tuple of string.')
      self._target_backend = tuple(self.target_backend)
    else:
      raise errors.ModelDefError(f'Unknown setting of '
                                 f'"target_backend": '
                                 f'{self.target_backend}')

  def ints(self, method='absolute'):
    """Collect all the integrators in the instance
    of DynamicSystem and the node instances.

    Parameters
    ----------
    method : str
      The prefix string for the integrator names.

    Returns
    -------
    collector : collector.Collector
      The collection contained the integrator name and the integrator function.
    """
    gather = collector.Collector()
    if method == 'relative':
      for k in dir(self):
        v = getattr(self, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(DE_PREFIX):
          gather[k] = v
        elif isinstance(v, DynamicSystem):
          for k2, v2 in v.ints(method=method).items():
            gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k in dir(self):
        v = getattr(self, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(DE_PREFIX):
          gather[f'{self.name}.{k}'] = v
        elif isinstance(v, DynamicSystem):
          gather.update(v.ints(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def _build(self, inputs, duration, rebuild=False):
    # backend checking
    check1 = self._target_backend[0] != 'general'
    check2 = math.get_backend_name() not in self._target_backend
    if check1 and check2:
      raise errors.ModelDefError(f'The model {self.name} is target '
                                 f'to run on {self._target_backend}, '
                                 f'but currently the selected backend '
                                 f'is {math.get_backend_name()}')

    # build main function the monitor function
    if self.driver is None:
      self.driver = get_driver()(self)
    formatted_inputs = utils.format_inputs(host=self,
                                           duration=duration,
                                           inputs=inputs)
    # build monitor and input functions
    self.driver.build(rebuild=rebuild, inputs=formatted_inputs)

  def _step_run(self, _t, _i):
    self.monitor_step(_t, _i)
    self.input_step(_t, _i)
    for step in self.steps.values():
      step(_t, _i)

  def run(self, duration, report=0., inputs=(), rebuild=False):
    """The running function.

    Parameters
    ----------
    inputs : list, tuple
      The inputs for this instance of DynamicSystem. It should the format
      of `[(target, value, [type, operation])]`, where `target` is the
      input target, `value` is the input value, `type` is the input type
      (such as "fixed" or "iter"), `operation` is the operation for inputs
      (such as "+", "-", "*", "/").
    duration : float, int, tuple, list
      The running duration.
    report : float
      The percent of progress to report. [0, 1]. If zero, the model
      will not output report progress.
    rebuild : bool
      Rebuild the running function.
    """

    # times
    start, end = utils.check_duration(duration)
    times = math.arange(start, end, math.get_dt())

    # build functions of monitor and inputs
    # ---
    # 1. build the inputs. All the inputs are wrapped into a single function.
    # 2. build the monitors. All the monitors are wrapped in a single function.
    self._build(duration=duration, inputs=inputs, rebuild=rebuild)

    # run the model
    # ----
    # 1. iteratively run the step function.
    # 2. report the running progress.
    # 3. return the overall running time.
    running_time = utils.run_model(run_func=self._step_run, times=times, report=report)

    # monitor
    # --
    # 1. add 'ts' variable to every monitor
    # 2. wrap the monitor iterm with the 'list' type
    #    into the 'ndarray' type
    times = times.numpy() if not isinstance(times, np.ndarray) else times
    for node in [self] + list(self.nodes().unique_values()):
      if node.mon.num_item > 0:
        node.mon.ts = times
      for key, val in list(node.mon.item_contents.items()):
        val = math.array(node.mon.item_contents[key])
        if not isinstance(val, np.ndarray):
          val = val.numpy()
        node.mon.item_contents[key] = val

    return running_time


class Container(DynamicSystem, dict):
  """Container object which is designed to add other DynamicalSystem instances.

  What's different from the other DynamicSystem objects is that Container has
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
  dynamic_systems : dict of (str, )
      The instance of DynamicSystem with the format of "key=dynamic_system".
  """

  def __init__(self, steps=None, monitors=None, name=None, **dynamic_systems):
    # initialize "dict"
    for val in dynamic_systems.values():
      if not isinstance(val, DynamicSystem):
        raise errors.ModelUseError(f'{self.__class__.__name__} receives '
                                   f'instances of DynamicSystem, however, '
                                   f'we got {type(val)}.')
    dict.__init__(self, **dynamic_systems)

    # check 'monitors'
    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    # initialize "DynamicSystem"
    if steps is None:
      steps = OrderedDict()
      for obj_key, obj in dynamic_systems.items():
        for step_key, step in obj.steps.items():
          steps[f'{obj_key}_{step_key}'] = step
    DynamicSystem.__init__(self, steps=steps, monitors=monitors, name=name)

  def vars(self, method='absolute'):
    """Collect all the variables (and their names) contained
    in the list and its children instance of DynamicSystem.

    Parameters
    ----------
    method : str
      string to prefix to the variable names.

    Returns
    -------
    gather collector.ArrayCollector
        A collection of all the variables.
    """
    gather = collector.ArrayCollector()
    if method == 'relative':
      for k, v in self.items():
        for k2, v2 in v.vars(method=method).items():
          gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.items():
        gather.update(v.vars(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    gather.update(super(Container, self).vars(method=method))
    return gather

  def ints(self, method='absolute'):
    gather = collector.Collector()
    if method == 'relative':
      for k, v in self.items():
        for k2, v2 in v.ints(method=method).items():
          gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.items():
        gather.update(v.ints(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    gather.update(super(Container, self).ints(method=method))
    return gather

  def nodes(self, method='absolute'):
    gather = collector.Collector()
    if method == 'relative':
      for k, v in self.items():
        gather[k] = v
        for k2, v2 in v.nodes(method=method).items():
          gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.items():
        gather[v.name] = v
        gather.update(v.nodes(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    gather.update(super(Container, self).nodes(method=method))
    return gather

  def __getattr__(self, item):
    if item in self:
      return self[item]
    else:
      return super(Container, self).__getattribute__(item)
