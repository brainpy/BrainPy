# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import math, errors, backend
from brainpy.backend.base import BaseDSDriver
from brainpy.integrators.integrators import Integrator
from brainpy.simulation import checking
from brainpy.simulation import collector
from brainpy.simulation import utils
from brainpy.simulation.monitor import Monitor

__all__ = [
  'DynamicSystem',
]

_error_msg = 'Unknown model type: {type}. ' \
             'Currently, BrainPy only supports: ' \
             'function, tuple/dict of functions, ' \
             'tuple of function names.'


class DynamicSystem(object):
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
  _KEYWORDS = ['vars', 'ints', 'nodes']

  target_backend = None

  def vars(self, prefix=''):
    """Collect all the variables in the instance of
    DynamicSystem and the node instances.

    Parameters
    ----------
    prefix : str
      The prefix string for the variable names.

    Returns
    -------
    gather : datastructures.ArrayCollector
      The collection contained the variable name and the variable data.
    """
    gather = collector.ArrayCollector()
    for k, v in self.__dict__.items():
      if isinstance(v, math.ndarray):
        gather[prefix + k] = v
        gather[f'{self.name}.{k}'] = v
      elif isinstance(v, DynamicSystem):
        gather.update(v.vars(prefix=prefix[:-1] if k == 'raw' else f'{prefix}{k}.'))
    return gather

  def ints(self, prefix=''):
    """Collect all the integrators in the instance
    of DynamicSystem and the node instances.

    Parameters
    ----------
    prefix : str
      The prefix string for the integrator names.

    Returns
    -------
    collector : collector.Collector
      The collection contained the integrator name and the integrator function.
    """
    gather = collector.Collector()
    for k, v in self.__dict__.items():
      if isinstance(v, Integrator):
        gather[prefix + k] = v
        gather[f'{self.name}.k'] = v
      elif isinstance(v, DynamicSystem):
        gather.update(v.ints(prefix=prefix[:-1] if k == 'raw' else prefix + k))
    return gather

  def nodes(self, prefix=''):
    """Collect all the nodes in the instance
    of DynamicSystem.

    Parameters
    ----------
    prefix : str
      The prefix string for the node names.

    Returns
    -------
    collector : collector.Collector
      The collection contained the integrator name and the integrator function.
    """
    gather = collector.Collector()
    for k, v in self.__dict__.items():
      if isinstance(v, DynamicSystem):
        gather[prefix + k] = v
        gather[f'{self.name}.{k}'] = v
        gather[v.name] = v
        gather.update(v.nodes(f'{prefix}{k}.'))
    return gather

  def __init__(self, steps=('update',), monitors=None, name=None):
    # runner and run function
    self.driver = None
    self.run_func = None
    self.input_step = lambda _t, _i: None
    self.monitor_step = lambda _t, _i: None

    # step functions
    self.steps = OrderedDict()
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

    # check whether the object has a unique name.
    self.name = self.unique_name(name=name, type='DynamicSystem')
    checking.check_name(name=name, obj=self)

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

  def update(self, _t, _i):
    raise NotImplementedError

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
      self.driver = backend.get_ds_driver()
      self.driver = self.driver(self)
    assert isinstance(self.driver, BaseDSDriver)
    run_length = utils.get_run_length_by_duration(duration)
    formatted_inputs = utils.format_inputs(host=self, duration=duration, inputs=inputs)
    run_func = self.driver.build(rebuild=rebuild,
                                 run_length=run_length,
                                 inputs=formatted_inputs)
    return run_func

  def run(self, duration, report=0., inputs=(), rebuild=False):
    """The running function.

    Parameters
    ----------
    inputs : list, tuple
      The inputs for this instance of DynamicSystem. It should the format of
      `[(target, value, [type, operation])]`, where `target` is the input target,
      `value` is the input value, `type` is the input type (such as "fixed" or "iter"),
      `operation` is the operation for inputs (such as "+", "-", "*", "/").
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

    # build run function
    self.run_func = self._build(duration=duration, inputs=inputs, rebuild=rebuild)

    # run the model
    running_time = utils.run_model(run_func=self.run_func, times=times, report=report)

    # monitor for times
    for node in [self] + list(self.nodes().unique_values()):
      node.mon.ts = times

    return running_time

  def unique_name(self, name=None, type=None):
    if name is None:
      assert type is not None
      return checking.get_name(type=type)
    else:
      checking.check_name(name=name, obj=self)
      return name

