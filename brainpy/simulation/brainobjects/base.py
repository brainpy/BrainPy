# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import backend
from brainpy import errors
from brainpy.backend import math
from brainpy.integrators.integrators import Integrator
from brainpy.simulation import utils
from brainpy.simulation.monitor import Monitor
from brainpy.simulation.drivers import BaseDSDriver
from brainpy.tools.collector import Collector

__all__ = [
  'DynamicSystem',
]

_DynamicSystem_NO = 0


class DynamicSystem(object):
  """Base Dynamic System Class.

  Any object has iterable step functions will be a dynamical system.
  That is to say, in BrainPy, the essence of the dynamical system
  is the "steps".

  Parameters
  ----------
  steps : function, list of function, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """
  _KEYWORDS = ['vars', 'pars', 'ints', 'nodes']

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
    collector : datastructures.Collector
      The collection contained the variable name and the variable data.
    """
    collector = Collector()
    prefix += f'({self.name}).'
    for k, v in self.__dict__.items():
      if isinstance(v, math.ndarray):
        collector[prefix + k] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.vars(prefix=prefix[:-1] if k == 'raw' else prefix + k))
    return collector

  def ints(self, prefix=''):
    """Collect all the integrators in the instance
    of DynamicSystem and the node instances.

    Parameters
    ----------
    prefix : str
      The prefix string for the integrator names.

    Returns
    -------
    collector : datastructures.Collector
      The collection contained the integrator name and the integrator function.
    """
    collector = Collector()
    prefix += f'({self.name}).'
    for k, v in self.__dict__.items():
      if isinstance(v, Integrator):
        collector[prefix + k] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.ints(prefix=prefix[:-1] if k == 'raw' else prefix + k))
    return collector

  def nodes(self, prefix=''):
    """Collect all the nodes in the instance
    of DynamicSystem.

    Parameters
    ----------
    prefix : str
      The prefix string for the integrator names.

    Returns
    -------
    collector : datastructures.Collector
      The collection contained the integrator name and the integrator function.
    """
    collector = Collector()
    prefix += f'{self.name}.'
    for k, v in self.__dict__.items():
      if isinstance(v, DynamicSystem):
        collector[v.name] = v
        collector[prefix + f'{k}'] = v
        collector.update(v.nodes(prefix + f'{k}.'))
    return collector

  def __init__(self, steps=None, monitors=None, name=None):
    # runner and run function
    self.driver = None
    self.run_func = None
    self.input_step = lambda _t, _i: None
    self.monitor_step = lambda _t, _i: None

    # step functions
    self.steps = OrderedDict()
    if steps is not None:
      if callable(steps):
        self.steps[steps.__name__] = steps
      elif isinstance(steps, (list, tuple)) and callable(steps[0]):
        for step in steps:
          self.steps[step.__name__] = step
      elif isinstance(steps, dict):
        self.steps.update(steps)
      else:
        raise errors.ModelDefError(f'Unknown model type: {type(steps)}. '
                                   f'Currently, BrainPy only supports: '
                                   f'function, list/tuple/dict of functions.')
    else:
      self.steps['call'] = self.update

    # name : useful in Numba Backend
    if name is None:
      global _DynamicSystem_NO
      name = f'DS{_DynamicSystem_NO}'
      _DynamicSystem_NO += 1
    if not name.isidentifier():
      raise errors.ModelUseError(f'"{name}" isn\'t a valid identifier '
                                 f'according to Python language definition. '
                                 f'Please choose another name.')
    self.name = name

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
      raise errors.ModelDefError('Must define "target_backend".')
    if isinstance(self.target_backend, str):
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
    check2 = backend.get_backend_name() not in self._target_backend
    if check1 and check2:
      raise errors.ModelDefError(f'The model {self.name} is target '
                                 f'to run on {self._target_backend}, '
                                 f'but currently the selected backend '
                                 f'is {backend.get_backend_name()}')

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
    times = math.arange(start, end, backend.get_dt())

    # build run function
    self.run_func = self._build(duration=duration, inputs=inputs, rebuild=rebuild)

    # run the model
    return utils.run_model(run_func=self.run_func, times=times, report=report)
