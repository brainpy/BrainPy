# -*- coding: utf-8 -*-

import gc
from collections import OrderedDict

from brainpy import backend
from brainpy import errors
from brainpy.backend import ops
from brainpy.simulation import utils
from brainpy.simulation.monitors import Monitor

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
  monitors : list, tuple, optional
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  show_code : bool
      Whether show the formatted codes.
  """
  target_backend = None

  def __init__(self, steps=None, monitors=None, name=None, show_code=False):
    # for Container Objects
    self.contained_members = OrderedDict()

    # all the children nodes: instances of DynamicSystem
    # which can be added by the attribute setting
    self.children_nodes = dict()

    # step functions
    # --------------
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

    # name
    # ----
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
    # ---------
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

    # runner
    # -------
    self.driver = backend.get_ds_driver()(target=self)

    # run function
    # ------------
    self.run_func = None

    # show code
    # ---------
    self.show_code = show_code

    # target backend
    # --------------
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

  def build(self,
            duration,
            inputs=(),
            inputs_is_formatted=False,
            return_format_code=False,
            show_code=False):
    """Build the object for running.

    Parameters
    ----------
    inputs : list, tuple, optional
        The object inputs.
    inputs_is_formatted : bool
        Whether the "inputs" is formatted.
    return_format_code : bool
        Whether return the formatted codes.
    duration : int, float, list of int, list of float, tuple of int, tuple of float
        The running duration.
    show_code : bool
        Whether show the code.

    Returns
    -------
    calls : list, tuple
        The code lines to call step functions.
    """

    # build input functions

    # build monitor functions

    mon_length = utils.get_run_length_by_duration(duration)
    if (self._target_backend[0] != 'general') and \
        (backend.get_backend_name() not in self._target_backend):
      raise errors.ModelDefError(f'The model {self.name} is target '
                                 f'to run on {self._target_backend}, '
                                 f'but currently the selected backend '
                                 f'is {backend.get_backend_name()}')
    if not inputs_is_formatted:
      inputs = utils.format_pop_level_inputs(inputs=inputs,
                                             host=self,
                                             duration=duration)
    return self.driver.build(formatted_inputs=inputs,
                             mon_length=mon_length,
                             return_format_code=return_format_code,
                             show_code=(self.show_code or show_code))

  def run(self, duration, inputs=(), report=False, report_percent=0.1):
    """The running function.

    Parameters
    ----------
    duration : float, int, tuple, list
        The running duration.
    inputs : list, tuple
        The model inputs with the format of ``[(key, value [operation])]``.
    report : bool
        Whether report the running progress.
    report_percent : float
        The percent of progress to report.
    """

    # times
    # ------
    start, end = utils.check_duration(duration)
    times = ops.arange(start, end, backend.get_dt())

    # build run function
    # ------------------
    self.run_func = self.build(inputs=inputs,
                               inputs_is_formatted=False,
                               duration=duration,
                               return_format_code=False)

    # run the model
    # -------------
    res = utils.run_model(self.run_func, times, report, report_percent)
    self.mon.ts = times
    return res

  def schedule(self):
    """Schedule the running order of the update functions.

    Returns
    -------
    schedules : tuple
        The running order of update functions.
    """
    steps = tuple()
    for member in self.contained_members.values():
      steps += tuple(member.steps.keys())
    steps += self.steps.keys()
    return ('input',) + steps + ('monitor',)

  def run2(self, duration, inputs=(), report=False, report_percent=0.1):
    """Run the simulation for the given duration.

    This function provides the most convenient way to run the network.
    For example:

    Parameters
    ----------
    duration : int, float, tuple, list
        The amount of simulation time to run for.
    inputs : list, tuple
        The receivers, external inputs and durations.
    report : bool
        Report the progress of the simulation.
    report_percent : float
        The speed to report simulation progress.
    """
    # preparation
    start, end = utils.check_duration(duration)
    dt = backend.get_dt()
    ts = ops.arange(start, end, dt)

    # build the network
    run_length = ts.shape[0]
    format_inputs = utils.format_net_level_inputs(inputs, run_length)
    self.run_func = self.driver.build(duration=duration,
                                      formatted_inputs=format_inputs,
                                      show_code=self.show_code)

    # run the network
    res = utils.run_model(self.run_func, times=ts, report=report,
                          report_percent=report_percent)

    # end
    for obj in self.contained_members.values():
      if obj.mon.num_item > 0:
        obj.mon.ts = ts
    return res

  def schedule2(self):
    """Network scheduling in a network.
    """
    for node in self.contained_members.values():
      for key in node.schedule():
        yield f'{node.name}.{key}'

  def __setattr__(self, key, value):
    if isinstance(value, DynamicSystem):
      self.children_nodes[key] = value
    else:
      object.__setattr__(self, key, value)

  # def __del__(self):
  #     pass
  #     # delete monitors
  #     if self.mon is not None:
  #         del self.mon
  #     # delete children
  #     for val in self.members.values():
  #         val.__del__()
  #     # delete driver
  #     del self.driver
  #     # delete self
  #     del self
  #     # garbage collection
  #     gc.collect()
