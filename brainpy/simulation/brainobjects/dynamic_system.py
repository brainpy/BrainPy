# -*- coding: utf-8 -*-

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

    Parameters
    ----------
    steps : callable, list of callable, dict
        The callable function, or a list of callable functions.
    monitors : list, tuple, None
        Variables to monitor.
    name : str
        The name of the dynamic system.
    show_code : bool
        Whether show the formatted codes.
    """

    target_backend = None

    def __init__(self, steps, monitors=None, name=None, show_code=False):
        # model
        # -----
        if callable(steps):
            self.steps = OrderedDict([(steps.__name__, steps)])
        elif isinstance(steps, (list, tuple)) and callable(steps[0]):
            self.steps = OrderedDict([(step.__name__, step) for step in steps])
        elif isinstance(steps, dict):
            self.steps = OrderedDict(steps)
        else:
            raise errors.ModelDefError(f'Unknown model type: {type(steps)}. Currently, '
                                       f'BrainPy only supports: function, list/tuple/dict '
                                       f'of functions.')

        # name
        # ----
        if name is None:
            global _DynamicSystem_NO
            name = f'DS{_DynamicSystem_NO}'
            _DynamicSystem_NO += 1
        if not name.isidentifier():
            raise errors.ModelUseError(f'"{name}" isn\'t a valid identifier according to '
                                       f'Python language definition. Please choose another '
                                       f'name.')
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
            raise errors.ModelUseError(f'"monitors" only supports list/tuple/dict/'
                                       f'instance of Monitor, not {type(monitors)}.')

        # runner
        # -------
        self.driver = backend.get_node_driver()(target=self)

        # run function
        # ------------
        self.run_func = None

        # others
        # ------
        self.show_code = show_code
        if self.target_backend is None:
            raise errors.ModelDefError('Must define "target_backend".')
        if isinstance(self.target_backend, str):
            self._target_backend = (self.target_backend, )
        elif isinstance(self.target_backend, (tuple, list)):
            if not isinstance(self.target_backend[0], str):
                raise errors.ModelDefError('"target_backend" must be a list/tuple of string.')
            self._target_backend = tuple(self.target_backend)
        else:
            raise errors.ModelDefError(f'Unknown setting of "target_backend": {self.target_backend}')

    def build(self, duration, inputs=(), inputs_is_formatted=False, return_format_code=False, show_code=False):
        """Build the object for running.

        Parameters
        ----------
        inputs : list, tuple, optional
            The object inputs.
        inputs_is_formatted : bool
            Whether the "inputs" is formatted.
        return_format_code : bool
            Whether return the formatted codes.
        duration : int, float, list, tuple
            The running duration.
        show_code : bool
            Whether show the code.

        Returns
        -------
        calls : list, tuple
            The code lines to call step functions.
        """
        mon_length = utils.get_run_length_by_duration(duration)
        if (self._target_backend[0] != 'general') and (backend.get_backend_name() not in self._target_backend):
            raise errors.ModelDefError(f'The model {self.name} is target to run on {self._target_backend}, '
                                       f'but currently the selected backend is {backend.get_backend_name()}')
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
        return ('input', ) + tuple(self.steps.keys()) + ('monitor',)
