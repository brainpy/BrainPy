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
    host : any
        The host to store data, including variables, functions, etc.
    show_code : bool
        Whether show the formatted codes.
    """

    target_backend = None

    def __init__(self, steps, monitors=None, name=None, host=None, show_code=False):
        # host of the data
        # ----------------
        if host is None:
            host = self
        self.host = host

        # model
        # -----
        if callable(steps):
            self.steps = OrderedDict([(steps.__name__, steps)])
        elif isinstance(steps, (list, tuple)) and callable(steps[0]):
            self.steps = OrderedDict([(step.__name__, step) for step in steps])
        elif isinstance(steps, dict):
            self.steps = steps
        else:
            raise errors.ModelDefError(f'Unknown model type: {type(steps)}. Currently, BrainPy '
                                       f'only supports: function, list/tuple/dict of functions.')

        # name
        # ----
        if name is None:
            global _DynamicSystem_NO
            name = f'DS{_DynamicSystem_NO}'
            _DynamicSystem_NO += 1
        if not name.isidentifier():
            raise errors.ModelUseError(f'"{name}" isn\'t a valid identifier according to Python '
                                       f'language definition. Please choose another name.')
        self.name = name

        # monitors
        # ---------
        if monitors is None:
            monitors = []
        self.mon = Monitor(target=self, variables=monitors)

        # runner
        # -------
        self.driver = backend.get_node_driver()(pop=self)

        # run function
        # ------------
        self.run_func = None

        # others
        # ---
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

    def build(self, inputs, inputs_is_formatted=False, return_code=True, mon_length=0, show_code=False):
        """Build the object for running.

        Parameters
        ----------
        inputs : list, tuple, optional
            The object inputs.
        inputs_is_formatted : bool
            Whether the "inputs" is formatted.
        return_code : bool
            Whether return the formatted codes.
        mon_length : int
            The monitor length.

        Returns
        -------
        calls : list, tuple
            The code lines to call step functions.
        """
        if (self._target_backend[0] != 'general') and (backend.get_backend_name() not in self._target_backend):
            raise errors.ModelDefError(f'The model {self.name} is target to run on {self._target_backend}, '
                                       f'but currently the selected backend is {backend.get_backend_name()}')
        if not inputs_is_formatted:
            inputs = utils.format_pop_level_inputs(inputs, self, mon_length)
        return self.driver.build(formatted_inputs=inputs,
                                 mon_length=mon_length,
                                 return_code=return_code,
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
        run_length = ops.shape(times)[0]

        # build run function
        # ------------------
        self.run_func = self.build(inputs, inputs_is_formatted=False, mon_length=run_length, return_code=False)

        # run the model
        # -------------
        res = utils.run_model(self.run_func, times, report, report_percent)
        self.mon.ts = times
        return res

    def get_schedule(self):
        """Get the schedule (running order) of the update functions.

        Returns
        -------
        schedule : list, tuple
            The running order of update functions.
        """
        return self.driver.get_schedule()

    def set_schedule(self, schedule):
        """Set the schedule (running order) of the update functions.

        For example, if the ``self.model`` has two step functions: `step1`, `step2`.
        Then, you can set the shedule by using:

        >>> pop = DynamicSystem(...)
        >>> pop.set_schedule(['input', 'step1', 'step2', 'monitor'])
        """
        self.driver.set_schedule(schedule)

