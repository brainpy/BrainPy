# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import backend
from brainpy import errors
from brainpy.simulation import constants
from brainpy.simulation import delay
from brainpy.simulation import utils
from brainpy.simulation.monitors import Monitor

__all__ = [
    'Population',
    'NeuGroup',
    'SynConn',
    'TwoEndConn',
]

_POPULATION_NO = 0
_NeuGroup_NO = 0
_TwoEndSyn_NO = 0


class Population(object):
    """Base Population Class.

    Parameters
    ----------
    name : str
        The name of the (neurons/synapses) ensemble.
    steps : callable, list of callable
        The callable function, or a list of callable functions.
    monitors : list, tuple, None
        Variables to monitor.
    pop_type : str
        Class type.
    """

    target_backend = None

    def __init__(self, steps, monitors=None, pop_type=None, name=None, host=None, show_code=False):
        # host of the data
        # ----------------
        if host is None:
            host = self
        self.host = host

        # ensemble type
        # -------------
        if pop_type is None:
            pop_type = constants.UNKNOWN_TYPE
        if pop_type not in constants.SUPPORTED_TYPES:
            print(f'Ensemble type {pop_type} is not registered in BrainPy. Currently, '
                  f'BrainPy has recognized "{constants.SUPPORTED_TYPES}".')
        self.ensemble_type = pop_type

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
            global _POPULATION_NO
            name = f'POP{_POPULATION_NO}'
            _POPULATION_NO += 1
        if not name.isidentifier():
            raise errors.ModelUseError(
                f'"{name}" isn\'t a valid identifier according to Python '
                f'language definition. Please choose another name.')
        self.name = name

        # monitors
        # ---------
        if monitors is None:
            monitors = []
        self.mon = Monitor(monitors)
        for var in self.mon['vars']:
            if not hasattr(self, var):
                raise errors.ModelDefError(f"Item {var} isn't defined in model {self}, "
                                           f"so it can not be monitored.")

        # runner
        # -------
        self.runner = backend.get_node_runner()(pop=self)

        # run function
        # ------------
        self.run_func = None

        # others
        # ---
        self.show_code = show_code
        if self.target_backend is None:
            raise errors.ModelDefError('Must define "target_backend".')
        if isinstance(self.target_backend, str):
            self.target_backend = [self.target_backend]
        assert isinstance(self.target_backend, (tuple, list)), 'target_backend must be a list/tuple.'

    def build(self, inputs, input_is_formatted=False, return_code=True, mon_length=0, show_code=False):
        """Build the object for running.

        Parameters
        ----------
        inputs : list, tuple, optional
            The object inputs.
        return_code : bool
            Whether return the formatted codes.
        mon_length : int
            The monitor length.

        Returns
        -------
        calls : list, tuple
            The code lines to call step functions.
        """
        if (self.target_backend[0] != 'general') and (backend.get_backend() not in self.target_backend):
            raise errors.ModelDefError(f'The model {self.name} is target to run on {self.target_backend},'
                                       f'but currently the default backend of BrainPy is '
                                       f'{backend.get_backend()}')
        if not input_is_formatted:
            inputs = utils.format_pop_level_inputs(inputs, self, mon_length)
        return self.runner.build(formatted_inputs=inputs,
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
        times = backend.arange(start, end, backend.get_dt())
        run_length = backend.shape(times)[0]

        # build run function
        # ------------------
        self.run_func = self.build(inputs, input_is_formatted=False, mon_length=run_length, return_code=False)

        # run the model
        # -------------
        utils.run_model(self.run_func, times, report, report_percent)
        self.mon['ts'] = times

    def get_schedule(self):
        """Get the schedule (running order) of the update functions.

        Returns
        -------
        schedule : list, tuple
            The running order of update functions.
        """
        return self.runner.get_schedule()

    def set_schedule(self, schedule):
        """Set the schedule (running order) of the update functions.

        For example, if the ``self.model`` has two step functions: `step1`, `step2`.
        Then, you can set the shedule by using:

        >>> pop = Population(...)
        >>> pop.set_schedule(['input', 'step1', 'step2', 'monitor'])
        """
        self.runner.set_schedule(schedule)

    def __str__(self):
        return self.name


class NeuGroup(Population):
    """Neuron Group.

    Parameters
    ----------
    steps : NeuType
        The instantiated neuron type model.
    size : int, tuple
        The neuron group geometry.
    monitors : list, tuple
        Variables to monitor.
    name : str
        The name of the neuron group.
    """

    def __init__(self, steps, size, monitors=None, name=None, host=None, show_code=False):
        # name
        # -----
        if name is None:
            name = ''
        else:
            name = '_' + name
        global _NeuGroup_NO
        _NeuGroup_NO += 1
        name = f'NG{_NeuGroup_NO}{name}'

        # size
        # ----
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise errors.ModelDefError('size must be int, or a tuple/list of int.')
            if not isinstance(size[0], int):
                raise errors.ModelDefError('size must be int, or a tuple/list of int.')
            size = tuple(size)
        elif isinstance(size, int):
            size = (size,)
        else:
            raise errors.ModelDefError('size must be int, or a tuple/list of int.')
        self.size = size

        # initialize
        # ----------
        super(NeuGroup, self).__init__(steps=steps,
                                       monitors=monitors,
                                       name=name,
                                       host=host,
                                       pop_type=constants.NEU_GROUP_TYPE,
                                       show_code=show_code)


class SynConn(Population):
    """Synaptic Connections.
    """

    def __init__(self, steps, **kwargs):
        # check delay update
        if callable(steps):
            steps = OrderedDict([(steps.__name__, steps)])
        elif isinstance(steps, (tuple, list)) and callable(steps[0]):
            steps = OrderedDict([(step.__name__, step) for step in steps])
        else:
            assert isinstance(steps, dict)
        if hasattr(self, 'constant_delays'):
            for key, delay_var in self.constant_delays.items():
                if delay_var.update not in steps:
                    delay_name = f'{key}_delay_update'
                    setattr(self, delay_name, delay_var.update)
                    steps[delay_name] = delay_var.update
        super(SynConn, self).__init__(steps=steps, **kwargs)

        for key, delay_var in self.constant_delays.items():
            delay_var.name = f'{self.name}_delay_{key}'

    def register_constant_delay(self, key, size, delay_time):
        if not hasattr(self, 'constant_delays'):
            self.constant_delays = {}
        if key in self.constant_delays:
            raise errors.ModelDefError(f'"{key}" has been registered as an constant delay.')
        self.constant_delays[key] = delay.ConstantDelay(size, delay_time)
        return self.constant_delays[key]


class TwoEndConn(SynConn):
    """Two End Synaptic Connections.

    Parameters
    ----------
    steps : SynType
        The instantiated neuron type model.
    pre : neurons.NeuGroup, neurons.NeuSubGroup
        Pre-synaptic neuron group.
    post : neurons.NeuGroup, neurons.NeuSubGroup
        Post-synaptic neuron group.
    monitors : list, tuple
        Variables to monitor.
    name : str
        The name of the neuron group.
    """

    def __init__(self, steps, pre, post, monitors=None, name=None, host=None, show_code=False):
        # name
        # ----
        if name is None:
            name = ''
        else:
            name = '_' + name
        global _TwoEndSyn_NO
        _TwoEndSyn_NO += 1
        name = f'TEC{_TwoEndSyn_NO}{name}'

        # pre or post neuron group
        # ------------------------
        if not isinstance(pre, NeuGroup):
            raise errors.ModelUseError('"pre" must be an instance of NeuGroup.')
        self.pre = pre
        if not isinstance(post, NeuGroup):
            raise errors.ModelUseError('"post" must be an instance of NeuGroup.')
        self.post = post

        # initialize
        # ----------
        super(TwoEndConn, self).__init__(steps=steps,
                                         name=name,
                                         monitors=monitors,
                                         pop_type=constants.TWO_END_TYPE,
                                         host=host,
                                         show_code=show_code)
