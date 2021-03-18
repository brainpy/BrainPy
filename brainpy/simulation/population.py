# -*- coding: utf-8 -*-

from brainpy import backend
from brainpy import connectivity
from brainpy import errors
from brainpy import profile
from brainpy.simulation import constants
from brainpy.simulation import utils
from brainpy.simulation.monitors import Monitor

__all__ = [
    'Population',
    'NeuGroup',
    'TwoEndConn',
]

_NeuGroup_NO = 0
_TwoEndSyn_NO = 0


class Population(object):
    """Base Population Class.

    Parameters
    ----------
    name : str
        The name of the (neurons/synapses) ensemble.
    size : int
        The number of the neurons/synapses.
    steps : function, list of function
        The callable function, or a list of callable functions.
    monitors : list, tuple, None
        Variables to monitor.
    ensemble_type : str
        Class type.
    """

    target_backend = None

    def __init__(self, size, steps, monitors, ensemble_type, name,
                 host=None, show_code=False):
        # host of the data
        # ----------------
        if host is None:
            host = self
        self.host = host

        # ensemble type
        # -------------
        if ensemble_type not in constants.SUPPORTED_TYPES:
            print(f'Ensemble type {ensemble_type} is not registered in BrainPy. Currently, '
                  f'BrainPy has recognized "{constants.SUPPORTED_TYPES}".')
        self.ensemble_type = ensemble_type

        # model
        # -----
        if callable(steps):
            self.steps = [steps]
        elif isinstance(steps, (list, tuple)) and callable(steps[0]):
            self.steps = list(steps)
        else:
            raise errors.ModelDefError(f'Unknown model type: {type(steps)}. Currently, BrainPy '
                                       f'only supports: function, list of functions.')

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

        # name
        # ----
        if not name.isidentifier():
            raise errors.ModelUseError(
                f'"{name}" isn\'t a valid identifier according to Python '
                f'language definition. Please choose another name.')
        self.name = name

        # monitors
        # ---------
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

    def build(self, format_inputs, return_code=True, mon_length=0):
        """Build the object for running.

        Parameters
        ----------
        format_inputs : list, tuple, optional
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
        if backend.get_backend() not in self.target_backend:
            raise errors.ModelDefError(f'The model {self.name} is target to run on {self.target_backend},'
                                       f'but currently the default backend of BrainPy is '
                                       f'{profile.get_backend()}')
        return self.runner.build(formatted_inputs=format_inputs,
                                 mon_length=mon_length,
                                 return_code=return_code,
                                 show_code=self.show_code)

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
        times = backend.arange(start, end, profile.get_dt())
        run_length = backend.shape(times)[0]

        # build run function
        # ------------------
        format_inputs = utils.format_pop_level_inputs(inputs, self, run_length, self.size)
        self.run_func = self.build(format_inputs, mon_length=run_length, return_code=False)

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

    def __init__(self, size, steps, monitors=None, name=None,
                 host=None, show_code=False):
        # name
        # -----
        if name is None:
            name = 'NeuGroup'
        global _NeuGroup_NO
        _NeuGroup_NO += 1
        name = f'NG{_NeuGroup_NO}_{name}'

        # initialize
        # ----------
        super(NeuGroup, self).__init__(size=size,
                                       steps=steps,
                                       monitors=monitors,
                                       name=name,
                                       host=host,
                                       ensemble_type=constants.NEU_GROUP_TYPE,
                                       show_code=show_code)


class TwoEndConn(Population):
    """Two End Synaptic Connections.

    Parameters
    ----------
    steps : SynType
        The instantiated neuron type model.
    pre : neurons.NeuGroup, neurons.NeuSubGroup
        Pre-synaptic neuron group.
    post : neurons.NeuGroup, neurons.NeuSubGroup
        Post-synaptic neuron group.
    conn : connectivity.Connector
        Connection method to create synaptic connectivity.
    monitors : list, tuple
        Variables to monitor.
    name : str
        The name of the neuron group.
    """

    def __init__(self, steps, pre=None, post=None, conn=None, monitors=None,
                 name=None, host=None, show_code=False):
        # name
        # ----
        if name is None:
            name = 'TwoEndConn'
        global _TwoEndSyn_NO
        _TwoEndSyn_NO += 1
        name = f'TEC{_TwoEndSyn_NO}_{name}'

        # pre or post neuron group
        # ------------------------
        self.pre = pre
        self.post = post
        self.conn = None
        if pre is not None and post is not None:
            if not isinstance(pre, NeuGroup):
                raise errors.ModelUseError('"pre" must be an instance of NeuGroup.')
            if not isinstance(post, NeuGroup):
                raise errors.ModelUseError('"post" must be an instance of NeuGroup.')

            if conn is not None:
                if isinstance(conn, connectivity.Connector):
                    self.conn = conn(pre.size, post.size)
                    self.conn = connectivity.Connector()

        size = 1  # TODO

        # initialize
        # ----------
        super(TwoEndConn, self).__init__(steps=steps,
                                         name=name,
                                         size=size,
                                         monitors=monitors,
                                         ensemble_type=constants.SYN_CONN_TYPE,
                                         host=host,
                                         show_code=show_code)
