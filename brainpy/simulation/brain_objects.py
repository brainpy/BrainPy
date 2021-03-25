# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import backend
from brainpy import errors
from brainpy.simulation import delay
from brainpy.simulation import utils
from brainpy.simulation.dynamic_system import DynamicSystem

__all__ = [
    'NeuGroup',
    'SynConn',
    'TwoEndConn',
    'Network',
]

_NeuGroup_NO = 0
_TwoEndSyn_NO = 0


class NeuGroup(DynamicSystem):
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

    def __init__(self, size, monitors=None, name=None, show_code=False):
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
        super(NeuGroup, self).__init__(steps={'update': self.update},
                                       monitors=monitors,
                                       name=name,
                                       show_code=show_code)

    def update(self, *args):
        raise NotImplementedError


class SynConn(DynamicSystem):
    """Synaptic Connections.
    """

    def __init__(self, steps, monitors=None, name=None, show_code=False):
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

        # initialize super class
        super(SynConn, self).__init__(steps=steps, monitors=monitors, name=name, show_code=show_code)

        # delay assignment
        if hasattr(self, 'constant_delays'):
            for key, delay_var in self.constant_delays.items():
                delay_var.name = f'{self.name}_delay_{key}'

    def register_constant_delay(self, key, size, delay_time):
        if not hasattr(self, 'constant_delays'):
            self.constant_delays = {}
        if key in self.constant_delays:
            raise errors.ModelDefError(f'"{key}" has been registered as an constant delay.')
        self.constant_delays[key] = delay.ConstantDelay(size, delay_time)
        return self.constant_delays[key]

    def update(self, *args):
        raise NotImplementedError


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

    def __init__(self, pre, post, monitors=None, name=None, show_code=False):
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
        super(TwoEndConn, self).__init__(steps={'update': self.update},
                                         name=name,
                                         monitors=monitors,
                                         show_code=show_code)


class Network(object):
    """The main simulation controller in ``BrainPy``.

    ``Network`` handles the running of a simulation. It contains a set
    of objects that are added with `add()`. The `run()` method actually
    runs the simulation. The main loop runs according to user add orders.
    The objects in the `Network` are accessible via their names, e.g.
    `net.name` would return the `object`.
    """

    def __init__(self, *args, show_code=False, **kwargs):
        # record the current step
        self.t_start = 0.
        self.t_end = 0.

        # store all nodes
        self.all_nodes = OrderedDict()

        # store the step function
        self.run_func = None
        self.show_code = show_code

        # add nodes
        self.add(*args, **kwargs)

    def __getattr__(self, item):
        if item in self.all_nodes:
            return self.all_nodes[item]
        else:
            return super(Network, self).__getattribute__(item)

    def _add_obj(self, obj, name=None):
        # 1. check object type
        if not isinstance(obj, DynamicSystem):
            raise ValueError(f'Unknown object type "{type(obj)}". '
                             f'Currently, Network only supports '
                             f'{NeuGroup.__name__} and '
                             f'{TwoEndConn.__name__}.')
        # 2. check object name
        name = obj.name if name is None else name
        if name in self.all_nodes:
            raise KeyError(f'Name "{name}" has been used in the network, '
                           f'please change another name.')
        # 3. add object to the network
        self.all_nodes[name] = obj
        if obj.name != name:
            self.all_nodes[obj.name] = obj

    def add(self, *args, **kwargs):
        """Add object (neurons or synapses) to the network.

        Parameters
        ----------
        args
            The nameless objects.
        kwargs
            The named objects, which can be accessed by `net.xxx`
            (xxx is the name of the object).
        """
        for obj in args:
            self._add_obj(obj)
        for name, obj in kwargs.items():
            self._add_obj(obj, name)

    def run(self, duration, inputs=(), report=False, report_percent=0.1):
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
        ts = backend.arange(start, end, dt)

        # build the network
        run_length = ts.shape[0]
        format_inputs = utils.format_net_level_inputs(inputs, run_length)
        net_runner = backend.get_net_runner()(all_nodes=self.all_nodes)
        self.run_func = net_runner.build(run_length=run_length,
                                         formatted_inputs=format_inputs,
                                         return_code=False,
                                         show_code=self.show_code)

        # run the network
        utils.run_model(self.run_func, times=ts, report=report, report_percent=report_percent)

        # end
        self.t_start, self.t_end = start, end
        for obj in self.all_nodes.values():
            if len(obj.mon['vars']) > 0:
                obj.mon['ts'] = ts

    @property
    def ts(self):
        """Get the time points of the network.
        """
        return backend.arange(self.t_start, self.t_end, backend.get_dt())
