# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import backend
from brainpy.backend import ops
from brainpy.simulation import utils
from brainpy.simulation.dynamic_system import DynamicSystem
from .neu_group import NeuGroup
from .syn_conn import SynConn

__all__ = [
    'Network',
]


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

        # driver
        self.driver = backend.get_net_driver()(host=self)

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
                             f'{NeuGroup.__name__} and {SynConn.__name__}.')
        # 2. check object name
        name = obj.name if name is None else name
        if name in self.all_nodes:
            raise KeyError(f'Name "{name}" has been used in the network, '
                           f'please change another name.')
        # 3. add object to the network
        self.all_nodes[name] = obj

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
        ts = ops.arange(start, end, dt)

        # build the network
        run_length = ts.shape[0]
        format_inputs = utils.format_net_level_inputs(inputs, run_length)
        self.run_func = self.driver.build(run_length=run_length,
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
        return ops.arange(self.t_start, self.t_end, backend.get_dt())
