# -*- coding: utf-8 -*-

from collections import OrderedDict

from brainpy import errors
from brainpy.simulation import delay
from brainpy.simulation.dynamic_system import DynamicSystem
from .neu_group import NeuGroup

__all__ = [
    'SynConn',
    'TwoEndConn',
]

_TwoEndSyn_NO = 0


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
