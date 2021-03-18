# -*- coding: utf-8 -*-

import abc
from brainpy import errors


__all__ = [
    'AbstractRunner',
    'NodeRunner',
    'NetRunner',
]


class AbstractRunner(abc.ABC):
    """
    Abstract base class for backend runner.
    """
    @abc.abstractmethod
    def build(self, *args, **kwargs):
        pass


class NodeRunner(AbstractRunner):
    """
    Abstract Node Runner.
    """
    def __init__(self, host, steps):
        self.host = host
        assert isinstance(steps, (list, tuple)) and callable(steps[0])
        self.steps = steps
        self.step_names = [step.__name__ for step in steps]
        self.schedule = ['input'] + self.step_names + ['monitor']

    def get_schedule(self):
        return self.schedule

    def set_schedule(self, schedule):
        if not isinstance(schedule, (list, tuple)):
            raise errors.ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + self.step_names
        for s in schedule:
            if s not in all_func_names:
                raise errors.ModelUseError(f'Unknown step function "{s}" for model "{self.state}".')
        self.schedule = schedule

    @abc.abstractmethod
    def set_data(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_input_func(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_monitor_func(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_steps_func(self, *args, **kwargs):
        pass


class NetRunner(AbstractRunner):
    """
    Abstract Network Runner.
    """
    def __init__(self, all_nodes):
        self.all_nodes = all_nodes
