# -*- coding: utf-8 -*-

import abc

from brainpy import errors

__all__ = [
    'AbstractDriver',
    'BaseNodeDriver',
    'BaseNetDriver',
    'BaseDiffIntDriver',
]


class AbstractDriver(abc.ABC):
    """
    Abstract base class for backend driver.
    """

    @abc.abstractmethod
    def build(self, *args, **kwargs):
        """Build the node or the network running function.
        """
        pass

    @abc.abstractmethod
    def upload(self, *args, **kwargs):
        """Upload the data or function to the node or the network.

        Establish the connection between the host and the driver. The
        driver can upload its specific data of functions to the host.
        Then, at the frontend of the host, users can call such functions
        or data by "host.func_name" or "host.some_data".
        """
        pass


class BaseNodeDriver(AbstractDriver):
    """Base Node Driver.
    """

    def __init__(self, host, steps):
        self.host = host
        self.steps = steps
        self.schedule = ['input'] + list(self.steps.keys()) + ['monitor']

    def upload(self, name, data_or_func):
        setattr(self.host, name, data_or_func)

    def get_schedule(self):
        """Get the running schedule.
        """
        return self.schedule

    def set_schedule(self, schedule):
        """Set the running schedule of the node.

        Parameters
        ----------
        schedule : list
            A list of the string, in which each item is the function name.
        """
        if not isinstance(schedule, (list, tuple)):
            raise errors.ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + list(self.steps.keys())
        for s in schedule:
            if s not in all_func_names:
                raise errors.ModelUseError(f'Unknown step function "{s}" for model "{self.host}".')
        self.schedule = schedule

    @abc.abstractmethod
    def get_input_func(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_monitor_func(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_steps_func(self, *args, **kwargs):
        pass


class BaseNetDriver(AbstractDriver):
    """Base Network Driver.
    """

    def __init__(self, host):
        self.host = host

    def upload(self, name, data_or_func):
        setattr(self.host, name, data_or_func)


class BaseDiffIntDriver(AbstractDriver):
    """Base Integration Driver for Differential Equations.
    """

    def __init__(self, code_scope, code_lines, func_name, uploads, show_code):
        self.code_scope = code_scope
        self.code_lines = code_lines
        self.func_name = func_name
        self.uploads = uploads
        self.show_code = show_code

    def upload(self, host, key, value):
        setattr(host, key, value)
