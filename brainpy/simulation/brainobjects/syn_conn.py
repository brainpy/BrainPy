# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.simulation.base import DynamicSystem
from brainpy.simulation.brainobjects.delays import ConstantDelay
from brainpy.simulation.brainobjects.neu_group import NeuGroup
from brainpy.simulation.connectivity.base import TwoEndConnector

__all__ = [
    'TwoEndConn',
]

_TwoEndSyn_NO = 0


class TwoEndConn(DynamicSystem):
    """Two End Synaptic Connections.

    Parameters
    ----------
    steps : function, list of function, tuple of function, dict of (str, function), optional
        The step functions.
    pre : NeuGroup
        Pre-synaptic neuron group.
    post : NeuGroup
        Post-synaptic neuron group.
    monitors : list of str, tuple of str
        Variables to monitor.
    name : str
        The name of the neuron group.
    show_code : bool
        Whether show the formatted code.
    """

    def __init__(self, pre, post, conn, name=None, steps=None, **kwargs):
        # name
        # ----
        if name is None:
            global _TwoEndSyn_NO
            _TwoEndSyn_NO += 1
            name = f'TwoEC{_TwoEndSyn_NO}{name}'
        else:
            if not name.isidentifier():
                raise errors.ModelUseError(f'"{name}" isn\'t a valid identifier '
                                           f'according to Python language definition. '
                                           f'Please choose another name.')

        # pre or post neuron group
        # ------------------------
        if not isinstance(pre, NeuGroup):
            raise errors.ModelUseError('"pre" must be an instance of NeuGroup.')
        self.pre = pre
        if not isinstance(post, NeuGroup):
            raise errors.ModelUseError('"post" must be an instance of NeuGroup.')
        self.post = post

        # connections
        # -----------
        if not isinstance(conn, TwoEndConnector):
            raise errors.ModelUseError(f'"conn" must be an instance of {TwoEndConnector}, '
                                       f'but we got {type(conn)}.')
        self.conn = conn
        self.conn(pre.size, post.size)

        # initialize
        # ----------
        if steps is None:
            steps = {'update': self.update}
        super(TwoEndConn, self).__init__(steps=steps, name=name, **kwargs)

    def update(self, _t, _i, _dt):
        raise NotImplementedError

    def register_constant_delay(self, key, size, delay_time):
        """Register a constant delay.

        Parameters
        ----------
        key : str
            The delay name.
        size : int, list/tuple of int
            The delay data size.
        delay_time : int, float
            The delay time length.

        Returns
        -------
        delay : ConstantDelay
            An instance of ConstantDelay.
        """

        if not hasattr(self, 'steps'):
            raise errors.ModelUseError('Please initialize the super class first. '
                                       'For example: \n\n'
                                       'super(YourClassName, self).__init__(**kwargs)')

        cdelay = ConstantDelay(size, delay_time)
        cdelay.name = f'{self.name}_delay_{key}'
        self.steps[f'{self.name}_{key}_update'] = cdelay.update

        return cdelay
