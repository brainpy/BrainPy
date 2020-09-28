# -*- coding: utf-8 -*-

import inspect

from .. import _numpy as bnp
from ..utils.helper import Dict

__all__ = [
    'ModelDefError',
    'BaseType',
    'BaseGroup'
]


class ModelDefError(Exception):
    pass


class StepFuncError(Exception):
    pass


_neu_no = 0
_syn_no = 0


class BaseType(object):
    """The base type of neurons and synapses."""

    def __init__(self, create_func, name=None, group_based=True, type_='neu'):
        # type
        # -----
        self.group_based = group_based

        # name
        # -----
        if name is None:
            if type_ == 'neu':
                global _neu_no
                self.name = f'NeuGroup{_neu_no}'
                _neu_no += 1
            elif type_ == 'syn':
                global _syn_no
                self.name = f'SynGroup{_syn_no}'
                _syn_no += 1
            else:
                raise KeyError('Unknown group type: ', type_)
        else:
            self.name = name

        # create_func
        # ------------
        try:
            func_return = create_func()
        except TypeError as e:
            raise ModelDefError(f'Arguments in "{create_func.__name__}" must provide default values.')
        if not isinstance(func_return, dict):
            raise ModelDefError('"create_func" must return a dict.')
        assert 'variables' in func_return, 'Keyword "variables" must be defined in the return dictionary.'
        assert 'step_funcs' in func_return, 'Keyword "step_funcs" must be defined in the return dictionary.'
        self.create_func = create_func

        # parameters
        # ------------
        parameters = inspect.getcallargs(create_func)
        self.parameters = Dict(parameters)

        # variables
        # ----------
        variables = func_return['variables']
        if variables is None:
            variables = Dict()
        elif isinstance(variables, (list, tuple)):
            variables = Dict((var_, 0.) for var_ in variables)
        elif isinstance(variables, dict):
            variables = Dict(variables)
        else:
            raise ValueError(f'Unknown variables type: {type(variables)}.')
        self.variables = variables

        # step functions
        # --------------
        step_funcs = func_return['step_funcs']
        if callable(step_funcs):
            step_funcs = [step_funcs]
        elif isinstance(step_funcs, (list, tuple)):
            step_funcs = list(step_funcs)
        else:
            raise ValueError('"step_funcs" must be a callable, or a list/tuple of callable functions.')
        self.step_funcs = step_funcs

    def __str__(self):
        return f'{self.name}'


class BaseGroup(object):
    __slots__ = ['model', '_mon_vars', 'mon', 'num', 'step_funcs']

    def init_monitor(self, length):
        for k in self._mon_vars:
            self.mon[k] = bnp.zeros((length, self.num), dtype=bnp.float_)

    def get_step_func_args(self):
        step_func_args = []
        for func in self.step_funcs:
            args = inspect.getfullargspec(func).args
            step_func_args.append(args)
        return step_func_args

    def check_step_func_args(self):
        for i, args in enumerate(self.get_step_func_args()):
            for arg in args:
                if not (arg in ['S', 't', 'i', 'P', 'din', 'dout']):
                    if not hasattr(self, arg):
                        raise StepFuncError(
                            f'Function "{self.step_funcs[i].__name__}" in "{self.model.name}"'
                            f'model requires "{arg}" as argument, but "{arg}" is not defined '
                            f'in the neuron group.')


def judge_spike(t, vth, S, k):
    """Judge and record the spikes of the given neuron group.

    Parameters
    ----------
    t : float
        The current time point.
    vth : int, float, bnp.ndarray
        Threshold to judge spike.
    S : bnp.ndarray
        The state of the neuron group.
    k : str
        The variable for spike judgement.

    Returns
    -------
    spike_indexes : list
        The neuron indexes that are spiking.
    """
    reach_threshold = (S[k] >= vth).astype(bnp.float_)
    spike_st = reach_threshold * (1. - S['above_th'])
    spike_idx = bnp.where(spike_st > 0.)[0]
    S['reach_th'] = reach_threshold
    S['spike'] = spike_st
    S['sp_time'][spike_idx] = t
    return spike_idx
