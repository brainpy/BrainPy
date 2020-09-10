# -*- coding: utf-8 -*-

from collections import OrderedDict

from .common_func import numbify_func
from .common_func import Group
from .. import _numpy as bnp
from .. import profile
from ..utils import helper

_group_no = 0

__all__ = [
    'NeuronGroup',
    'create_neuron_model'
]


class NeuronGroup(Group):
    def __init__(self, geometry, monitors=None, vars_init=None, pars_updates=None):
        # num and geometry
        # -----------------
        if isinstance(geometry, (int, float)):
            geometry = (1, int(geometry))
        elif isinstance(geometry, (tuple, list)):
            # a tuple is given, can be 1 .. N dimensional
            if len(geometry) == 1:
                height, width = 1, geometry[0]
            elif len(geometry) == 2:
                height, width = geometry[0], geometry[1]
            else:
                raise ValueError('Do not support 3+ dimensional networks.')
            geometry = (height, width)
        else:
            raise ValueError()
        num = int(bnp.prod(geometry))
        self.num, self.geometry = num, geometry

        # super class initialization
        super(NeuronGroup, self).__init__(vars_init=vars_init, pars_updates=pars_updates)

        # monitors
        # ----------
        self.mon = dict()
        self._mon_vars = monitors
        self._mon_update = None

        if monitors is not None:
            for k in monitors:
                self.mon[k] = bnp.zeros((1, 1), dtype=bnp.float_)

            # generate function
            if profile.is_numba_bk():
                def update(i):
                    for k in self._mon_vars:
                        self.mon[k][i] = self.state[self.var2index[k]]
            else:
                def update(i):
                    for k in self._mon_vars:
                        self.mon[k][i] = self.state[k]
            self._mon_update = update




def create_neuron_model(parameters=None, variables=None, update_func=None, name=None):
    # handle "update"
    # -----------------
    assert update_func is not None, '"update_func" cannot be None.'

    # handle "name"
    # --------------
    if name is None:
        global _group_no
        name_ = 'neu_group_{}'.format(_group_no)
        _group_no += 1
    else:
        name_ = name

    # handle "parameters"
    # --------------------
    if parameters is None:
        parameters = OrderedDict()
    elif isinstance(parameters, (list, tuple)):
        parameters = OrderedDict((par, 0.) for par in parameters)
    elif isinstance(parameters, dict):
        parameters = OrderedDict(parameters)
    else:
        raise ValueError('Unknown parameters type: {}'.format(type(parameters)))

    # handle "variables"
    # --------------------
    if variables is None:
        variables = OrderedDict()
    elif isinstance(variables, (list, tuple)):
        variables = OrderedDict((var_, 0.) for var_ in variables)
    elif isinstance(variables, dict):
        variables = OrderedDict(variables)
    else:
        raise ValueError('Unknown variables type: {}'.format(type(variables)))

    variables['not_ref'] = 1.
    variables['above_th'] = 0.
    variables['spike'] = 0.
    variables['sp_time'] = -1e7
    variables['input'] = 0.

    # generate class
    # --------------------

    class CreatedNeuronGroup(NeuronGroup):
        pars = parameters
        vars = variables
        update = update_func
        name = name_

        def __str__(self):
            return self.name

        def __repr__(self):
            return self.name

    return CreatedNeuronGroup
