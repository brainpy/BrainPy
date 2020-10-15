# -*- coding: utf-8 -*-

import inspect
from copy import deepcopy

from .. import numpy as np
from .. import profile
from ..core_system.types import NeuState
from ..tools import DictPlus


class DynamicsAnalyst(object):
    def __init__(self, model):
        self.model = model

    def simulate(self, duration, monitors, vars_init=None, pars_update=None):
        # times
        if isinstance(duration, (int, float)):
            start, end = 0, duration
        elif isinstance(duration, (tuple, list)):
            assert len(duration) == 2, 'Only support duration with the format of "(start, end)".'
            start, end = duration
        else:
            raise ValueError(f'Unknown duration type: {type(duration)}')
        dt = profile.get_dt()
        times = np.arange(start, end, dt)

        # monitors
        mon = DictPlus()
        for k in monitors:
            mon[k] = np.zeros(len(times))

        # variables
        variables = deepcopy(self.model.variables)
        if vars_init is not None:
            assert isinstance(vars_init, dict)
            for k, v in vars_init.items():
                variables[k] = v

        # parameters
        parameters = deepcopy(self.model.parameters)
        if pars_update is not None:
            for k, v in pars_update.items():
                parameters[k] = v

        # get update function
        update = self.model.steps[0]
        assert callable(update)

        # initialize corresponding state
        ST = NeuState(variables)(1)

        # get the running _code
        code_scope = {'update': update, 'monitor': mon, 'ST': ST,
                      'mon_keys': monitors, 'dt': dt, 'times': times}
        code_args = inspect.getfullargspec(update).args
        mapping = {'ST': 'ST', '_t_': 't', '_i_': 'i', '_dt_': 'dt'}
        code_arg2call = [mapping[arg] for arg in code_args]
        code_lines = [f'def run_func():']
        code_lines.append(f'  for i, t in enumerate(times):')
        code_lines.append(f'    update({", ".join(code_arg2call)})')
        code_lines.append(f'    for k in mon_keys:')
        if self.model.vector_based:
            code_lines.append(f'      monitor[k][i] = ST[k][0]')
        else:
            code_lines.append(f'      monitor[k][i] = ST[k]')

        # run the model
        codes = '\n'.join(code_lines)
        exec(compile(codes, '', 'exec'), code_scope)
        code_scope['run_func']()
        return mon
