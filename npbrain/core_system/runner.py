# -*- coding: utf-8 -*-


import inspect
from copy import deepcopy

import autopep8

from .base_objects import ModelDefError
from .base_objects import ModelUseError
from .base_objects import _ARG_KEYWORDS
from .types import NeuState
from .types import SynState
from .. import numpy as np
from .. import profile
from .. import tools
from ..tools import DictPlus

class SingleModelRunner(object):
    def __init__(self, model, duration, monitors, vars_init=None, inputs=None):
        # model
        self.model = model

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
        mon['ts'] = times

        # variables
        variables = deepcopy(self.model.variables)
        if vars_init is not None:
            assert isinstance(vars_init, dict)
            for k, v in vars_init.items():
                variables[k] = v

        # initialize model attributes
        model_attrs = DictPlus()
        for func in self.model.steps:
            for arg in inspect.getfullargspec(func).args:
                if arg in _ARG_KEYWORDS:
                    continue
                if arg not in model.requires:
                    raise ModelDefError(f'"{model.name}" requires "{arg}" as argument, but "{arg}" isn\'t declared in '
                                        f'"requires". NumpyBrain cannot automatically initialize it.')
                state = model.requires[arg]
                if arg in model_attrs:
                    continue
                else:
                    if isinstance(state, NeuState):
                        model_attrs[arg] = state.make_copy(1)
                    elif isinstance(state, SynState):
                        raise NotImplementedError
                        model_attrs[arg] = state.make_copy(1, delay=None, delay_vars=model._de)
                    else:
                        raise NotImplementedError

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
        if self.vector_based:
            code_lines.append(f'      monitor[k][i] = ST[k][0]')
        else:
            code_lines.append(f'      monitor[k][i] = ST[k]')

        # run the model
        codes = '\n'.join(code_lines)
        exec(compile(codes, '', 'exec'), code_scope)
        code_scope['run_func']()
        return mon

    def _format_inputs(self, inputs, run_length):
        # check
        try:
            assert isinstance(inputs, (tuple, list))
        except AssertionError:
            raise ModelUseError('"inputs" must be a tuple/list.')
        if not isinstance(inputs[0], (list, tuple)):
            if isinstance(inputs[0], str):
                inputs = [inputs]
            else:
                raise ModelUseError('Unknown input structure.')
        for inp in inputs:
            try:
                assert 2 <= len(inp) <= 3
            except AssertionError:
                raise ModelUseError('For each target, you must specify "(key, value, [operation])".')
            if len(inp) == 3:
                try:
                    assert inp[2] in ['+', '-', 'x', '/', '=']
                except AssertionError:
                    raise ModelUseError(f'Input operation only support "+, -, x, /, =", not "{inp[2]}".')

        # format input
        formatted_inputs = []
        for inp in inputs:
            # key
            try:
                assert isinstance(inp[0], str)
            except AssertionError:
                raise ModelUseError('For each input, input[0] must be a string '
                                    'to specify variable of the target.')
            key = inp[0]

            # value and data type
            if isinstance(inp[1], (int, float)):
                val = inp[1]
                data_type = 'fix'
            elif isinstance(inp[1], np.ndarray):
                val = inp[1]
                if val.shape[0] == run_length:
                    data_type = 'iter'
                else:
                    data_type = 'fix'
            else:
                raise ModelUseError('For each input, input[1] must be a numerical value to specify input values.')

            # operation
            if len(inp) == 3:
                ops = inp[2]
            else:
                ops = '+'

            format_inp = (key, val, ops, data_type)
            formatted_inputs.append(format_inp)

        return formatted_inputs

    def _add_input(self, key_val_ops_types):
        code_scope, code_args, code_arg2call, code_lines = {}, set(), {}, []
        input_idx = 0

        # check datatype of the input
        # ----------------------------
        has_iter = False
        for _, _, _, t in key_val_ops_types:
            if t == 'iter':
                has_iter = True
        if has_iter:
            code_args.add('_i_')
            code_arg2call['_i_'] = '_i_'

        # generate code of input function
        # --------------------------------
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            # ----------------- #

            # if "item" is the model attribute
            if len(attr_item) == 1 and (attr_item[0] not in self.model.requires['ST']._keys):
                raise ModelUseError('In NeuType mode, only support inputs for NeuState/SynState.')
            else:
                if len(attr_item) == 1:
                    attr, item = 'ST', attr_item[0]
                elif len(attr_item) == 2:
                    attr, item = attr_item[0], attr_item[1]
                else:
                    raise ModelUseError(f'Unknown target : {key}.')
                try:
                    assert item in getattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'"{self.name}.{attr}" doesn\'t have "{item}" field.')

                if profile.is_numpy_bk():
                    left = f'{self.name}.{attr}["{item}"]'
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    left = f'{self.name}_{attr}[{idx}]'
                    code_args.add(f'{self.name}_{attr}')
                    code_arg2call[f'{self.name}_{attr}'] = f'{self.name}.{attr}["_data"]'

            # get the right side #
            right = f'{self.name}_input{input_idx}_{attr}_{item}_{ops2str[ops]}'
            code_scope[right] = val
            if data_type == 'iter':
                right = right + '[_i_]'
            input_idx += 1

            # final code line #
            if ops == '=':
                code_lines.append(left + " = " + right)
            else:
                code_lines.append(left + f" {ops}= " + right)

        # final code
        # ----------
        if len(key_val_ops_types) > 0:
            code_lines.insert(0, f'# "input" step function of {self.name}')

        if profile.is_numpy_bk():
            if len(key_val_ops_types) > 0:
                code_args = sorted(list(code_args))
                code_lines.insert(0, f'\ndef input_step({", ".join(code_args)}):')

                # compile function
                func_code = '\n  '.join(code_lines)
                if profile._auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.input_step = code_scope['input_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.input_step({", ".join(code_arg2call)})'

                if profile._show_formatted_code:
                    print(func_code)
                    print()
                    tools.show_code_scope(code_scope, ['__builtins__', 'input_step'])
                    print()
            else:
                self.input_step = None
                func_call = ''

            self._codegen['input'] = {'func': self.input_step, 'call': func_call}

        else:
            code_lines.append('\n')
            self._codegen['input'] = {'scopes': code_scope, 'args': code_args,
                                      'arg2calls': code_arg2call, 'codes': code_lines}

