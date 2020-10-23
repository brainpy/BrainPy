# -*- coding: utf-8 -*-


import inspect
from copy import deepcopy

import autopep8

from .base_objects import ModelDefError
from .base_objects import ModelUseError
from .base_objects import _ARG_KEYWORDS
from .constants import INPUT_OPERATIONS
from .types import NeuState
from .types import SynState
from .. import numpy as np
from .. import profile
from .. import tools
from ..tools import DictPlus


class Runner(object):
    def __init__(self, model):
        self.model = model
        self._codegen = dict()

    def format_input(self, mode, key_val_ops_types):
        try:
            assert len(key_val_ops_types) > 0
        except AssertionError:
            raise ModelUseError(f'{self.model.name} has no input, cannot call this function.')

        code_scope = {self.model.name: self.model}
        code_args, code_arg2call, code_lines = set(), {}, []
        input_idx = 0

        # check datatype of the input
        # ----------------------------
        has_iter = False
        for _, _, _, t in key_val_ops_types:
            try:
                assert t in ['iter', 'fix']
            except AssertionError:
                raise ModelUseError('Only support inputs of "iter" and "fix" types.')
            if t == 'iter':
                has_iter = True
        if has_iter:
            code_args.add('_i_')
            code_arg2call['_i_'] = '_i_'

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            try:
                assert ops in INPUT_OPERATIONS
            except AssertionError:
                raise ModelUseError(f'Only support five input operations: {list(INPUT_OPERATIONS.keys())}')

        # generate code of input function
        # --------------------------------
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.model.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                try:
                    assert hasattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self.model.name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self.model.name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self, attr), np.ndarray)
                except AssertionError:
                    raise ModelUseError(f'NumpyBrain only support input to arrays.')

                if mode == 'numpy':
                    left = f'{self.model.name}.{attr}'
                else:
                    left = f'{self.model.name}_{attr}'
                    code_args.add(left)
                    code_arg2call[left] = f'{self.model.name}.{attr}'
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
                    raise ModelUseError(f'"{self.model.name}.{attr}" doesn\'t have "{item}" field.')

                if mode == 'numpy':
                    left = f'{self.model.name}.{attr}["{item}"]'
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    left = f'{self.model.name}_{attr}[{idx}]'
                    code_args.add(f'{self.model.name}_{attr}')
                    code_arg2call[f'{self.model.name}_{attr}'] = f'{self.model.name}.{attr}["_data"]'

            # get the right side #
            right = f'{self.model.name}_inp{input_idx}'
            code_scope[right] = val
            if data_type == 'iter':
                right = right + '[_i_]'
            input_idx += 1

            # final code line #
            if ops == '=':
                code_lines.append(f"{left} = {right}")
            else:
                code_lines.append(f"{left} {ops}= {right}")

        # final code
        # ----------
        code_lines.insert(0, f'# "input" step function of {self.model.name}')
        code_lines.append('\n')

        # if mode == 'numpy':
        #     code_args = sorted(list(code_args))
        #     code_lines.insert(0, f'\ndef input_step({", ".join(code_args)}):')
        #
        #     # compile function
        #     func_code = '\n  '.join(code_lines)
        #     if profile._auto_pep8:
        #         func_code = autopep8.fix_code(func_code)
        #     exec(compile(func_code, '', 'exec'), code_scope)
        #     self.input_step = code_scope['input_step']
        #
        #     # format function call
        #     code_arg2call = [code_arg2call[arg] for arg in code_args]
        #     func_call = f'{self.model.name}.runner.input_step({", ".join(code_arg2call)})'
        #
        #     if profile._show_formatted_code:
        #         tools.show_code_str(func_code)
        #         tools.show_code_scope(code_scope, ['__builtins__', 'input_step'])
        #
        #     self._codegen[f'input-{mode}'] = {'call': func_call}
        #
        # else:
        #     self._codegen[f'input-{mode}'] = {'scopes': code_scope, 'args': code_args,
        #                                       'arg2calls': code_arg2call, 'codes': code_lines}

        return {'scopes': code_scope, 'args': code_args,
                'arg2calls': code_arg2call, 'codes': code_lines}

    def _add_monitor(self, run_length):
        code_scope, code_args, code_arg2call, code_lines = {self.name: self}, set(), {}, []
        idx_no = 0

        # generate code of monitor function
        # ---------------------------------
        for key, indices in self._mon_vars:
            # check indices #
            if indices is not None:
                if isinstance(indices, list):
                    try:
                        isinstance(indices[0], int)
                    except AssertionError:
                        raise ModelUseError('Monitor index only supports list [int] or 1D array.')
                elif isinstance(indices, np.ndarray):
                    try:
                        assert np.ndim(indices) == 1
                    except AssertionError:
                        raise ModelUseError('Monitor index only supports list [int] or 1D array.')
                else:
                    raise ModelUseError(f'Unknown monitor index type: {type(indices)}.')

            attr_item = key.split('.')

            # get the code line #
            if (len(attr_item) == 1) and (attr_item[0] not in getattr(self, 'ST')):
                attr = attr_item[0]
                try:
                    assert hasattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self.name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self.name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self, attr), np.ndarray)
                except AssertionError:
                    assert ModelUseError(f'NumpyBrain only support monitor of arrays.')

                shape = getattr(self, attr).shape

                idx_name = f'{self.name}_idx{idx_no}_{attr}'
                if profile.is_numpy_bk():
                    if indices is None:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}'
                    else:
                        line = f'{self.name}.mon["{key}"][i] = {self.name}.{attr}[{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1

                else:
                    mon_name = f'{self.name}_mon_{attr}'
                    target_name = f'{self.name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[_i_] = {target_name}'
                    else:
                        line = f'{mon_name}[_i_] = {target_name}[{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self.name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self.name}.{attr}'
            else:
                if len(attr_item) == 1:
                    item, attr = attr_item[0], 'ST'
                elif len(attr_item) == 2:
                    attr, item = attr_item
                else:
                    raise ModelUseError(f'Unknown target : {key}.')

                shape = getattr(self, attr)[item].shape

                idx_name = f'{self.name}_idx{idx_no}_{attr}_{item}'
                if profile.is_numpy_bk():
                    if indices is None:
                        line = f'{self.name}.mon["{key}"][_i_] = {self.name}.{attr}["{item}"]'
                    else:
                        line = f'{self.name}.mon["{key}"][_i_] = {self.name}.{attr}["{item}"][{idx_name}]'
                        code_scope[idx_name] = indices
                        idx_no += 1
                else:
                    idx = getattr(self, attr)['_var2idx'][item]
                    mon_name = f'{self.name}_mon_{attr}_{item}'
                    target_name = f'{self.name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[_i_] = {target_name}[{idx}]'
                    else:
                        line = f'{mon_name}[_i_] = {target_name}[{idx}][{idx_name}]'
                        code_scope[idx_name] = indices
                    idx_no += 1
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self.name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self.name}.{attr}["_data"]'

            # initialize monitor array #
            key = key.replace(',', '_')
            if indices is None:
                self.mon[key] = np.zeros((run_length,) + shape, dtype=np.float_)
            else:
                self.mon[key] = np.zeros((run_length, len(indices)) + shape[1:], dtype=np.float_)

            # add line #
            code_lines.append(line)

        # final code
        # ----------
        if len(self._mon_vars):
            code_args.add('_i_')
            code_arg2call['_i_'] = '_i_'
            code_lines.insert(0, f'# "monitor" step function of {self.name}')

        if profile.is_numpy_bk():
            if len(self._mon_vars):
                code_args = sorted(list(code_args))
                code_lines.insert(0, f'\ndef monitor_step({", ".join(code_args)}):')

                # compile function
                func_code = '\n  '.join(code_lines)
                if profile._auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scope)
                self.monitor_step = code_scope['monitor_step']

                # format function call
                code_arg2call = [code_arg2call[arg] for arg in code_args]
                func_call = f'{self.name}.monitor_step({", ".join(code_arg2call)})'

                if profile._show_formatted_code:
                    tools.show_code_str(func_code)
                    tools.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))
            else:
                self.monitor_step = None
                func_call = ''

            self._codegen['monitor'] = {'func': self.monitor_step, 'call': func_call}

        else:
            code_lines.append('\n')
            self._codegen['monitor'] = {'scopes': code_scope, 'args': code_args,
                                        'arg2calls': code_arg2call, 'codes': code_lines}



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
