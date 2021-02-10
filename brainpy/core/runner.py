# -*- coding: utf-8 -*-

import ast
import inspect
import math
import re

import numba
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float64

from . import constants
from . import types
from . import utils
from .. import errors
from .. import integration
from .. import profile
from .. import tools
from ..tools import NoiseHandler

__all__ = [
    'Runner',
    'TrajectoryRunner',
]



class Runner(object):
    """Basic runner class.

    Parameters
    ----------
    ensemble : NeuGroup, SynConn
        The ensemble of the models.
    """

    def __init__(self, ensemble):
        # ensemble: NeuGroup / SynConn
        self.ensemble = ensemble
        # ensemble model
        self._model = ensemble.model
        # ensemble name
        self._name = ensemble.name
        # ensemble parameters
        self._pars = ensemble.pars
        # model delay keys
        self._delay_keys = ensemble.model._delay_keys
        # model step functions
        self._steps = ensemble.model.steps
        self._step_names = ensemble.model.step_names
        # model update schedule
        self._schedule = ['input'] + ensemble.model.step_names + ['monitor']
        self._inputs = {}
        self.gpu_data = {}

    def check_attr(self, attr):
        if not hasattr(self, attr):
            raise errors.ModelUseError(f'Model "{self._name}" doesn\'t have "{attr}" attribute", '
                                       f'and "{self._name}.ST" doesn\'t have "{attr}" field.')

    def get_codes_of_input(self, key_val_ops_types):
        """Format the code of external input.

        Parameters
        ----------
        key_val_ops_types : list, tuple
            The inputs.

        Returns
        -------
        code : dict
            The formatted code.
        """
        if len(key_val_ops_types) <= 0:
            raise errors.ModelUseError(f'{self._name} has no input, cannot call this function.')

        # check datatype of the input
        # ----------------------------
        has_iter = False
        all_inputs = set()
        for key, val, ops, t in key_val_ops_types:
            if t not in ['iter', 'fix']:
                raise errors.ModelUseError('Only support inputs of "iter" and "fix" types.')
            if t == 'iter':
                has_iter = True
            if key in all_inputs:
                raise errors.ModelUseError('Only support assignment for each key once.')
            else:
                self._inputs[key] = (val, ops, t)
                all_inputs.add(key)

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            if ops not in constants.INPUT_OPERATIONS:
                raise errors.ModelUseError(
                    f'Only support five input operations: {list(constants.INPUT_OPERATIONS.keys())}')

        # generate code of input function
        # --------------------------------
        if profile.run_on_cpu():
            code_scope = {self._name: self.ensemble, f'{self._name}_runner': self}
            code_args, code_arg2call, code_lines = set(), {}, []
            if has_iter:
                code_args.add('_i')
                code_arg2call['_i'] = '_i'

            input_idx = 0
            for key, val, ops, data_type in key_val_ops_types:
                # get the left side #
                attr_item = key.split('.')
                if len(attr_item) == 1 and (attr_item[0] not in self.ensemble.ST):
                    # if "item" is the model attribute
                    attr, item = attr_item[0], ''
                    target = getattr(self.ensemble, attr)
                    self.check_attr(attr)
                    if not isinstance(target, np.ndarray):
                        raise errors.ModelUseError(f'BrainPy only support input to arrays.')
                    left = attr
                    code_args.add(left)
                    code_arg2call[left] = f'{self._name}.{attr}'
                else:
                    if len(attr_item) == 1:
                        attr, item = 'ST', attr_item[0]
                    elif len(attr_item) == 2:
                        attr, item = attr_item[0], attr_item[1]
                    else:
                        raise errors.ModelUseError(f'Unknown target : {key}.')
                    data = getattr(self.ensemble, attr)
                    if item not in data:
                        raise errors.ModelUseError(f'"{self._name}.{attr}" doesn\'t have "{item}" field.')
                    idx = data['_var2idx'][item]
                    left = f'{attr}[{idx}]'
                    code_args.add(attr)
                    code_arg2call[attr] = f'{self._name}.{attr}["_data"]'

                # get the right side #
                right = f'{key.replace(".", "_")}_inp'
                code_args.add(right)
                code_arg2call[right] = f'{self._name}_runner.{right}'
                self.set_data(right, val)
                if data_type == 'iter':
                    right = right + '[_i]'
                    if np.ndim(val) > 1:
                        pass
                input_idx += 1

                # final code line #
                if ops == '=':
                    code_lines.append(f"{left} = {right}")
                else:
                    code_lines.append(f"{left} {ops}= {right}")

            # final code
            # ----------
            code_lines.insert(0, f'# "input" step function of {self._name}')
            code_lines.append('\n')

            # compile function
            code_to_compile = [f'def input_step({tools.func_call(code_args)}):'] + code_lines
            func_code = '\n  '.join(code_to_compile)
            exec(compile(func_code, '', 'exec'), code_scope)
            input_step = code_scope['input_step']
            # if profile.is_jit():
            #     input_step = tools.jit(input_step)
            self.input_step = input_step
            if not profile.is_merge_steps():
                if profile.show_format_code():
                    utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                if profile.show_code_scope():
                    utils.show_code_scope(code_scope, ['__builtins__', 'input_step'])

            # format function call
            arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}_runner.input_step({tools.func_call(arg2call)})'

            return {'input': {'scopes': code_scope,
                              'args': code_args,
                              'arg2calls': code_arg2call,
                              'codes': code_lines,
                              'call': func_call}}

        else:
            input_idx = 0
            results = {}
            for key, val, ops, data_type in key_val_ops_types:
                code_scope = {self._name: self.ensemble, f'{self._name}_runner': self, 'cuda': cuda}
                code_args, code_arg2call, code_lines = set(), {}, []
                if has_iter:
                    code_args.add('_i')
                    code_arg2call['_i'] = '_i'

                attr_item = key.split('.')
                if len(attr_item) == 1 and (attr_item[0] not in self.ensemble.ST):
                    # if "item" is the model attribute
                    attr, item = attr_item[0], ''
                    self.check_attr(attr)
                    target = getattr(self.ensemble, attr)
                    if not isinstance(target, np.ndarray):
                        raise errors.ModelUseError(f'BrainPy only supports input to arrays.')
                    # get the left side
                    left = f'{attr}[cuda_i]'
                    self.set_gpu_data(f'{attr}_cuda', target)
                else:
                    # if "item" is the ObjState
                    if len(attr_item) == 1:
                        attr, item = 'ST', attr_item[0]
                    elif len(attr_item) == 2:
                        attr, item = attr_item[0], attr_item[1]
                    else:
                        raise errors.ModelUseError(f'Unknown target : {key}.')
                    data = getattr(self.ensemble, attr)
                    if item not in data:
                        raise errors.ModelUseError(f'"{self._name}.{attr}" doesn\'t have "{item}" field.')
                    # get the left side
                    target = data[item]
                    idx = data['_var2idx'][item]
                    left = f'{attr}[{idx}, cuda_i]'
                    self.set_gpu_data(f'{attr}_cuda', data)
                code_args.add(f'{attr}')
                code_arg2call[f'{attr}'] = f'{self._name}_runner.{attr}_cuda'

                # get the right side #
                right = f'{key.replace(".", "_")}_inp'
                self.set_data(right, val)
                code_args.add(right)
                code_arg2call[right] = f'{self._name}_runner.{right}'

                # check data type
                iter_along_time = data_type == 'iter'
                if np.isscalar(val):
                    iter_along_data = False
                else:
                    if iter_along_time:
                        if np.isscalar(val[0]):
                            iter_along_data = False
                        else:
                            assert len(val[0]) == len(target)
                            iter_along_data = True
                    else:
                        assert len(val) == len(target)
                        iter_along_data = True
                if iter_along_time and iter_along_data:
                    right = right + '[_i, cuda_i]'
                elif iter_along_time:
                    right = right + '[_i]'
                elif iter_along_data:
                    right = right + '[cuda_i]'
                else:
                    right = right

                # final code line
                if ops == '=':
                    code_lines.append(f"{left} = {right}")
                else:
                    code_lines.append(f"{left} {ops}= {right}")
                code_lines = ['  ' + line for line in code_lines]
                code_lines.insert(0, f'if cuda_i < {len(target)}:')

                # final code
                func_name = f'input_of_{attr}_{item}'
                code_to_compile = [f'# "input" of {self._name}.{attr}.{item}',
                                   f'def {func_name}({tools.func_call(code_args)}):',
                                   f'  cuda_i = cuda.grid(1)']
                code_to_compile += [f'  {line}' for line in code_lines]

                # compile function
                func_code = '\n'.join(code_to_compile)
                exec(compile(func_code, '', 'exec'), code_scope)
                step_func = code_scope[func_name]
                step_func = cuda.jit(step_func)
                setattr(self, func_name, step_func)
                if not profile.is_merge_steps():
                    if profile.show_format_code():
                        utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                    if profile.show_code_scope():
                        utils.show_code_scope(code_scope, ['__builtins__', 'input_step'])

                # format function call
                if len(target) <= profile.get_num_thread_gpu():
                    num_thread = len(target)
                    num_block = 1
                else:
                    num_thread = profile.get_num_thread_gpu()
                    num_block = math.ceil(len(target) / profile.get_num_thread_gpu())
                arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
                func_call = f'{self._name}_runner.{func_name}[{num_block}, {num_thread}]({tools.func_call(arg2call)})'

                # function result
                results[f'input-{input_idx}'] = {'scopes': code_scope,
                                                 'args': code_args,
                                                 'arg2calls': code_arg2call,
                                                 'codes': code_lines,
                                                 'call': func_call,
                                                 'num_data': len(target)}

                # iteration
                input_idx += 1

            return results

    def get_codes_of_monitor(self, mon_vars, run_length):
        """Get the code of the monitors.

        Parameters
        ----------
        mon_vars : tuple, list
            The variables to monitor.
        run_length

        Returns
        -------
        code : dict
            The formatted code.
        """
        if len(mon_vars) <= 0:
            raise errors.ModelUseError(f'{self._name} has no monitor, cannot call this function.')

        # check indices #
        for key, indices in mon_vars:
            if indices is not None:
                if isinstance(indices, list):
                    if not isinstance(indices[0], int):
                        raise errors.ModelUseError('Monitor index only supports list [int] or 1D array.')
                elif isinstance(indices, np.ndarray):
                    if np.ndim(indices) != 1:
                        raise errors.ModelUseError('Monitor index only supports list [int] or 1D array.')
                else:
                    raise errors.ModelUseError(f'Unknown monitor index type: {type(indices)}.')

        if profile.run_on_cpu():
            # monitor
            mon = tools.DictPlus()

            code_scope = {self._name: self.ensemble, f'{self._name}_runner': self}
            code_args, code_arg2call, code_lines = set(), {}, []

            # generate code of monitor function
            # ---------------------------------
            mon_idx = 0
            for key, indices in mon_vars:
                if indices is not None:
                    indices = np.asarray(indices)
                attr_item = key.split('.')

                # get the code line #
                if (len(attr_item) == 1) and (attr_item[0] not in self.ensemble.ST):
                    attr = attr_item[0]
                    self.check_attr(attr)
                    data = getattr(self.ensemble, attr)
                    if not isinstance(data, np.ndarray):
                        assert errors.ModelUseError(f'BrainPy only supports monitor of arrays.')
                    shape = data.shape
                    mon_name = f'mon_{attr}'
                    target_name = attr
                    if indices is None:
                        line = f'{mon_name}[_i] = {target_name}'
                    else:
                        idx_name = f'idx{mon_idx}_{attr}'
                        line = f'{mon_name}[_i] = {target_name}[{idx_name}]'
                        code_scope[idx_name] = indices
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self._name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self._name}.{attr}'
                else:
                    if len(attr_item) == 1:
                        item, attr = attr_item[0], 'ST'
                    elif len(attr_item) == 2:
                        attr, item = attr_item
                    else:
                        raise errors.ModelUseError(f'Unknown target : {key}.')
                    data = getattr(self.ensemble, attr)
                    shape = data[item].shape
                    idx = data['_var2idx'][item]
                    mon_name = f'mon_{attr}_{item}'
                    target_name = attr
                    if indices is None:
                        line = f'{mon_name}[_i] = {target_name}[{idx}]'
                    else:
                        idx_name = f'idx{mon_idx}_{attr}_{item}'
                        line = f'{mon_name}[_i] = {target_name}[{idx}][{idx_name}]'
                        code_scope[idx_name] = indices
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self._name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self._name}.{attr}["_data"]'
                mon_idx += 1

                # initialize monitor array #
                key = key.replace('.', '_')
                if indices is None:
                    mon[key] = np.zeros((run_length,) + shape, dtype=np.float_)
                else:
                    mon[key] = np.zeros((run_length, len(indices)) + shape[1:], dtype=np.float_)

                # add line #
                code_lines.append(line)

            # final code
            # ----------
            code_lines.insert(0, f'# "monitor" step function of {self._name}')
            code_lines.append('\n')
            code_args.add('_i')
            code_arg2call['_i'] = '_i'

            # compile function
            code_to_compile = [f'def monitor_step({tools.func_call(code_args)}):'] + code_lines
            func_code = '\n  '.join(code_to_compile)

            if not profile.is_merge_steps():
                if profile.show_format_code():
                    utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                if profile.show_code_scope():
                    utils.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))

            exec(compile(func_code, '', 'exec'), code_scope)
            monitor_step = code_scope['monitor_step']
            # if profile.is_jit():
            #     monitor_step = tools.jit(monitor_step)
            self.monitor_step = monitor_step

            # format function call
            arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}_runner.monitor_step({tools.func_call(arg2call)})'

            return mon, {'monitor': {'scopes': code_scope,
                                     'args': code_args,
                                     'arg2calls': code_arg2call,
                                     'codes': code_lines,
                                     'call': func_call}}

        else:
            results = {}
            mon = tools.DictPlus()

            # generate code of monitor function
            # ---------------------------------
            mon_idx = 0
            for key, indices in mon_vars:
                if indices is not None:
                    indices = np.asarray(indices)
                code_scope = {self._name: self.ensemble, f'{self._name}_runner': self}
                code_args, code_arg2call, code_lines = set(), {}, []

                attr_item = key.split('.')
                key = key.replace(".", "_")
                # get the code line #
                if (len(attr_item) == 1) and (attr_item[0] not in self.ensemble.ST):
                    attr, item = attr_item[0], ''
                    self.check_attr(attr)
                    if not isinstance(getattr(self.ensemble, attr), np.ndarray):
                        assert errors.ModelUseError(f'BrainPy only supports monitor of arrays.')
                    data = getattr(self.ensemble, attr)
                    shape = data.shape
                    mon_name = f'mon_{attr}'
                    target_name = f'{attr}_cuda'
                    if indices is None:
                        num_data = shape[0]
                        line = f'{mon_name}[_i, cuda_i] = {target_name}[cuda_i]'
                    else:
                        num_data = len(indices)
                        idx_name = f'idx{mon_idx}_{attr}'
                        code_lines.append(f'mon_idx = {idx_name}[cuda_i]')
                        line = f'{mon_name}[_i, cuda_i] = {target_name}[mon_idx]'
                        code_scope[idx_name] = cuda.to_device(indices)
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self._name}_runner.mon_{key}_cuda'
                    self.set_gpu_data(f'{attr}_cuda', data)
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self._name}_runner.{attr}_cuda'
                else:
                    if len(attr_item) == 1:
                        item, attr = attr_item[0], 'ST'
                    elif len(attr_item) == 2:
                        attr, item = attr_item
                    else:
                        raise errors.ModelUseError(f'Unknown target : {key}.')
                    data = getattr(self.ensemble, attr)
                    shape = data[item].shape
                    idx = getattr(self.ensemble, attr)['_var2idx'][item]
                    mon_name = f'mon_{attr}_{item}'
                    target_name = attr
                    if indices is None:
                        num_data = shape[0]
                        line = f'{mon_name}[_i, cuda_i] = {target_name}[{idx}, cuda_i]'
                    else:
                        num_data = len(indices)
                        idx_name = f'idx{mon_idx}_{attr}_{item}'
                        code_lines.append(f'mon_idx = {idx_name}[cuda_i]')
                        line = f'{mon_name}[_i, cuda_i] = {target_name}[{idx}, mon_idx]'
                        code_scope[idx_name] = cuda.to_device(indices)
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self._name}_runner.mon_{key}_cuda'
                    self.set_gpu_data(f'{attr}_cuda', data)
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self._name}_runner.{attr}_cuda'

                # initialize monitor array #
                if indices is None:
                    mon[key] = np.zeros((run_length,) + shape, dtype=np.float_)
                else:
                    mon[key] = np.zeros((run_length, num_data) + shape[1:], dtype=np.float_)
                self.set_gpu_data(f'mon_{key}_cuda', mon[key])

                # add line #
                code_args.add('_i')
                code_arg2call['_i'] = '_i'
                code_scope['cuda'] = cuda

                # final code
                # ----------
                code_lines.append(line)
                code_lines = ['  ' + line for line in code_lines]
                code_lines.insert(0, f'if cuda_i < {num_data}:')

                # compile function
                func_name = f'monitor_of_{attr}_{item}'
                code_to_compile = [f'# "monitor" of {self._name}.{attr}.{item}',
                                   f'def {func_name}({tools.func_call(code_args)}):',
                                   f'  cuda_i = cuda.grid(1)']
                code_to_compile += [f'  {line}' for line in code_lines]
                func_code = '\n'.join(code_to_compile)
                exec(compile(func_code, '', 'exec'), code_scope)
                monitor_step = code_scope[func_name]
                monitor_step = cuda.jit(monitor_step)
                setattr(self, func_name, monitor_step)

                if not profile.is_merge_steps():
                    if profile.show_format_code():
                        utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                    if profile.show_code_scope():
                        utils.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))

                # format function call
                if num_data <= profile.get_num_thread_gpu():
                    num_thread = num_data
                    num_block = 1
                else:
                    num_thread = profile.get_num_thread_gpu()
                    num_block = math.ceil(num_data / profile.get_num_thread_gpu())
                arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
                func_call = f'{self._name}_runner.{func_name}[{num_block}, {num_thread}]({tools.func_call(arg2call)})'

                results[f'monitor-{mon_idx}'] = {'scopes': code_scope,
                                                 'args': code_args,
                                                 'arg2calls': code_arg2call,
                                                 'codes': code_lines,
                                                 'call': func_call,
                                                 'num_data': num_data}

                mon_idx += 1

            return mon, results

    def get_codes_of_steps(self):
        """Get the code of user defined update steps.

        Returns
        -------
        code : dict
            The formatted code.
        """
        if self._model.mode == constants.SCALAR_MODE:
            return self.step_scalar_model()
        else:
            return self.step_vector_model()

    def format_step_code(self, func_code):
        """Format code of user defined step function.

        Parameters
        ----------
        func_code : str
            The user defined function codes.
        """
        tree = ast.parse(func_code.strip())
        formatter = tools.CodeLineFormatter()
        formatter.visit(tree)
        return formatter

    def merge_integrators(self, func):
        """Substitute the user defined integrators into the main step functions.

        Parameters
        ----------
        func : callable
            The user defined (main) step function.

        Returns
        -------
        results : tuple
            The codes and code scope.
        """
        # get code and code lines
        func_code = tools.deindent(tools.get_main_code(func))
        formatter = self.format_step_code(func_code)
        code_lines = formatter.lines

        # get function scope
        vars = inspect.getclosurevars(func)
        code_scope = dict(vars.nonlocals)
        code_scope.update(vars.globals)
        code_scope.update({self._name: self.ensemble})
        code_scope.update(formatter.scope)
        if len(code_lines) == 0:
            return '', code_scope

        # code scope update
        scope_to_add = {}
        scope_to_del = set()
        need_add_mapping_scope = False
        for k, v in code_scope.items():
            if isinstance(v, integration.Integrator):
                if profile.is_merge_integrators():
                    need_add_mapping_scope = True

                    # locate the integration function
                    need_replace = False
                    int_func_name = v.py_func_name
                    for line_no, line in enumerate(code_lines):
                        if int_func_name in tools.get_identifiers(line):
                            need_replace = True
                            break
                    if not need_replace:
                        scope_to_del.add(k)
                        continue

                    # get integral function line indent
                    line_indent = tools.get_line_indent(line)
                    indent = ' ' * line_indent

                    # get the replace line and arguments need to replace
                    new_line, args, kwargs = tools.replace_func(line, int_func_name)
                    # append code line of argument replacement
                    func_args = v.diff_eq.func_args
                    append_lines = [indent + f'_{func_args[i]} = {args[i]}' for i in range(len(args))]
                    for arg in func_args[len(args):]:
                        append_lines.append(indent + f'_{arg} = {kwargs[arg]}')

                    # append numerical integration code lines
                    append_lines.extend([indent + l for l in v.update_code.split('\n')])
                    append_lines.append(indent + new_line)

                    # add appended lines into the main function code lines
                    code_lines = code_lines[:line_no] + append_lines + code_lines[line_no + 1:]

                    # get scope variables to delete
                    scope_to_del.add(k)
                    for k_, v_ in v.code_scope.items():
                        if profile.is_jit() and callable(v_):
                            v_ = tools.numba_func(v_, params=self._pars.updates)
                        scope_to_add[k_] = v_

                else:
                    if self._model.mode == constants.SCALAR_MODE:
                        for ks, vs in utils.get_func_scope(v.update_func, include_dispatcher=True).items():
                            if ks in self._pars.heters:
                                raise errors.ModelUseError(
                                    f'Heterogeneous parameter "{ks}" is not in step functions, '
                                    f'it will not work. Please set "brainpy.profile.set(merge_integrators=True)" '
                                    f'to try to merge parameter "{ks}" into the step functions.')
                    if profile.is_jit():
                        code_scope[k] = tools.numba_func(v.update_func, params=self._pars.updates)

            elif type(v).__name__ == 'function':
                if profile.is_jit():
                    code_scope[k] = tools.numba_func(v, params=self._pars.updates)

        # update code scope
        if need_add_mapping_scope:
            code_scope.update(integration.get_mapping_scope())
        code_scope.update(scope_to_add)
        for k in scope_to_del:
            code_scope.pop(k)

        # return code lines and code scope
        return '\n'.join(code_lines), code_scope, formatter

    def step_vector_model(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        delay_keys = self._delay_keys

        for func in self._steps:
            # information about the function
            func_name = func.__name__
            stripped_fname = tools.get_func_name(func, replace=True)
            func_args = inspect.getfullargspec(func).args

            # initialize code namespace
            used_args, code_arg2call = set(), {}
            func_code, code_scope, formatter = self.merge_integrators(func)
            code_scope[f'{self._name}_runner'] = self

            # check function code
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in constants.ARG_KEYWORDS and
                          isinstance(getattr(self.ensemble, k), types.ObjState)}
            except AttributeError:
                raise errors.ModelUseError(f'Model "{self._name}" does not have all the '
                                           f'required attributes: {func_args}.')
            add_args = set()
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg in states:
                    st = states[arg]
                    var2idx = st['_var2idx']

                    if self.ensemble._is_state_attr(arg):
                        # Function with "delayed" decorator should use
                        # ST pulled from the delay queue
                        if func_name.startswith('_brainpy_delayed_'):
                            if len(delay_keys):
                                dout = f'{arg}_dout'
                                add_args.add(dout)
                                code_arg2call[dout] = f'{self._name}.{arg}._delay_out'
                                for st_k in delay_keys:
                                    p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                    r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {dout}]"
                                    func_code = re.sub(r'' + p, r, func_code)
                        else:
                            # Function without "delayed" decorator should push their
                            # updated ST to the delay queue
                            if len(delay_keys):
                                func_code_left = '\n'.join(formatter.lefts)
                                func_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                                func_delay_keys = func_keys.intersection(delay_keys)
                                if len(func_delay_keys) > 0:
                                    din = f'{arg}_din'
                                    add_args.add(din)
                                    code_arg2call[din] = f'{self._name}.{arg}._delay_in'
                                    for st_k in func_delay_keys:
                                        right = f'{arg}[{var2idx[st_k]}]'
                                        left = f"{arg}[{var2idx['_' + st_k + '_offset']} + {din}]"
                                        func_code += f'\n{left} = {right}'

                    # replace key access to index access
                    for st_k in st._keys:
                        p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                        r = f"{arg}[{var2idx[st_k]}]"
                        func_code = re.sub(r'' + p, r, func_code)

            # substitute arguments
            code_args = add_args
            for arg in used_args:
                if arg in constants.ARG_KEYWORDS:
                    code_arg2call[arg] = arg
                else:
                    if isinstance(getattr(self.ensemble, arg), types.ObjState):
                        code_arg2call[arg] = f'{self._name}.{arg}["_data"]'
                    else:
                        code_arg2call[arg] = f'{self._name}.{arg}'
                code_args.add(arg)

            # substitute "range" to "numba.prange"
            arg_substitute = {}
            if ' range' in func_code:
                arg_substitute['range'] = 'numba.prange'
                code_scope['numba'] = numba
                func_code = tools.word_replace(func_code, arg_substitute)

            # update code scope
            for k in list(code_scope.keys()):
                if k in self._pars.updates:
                    code_scope[k] = self._pars.updates[k]

            # handle the "_normal_like_"
            func_code = NoiseHandler.normal_pattern.sub(NoiseHandler.vector_replace_f, func_code)
            code_scope['numpy'] = np

            # final
            code_lines = func_code.split('\n')
            code_lines.insert(0, f'# "{stripped_fname}" step function of {self._name}')
            code_lines.append('\n')

            # code to compile
            code_to_compile = [f'def {stripped_fname}({tools.func_call(code_args)}):']
            code_to_compile += code_lines
            func_code = '\n '.join(code_to_compile)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = code_scope[stripped_fname]
            if profile.is_jit():
                func = tools.jit(func)
            if not profile.is_merge_steps():
                if profile.show_format_code():
                    utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                if profile.show_code_scope():
                    utils.show_code_scope(code_scope, ['__builtins__', stripped_fname])

            # set the function to the model
            setattr(self, stripped_fname, func)
            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}_runner.{stripped_fname}({tools.func_call(arg2calls)})'

            results[stripped_fname] = {'scopes': code_scope,
                                       'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines,
                                       'call': func_call}

        return results

    def step_scalar_model(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        delay_keys = self._delay_keys
        all_heter_pars = set(self._pars.heters.keys())

        for i, func in enumerate(self._steps):
            func_name = func.__name__

            # get necessary code data
            # -----------------------
            # 1. code arguments
            # 2. code argument_to_call
            # 3. code lines
            # 4. code scope variables
            used_args, code_arg2call = set(), {}
            func_args = inspect.getfullargspec(func).args
            func_code, code_scope, formatter = self.merge_integrators(func)
            code_scope[f'{self._name}_runner'] = self
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in constants.ARG_KEYWORDS and
                          isinstance(getattr(self.ensemble, k), types.ObjState)}
            except AttributeError:
                raise errors.ModelUseError(f'Model "{self._name}" does not have all the '
                                           f'required attributes: {func_args}.')

            # update functions in code scope
            # 1. recursively jit the function
            # 2. update the function parameters
            for k, v in code_scope.items():
                if profile.is_jit() and callable(v):
                    code_scope[k] = tools.numba_func(func=v, params=self._pars.updates)

            add_args = set()
            # substitute STATE item access to index
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg not in states:
                    continue

                st = states[arg]
                var2idx = st['_var2idx']
                if self.ensemble._is_state_attr(arg):
                    if func_name.startswith('_brainpy_delayed_'):
                        if len(delay_keys):
                            dout = f'{arg}_dout'
                            add_args.add(dout)
                            code_arg2call[dout] = f'{self._name}.{arg}._delay_out'
                            # Function with "delayed" decorator should use ST pulled from the delay queue
                            for st_k in delay_keys:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {dout}, _obj_i]"
                                func_code = re.sub(r'' + p, r, func_code)
                    else:
                        if len(delay_keys):
                            # Function without "delayed" decorator should push
                            # their updated ST to the delay queue
                            func_code_left = '\n'.join(formatter.lefts)
                            func_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                            func_delay_keys = func_keys.intersection(delay_keys)
                            if len(func_delay_keys) > 0:
                                din = f'{arg}_din'
                                add_args.add(din)
                                code_arg2call[din] = f'{self._name}.{arg}._delay_in'
                                for st_k in func_delay_keys:
                                    right = f'{arg}[{var2idx[st_k]}, _obj_i]'
                                    left = f"{arg}[{var2idx['_' + st_k + '_offset']} + {din}, _obj_i]"
                                    func_code += f'\n{left} = {right}'
                        for st_k in st._keys:
                            p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                            r = f"{arg}[{var2idx[st_k]}, _obj_i]"
                            func_code = re.sub(r'' + p, r, func_code)
                elif arg == 'pre':
                    # 1. implement the atomic operations for "pre"
                    if profile.run_on_gpu():
                        code_lines = func_code.split('\n')
                        add_cuda = False
                        line_no = 0
                        while line_no < len(code_lines):
                            line = code_lines[line_no]
                            blank_no = len(line) - len(line.lstrip())
                            line = line.strip()
                            if line.startswith('pre'):
                                pre_transformer = tools.find_atomic_op(line, var2idx)
                                if pre_transformer.left is not None:
                                    left = pre_transformer.left
                                    right = pre_transformer.right
                                    code_lines[line_no] = ' ' * blank_no + f'cuda.atomic.add({left}, _pre_i, {right})'
                                    add_cuda = True
                            line_no += 1
                        if add_cuda:
                            code_scope['cuda'] = cuda
                        func_code = '\n'.join(code_lines)
                    # 2. transform the key access to index access
                    for st_k in st._keys:
                        p = f'pre\[([\'"]{st_k}[\'"])\]'
                        r = f"pre[{var2idx[st_k]}, _pre_i]"
                        func_code = re.sub(r'' + p, r, func_code)
                elif arg == 'post':
                    # 1. implement the atomic operations for "post"
                    if profile.run_on_gpu():
                        code_lines = func_code.split('\n')
                        add_cuda = False
                        line_no = 0
                        while line_no < len(code_lines):
                            line = code_lines[line_no]
                            blank_no = len(line) - len(line.lstrip())
                            line = line.strip()
                            if line.startswith('post'):
                                post_transformer = tools.find_atomic_op(line, var2idx)
                                if post_transformer.left is not None:
                                    left = post_transformer.left
                                    right = post_transformer.right
                                    code_lines[line_no] = ' ' * blank_no + f'cuda.atomic.add({left}, _post_i, {right})'
                                    add_cuda = True
                            line_no += 1
                        if add_cuda:
                            code_scope['cuda'] = cuda
                        func_code = '\n'.join(code_lines)
                    # 2. transform the key access to index access
                    for st_k in st._keys:
                        p = f'post\[([\'"]{st_k}[\'"])\]'
                        r = f"post[{var2idx[st_k]}, _post_i]"
                        func_code = re.sub(r'' + p, r, func_code)
                else:
                    raise ValueError

            # get formatted function arguments
            # --------------------------------
            # 1. For argument in "ARG_KEYWORDS", keep it unchanged
            # 2. For argument is an instance of ObjState, get it's cuda data
            # 3. For other argument, get it's cuda data
            code_args = add_args
            for arg in used_args:
                if arg in constants.ARG_KEYWORDS:
                    code_arg2call[arg] = arg
                else:
                    data = getattr(self.ensemble, arg)
                    if profile.run_on_cpu():
                        if isinstance(data, types.ObjState):
                            code_arg2call[arg] = f'{self._name}.{arg}["_data"]'
                        else:
                            code_arg2call[arg] = f'{self._name}.{arg}'
                    else:
                        if isinstance(data, types.ObjState):
                            code_arg2call[arg] = f'{self._name}_runner.{arg}_cuda'
                        else:
                            code_arg2call[arg] = f'{self._name}_runner.{arg}_cuda'
                        self.set_gpu_data(f'{arg}_cuda', data)
                code_args.add(arg)

            # add the for loop in the start of the main code
            has_pre = 'pre' in func_args
            has_post = 'post' in func_args
            if profile.run_on_cpu():
                code_lines = [f'for _obj_i in numba.prange({self.ensemble.num}):']
                code_scope['numba'] = numba
            else:
                code_lines = [f'_obj_i = cuda.grid(1)',
                              f'if _obj_i < {self.ensemble.num}:']
                code_scope['cuda'] = cuda

            if has_pre:
                code_args.add(f'pre_ids')
                code_arg2call[f'pre_ids'] = f'{self._name}_runner.pre_ids'
                code_lines.append(f'  _pre_i = pre_ids[_obj_i]')
                self.set_data('pre_ids', getattr(self.ensemble, 'pre_ids'))
            if has_post:
                code_args.add(f'post_ids')
                code_arg2call[f'post_ids'] = f'{self._name}_runner.post_ids'
                code_lines.append(f'  _post_i = post_ids[_obj_i]')
                self.set_data('post_ids', getattr(self.ensemble, 'post_ids'))

            # substitute heterogeneous parameter "p" to "p[_obj_i]"
            # ------------------------------------------------------
            arg_substitute = {}
            for p in self._pars.heters.keys():
                if p in code_scope:
                    arg_substitute[p] = f'{p}[_obj_i]'
            if len(arg_substitute):
                func_code = tools.word_replace(func_code, arg_substitute)

            # add the main code (user defined)
            # ------------------
            for l in func_code.split('\n'):
                code_lines.append('  ' + l)
            code_lines.append('\n')
            stripped_fname = tools.get_func_name(func, replace=True)
            code_lines.insert(0, f'# "{stripped_fname}" step function of {self._name}')

            # update code scope
            # ------------------
            for k in list(code_scope.keys()):
                if k in self._pars.updates:
                    if profile.run_on_cpu():
                        # run on cpu :
                        # 1. update the parameter
                        # 2. remove the heterogeneous parameter
                        code_scope[k] = self._pars.updates[k]
                        if k in all_heter_pars:
                            all_heter_pars.remove(k)
                    else:
                        # run on gpu :
                        # 1. update the parameter
                        # 2. transform the heterogeneous parameter to function argument
                        if k in all_heter_pars:
                            code_args.add(k)
                            code_arg2call[k] = cuda.to_device(self._pars.updates[k])
                        else:
                            code_scope[k] = self._pars.updates[k]

            # handle the "_normal_like_"
            # ---------------------------
            func_code = '\n'.join(code_lines)
            if len(NoiseHandler.normal_pattern.findall(func_code)):
                if profile.run_on_gpu():  # gpu noise
                    func_code = NoiseHandler.normal_pattern.sub(NoiseHandler.cuda_replace_f, func_code)
                    code_scope['xoroshiro128p_normal_float64'] = xoroshiro128p_normal_float64
                    num_block, num_thread = tools.get_cuda_size(self.ensemble.num)
                    code_args.add('rng_states')
                    code_arg2call['rng_states'] = f'{self._name}_runner.rng_states'
                    rng_state = create_xoroshiro128p_states(num_block * num_thread, seed=np.random.randint(100000))
                    setattr(self, 'rng_states', rng_state)
                else:  # cpu noise
                    func_code = NoiseHandler.normal_pattern.sub(NoiseHandler.scalar_replace_f, func_code)
                    code_scope['numpy'] = np
                code_lines = func_code.split('\n')

            # code to compile
            # -----------------
            # 1. get the codes to compile
            code_to_compile = [f'def {stripped_fname}({tools.func_call(code_args)}):']
            code_to_compile += code_lines
            func_code = '\n  '.join(code_to_compile)
            exec(compile(func_code, '', 'exec'), code_scope)
            # 2. output the function codes
            if not profile.is_merge_steps():
                if profile.show_format_code():
                    utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                if profile.show_code_scope():
                    utils.show_code_scope(code_scope, ['__builtins__', stripped_fname])
            # 3. jit the compiled function
            func = code_scope[stripped_fname]
            if profile.run_on_cpu():
                if profile.is_jit():
                    func = tools.jit(func)
            else:
                func = cuda.jit(func)
            # 4. set the function to the model
            setattr(self, stripped_fname, func)

            # get function call
            # -----------------
            # 1. get the functional arguments
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            arg_code = tools.func_call(arg2calls)
            if profile.run_on_cpu():
                # 2. function call on cpu
                func_call = f'{self._name}_runner.{stripped_fname}({arg_code})'
            else:
                # 3. function call on gpu
                num_block, num_thread = tools.get_cuda_size(self.ensemble.num)
                func_call = f'{self._name}_runner.{stripped_fname}[{num_block}, {num_thread}]({arg_code})'

            # the final result
            # ------------------
            results[stripped_fname] = {'scopes': code_scope,
                                       'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines,
                                       'call': func_call,
                                       'num_data': self.ensemble.num}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise errors.ModelDefError(f'''
Heterogeneous parameters "{list(all_heter_pars)}" are not defined 
in main step function. BrainPy can not recognize. 

This error may be caused by:
1. Heterogeneous par is defined in other non-main step functions.
2. Heterogeneous par is defined in "integrators", but do not call 
   "profile.set(merge_integrators=True)".

Several ways to correct this error is:
1. Define the heterogeneous parameter in the "ST".
2. Call "profile.set(merge_integrators=True)" define the network definition.

''')

        return results

    def merge_codes(self, compiled_result):
        codes_of_calls = []  # call the compiled functions

        if profile.run_on_cpu():
            if profile.is_merge_steps():
                lines, code_scopes, args, arg2calls = [], dict(), set(), dict()
                for item in self.get_schedule():
                    if item in compiled_result:
                        lines.extend(compiled_result[item]['codes'])
                        code_scopes.update(compiled_result[item]['scopes'])
                        args = args | compiled_result[item]['args']
                        arg2calls.update(compiled_result[item]['arg2calls'])

                args = sorted(list(args))
                arg2calls_list = [arg2calls[arg] for arg in args]
                lines.insert(0, f'\n# {self._name} "merge_func"'
                                f'\ndef merge_func({tools.func_call(args)}):')
                func_code = '\n  '.join(lines)
                exec(compile(func_code, '', 'exec'), code_scopes)

                func = code_scopes['merge_func']
                if profile.is_jit():
                    func = tools.jit(func)
                self.merge_func = func
                func_call = f'{self._name}_runner.merge_func({tools.func_call(arg2calls_list)})'
                codes_of_calls.append(func_call)

                if profile.show_format_code():
                    utils.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                if profile.show_code_scope():
                    utils.show_code_scope(code_scopes, ('__builtins__', 'merge_func'))

            else:
                for item in self.get_schedule():
                    if item in compiled_result:
                        func_call = compiled_result[item]['call']
                        codes_of_calls.append(func_call)

        else:
            if profile.is_merge_steps():
                print('WARNING: GPU mode do not support to merge steps.')

            for item in self.get_schedule():
                for compiled_key in compiled_result.keys():
                    if compiled_key.startswith(item):
                        func_call = compiled_result[compiled_key]['call']
                        codes_of_calls.append(func_call)
                        codes_of_calls.append('cuda.synchronize()')

        return codes_of_calls

    def get_schedule(self):
        return self._schedule

    def set_schedule(self, schedule):
        if not isinstance(schedule, (list, tuple)):
            raise errors.ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + self._step_names
        for s in schedule:
            if s not in all_func_names:
                raise errors.ModelUseError(f'Unknown step function "{s}" for model "{self._name}".')
        self._schedule = schedule

    def set_data(self, key, data):
        if profile.run_on_gpu():
            if np.isscalar(data):
                data_cuda = data
            else:
                data_cuda = cuda.to_device(data)
            setattr(self, key, data_cuda)
        else:
            setattr(self, key, data)

    def set_gpu_data(self, key, val):
        if key not in self.gpu_data:
            if isinstance(val, np.ndarray):
                val = cuda.to_device(val)
            elif isinstance(val, types.ObjState):
                val = val.get_cuda_data()
            setattr(self, key, val)
            self.gpu_data[key] = val

    def gpu_data_to_cpu(self):
        for val in self.gpu_data.values():
            val.to_host()


class TrajectoryRunner(Runner):
    """Runner class for trajectory.

    Parameters
    ----------
    ensemble : NeuGroup
        The neuron ensemble.
    target_vars : tuple, list
        The targeted variables for trajectory.
    fixed_vars : dict
        The fixed variables.
    """

    def __init__(self, ensemble, target_vars, fixed_vars=None):
        # check ensemble
        from brainpy.core.neurons import NeuGroup
        if not isinstance(ensemble, NeuGroup):
            raise errors.ModelUseError(f'{self.__name__} only supports the instance of NeuGroup.')

        # initialization
        super(TrajectoryRunner, self).__init__(ensemble=ensemble)

        # check targeted variables
        if not isinstance(target_vars, (list, tuple)):
            raise errors.ModelUseError('"target_vars" must be a list/tuple.')
        for var in target_vars:
            if var not in self._model.variables:
                raise errors.ModelUseError(f'"{var}" in "target_vars" is not defined in model "{self._model.name}".')
        self.target_vars = target_vars

        # check fixed variables
        try:
            if fixed_vars is not None:
                isinstance(fixed_vars, dict)
            else:
                fixed_vars = dict()
        except AssertionError:
            raise errors.ModelUseError('"fixed_vars" must be a dict.')
        self.fixed_vars = dict()
        for integrator in self._model.integrators:
            var_name = integrator.diff_eq.var_name
            if var_name not in target_vars:
                if var_name in fixed_vars:
                    self.fixed_vars[var_name] = fixed_vars.get(var_name)
                else:
                    self.fixed_vars[var_name] = self._model.variables.get(var_name)
        for var in fixed_vars.keys():
            if var not in self.fixed_vars:
                self.fixed_vars[var] = fixed_vars.get(var)

    def format_step_code(self, func_code):
        """Format code of user defined step function.

        Parameters
        ----------
        func_code : str
            The user defined function.
        """
        tree = ast.parse(func_code.strip())
        formatter = tools.LineFormatterForTrajectory(self.fixed_vars)
        formatter.visit(tree)
        return formatter
