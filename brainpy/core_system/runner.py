# -*- coding: utf-8 -*-

import inspect
import re
from importlib import import_module

import autopep8

from .constants import ARG_KEYWORDS
from .constants import INPUT_OPERATIONS
from .types import ObjState
from .. import numpy as np
from .. import profile
from .. import tools
from ..errors import ModelDefError
from ..errors import ModelUseError
from ..integration.integrator import Integrator
from ..integration.sympy_tools import get_mapping_scope


class Runner(object):
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

    def format_input_code(self, key_val_ops_types, mode):
        try:
            assert len(key_val_ops_types) > 0
        except AssertionError:
            raise ModelUseError(f'{self._name} has no input, cannot call this function.')

        code_scope = {self._name: self.ensemble}
        code_args, code_arg2call, code_lines = set(), {}, []

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
        input_idx = 0
        for key, val, ops, data_type in key_val_ops_types:
            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.ensemble.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                try:
                    assert hasattr(self, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self._name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self._name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self.ensemble, attr), np.ndarray)
                except AssertionError:
                    raise ModelUseError(f'BrainPy only support input to arrays.')

                if mode == 'numpy':
                    left = f'{self._name}.{attr}'
                else:
                    left = f'{self._name}_{attr}'
                    code_args.add(left)
                    code_arg2call[left] = f'{self._name}.{attr}'
            else:
                if len(attr_item) == 1:
                    attr, item = 'ST', attr_item[0]
                elif len(attr_item) == 2:
                    attr, item = attr_item[0], attr_item[1]
                else:
                    raise ModelUseError(f'Unknown target : {key}.')
                try:
                    assert item in getattr(self.ensemble, attr)
                except AssertionError:
                    raise ModelUseError(f'"{self._name}.{attr}" doesn\'t have "{item}" field.')

                if mode == 'numpy':
                    left = f'{self._name}.{attr}["{item}"]'
                else:
                    idx = getattr(self.ensemble, attr)['_var2idx'][item]
                    left = f'{self._name}_{attr}[{idx}]'
                    code_args.add(f'{self._name}_{attr}')
                    code_arg2call[f'{self._name}_{attr}'] = f'{self._name}.{attr}["_data"]'

            # get the right side #
            right = f'{self._name}_inp{input_idx}'
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
        code_lines.insert(0, f'# "input" step function of {self._name}')
        code_lines.append('\n')

        # compile function
        code_to_compile = [f'def input_step({tools.func_call(code_args)}):'] + code_lines
        func_code = '\n  '.join(code_to_compile)
        if profile._auto_pep8:
            func_code = autopep8.fix_code(func_code)
        exec(compile(func_code, '', 'exec'), code_scope)
        self.input_step = code_scope['input_step']
        if mode != 'numpy':
            self.input_step = tools.jit(self.input_step)
        if profile._show_formatted_code:
            if not (profile._merge_steps and mode != 'numpy'):
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ['__builtins__', 'input_step'])

        # format function call
        arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
        func_call = f'{self._name}.runner.input_step({tools.func_call(arg2call)})'

        return {'input': {'scopes': code_scope, 'args': code_args,
                          'arg2calls': code_arg2call,
                          'codes': code_lines, 'call': func_call}}

    def format_monitor_code(self, mon_vars, run_length, mode):
        try:
            assert len(mon_vars) > 0
        except AssertionError:
            raise ModelUseError(f'{self._name} has no monitor, cannot call this function.')

        code_scope = {self._name: self.ensemble}
        code_args, code_arg2call, code_lines = set(), {}, []

        # monitor
        mon = tools.DictPlus()

        # generate code of monitor function
        # ---------------------------------
        mon_idx = 0
        for key, indices in mon_vars:
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
            if (len(attr_item) == 1) and (attr_item[0] not in self.ensemble.ST):
                attr = attr_item[0]
                try:
                    assert hasattr(self.ensemble, attr)
                except AssertionError:
                    raise ModelUseError(f'Model "{self._name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self._name}.ST" doesn\'t have "{attr}" field.')
                try:
                    assert isinstance(getattr(self.ensemble, attr), np.ndarray)
                except AssertionError:
                    assert ModelUseError(f'BrainPy only support monitor of arrays.')

                shape = getattr(self.ensemble, attr).shape

                idx_name = f'{self._name}_idx{mon_idx}_{attr}'
                if mode == 'numpy':
                    if indices is None:
                        line = f'{self._name}.mon["{key}"][i] = {self._name}.{attr}'
                    else:
                        line = f'{self._name}.mon["{key}"][i] = {self._name}.{attr}[{idx_name}]'
                        code_scope[idx_name] = indices
                        mon_idx += 1

                else:
                    mon_name = f'{self._name}_mon_{attr}'
                    target_name = f'{self._name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[_i_] = {target_name}'
                    else:
                        line = f'{mon_name}[_i_] = {target_name}[{idx_name}]'
                        code_scope[idx_name] = indices
                        mon_idx += 1
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
                    raise ModelUseError(f'Unknown target : {key}.')

                shape = getattr(self.ensemble, attr)[item].shape

                idx_name = f'{self._name}_idx{mon_idx}_{attr}_{item}'
                if mode == 'numpy':
                    if indices is None:
                        line = f'{self._name}.mon["{key}"][_i_] = {self._name}.{attr}["{item}"]'
                    else:
                        line = f'{self._name}.mon["{key}"][_i_] = {self._name}.{attr}["{item}"][{idx_name}]'
                        code_scope[idx_name] = indices
                        mon_idx += 1
                else:
                    idx = getattr(self.ensemble, attr)['_var2idx'][item]
                    mon_name = f'{self._name}_mon_{attr}_{item}'
                    target_name = f'{self._name}_{attr}'
                    if indices is None:
                        line = f'{mon_name}[_i_] = {target_name}[{idx}]'
                    else:
                        line = f'{mon_name}[_i_] = {target_name}[{idx}][{idx_name}]'
                        code_scope[idx_name] = indices
                    mon_idx += 1
                    code_args.add(mon_name)
                    code_arg2call[mon_name] = f'{self._name}.mon["{key}"]'
                    code_args.add(target_name)
                    code_arg2call[target_name] = f'{self._name}.{attr}["_data"]'

            # initialize monitor array #
            key = key.replace(',', '_')
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
        code_args.add('_i_')
        code_arg2call['_i_'] = '_i_'

        # compile function
        code_to_compile = [f'def monitor_step({tools.func_call(code_args)}):'] + code_lines
        func_code = '\n  '.join(code_to_compile)
        if profile._auto_pep8:
            func_code = autopep8.fix_code(func_code)
        exec(compile(func_code, '', 'exec'), code_scope)
        monitor_step = code_scope['monitor_step']
        if mode != 'numpy':
            monitor_step = tools.jit(monitor_step)
        self.monitor_step = monitor_step

        # format function call
        arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
        func_call = f'{self._name}.runner.monitor_step({tools.func_call(arg2call)})'

        if profile._show_formatted_code:
            if not (profile._merge_steps and mode != 'numpy'):
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))

        return mon, {'monitor': {'scopes': code_scope, 'args': code_args,
                                 'arg2calls': code_arg2call,
                                 'codes': code_lines, 'call': func_call}}

    def format_step_codes(self, mode):
        if self._model.vector_based:
            if mode == 'numpy':
                return self.__step_mode_np_vector()
            elif mode == 'numba':
                return self.__step_mode_nb_vector()
            else:
                raise NotImplementedError

        else:
            if mode == 'numpy':
                return self.__step_mode_np_scalar()
            elif mode == 'numba':
                return self.__step_mode_nb_scalar()
            else:
                raise NotImplementedError

    def __step_mode_np_vector(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        if len(self._pars.heters) > 0:
            raise ModelUseError(f'This model has heterogeneous parameters '
                                f'"{list(self._pars.heters.keys())}", '
                                f'it cannot be compiled in numpy mode.')

        # get the delay keys
        delay_keys = self._delay_keys
        for func in self._steps:
            func_name = func.__name__
            stripped_fname = tools.get_func_name(func, replace=True)
            func_args = inspect.getfullargspec(func).args

            if len(delay_keys) > 0:
                func_code = tools.get_main_code(func)
                if func_name.startswith('_npbrain_delayed_'):
                    # In the delayed function,
                    # synapse state should pull out from the delay queues
                    code_scope = {stripped_fname: func}
                    code_lines = [f'def {stripped_fname}_enhanced({tools.func_call(func_args)}):']
                    # pull delayed keys
                    for arg in delay_keys.keys():
                        if arg not in func_args:
                            continue
                        code_lines.append(f'  new_{arg} = dict()')
                        func_delay_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code))
                        for key in func_delay_keys:
                            code_lines.append(f'  new_{arg}["{key}"] = {arg}.delay_pull("{key}")')
                        code_lines.append(f'  {arg} = new_{arg}')
                    code_lines.append(f'  {stripped_fname}({tools.func_call(func_args)})')

                else:
                    # In other un-delayed function,
                    # the calculated values of delayed keys should be push into the delay queues
                    code_lines = []
                    code_scope = {}
                    func_code_left = '\n'.join(tools.format_code(func_code).lefts)
                    # push delayed keys
                    for arg in delay_keys.keys():
                        if arg not in func_args:
                            continue
                        func_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                        func_delay_keys = func_keys.intersection(delay_keys[arg])
                        for key in func_delay_keys:
                            code_lines.append(f'  {arg}.delay_push({arg}["{key}"], var="{key}")')
                    if len(code_lines):
                        code_scope = {stripped_fname: func}
                        code_lines = [f'def {stripped_fname}_enhanced({tools.func_call(func_args)}):',
                                      f'  {stripped_fname}({tools.func_call(func_args)})'] + code_lines

                if len(code_lines):
                    # Compile the modified step function
                    func_code = '\n'.join(code_lines)
                    if profile._auto_pep8:
                        func_code = autopep8.fix_code(func_code)
                    exec(compile(func_code, '', 'exec'), code_scope)
                    func = code_scope[stripped_fname + '_enhanced']

                    if profile._show_formatted_code:
                        tools.show_code_str(func_code)
                        tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])

            # set the function to the this model
            setattr(self, stripped_fname, func)

            # get the function call
            arg_calls = []
            for arg in func_args:
                arg_calls.append(arg if arg in ARG_KEYWORDS else f"{self._name}.{arg}")
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg_calls)})'

            # get the function result
            results[stripped_fname] = {'call': func_call}

        return results

    def __step_mode_np_scalar(self):
        results = dict()

        # check number of the neurons/synapses,
        # too huge number of neurons/synapses sharply reduce running speed
        if self.ensemble.num > 4000:
            raise ModelUseError(f'The number of elements in {self._name} is too huge (>4000), '
                                f'please use numba backend or define vector_based model.')

        # check whether the model include heterogeneous parameters
        if len(self._pars.heters) > 0:
            raise ModelUseError(f'This model has heterogeneous parameters '
                                f'"{list(self._pars.heters.keys())}", '
                                f'it cannot be compiled in numpy mode.')

        # get the delay keys
        delay_keys = self._delay_keys

        for func in self._steps:
            func_name = func.__name__
            stripped_fname = tools.get_func_name(func, replace=True)

            # function argument
            func_args = inspect.getfullargspec(func).args

            # arg and arg2call
            code_arg, code_arg2call = [], {}
            for arg in func_args:
                if arg in ARG_KEYWORDS:
                    code_arg2call[arg] = arg
                    code_arg.append(arg)
                else:
                    try:
                        attr = getattr(self.ensemble, arg)
                    except AttributeError:
                        raise ModelUseError(f'Model "{self._name}" does not have the '
                                            f'required attribute "{arg}".')
                    if isinstance(attr, ObjState):
                        code_arg2call[f'{self._name}_{arg}'] = f'{self._name}.{arg}'
                        code_arg.append(f'{self._name}_{arg}')
                    else:
                        code_arg2call[arg] = f'{self._name}.{arg}'
                        code_arg.append(arg)
            code_arg = set(code_arg)

            # scope
            code_scope = {f'{stripped_fname}_origin': func}

            # codes
            has_ST = 'ST' in func_args
            has_pre = 'pre' in func_args
            has_post = 'post' in func_args
            if has_ST:  # have ST
                if has_pre and has_post:
                    code_arg.update(['pre2syn', 'post_ids', 'pre_indices'])
                    code_arg2call['pre2syn'] = f'{self._name}.pre2syn'
                    code_arg2call['post_ids'] = f'{self._name}.post_ids'
                    code_arg2call['pre_indices'] = f'{self._name}.pre_group.indices'

                    code_lines = [f'def {stripped_fname}({tools.func_call(code_arg)}):',
                                  f'  for pre_i in pre_indices.flatten():',
                                  f'    pre = {self._name}_pre.extract_by_index(pre_i)',
                                  f'    for _obj_i_ in pre2syn[pre_i]:',
                                  f'      post_i = post_ids[_obj_i_]',
                                  f'      post = {self._name}_post.extract_by_index(post_i)']
                    prefix = '  ' * 3
                    post_lines = [f'      {self._name}_post.update_by_index(post_i, post)',
                                  f'    {self._name}_pre.update_by_index(pre_i, pre)']
                elif has_pre:
                    code_arg.update(['pre2syn', 'pre_indices'])
                    code_arg2call['pre2syn'] = f'{self._name}.pre2syn'
                    code_arg2call['pre_indices'] = f'{self._name}.pre_group.indices'

                    code_lines = [f'def {stripped_fname}({tools.func_call(code_arg)}):',
                                  f'  for pre_i in pre_indices.flatten():',
                                  f'    pre = {self._name}_pre.extract_by_index(pre_i)',
                                  f'    for _obj_i_ in pre2syn[pre_i]:']
                    prefix = '  ' * 3
                    post_lines = [f'    {self._name}_pre.update_by_index(pre_i, pre)']
                elif has_post:
                    code_arg.update(['post2syn', 'post_indices'])
                    code_arg2call['post2syn'] = f'{self._name}.post2syn'
                    code_arg2call['post_indices'] = f'{self._name}.post_group.indices'
                    code_lines = [f'def {stripped_fname}({tools.func_call(code_arg)}):',
                                  f'  for post_i in post_indices.flatten():',
                                  f'    post = {self._name}_post.extract_by_index(post_i)',
                                  f'    for _obj_i_ in post2syn[post_i]:']
                    prefix = '  ' * 3
                    post_lines = [f'    {self._name}_post.update_by_index(post_i, post)']
                else:
                    code_lines = [f'def {stripped_fname}({tools.func_call(code_arg)}):',
                                  f'  for _obj_i_ in range({self.ensemble.num}):']
                    prefix = '  ' * 2
                    post_lines = []

                if func_name.startswith('_npbrain_delayed_'):
                    # Function with "delayed" decorator should use STATE pulled from the delay queue
                    for k in delay_keys.keys():
                        if k not in func_args: continue
                        code_lines.append(prefix + f'{k} = {self._name}_{k}.extract_by_index(_obj_i_, delay_pull=True)')
                    code_lines.append(prefix + f'{stripped_fname}_origin({tools.func_call(func_args)})')
                    code_lines.extend(post_lines)
                else:
                    if len(delay_keys):
                        # Other function without "delayed" decorator
                        for k in delay_keys.keys():
                            if k not in func_args: continue
                            code_lines.append(prefix + f'{k} = {self._name}_{k}.extract_by_index(_obj_i_)')
                        code_lines.append(prefix + f'{stripped_fname}_origin({tools.func_call(func_args)})')
                        for k in delay_keys.keys():
                            if k not in func_args: continue
                            code_lines.append(prefix + f'{self._name}_{k}.update_by_index(_obj_i_, {k})')
                        code_lines.extend(post_lines)

                        # Function without "delayed" decorator should push their
                        # updated STATE to the delay queue
                        func_code = tools.get_main_code(func)
                        func_code_left = '\n'.join(tools.format_code(func_code).lefts)
                        for k, v in delay_keys.items():
                            func_keys = set(re.findall(r'' + k + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                            func_delay_keys = func_keys.intersection(v)
                            if len(func_delay_keys) > 0:
                                for key in func_delay_keys:
                                    code_lines.append(f'  {self._name}_{k}.delay_push('
                                                      f'{self._name}_{k}["{key}"], "{key}")')
                    else:
                        code_lines.append(prefix + f'ST = {self._name}_ST.extract_by_index(_obj_i_)')
                        code_lines.append(prefix + f'{stripped_fname}_origin({tools.func_call(func_args)})')
                        code_lines.append(prefix + f'{self._name}_ST.update_by_index(_obj_i_, ST)')

            else:  # doesn't have ST
                try:
                    assert not has_post and not has_pre
                except AssertionError:
                    raise ModelDefError(f'Unknown "{stripped_fname}" function structure.')
                code_lines = [f'def {stripped_fname}({tools.func_call(code_arg)}):',
                              f'  for _obj_i_ in range({self.ensemble.num}):',
                              f'    {stripped_fname}_origin({tools.func_call(func_args)})']

            # append the final results
            code_lines.insert(0, f'# "{stripped_fname}" step function in {self._name}')

            # compile the updated function
            func_code = '\n'.join(code_lines)
            if profile._auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = code_scope[stripped_fname]
            if profile._show_formatted_code:
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])

            # set the function to the model
            setattr(self, stripped_fname, func)

            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_arg))]
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg2calls)})'

            # final
            results[stripped_fname] = {'call': func_call}

        return results

    def __step_substitute_integrator(self, func):
        # get code and code lines
        func_code = tools.deindent(tools.get_main_code(func))
        code_lines = tools.format_code(func_code).lines

        # get function scope
        vars = inspect.getclosurevars(func)
        code_scope = dict(vars.nonlocals)
        code_scope.update(vars.globals)
        code_scope.update({self._name: self.ensemble})
        if len(code_lines) == 0:
            return '', code_scope

        # code scope update
        scope_to_add = {}
        scope_to_del = set()
        need_add_mapping_scope = False
        for k, v in code_scope.items():
            if isinstance(v, Integrator):
                if profile._merge_steps:
                    need_add_mapping_scope = True

                    # locate the integration function
                    int_func_name = v.py_func_name
                    for line_no, line in enumerate(code_lines):
                        if int_func_name in tools.get_identifiers(line):
                            break

                    # get integral function line indent
                    line_indent = tools.get_line_indent(line)
                    indent = ' ' * line_indent

                    # get the replace line and arguments need to replace
                    new_line, args, kwargs = tools.replace_func(line, int_func_name)

                    # append code line of argument replacement
                    func_args = v.diff_eq.func_args
                    append_lines = [indent + f'_{v.py_func_name}_{func_args[i]} = {args[i]}'
                                    for i in range(len(args))]
                    for arg in func_args[len(args):]:
                        append_lines.append(indent + f'_{v.py_func_name}_{arg} = {kwargs[arg]}')

                    # append numerical integration code lines
                    try:
                        append_lines.extend([indent + l for l in v.update_code.split('\n')])
                    except AttributeError:
                        raise ModelUseError(f'Integrator {v} has no "update_code". This may be caused by \n'
                                            f'the declaration of "profile.set(backend="numba")" is not \n'
                                            f'before the definition of the model.')
                    append_lines.append(indent + new_line)

                    # add appended lines into the main function code lines
                    code_lines = code_lines[:line_no] + append_lines + code_lines[line_no + 1:]

                    # get scope variables to delete
                    scope_to_del.add(k)
                    for k_, v_ in v.code_scope.items():
                        if callable(v_):
                            v_ = tools.numba_func(v_, params=self._pars.updates)
                        scope_to_add[k_] = v_

                else:
                    if not self._model.vector_based:
                        for ks, vs in tools.get_func_scope(v.update_func, include_dispatcher=True).items():
                            if ks in self._pars.heters:
                                raise ModelUseError(f'Heterogeneous parameter "{ks}" is not in step functions, '
                                                    f'it will not work.\n'
                                                    f'Please set "brainpy.profile._merge_steps = True" to try to '
                                                    f'merge parameter "{ks}" into the step functions.')
                    code_scope[k] = tools.numba_func(v.update_func, params=self._pars.updates)

            elif type(v).__name__ == 'function':
                code_scope[k] = tools.numba_func(v, params=self._pars.updates)


        # update code scope
        if need_add_mapping_scope:
            code_scope.update(get_mapping_scope())
        code_scope.update(scope_to_add)
        for k in scope_to_del:
            code_scope.pop(k)

        # return code lines and code scope
        return '\n'.join(code_lines), code_scope

    def __step_mode_nb_vector(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        delay_keys = self._delay_keys
        all_heter_pars = list([k for k in self._model.heter_params_replace.keys()
                               if k in self._pars.updates])

        for func in self._steps:
            # information about the function
            func_name = func.__name__
            stripped_fname = tools.get_func_name(func, replace=True)
            func_args = inspect.getfullargspec(func).args

            # initialize code namespace
            used_args, code_arg2call, code_lines = set(), {}, []
            func_code, code_scope = self.__step_substitute_integrator(func)

            # check function code
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in ARG_KEYWORDS and
                          isinstance(getattr(self.ensemble, k), ObjState)}
            except AttributeError:
                raise ModelUseError(f'Model "{self._name}" does not have all the '
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
                        # Function with "delayed" decorator should use ST pulled from the delay queue
                        if func_name.startswith('_npbrain_delayed_'):
                            if arg in delay_keys:
                                dout = f'{self._name}_{arg}_dout'
                                add_args.add(dout)
                                code_arg2call[dout] = f'{self._name}.{arg}._delay_out'
                                for st_k in delay_keys[arg]:
                                    p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                    r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {dout}]"
                                    func_code = re.sub(r'' + p, r, func_code)
                        else:
                            # Function without "delayed" decorator should push their
                            # updated ST to the delay queue
                            if arg in delay_keys:
                                func_code_left = '\n'.join(tools.format_code(func_code).lefts)
                                func_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                                func_delay_keys = func_keys.intersection(delay_keys[arg])
                                if len(func_delay_keys) > 0:
                                    din = f'{self._name}_{arg}_din'
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
            arg_substitute = {}
            for arg in used_args:
                if arg in ARG_KEYWORDS:
                    new_arg = arg
                    code_arg2call[arg] = arg
                else:
                    new_arg = f'{self._name}_{arg}'
                    arg_substitute[arg] = new_arg
                    if isinstance(getattr(self.ensemble, arg), ObjState):
                        code_arg2call[new_arg] = f'{self._name}.{arg}["_data"]'
                    else:
                        code_arg2call[new_arg] = f'{self._name}.{arg}'
                code_args.add(new_arg)
            # substitute heterogeneous parameters
            for k in code_scope.keys():
                if k in self._model.heter_params_replace:
                    arg_substitute[k] = self._model.heter_params_replace[k]
                    if k in all_heter_pars:
                        all_heter_pars.remove(k)
            # substitute
            func_code = tools.word_replace(func_code, arg_substitute)

            # update code scope
            for k in list(code_scope.keys()):
                if k in self._pars.updates:
                    code_scope[k] = self._pars.updates[k]

            # final
            code_lines = func_code.split('\n')
            code_lines.insert(0, f'# "{stripped_fname}" step function of {self._name}')
            code_lines.append('\n')

            # code to compile
            code_to_compile = [f'def {stripped_fname}({tools.func_call(code_args)}):'] + code_lines
            func_code = '\n '.join(code_to_compile)
            if profile._auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = tools.jit(code_scope[stripped_fname])
            if profile._show_formatted_code and not profile._merge_steps:
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])

            # set the function to the model
            setattr(self, stripped_fname, func)
            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg2calls)})'

            results[stripped_fname] = {'scopes': code_scope, 'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines, 'call': func_call}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. BrainPy cannot recognize. Please check.')

        return results

    def __step_mode_nb_scalar(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        delay_keys = self._delay_keys
        all_heter_pars = set(self._pars.heters.keys())

        for i, func in enumerate(self._steps):
            func_name = func.__name__

            # get code scope
            used_args, code_arg2call, code_lines = set(), {}, []
            func_args = inspect.getfullargspec(func).args
            func_code, code_scope = self.__step_substitute_integrator(func)
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in ARG_KEYWORDS and
                          isinstance(getattr(self.ensemble, k), ObjState)}
            except AttributeError:
                raise ModelUseError(f'Model "{self._name}" does not have all the '
                                    f'required attributes: {func_args}.')

            # update functions in code scope
            for k, v in code_scope.items():
                if callable(v):
                    code_scope[k] = tools.numba_func(func=v, params=self._pars.updates)

            add_args = set()
            # substitute STATE item access to index
            for i, arg in enumerate(func_args):
                used_args.add(arg)
                if len(states) == 0:
                    continue
                if arg in states:
                    st = states[arg]
                    var2idx = st['_var2idx']
                    if self.ensemble._is_state_attr(arg):
                        if func_name.startswith('_npbrain_delayed_'):
                            if arg in delay_keys:
                                dout = f'{self._name}_{arg}_dout'
                                add_args.add(dout)
                                code_arg2call[dout] = f'{self._name}.{arg}._delay_out'
                                # Function with "delayed" decorator should use ST pulled from the delay queue
                                for st_k in delay_keys[arg]:
                                    p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                    r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {dout}, _obj_i_]"
                                    func_code = re.sub(r'' + p, r, func_code)
                        else:
                            if arg in delay_keys:
                                # Function without "delayed" decorator should push
                                # their updated ST to the delay queue
                                func_code_left = '\n'.join(tools.format_code(func_code).lefts)
                                func_keys = set(re.findall(r'' + arg + r'\[[\'"](\w+)[\'"]\]', func_code_left))
                                func_delay_keys = func_keys.intersection(delay_keys[arg])
                                if len(func_delay_keys) > 0:
                                    din = f'{self._name}_{arg}_din'
                                    add_args.add(din)
                                    code_arg2call[din] = f'{self._name}.{arg}._delay_in'
                                    for st_k in func_delay_keys:
                                        right = f'{arg}[{var2idx[st_k]}]'
                                        left = f"{arg}[{var2idx['_' + st_k + '_offset']} + {din}]"
                                        func_code += f'\n{left} = {right}'
                            for st_k in st._keys:
                                p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                r = f"{arg}[{var2idx[st_k]}, _obj_i_]"
                                func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'pre':
                        for st_k in st._keys:
                            p = f'pre\[([\'"]{st_k}[\'"])\]'
                            r = f"pre[{var2idx[st_k]}, _pre_i_]"
                            func_code = re.sub(r'' + p, r, func_code)
                    elif arg == 'post':
                        for st_k in st._keys:
                            p = f'post\[([\'"]{st_k}[\'"])\]'
                            r = f"post[{var2idx[st_k]}, _post_i_]"
                            func_code = re.sub(r'' + p, r, func_code)
                    else:
                        raise ValueError

            # substitute arguments
            code_args = add_args
            arg_substitute = {}
            for arg in used_args:
                if arg in ARG_KEYWORDS:
                    new_arg = arg
                    code_arg2call[arg] = arg
                else:
                    new_arg = f'{self._name}_{arg}'
                    arg_substitute[arg] = new_arg
                    if isinstance(getattr(self.ensemble, arg), ObjState):
                        code_arg2call[new_arg] = f'{self._name}.{arg}["_data"]'
                    else:
                        code_arg2call[new_arg] = f'{self._name}.{arg}'
                code_args.add(new_arg)
            # substitute multi-dimensional parameter "p" to "p[_ni_]"
            for p in self._pars.heters.keys():
                if p in code_scope:
                    arg_substitute[p] = f'{p}[_obj_i_]'
            # substitute
            func_code = tools.word_replace(func_code, arg_substitute)

            # add the for loop in the start of the main code
            try:
                assert 'ST' in func_args
            except AssertionError:
                raise ModelUseError("In numba mode, scalar-based model only support function has 'ST' argument.")
            has_pre = 'pre' in func_args
            has_post = 'post' in func_args
            if has_pre and has_post:
                code_args.add(f'{self._name}_post_ids')
                code_arg2call[f'{self._name}_post_ids'] = f'{self._name}.post_ids'
                code_args.add(f'{self._name}_pre2syn')
                code_arg2call[f'{self._name}_pre2syn'] = f'{self._name}.pre2syn'
                code_args.add(f'{self._name}_pre_indices')
                code_arg2call[f'{self._name}_pre_indices'] = f'{self._name}.pre_group.indices'
                # f'for _pre_i_ in numba.prange({self.ensemble.pre_group.num}):',
                code_lines = [f'{self._name}_pre_indices = {self._name}_pre_indices.flatten()',
                              f'for _pre_i_ in numba.prange({self.ensemble.pre_group.num}):',
                              f'  _pre_i_ = {self._name}_pre_indices[_pre_i_]',
                              f'  for _obj_i_ in {self._name}_pre2syn[_pre_i_]:',
                              f'    _post_i_ = {self._name}_post_ids[_obj_i_]']
                blank = '  ' * 2
            elif has_pre:
                code_args.add(f'{self._name}_pre2syn')
                code_arg2call[f'{self._name}_pre2syn'] = f'{self._name}.pre2syn'
                code_args.add(f'{self._name}_pre_indices')
                code_arg2call[f'{self._name}_pre_indices'] = f'{self._name}.pre_group.indices'
                code_lines = [f'{self._name}_pre_indices = {self._name}_pre_indices.flatten()',
                              f'for _pre_i_ in numba.prange({self.ensemble.pre_group.num}):',
                              f'  _pre_i_ = {self._name}_pre_indices[_pre_i_]',
                              f'  for _obj_i_ in {self._name}_pre2syn[_pre_i_]:']
                blank = '  ' * 2
            elif has_post:
                code_args.add(f'{self._name}_post2syn')
                code_arg2call[f'{self._name}_post2syn'] = f'{self._name}.post2syn'
                code_args.add(f'{self._name}_post_indices')
                code_arg2call[f'{self._name}_post_indices'] = f'{self._name}.post_group.indices'
                code_lines = [f'{self._name}_post_indices = {self._name}_post_indices.flatten()',
                              f'for _post_i_ in numba.prange({self.ensemble.post_group.num}):',
                              f'  _post_i_ = {self._name}_post_indices[_post_i_]',
                              f'  for _obj_i_ in {self._name}_post2syn[_post_i_]:']
                blank = '  ' * 2
            else:
                code_lines = [f'for _obj_i_ in numba.prange({self.ensemble.num}):']
                blank = '  ' * 1

            # add the main code (user defined)
            code_lines.extend([blank + l for l in func_code.split('\n')])
            code_lines.append('\n')
            stripped_fname = tools.get_func_name(func, replace=True)
            code_lines.insert(0, f'# "{stripped_fname}" step function of {self._name}')

            # update code scope
            code_scope['numba'] = import_module('numba')
            for k in list(code_scope.keys()):
                if k in self._pars.updates:
                    code_scope[k] = self._pars.updates[k]
                if k in all_heter_pars:
                    all_heter_pars.remove(k)

            # code to compile
            code_to_compile = [f'def {stripped_fname}({tools.func_call(code_args)}):'] + code_lines
            func_code = '\n '.join(code_to_compile)
            if profile._auto_pep8:
                func_code = autopep8.fix_code(func_code)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = tools.jit(code_scope[stripped_fname])
            if profile._show_formatted_code and not profile._merge_steps:
                tools.show_code_str(func_code)
                tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])
            # set the function to the model
            setattr(self, stripped_fname, func)
            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg2calls)})'

            # the final results
            results[stripped_fname] = {'scopes': code_scope, 'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines, 'call': func_call}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. NumpyBrain cannot recognize. Please check.')

        return results

    def merge_steps(self, compiled_result, mode):
        codes_of_calls = []  # call the compiled functions

        if mode == 'numpy':  # numpy mode
            for item in self._schedule:
                if item in compiled_result:
                    func_call = compiled_result[item]['call']
                    codes_of_calls.append(func_call)

        elif mode == 'numba':  # numba mode

            if profile._merge_steps:
                lines, code_scopes, args, arg2calls = [], dict(), set(), dict()
                for item in self._schedule:
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
                if profile._auto_pep8:
                    func_code = autopep8.fix_code(func_code)
                exec(compile(func_code, '', 'exec'), code_scopes)

                self.merge_func = tools.jit(code_scopes['merge_func'])
                func_call = f'{self._name}.runner.merge_func({tools.func_call(arg2calls_list)})'
                codes_of_calls.append(func_call)

                if profile._show_formatted_code:
                    tools.show_code_str(func_code)
                    tools.show_code_scope(code_scopes, ('__builtins__', 'merge_func'))

            else:
                for item in self._schedule:
                    if item in compiled_result:
                        func_call = compiled_result[item]['call']
                        codes_of_calls.append(func_call)

        else:
            raise NotImplementedError

        return codes_of_calls

    def get_schedule(self):
        return self._schedule

    def set_schedule(self, schedule):
        try:
            assert isinstance(schedule, (list, tuple))
        except AssertionError:
            raise ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + self._step_names
        for s in schedule:
            try:
                assert s in all_func_names
            except AssertionError:
                raise ModelUseError(f'Unknown step function "{s}" for model "{self._name}".')
        self._schedule = schedule
