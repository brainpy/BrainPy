# -*- coding: utf-8 -*-

import inspect
import re
from importlib import import_module

from . import constants
from .types import ObjState
from .. import numpy as np
from .. import profile
from .. import tools
from ..errors import ModelDefError
from ..errors import ModelUseError
from ..integration.integrator import Integrator
from ..integration.sympy_tools import get_mapping_scope

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
            code_args.add('_i')
            code_arg2call['_i'] = '_i'

        # check data operations
        # ----------------------
        for _, _, ops, _ in key_val_ops_types:
            if ops not in constants.INPUT_OPERATIONS:
                raise ModelUseError(f'Only support five input operations: {list(constants.INPUT_OPERATIONS.keys())}')

        # generate code of input function
        # --------------------------------
        input_idx = 0
        for key, val, ops, data_type in key_val_ops_types:
            if key in self._inputs:
                raise ModelUseError('Only support assignment for each key once.')
            else:
                self._inputs[key] = (val, ops, data_type)

            attr_item = key.split('.')

            # get the left side #
            if len(attr_item) == 1 and (attr_item[0] not in self.ensemble.ST):  # if "item" is the model attribute
                attr, item = attr_item[0], ''
                if not hasattr(self, attr):
                    raise ModelUseError(f'Model "{self._name}" doesn\'t have "{attr}" attribute", '
                                        f'and "{self._name}.ST" doesn\'t have "{attr}" field.')
                if not isinstance(getattr(self.ensemble, attr), np.ndarray):
                    raise ModelUseError(f'BrainPy only support input to arrays.')

                left = attr
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

                idx = getattr(self.ensemble, attr)['_var2idx'][item]
                left = f'{attr}[{idx}]'
                code_args.add(attr)
                code_arg2call[attr] = f'{self._name}.{attr}["_data"]'

            # get the right side #
            right = f'{key.replace(".", "_")}_inp'
            setattr(self, right, val)
            code_args.add(right)
            code_arg2call[right] = f'{self._name}.runner.{right}'
            if data_type == 'iter':
                right = right + '[_i]'
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
        self.input_step = code_scope['input_step']
        # if profile.is_jit_backend():
        #     self.input_step = tools.jit(self.input_step)
        if profile._show_format_code:
            if not profile._merge_integrators:
                tools.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                tools.show_code_scope(code_scope, ['__builtins__', 'input_step'])

        # format function call
        arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
        func_call = f'{self._name}.runner.input_step({tools.func_call(arg2call)})'

        return {'input': {'scopes': code_scope,
                          'args': code_args,
                          'arg2calls': code_arg2call,
                          'codes': code_lines,
                          'call': func_call}}

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

                idx_name = f'idx{mon_idx}_{attr}'
                mon_name = f'mon_{attr}'
                target_name = attr
                if indices is None:
                    line = f'{mon_name}[_i] = {target_name}'
                else:
                    line = f'{mon_name}[_i] = {target_name}[{idx_name}]'
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

                idx_name = f'idx{mon_idx}_{attr}_{item}'
                idx = getattr(self.ensemble, attr)['_var2idx'][item]
                mon_name = f'mon_{attr}_{item}'
                target_name = attr
                if indices is None:
                    line = f'{mon_name}[_i] = {target_name}[{idx}]'
                else:
                    line = f'{mon_name}[_i] = {target_name}[{idx}][{idx_name}]'
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
        code_args.add('_i')
        code_arg2call['_i'] = '_i'

        # compile function
        code_to_compile = [f'def monitor_step({tools.func_call(code_args)}):'] + code_lines
        func_code = '\n  '.join(code_to_compile)
        exec(compile(func_code, '', 'exec'), code_scope)
        monitor_step = code_scope['monitor_step']
        # if profile.is_jit_backend():
        #     monitor_step = tools.jit(monitor_step)
        self.monitor_step = monitor_step

        # format function call
        arg2call = [code_arg2call[arg] for arg in sorted(list(code_args))]
        func_call = f'{self._name}.runner.monitor_step({tools.func_call(arg2call)})'

        if profile._show_format_code:
            if not profile._merge_steps:
                tools.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                tools.show_code_scope(code_scope, ('__builtins__', 'monitor_step'))

        return mon, {'monitor': {'scopes': code_scope,
                                 'args': code_args,
                                 'arg2calls': code_arg2call,
                                 'codes': code_lines,
                                 'call': func_call}}

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

    def format_step_code(self, func):
        """Format code of user defined step function.

        Parameters
        ----------
        func : callable
            The user defined function.

        Returns
        -------
        code_lines : list, tuple
            The code lines.
        """
        func_code = tools.deindent(tools.get_main_code(func))
        return tools.format_code(func_code).lines

    def step_substitute_integrator(self, func):
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
        code_lines = self.format_step_code(func)

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
                if profile._merge_integrators:
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
                    append_lines = [indent + f'_{v.py_func_name}_{func_args[i]} = {args[i]}'
                                    for i in range(len(args))]
                    for arg in func_args[len(args):]:
                        append_lines.append(indent + f'_{v.py_func_name}_{arg} = {kwargs[arg]}')

                    # append numerical integration code lines
                    try:
                        append_lines.extend([indent + l for l in v.update_code.split('\n')])
                    except:
                        raise ModelUseError(f'Integrator {v} has no "update_code". This may be caused by \n'
                                            f'that "profile.set(backend="numba")" is not declared \n'
                                            f'before the definition of the model.')
                    append_lines.append(indent + new_line)

                    # add appended lines into the main function code lines
                    code_lines = code_lines[:line_no] + append_lines + code_lines[line_no + 1:]

                    # get scope variables to delete
                    scope_to_del.add(k)
                    for k_, v_ in v.code_scope.items():
                        if profile.is_jit_backend() and callable(v_):
                            v_ = tools.numba_func(v_, params=self._pars.updates)
                        scope_to_add[k_] = v_

                else:
                    if self._model.mode == constants.SCALAR_MODE:
                        for ks, vs in tools.get_func_scope(v.update_func, include_dispatcher=True).items():
                            if ks in self._pars.heters:
                                raise ModelUseError(f'Heterogeneous parameter "{ks}" is not in step functions, '
                                                    f'it will not work.\n'
                                                    f'Please set "brainpy.profile.set(merge_steps=True)" to try to '
                                                    f'merge parameter "{ks}" into the step functions.')
                    if profile.is_jit_backend():
                        code_scope[k] = tools.numba_func(v.update_func, params=self._pars.updates)

            elif type(v).__name__ == 'function':
                if profile.is_jit_backend():
                    code_scope[k] = tools.numba_func(v, params=self._pars.updates)

        # update code scope
        if need_add_mapping_scope:
            code_scope.update(get_mapping_scope())
        code_scope.update(scope_to_add)
        for k in scope_to_del:
            code_scope.pop(k)

        # return code lines and code scope
        return '\n'.join(code_lines), code_scope

    def step_vector_model(self):
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
            func_code, code_scope = self.step_substitute_integrator(func)

            # check function code
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in constants.ARG_KEYWORDS and
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
                        # Function with "delayed" decorator should use
                        # ST pulled from the delay queue
                        if func_name.startswith('_npbrain_delayed_'):
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
                                func_code_left = '\n'.join(tools.format_code(func_code).lefts)
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
                    if isinstance(getattr(self.ensemble, arg), ObjState):
                        code_arg2call[arg] = f'{self._name}.{arg}["_data"]'
                    else:
                        code_arg2call[arg] = f'{self._name}.{arg}'
                code_args.add(arg)
            arg_substitute = {}
            # substitute heterogeneous parameters
            for k in code_scope.keys():
                if k in self._model.heter_params_replace:
                    arg_substitute[k] = self._model.heter_params_replace[k]
                    if k in all_heter_pars:
                        all_heter_pars.remove(k)
            # substitute
            if len(arg_substitute):
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
            exec(compile(func_code, '', 'exec'), code_scope)
            func = tools.jit(code_scope[stripped_fname]) \
                if profile.is_jit_backend() \
                else code_scope[stripped_fname]
            if profile._show_format_code and not profile._merge_steps:
                tools.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])

            # set the function to the model
            setattr(self, stripped_fname, func)
            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg2calls)})'

            results[stripped_fname] = {'scopes': code_scope,
                                       'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines,
                                       'call': func_call}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. BrainPy cannot recognize. Please check.')

        return results

    def step_scalar_model(self):
        results = dict()

        # check whether the model include heterogeneous parameters
        delay_keys = self._delay_keys
        all_heter_pars = set(self._pars.heters.keys())

        for i, func in enumerate(self._steps):
            func_name = func.__name__

            # get code scope
            used_args, code_arg2call, code_lines = set(), {}, []
            func_args = inspect.getfullargspec(func).args
            func_code, code_scope = self.step_substitute_integrator(func)
            try:
                states = {k: getattr(self.ensemble, k) for k in func_args
                          if k not in constants.ARG_KEYWORDS and
                          isinstance(getattr(self.ensemble, k), ObjState)}
            except AttributeError:
                raise ModelUseError(f'Model "{self._name}" does not have all the '
                                    f'required attributes: {func_args}.')

            # update functions in code scope
            for k, v in code_scope.items():
                if profile.is_jit_backend() and callable(v):
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
                            if len(delay_keys):
                                dout = f'{arg}_dout'
                                add_args.add(dout)
                                code_arg2call[dout] = f'{self._name}.{arg}._delay_out'
                                # Function with "delayed" decorator should use ST pulled from the delay queue
                                for st_k in delay_keys:
                                    p = f'{arg}\[([\'"]{st_k}[\'"])\]'
                                    r = f"{arg}[{var2idx['_' + st_k + '_offset']} + {dout}, _obj_i_]"
                                    func_code = re.sub(r'' + p, r, func_code)
                        else:
                            if len(delay_keys):
                                # Function without "delayed" decorator should push
                                # their updated ST to the delay queue
                                func_code_left = '\n'.join(tools.format_code(func_code).lefts)
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
            for arg in used_args:
                if arg in constants.ARG_KEYWORDS:
                    code_arg2call[arg] = arg
                else:
                    if isinstance(getattr(self.ensemble, arg), ObjState):
                        code_arg2call[arg] = f'{self._name}.{arg}["_data"]'
                    else:
                        code_arg2call[arg] = f'{self._name}.{arg}'
                code_args.add(arg)
            # substitute multi-dimensional parameter "p" to "p[_ni_]"
            arg_substitute = {}
            for p in self._pars.heters.keys():
                if p in code_scope:
                    arg_substitute[p] = f'{p}[_obj_i_]'
            # substitute
            if len(arg_substitute):
                func_code = tools.word_replace(func_code, arg_substitute)

            # add the for loop in the start of the main code
            has_pre = 'pre' in func_args
            has_post = 'post' in func_args
            code_lines = [f'for _obj_i_ in numba.prange({self.ensemble.num}):']
            if has_pre:
                code_args.add(f'pre_ids')
                code_arg2call[f'pre_ids'] = f'{self._name}.pre_ids'
                code_lines.append(f'  _pre_i_ = pre_ids[_obj_i_]')
            if has_post:
                code_args.add(f'post_ids')
                code_arg2call[f'post_ids'] = f'{self._name}.post_ids'
                code_lines.append(f'  _post_i_ = post_ids[_obj_i_]')

            # add the main code (user defined)
            code_lines.extend(['  ' + l for l in func_code.split('\n')])
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
            func_code = '\n  '.join(code_to_compile)
            exec(compile(func_code, '', 'exec'), code_scope)
            func = tools.jit(code_scope[stripped_fname]) if profile.is_jit_backend() \
                else code_scope[stripped_fname]
            if profile._show_format_code and not profile._merge_steps:
                tools.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                tools.show_code_scope(code_scope, ['__builtins__', stripped_fname])
            # set the function to the model
            setattr(self, stripped_fname, func)
            # function call
            arg2calls = [code_arg2call[arg] for arg in sorted(list(code_args))]
            func_call = f'{self._name}.runner.{stripped_fname}({tools.func_call(arg2calls)})'

            # the final results
            results[stripped_fname] = {'scopes': code_scope,
                                       'args': code_args,
                                       'arg2calls': code_arg2call,
                                       'codes': code_lines,
                                       'call': func_call}

        # WARNING: heterogeneous parameter may not in the main step functions
        if len(all_heter_pars) > 0:
            raise ModelDefError(f'Heterogeneous parameters "{list(all_heter_pars)}" are not defined '
                                f'in main step function. NumpyBrain cannot recognize. Please check.')

        return results

    def merge_codes(self, compiled_result):
        codes_of_calls = []  # call the compiled functions

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
            exec(compile(func_code, '', 'exec'), code_scopes)

            if profile.is_jit_backend():
                func = tools.jit(code_scopes['merge_func'])
            else:
                func = code_scopes['merge_func']
            self.merge_func = func
            func_call = f'{self._name}.runner.merge_func({tools.func_call(arg2calls_list)})'
            codes_of_calls.append(func_call)

            if profile._show_format_code:
                tools.show_code_str(func_code.replace('def ', f'def {self._name}_'))
                tools.show_code_scope(code_scopes, ('__builtins__', 'merge_func'))

        else:
            for item in self._schedule:
                if item in compiled_result:
                    func_call = compiled_result[item]['call']
                    codes_of_calls.append(func_call)

        return codes_of_calls

    def get_schedule(self):
        return self._schedule

    def set_schedule(self, schedule):
        if not isinstance(schedule, (list, tuple)):
            raise ModelUseError('"schedule" must be a list/tuple.')
        all_func_names = ['input', 'monitor'] + self._step_names
        for s in schedule:
            if s not in all_func_names:
                raise ModelUseError(f'Unknown step function "{s}" for model "{self._name}".')
        self._schedule = schedule


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
        from brainpy.core_system.neurons import NeuGroup
        try:
            isinstance(ensemble, NeuGroup)
        except AssertionError:
            raise ModelUseError(f'{self.__name__} only supports the instance of NeuGroup.')

        # initialization
        super(TrajectoryRunner, self).__init__(ensemble=ensemble)

        # check targeted variables
        try:
            isinstance(target_vars, (list, tuple))
        except AssertionError:
            raise ModelUseError('"target_vars" must be a list/tuple.')
        for var in target_vars:
            try:
                assert var in self._model.variables
            except AssertionError:
                raise ModelUseError(f'"{var}" in "target_vars" is not defined in model "{self._model.name}".')
        self.target_vars = target_vars

        # check fixed variables
        try:
            if fixed_vars is not None:
                isinstance(fixed_vars, dict)
            else:
                fixed_vars = dict()
        except AssertionError:
            raise ModelUseError('"fixed_vars" must be a dict.')
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

    def format_step_code(self, func):
        """Format code of user defined step function.

        Parameters
        ----------
        func : callable
            The user defined function.

        Returns
        -------
        code_lines : list, tuple
            The code lines.
        """
        func_code = tools.deindent(tools.get_main_code(func))
        return tools.format_code_for_trajectory(func_code, self.fixed_vars).lines
