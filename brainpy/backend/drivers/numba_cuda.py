# -*- coding: utf-8 -*-

import ast

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.simulation.brainobjects import SynConn, NeuGroup
from .numba_cpu import NumbaCPUNodeDriver
from .numba_cpu import StepFuncReader
from .numba_cpu import analyze_step_func
from . import utils

try:
    import numba
    from numba import cuda

    if not cuda.is_available():
        raise errors.PackageMissingError('cuda should be installed when using numba-cuda backend.')

except ModuleNotFoundError:
    raise errors.BackendNotInstalled('numba')



__all__ = [
    'NumbaCudaNodeDriver',
]


class CudaStepFuncReader(StepFuncReader):
    """The tasks done in "CudaStepFuncReader" are:

    - Find all expressions, including Assign, AugAssign, For loop, If-else condition.
    - Find all delay push and pull.
    - Find all atomic operations.

    """

    def __init__(self, host):
        super(CudaStepFuncReader, self).__init__(host=host)
        self.need_add_cuda = False

    def check_atomic_ops(self, target):
        if isinstance(self.host, SynConn) and isinstance(target, ast.Subscript):
            values = self.visit_attr(target.value)
            slice_ = tools.ast2code(ast.fix_missing_locations(target.slice))
            if len(values) >= 3 and values[-1] in backend.CLASS_KEYWORDS:
                obj = getattr(self.host, values[-2])
                if isinstance(obj, NeuGroup):
                    target_ = '.'.join(values[::-1])
                    return target_, slice_
        return None

    def visit_Assign(self, node, level=0):
        if node not in self.visited_nodes:
            prefix = '  ' * level

            success = False
            check = None
            if len(node.targets) == 1:
                check = self.check_atomic_ops(node.targets[0])
            if check is not None:
                target, slice_ = check
                left = f'{target}[{slice_}]'
                if isinstance(node.value, ast.BinOp):
                    r_left = tools.ast2code(ast.fix_missing_locations(node.value.left))
                    r_right = tools.ast2code(ast.fix_missing_locations(node.value.right))
                    op = tools.ast2code(ast.fix_missing_locations(node.value.op))
                    if op in ['+', '-']:
                        if r_left == left:
                            if op == '+':
                                expr = r_right
                            if op == '-':
                                expr = f'- {r_right}'
                        elif r_left == '-' + left:
                            if op == '+':
                                expr = f"-2 * {left} + {r_right}"
                            if op == '-':
                                expr = f"-2 * {left} - {r_right}"
                        elif r_right == left:
                            if op == '+':
                                expr = r_left
                            if op == '-':
                                expr = f"{r_left} - 2 * {left}"
                        elif r_right == '-' + left:
                            if op == '+':
                                expr = f"{r_left} - 2 * {left}"
                            if op == '-':
                                expr = r_left
                        else:
                            raise ValueError(f'Cannot assign an automic operation for this '
                                             f'expression: {r_left} {op} {r_right}')
                        self.lefts.append(left)
                        self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')
                        success = True
                        self.need_add_cuda = True
            expr = tools.ast2code(ast.fix_missing_locations(node.value))
            self.rights.append(expr)
            if not success:
                targets = []
                for target in node.targets:
                    targets.append(tools.ast2code(ast.fix_missing_locations(target)))
                _target = ' = '.join(targets)
                self.lefts.append(_target)
                self.lines.append(f'{prefix}{_target} = {expr}')

            self.visited_nodes.add(node)

        self.generic_visit(node)

    def visit_AugAssign(self, node, level=0):
        if node not in self.visited_nodes:

            prefix = '  ' * level
            op = tools.ast2code(ast.fix_missing_locations(node.op))
            expr = tools.ast2code(ast.fix_missing_locations(node.value))

            check = self.check_atomic_ops(node.target)
            if check is None:
                target = tools.ast2code(ast.fix_missing_locations(node.target))
                self.lefts.append(target)
                self.rights.append(f'{target} {op} {expr}')
                self.lines.append(f"{prefix}{target} = {target} {op} {expr}")
            else:
                target, slice_ = check
                if op == '+':
                    expr = expr
                elif op == '-':
                    expr = '-' + expr
                else:
                    raise ValueError(f'Cannot assign an automic operation for this '
                                     f'expression: {target}[{slice_}] {op}= {expr}')
                self.lefts.append(target)
                self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')
                self.need_add_cuda = True

            self.visited_nodes.add(node)

        self.generic_visit(node)


def class2func(cls_func, host, func_name=None, show_code=False):
    """Transform the function in a class into the ordinary function which is
    compatible with the Numba JIT compilation.

    Parameters
    ----------
    cls_func : function
        The function of the instantiated class.
    func_name : str
        The function name. If not given, it will get the function by `cls_func.__name__`.
    show_code : bool
        Whether show the code.

    Returns
    -------
    new_func : function
        The transformed function.
    """
    class_arg, arguments = utils.get_args(cls_func)
    func_name = cls_func.__name__ if func_name is None else func_name
    host_name = host.name

    # get code analysis
    # --------
    analyzed_results = analyze_step_func(host=host, f=cls_func)
    delay_call = analyzed_results['delay_call']
    main_code = analyzed_results['code_string']
    code_scope = analyzed_results['code_scope']
    self_data_in_right = analyzed_results['self_data_in_right']
    self_data_with_index_in_left = analyzed_results['self_data_with_index_in_left']
    data_need_pass = sorted(list(set(self_data_in_right + self_data_with_index_in_left)))

    # reprocess the normal function


    # transform the cpu data to cuda data


    # arguments 1: the function intrinsic needed arguments
    # -----------
    calls = []
    for arg in arguments:
        if hasattr(host, arg):
            calls.append(f'{host_name}.{arg}')
        elif arg in backend.SYSTEM_KEYWORDS:
            calls.append(arg)
        else:
            raise errors.ModelDefError(f'Step function "{func_name}" of {host} '
                                       f'define an unknown argument "{arg}" which is not '
                                       f'an attribute of {host} nor the system keywords '
                                       f'{backend.SYSTEM_KEYWORDS}.')

    # reprocess delay function
    # -----------
    replaces_early = {}
    replaces_later = {}
    if len(delay_call) > 0:
        for delay_ in delay_call.values():
            # method 2: : delay push / delay pull
            # ------
            # delay_ = dict(type=calls[-1],
            #               args=args,
            #               kws_append=kws_append,
            #               func=func,
            #               org_call=org_call,
            #               rep_call=rep_expression,
            #               data_need_pass=data_need_pass)
            data_need_pass.extend(delay_['data_need_pass'])
            replaces_early[delay_['org_call']] = delay_['rep_call']
    for target, dest in replaces_early.items():
        main_code = main_code.replace(target, dest)

    # arguments 2: data need pass
    # -----------
    new_args = arguments + []
    for data in sorted(set(data_need_pass)):
        splits = data.split('.')
        replaces_later[data] = utils.attr_replace(data)
        obj = host
        for attr in splits[1:]:
            obj = getattr(obj, attr)
        if callable(obj):
            code_scope[utils.attr_replace(data)] = obj
            continue
        new_args.append(utils.attr_replace(data))
        calls.append('.'.join([host_name] + splits[1:]))

    # code scope
    code_scope[host_name] = host

    # codes
    header = f'def new_{func_name}({", ".join(new_args)}):\n'
    main_code = header + tools.indent(main_code, spaces_per_tab=2)
    main_code = tools.word_replace(main_code, replaces_later)
    if show_code:
        print(main_code)
        print(code_scope)
        print()

    # recompile
    exec(compile(main_code, '', 'exec'), code_scope)
    func = code_scope[f'new_{func_name}']
    func = cuda.jit(func)
    return func, calls


class NumbaCudaNodeDriver(NumbaCPUNodeDriver):
    def get_input_func(self, formatted_inputs, show_code=False):
        need_rebuild = False
        # check whether the input is changed
        # --
        new_inputs = {}
        input_keep_same = True
        old_input_keys = list(self.last_inputs.keys())
        for key, val, ops, data_type in formatted_inputs:
            # set data
            self.set_data(self.input_data_name(key), val)
            # compare
            if key in old_input_keys:
                old_input_keys.remove(key)
                if backend.shape(self.last_inputs[key][0]) != backend.shape(val):
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input shape {backend.shape(val)} is different '
                              f'from the last input shape {backend.shape(self.last_inputs[key][0])}.')
                if self.last_inputs[key][1] != ops:
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input operation "{ops}" is different '
                              f'from the last operation "{self.last_inputs[key][1]}". ')
            else:
                input_keep_same = False
                if show_code:
                    print(f'The input to a new key "{key}" in {self.host}.')
            new_inputs[key] = (val, ops, data_type)
        self.last_inputs = new_inputs
        if len(old_input_keys):
            input_keep_same = False
            if show_code:
                print(f'The inputs of {old_input_keys} in {self.host} are not provided.')

        # get the function of the input
        # ---
        if not input_keep_same:
            # codes
            input_func_name = 'input_step'
            host_name = self.host.name
            code_scope = {host_name: self.host}
            code_lines = [f'def {input_func_name}(_i):']
            for key, val, ops, data_type in formatted_inputs:
                if ops == '=':
                    line = f'  {host_name}.{key} = {host_name}.{self.input_data_name(key)}'
                else:
                    line = f'  {host_name}.{key} {ops}= {host_name}.{self.input_data_name(key)}'
                if data_type == 'iter':
                    line = line + '[_i]'
                code_lines.append(line)
            if len(formatted_inputs) == 0:
                code_lines.append('  pass')

            # function
            code = '\n'.join(code_lines)
            if show_code:
                print(code)
                print(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            self.set_data(input_func_name, code_scope[input_func_name])
            # results
            self.formatted_funcs['input'] = {
                'func': code_scope[input_func_name],
                'scope': {host_name: self.host},
                'call': [f'{host_name}.{input_func_name}(_i)'],
            }
            need_rebuild = True
        return need_rebuild

    def get_monitor_func(self, mon_length, show_code=False):
        mon = self.host.mon
        if len(mon['vars']) > 0:
            monitor_func_name = 'monitor_step'
            host = self.host.name
            code_scope = {host: self.host}
            code_lines = [f'def {monitor_func_name}(_i):']
            for key in mon['vars']:
                if not hasattr(self.host, key):
                    raise errors.ModelUseError(f'{self.host} do not have {key}, '
                                               f'thus it cannot be monitored.')

                # initialize monitor array #
                shape = backend.shape(getattr(self.host, key))
                mon[key] = backend.zeros((mon_length,) + shape)

                # add line #
                line = f'  {host}.mon["{key}"][_i] = {host}.{key}'
                code_lines.append(line)

            # function
            code = '\n'.join(code_lines)
            if show_code:
                print(code)
                print(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            self.set_data(monitor_func_name, code_scope[monitor_func_name])
            # results
            self.formatted_funcs['monitor'] = {
                'func': code_scope[monitor_func_name],
                'scope': {host: self.host},
                'call': [f'{host}.{monitor_func_name}(_i)'],
            }


class NumbaCudaNetDriver():
    pass
