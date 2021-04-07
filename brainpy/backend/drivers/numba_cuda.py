# -*- coding: utf-8 -*-

import ast

from brainpy import backend
from brainpy import tools
from brainpy.simulation.brainobjects import SynConn, NeuGroup
from .numba_cpu import NumbaCPUNodeDriver
from .numba_cpu import StepFuncReader

from brainpy import errors

try:
    import numba
except ModuleNotFoundError:
    raise errors.BackendNotInstalled('numba')

__all__ = [
    'NumbaCudaNodeDriver',
]


class CudaStepFuncReader(StepFuncReader):
    def __init__(self, host):
        super(CudaStepFuncReader, self).__init__(host=host)

        self.need_add_cuda = False
        # get pre assignment
        self.pre_assign = []
        # get post assignment
        self.post_assign = []

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
        self.generic_visit(node)
        prefix = '  ' * level
        expr = tools.ast2code(ast.fix_missing_locations(node.value))
        self.rights.append(expr)

        check = None
        if len(node.targets) == 1:
            check = self.check_atomic_ops(node.targets[0])

        if check is None:
            targets = []
            for target in node.targets:
                targets.append(tools.ast2code(ast.fix_missing_locations(target)))
            _target = ' = '.join(targets)
            self.lefts.append(_target)
            self.lines.append(f'{prefix}{_target} = {expr}')
        else:
            target, slice_ = check
            self.lefts.append(target)
            self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')

    def visit_AugAssign(self, node, level=0):
        self.generic_visit(node)
        prefix = '  ' * level
        op = tools.ast2code(ast.fix_missing_locations(node.op))
        expr = tools.ast2code(ast.fix_missing_locations(node.value))

        check = self.check_atomic_ops(node.target)
        if check is None:
            target = tools.ast2code(ast.fix_missing_locations(node.target))
            self.lefts.append(target)
            self.rights.append(expr)
            self.lines.append(f"{prefix}{target} {op}= {expr}")
        else:
            if op == '+':
                expr = expr
            elif op == '-':
                expr = '-' + expr
            else:
                raise ValueError
            target, slice_ = check
            self.lefts.append(target)
            self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')


def analyze_step_func(f, host):
    """Analyze the step functions in a population.

    Parameters
    ----------
    f : callable
        The step function.
    host : Population
        The data and the function host.

    Returns
    -------
    results : dict
        The code string of the function, the code scope,
        the data need pass into the arguments,
        the data need return.
    """

    code_string = tools.deindent(inspect.getsource(f)).strip()
    tree = ast.parse(code_string)

    # arguments
    # ---
    args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

    # code lines
    # ---
    formatter = StepFuncReader(host=host)
    formatter.visit(tree)

    # data assigned by self.xx in line right
    # ---
    self_data_in_right = []
    if args[0] in backend.CLASS_KEYWORDS:
        code = ', \n'.join(formatter.rights)
        self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
        self_data_in_right = list(set(self_data_in_right))

    # data assigned by self.xxx in line left
    # ---
    code = ', \n'.join(formatter.lefts)
    self_data_without_index_in_left = []
    self_data_with_index_in_left = []
    if args[0] in backend.CLASS_KEYWORDS:
        class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
        self_data_without_index_in_left = set(re.findall(class_p1, code))
        class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
        self_data_with_index_in_left = set(re.findall(class_p2, code))
        self_data_without_index_in_left -= self_data_with_index_in_left
        self_data_without_index_in_left = list(self_data_without_index_in_left)

    # code scope
    # ---
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)

    # final
    # ---
    self_data_in_right = sorted(self_data_in_right)
    self_data_without_index_in_left = sorted(self_data_without_index_in_left)
    self_data_with_index_in_left = sorted(self_data_with_index_in_left)

    analyzed_results = {
        'code_string': code_string,
        'code_scope': code_scope,
        'self_data_in_right': self_data_in_right,
        'self_data_without_index_in_left': self_data_without_index_in_left,
        'self_data_with_index_in_left': self_data_with_index_in_left,
    }

    return analyzed_results


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