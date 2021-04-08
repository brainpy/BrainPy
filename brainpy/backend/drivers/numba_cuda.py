# -*- coding: utf-8 -*-

import ast
import inspect
import re
from pprint import pprint

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.integrators import constants as diffint_cons
from brainpy.simulation import drivers
from brainpy.simulation.brainobjects import SynConn, NeuGroup
from . import utils
from .numba_cpu import NumbaCpuNodeDriver
from .numba_cpu import _StepFuncReader

try:
    import numba
    from numba import cuda
    from numba import cuda
    from numba.cuda.compiler import DeviceFunctionTemplate
    from numba.cuda.compiler import Dispatcher

    # if not cuda.is_available():
    #     raise errors.PackageMissingError('User choose the numba-cuda backend, '
    #                                      'while cuda is not available.')

except ModuleNotFoundError:
    raise errors.BackendNotInstalled('numba')

__all__ = [
    'NumbaCudaNodeDriver',
    'NumbaCudaDiffIntDriver',
]


class NumbaCudaDiffIntDriver(drivers.BaseDiffIntDriver):
    def build(self, *args, **kwargs):
        # code
        code = '\n'.join(self.code_lines)
        if self.show_code:
            print(code)
            print()
            pprint(self.code_scope)
            print()

        # jit original functions
        has_jitted = False
        if isinstance(self.code_scope['f'], DeviceFunctionTemplate):
            has_jitted = True
        elif isinstance(self.code_scope['f'], Dispatcher):
            raise ValueError('Cannot call cuda.jit function in a cuda.jit function, '
                             'only support cuda.jit(device=True) function.')
        if not has_jitted:
            if self.func_name.startswith(diffint_cons.ODE_PREFIX):
                self.code_scope['f'] = cuda.jit(self.code_scope['f'], device=True)
            elif self.func_name.startswith(diffint_cons.SDE_PREFIX):
                self.code_scope['f'] = cuda.jit(self.code_scope['f'], device=True)
                self.code_scope['g'] = cuda.jit(self.code_scope['g'], device=True)
            else:
                raise NotImplementedError

        # compile
        exec(compile(code, '', 'exec'), self.code_scope)

        # attribute assignment
        new_f = self.code_scope[self.func_name]
        for key, value in self.uploads.items():
            self.upload(host=new_f, key=key, value=value)
        if not has_jitted:
            new_f = cuda.jit(new_f)
        return new_f


class _CudaStepFuncReader(_StepFuncReader):
    """The tasks done in "CudaStepFuncReader" are:

    - Find all expressions, including Assign, AugAssign, For loop, If-else condition.
    - Find all delay push and pull.
    - Find all atomic operations.


    `CudaStepFuncReader` can analyze two kinds of coding schema.
    When users define the model with the explicit for-loop, such like:

    .. code-block:: python

       for i in prange(self.size):
            pre_id = self.pre_ids[i]

            self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau)
            self.x[i] += self.pre.spike[pre_id]

            self.I_syn.push(i, self.w[i] * self.s[i])

            # output
            post_id = self.post_ids[i]
            self.post.input[post_id] += self.I_syn.pull(i)

    It will recognize the loop body as the cuda kernel.

    However, when the function is not started with the for-loop, such like:

    .. code-block:: python

       i = cuda.grid(1)
       pre_id = self.pre_ids[i]

       self.s[i], self.x[i] = self.integral(self.s[i], self.x[i], _t, self.tau)
       self.x[i] += self.pre.spike[pre_id]

       self.I_syn.push(i, self.w[i] * self.s[i])

       # output
       post_id = self.post_ids[i]
       self.post.input[post_id] += self.I_syn.pull(i)

    It will recognize it as a customized function all controlled by users. In such case,
    the user should provide the "num" attribute.

    """

    def __init__(self, host):
        super(_CudaStepFuncReader, self).__init__(host=host)
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


def _analyze_step_func(host, f, func_name=None, show_code=False):
    """Analyze the step functions in a population.

    Parameters
    ----------
    host : DynamicSystem
        The data and the function host.
    f : callable
        The step function.

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
    # ---------
    host_name = host.name
    func_name = f.__name__ if func_name is None else func_name
    args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

    # judge step function type
    # ---------
    func_body = tree.body[0].body
    if len(func_body) == 1 and isinstance(func_body[0], ast.For):
        type_ = 'for_loop'
        iter_target = func_body[0].target.id
        iter_args = func_body[0].iter.args
        if len(iter_args) == 1:
            iter_seq = tools.ast2code(ast.fix_missing_locations(iter_args[0]))
        else:
            raise ValueError
        tree_to_analyze = func_body[0].body
    else:
        type_ = 'customize_type'
        tree_to_analyze = tree

    # AST analysis
    # -------
    formatter = _CudaStepFuncReader(host=host)
    formatter.visit(tree_to_analyze)

    # data assigned by self.xx in line right
    self_data_in_right = []
    if args[0] in backend.CLASS_KEYWORDS:
        code = ', \n'.join(formatter.rights)
        self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
        self_data_in_right = list(set(self_data_in_right))

    # data assigned by self.xxx in line left
    code = ', \n'.join(formatter.lefts)
    self_data_without_index_in_left = []
    self_data_with_index_in_left = []
    if args[0] in backend.CLASS_KEYWORDS:
        class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
        self_data_without_index_in_left = set(re.findall(class_p1, code))
        # class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\[.*\\]'
        class_p2 = '(\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*)\\['
        self_data_with_index_in_left = set(re.findall(class_p2, code))
        self_data_with_index_in_left = list(self_data_with_index_in_left)
        self_data_without_index_in_left = list(self_data_without_index_in_left)

    self_data_in_right = sorted(self_data_in_right)
    self_data_without_index_in_left = sorted(self_data_without_index_in_left)
    self_data_with_index_in_left = sorted(self_data_with_index_in_left)
    data_need_pass = sorted(list(set(self_data_in_right + self_data_with_index_in_left)))

    # main code
    main_code = '\n'.join(formatter.lines)

    # MAIN Task 1: reprocess delay function
    # -----------
    replaces_early = {}
    if len(formatter.delay_call) > 0:
        for delay_ in formatter.delay_call.values():
            data_need_pass.extend(delay_['data_need_pass'])
            replaces_early[delay_['org_call']] = delay_['rep_call']
    for target, dest in replaces_early.items():
        main_code = main_code.replace(target, dest)

    # MAIN Task 2: recompile the integrators
    # -----------

    integrators_to_recompile = {}

    # code scope
    closure_vars = inspect.getclosurevars(f)
    code_scope = dict(closure_vars.nonlocals)
    code_scope.update(closure_vars.globals)
    code_scope[host_name] = host
    if formatter.need_add_cuda:
        code_scope['cuda'] = cuda

    # arguments 1: the function intrinsic needed arguments
    calls = []
    for arg in args[1:]:
        if hasattr(host, arg):
            calls.append(f'{host_name}.{arg}')
        elif arg in backend.SYSTEM_KEYWORDS:
            calls.append(arg)
        else:
            raise errors.ModelDefError(f'Step function "{func_name}" of {host} define an '
                                       f'unknown argument "{arg}" which is not an attribute '
                                       f'of {host} nor the system keywords '
                                       f'{backend.SYSTEM_KEYWORDS}.')

    # arguments 2: data need pass
    replaces_later = {}
    new_args = args[1:]
    for data in sorted(set(data_need_pass)):
        splits = data.split('.')
        replaces_later[data] = utils.attr_replace(data)
        obj = host
        for attr in splits[1:]:
            obj = getattr(obj, attr)
        if callable(obj):
            if isinstance(obj, Dispatcher):
                if obj.py_func.__name__.startswith(DE_PREFIX):
                    integrators_to_recompile[utils.attr_replace(data)] = obj.py_func
                else:
                    raise ValueError('Cannot call a cuda.jit function, please change it '
                                     'to a device jit function.')
            elif isinstance(obj, DeviceFunctionTemplate):
                code_scope[utils.attr_replace(data)] = obj
            else:
                raise ValueError(f'Only support the device numba.cuda.jit function, not {type(obj)}.')
            continue
        else:
            new_args.append(utils.attr_replace(data))
            calls.append('.'.join([host_name] + splits[1:]))

    # MAIN Task 3: transform the cpu data to cuda data

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

    # final
    # -----

    analyzed_results = {
        'delay_call': formatter.delay_call,
        'code_string': '\n'.join(formatter.lines),
        'code_scope': code_scope,
        'self_data_in_right': self_data_in_right,
        'self_data_without_index_in_left': self_data_without_index_in_left,
        'self_data_with_index_in_left': self_data_with_index_in_left,
    }

    return analyzed_results



class NumbaCudaNodeDriver(NumbaCpuNodeDriver):
    def get_input_func(self, formatted_inputs, show_code=False):
        need_rebuild = False
        # check whether the input is changed
        # --
        new_inputs = {}
        input_keep_same = True
        old_input_keys = list(self.last_inputs.keys())
        for key, val, ops, data_type in formatted_inputs:
            # set data
            self.upload(self.input_data_name(key), val)
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
            self.upload(input_func_name, code_scope[input_func_name])
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
            self.upload(monitor_func_name, code_scope[monitor_func_name])
            # results
            self.formatted_funcs['monitor'] = {
                'func': code_scope[monitor_func_name],
                'scope': {host: self.host},
                'call': [f'{host}.{monitor_func_name}(_i)'],
            }


class NumbaCudaNetDriver():
    pass
