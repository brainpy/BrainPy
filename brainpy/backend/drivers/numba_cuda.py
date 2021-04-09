# -*- coding: utf-8 -*-

import ast
import inspect
import math
import re
from collections import OrderedDict
from pprint import pprint

import numpy as np

from brainpy import backend
from brainpy import errors
from brainpy import tools
from brainpy.backend import ops
from brainpy.integrators import constants as diffint_cons
from brainpy.integrators import utils as diffint_utils
from brainpy.simulation import drivers
from brainpy.simulation.brainobjects import SynConn, NeuGroup
from brainpy.simulation.delays import ConstantDelay
from . import utils
from .numba_cpu import NumbaCPUNodeDriver
from .numba_cpu import _CPUReader

try:
    import numba
    from numba import cuda
    from numba.cuda.compiler import DeviceFunctionTemplate, Dispatcher
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    # if not cuda.is_available():
    #     raise errors.PackageMissingError('User choose the numba-cuda backend, '
    #                                      'while cuda is not available.')

except ModuleNotFoundError:
    raise errors.BackendNotInstalled('numba')

__all__ = [
    'NumbaCUDANodeDriver',
    'NumbaCudaDiffIntDriver',
    'set_num_thread_gpu',
    'get_num_thread_gpu',
]

_num_thread_gpu = 1024


def set_num_thread_gpu(num_thread):
    global _num_thread_gpu
    _num_thread_gpu = num_thread


def get_num_thread_gpu():
    return _num_thread_gpu


def get_cuda_size(num):
    if num < get_num_thread_gpu():
        num_block, num_thread = 1, num
    else:
        num_thread = get_num_thread_gpu()
        num_block = math.ceil(num / num_thread)
    return num_block, num_thread


class NumbaCudaDiffIntDriver(drivers.BaseDiffIntDriver):
    def build(self, *args, **kwargs):
        if self.uploads['var_type'] != diffint_cons.SCALAR_VAR:
            raise errors.IntegratorError(f'Numba Cuda backend only supports scalar variable, but we got a differential '
                                         f'equation defined with {self.uploads["var_type"]} variable.')

        # reprocess the wiener process
        need_rng_states = False
        if self.func_name.startswith(diffint_cons.SDE_PREFIX):
            for line_id, line in enumerate(self.code_lines):
                if 'ops.normal(0.000, dt_sqrt,' in line:
                    parts = line.split('=')
                    self.code_lines[line_id] = f'{parts[0]} = xoroshiro128p_uniform_float64(rng_states, thread_id)'
                    self.code_scope['xoroshiro128p_uniform_float64'] = xoroshiro128p_uniform_float64
                    need_rng_states = True

        # add "rng_states" in the integrator arguments
        if need_rng_states:
            _, _, _, arguments = diffint_utils.get_args(self.code_scope['f'])
            for line_id, line in enumerate(self.code_lines):
                if line.startswith(f'def {self.func_name}'):
                    arguments += ["thread_id=0", "rng_states=None"]
                    self.code_lines[line_id] = f'def {self.func_name}({", ".join(arguments)}):'
                    break

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


class _CUDATransformer(ast.NodeTransformer):
    def __init__(self, host):
        self.host = host
        self.need_add_rng_states = False

    def visit_attr(self, node):
        if isinstance(node, ast.Attribute):
            r = self.visit_attr(node.value)
            return [node.attr] + r
        elif isinstance(node, ast.Name):
            return [node.id]
        else:
            raise ValueError

    def visit_Call(self, node):
        if getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        if getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")

        # get the calling string
        calls = self.visit_attr(node.func)
        calls = calls[::-1]

        # get the object and the function
        if calls[0] not in backend.CLASS_KEYWORDS:
            return node
        obj = self.host
        for data in calls[1:-1]:
            obj = getattr(obj, data)
        obj_func = getattr(obj, calls[-1])

        # get function arguments
        args = []
        for arg in node.args:
            args.append(tools.ast2code(ast.fix_missing_locations(arg)))
        kw_args = OrderedDict()
        for keyword in node.keywords:
            kw_args[keyword.arg] = tools.ast2code(ast.fix_missing_locations(keyword.value))

        # TASK 2 : check integrator functions
        # ------
        # If the integrator calling does not have "rng_states", add it
        # ------
        add_keywords = []
        if isinstance(obj_func, (Dispatcher, DeviceFunctionTemplate)):
            py_func = obj_func.py_func
            _, _, _, py_func_args = diffint_utils.get_args(py_func)
            if py_func.__name__.startswith(diffint_cons.SDE_PREFIX):
                if 'rng_states' not in kw_args and 'thread_id' not in kw_args:
                    arg_length = len(args) + len(kw_args)
                    if len(py_func_args) == arg_length + 2:
                        self.need_add_rng_states = True
                        rng = ast.keyword(arg='rng_states',
                                          value=ast.Attribute(attr='rng_states',
                                                              value=ast.Name(id=calls[0])))
                        thread_id = ast.keyword(arg='thread_id', value=ast.Name(id='_thread_id_'))
                        add_keywords.extend([rng, thread_id])
                else:
                    raise ValueError(f'Cannot parse: {tools.ast2code(ast.fix_missing_locations(node))}')
        return ast.Call(func=node.func, args=node.args, keywords=node.keywords + add_keywords)


class _CUDAReader(_CPUReader):
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
        super(_CUDAReader, self).__init__(host=host)
        self.need_add_cuda_to_scope = False

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
        """Two main tasks:

        1. parse the left, right separations
        2. add cuda.automic operations
        """
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
                        self.need_add_cuda_to_scope = True
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
        """Two main tasks:

        1. parse the left, right separations
        2. add cuda.automic operations
        """
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
                self.need_add_cuda_to_scope = True

            self.visited_nodes.add(node)

        self.generic_visit(node)


class NumbaCUDANodeDriver(NumbaCPUNodeDriver):
    FOR_LOOP_TYPE = 'for_loop'
    CUSTOMIZE_TYPE = 'customize'

    @staticmethod
    def transfer_data_cpu2gpu(host, key, data):
        if not hasattr(host, 'numba_gpu_data'):
            setattr(host, 'numba_gpu_data', {})
        if key not in host.numba_gpu_data:
            setattr(host, key, data)
            host.numba_gpu_data[key] = data

    def _reprocess_steps(self, f, func_name=None, show_code=False):
        """Analyze the step functions in a DynamicSystem.
        """
        code_string = tools.deindent(inspect.getsource(f)).strip()
        tree = ast.parse(code_string)

        # arguments
        # ---------
        host_name = self.host.name
        func_name = f.__name__ if func_name is None else func_name
        args = tools.ast2code(ast.fix_missing_locations(tree.body[0].args)).split(',')

        # judge step function type
        # ---------
        func_body = tree.body[0].body
        if len(func_body) == 1 and isinstance(func_body[0], ast.For):
            type_ = self.FOR_LOOP_TYPE
            iter_target = func_body[0].target.id
            iter_args = func_body[0].iter.args
            if len(iter_args) == 1:
                iter_seq = tools.ast2code(ast.fix_missing_locations(iter_args[0]))
            else:
                raise NotImplementedError

            # MAIN Task 1: add "rng_states" to sde integrals
            # ------
            transformer = _CUDATransformer(host=self.host)
            tree = transformer.visit(ast.parse(code_string))
            if transformer.need_add_rng_states:
                if not hasattr(self.host, 'num'):
                    raise errors.ModelDefError(
                        f'Must define "num" in {self.host} when user uses the Numba CUDA backend.')
                num_block, num_thread = get_cuda_size(self.host.num)
                rng_state = create_xoroshiro128p_states(num_block * num_thread, seed=np.random.randint(100000))
                self.upload('rng_states', rng_state)

            tree_to_analyze = tree.body[0].body[0].body
        else:
            type_ = self.CUSTOMIZE_TYPE
            tree_to_analyze = tree

        # AST reader
        # -------
        reader = _CUDAReader(host=self.host)
        reader.visit(tree_to_analyze)

        # data assigned by self.xx in line right
        self_data_in_right = []
        if args[0] in backend.CLASS_KEYWORDS:
            code = ', \n'.join(reader.rights)
            self_data_in_right = re.findall('\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b', code)
            self_data_in_right = list(set(self_data_in_right))

        # data assigned by self.xxx in line left
        code = ', \n'.join(reader.lefts)
        self_data_without_index_in_left = []
        if args[0] in backend.CLASS_KEYWORDS:
            class_p1 = '\\b' + args[0] + '\\.[A-Za-z_][A-Za-z0-9_.]*\\b'
            self_data_without_index_in_left = set(re.findall(class_p1, code))
            self_data_without_index_in_left = list(self_data_without_index_in_left)
        self_data_in_right = sorted(self_data_in_right)
        self_data_without_index_in_left = sorted(self_data_without_index_in_left)
        data_need_pass = sorted(list(set(self_data_in_right + self_data_without_index_in_left)))

        # main code
        main_code = '\n'.join(reader.lines)

        # MAIN Task 2: reprocess delay function
        # -----------
        replaces_early = {}
        if len(reader.visited_calls) > 0:
            for delay_ in reader.visited_calls.values():
                data_need_pass.extend(delay_['data_need_pass'])
                replaces_early[delay_['org_call']] = delay_['rep_call']
        for target, dest in replaces_early.items():
            main_code = main_code.replace(target, dest)

        # MAIN Task 3: check the integrators jit
        # -----------

        # code scope
        closure_vars = inspect.getclosurevars(f)
        code_scope = dict(closure_vars.nonlocals)
        code_scope.update(closure_vars.globals)
        code_scope[host_name] = self.host
        code_scope['cuda'] = cuda
        if reader.need_add_cuda_to_scope:
            pass

        # arguments 1: the user defined function arguments
        calls = []
        for arg in args[1:]:
            if hasattr(self.host, arg):
                calls.append(f'{host_name}.{arg}')
            elif arg in backend.SYSTEM_KEYWORDS:
                calls.append(arg)
            else:
                msg = f'Step function "{func_name}" of {self.host} define an ' \
                      f'unknown argument "{arg}" which is not an attribute ' \
                      f'of {self.host} nor the system keywords ' \
                      f'{backend.SYSTEM_KEYWORDS}.'
                raise errors.ModelDefError(msg)

        # MAIN Task 4: transform the cpu data to cuda data,
        #              check jitted integrators,
        #              data need pass
        replaces_later = {}
        new_args = args[1:]
        for data in sorted(set(data_need_pass)):
            splits = data.split('.')
            replaces_later[data] = utils.attr_replace(data)
            host = self.host
            for attr in splits[1:-1]:
                host = getattr(host, attr)
            obj = getattr(host, splits[-1])

            if isinstance(obj, np.ndarray):  # 1. transform the cpu data to cuda data
                splits[-1] = f'{splits[-1]}_cuda'
                self.transfer_data_cpu2gpu(host, splits[-1], cuda.to_device(obj))

            elif isinstance(obj, DeviceNDArray):
                self.transfer_data_cpu2gpu(host, splits[-1], obj)

            elif callable(obj):  # 2. check jitted integrators
                if isinstance(obj, Dispatcher):
                    if diffint_cons.NAME_PREFIX in obj.py_func.__name__:
                        code_scope[utils.attr_replace(data)] = cuda.jit(obj.py_func, device=True)
                    else:
                        raise ValueError('Cannot call a cuda.jit function, please change it '
                                         'to a numba.cuda.jit device function.')
                elif isinstance(obj, DeviceFunctionTemplate):
                    code_scope[utils.attr_replace(data)] = obj
                else:
                    raise ValueError(f'Only support the numba.cuda.jit(device=True) function in '
                                     f'the step function, not {type(obj)}.')
                continue

            # 3. data need pass
            new_args.append(utils.attr_replace(data))
            calls.append('.'.join([host_name] + splits[1:]))

        # format final codes
        if type_ == self.FOR_LOOP_TYPE:
            replaces_later['_thread_id_'] = iter_target
            for_loop = f'{iter_target} = cuda.grid(1)\n' \
                       f'if {iter_target} < {iter_seq}:\n'
            main_code = for_loop + tools.indent(main_code, spaces_per_tab=2)
        elif type_ == self.CUSTOMIZE_TYPE:
            pass
        else:
            raise NotImplementedError(f'Unknown coding type: {type_}')
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

        return func, calls, []

    def _reprocess_delays(self, host, f, func_name, show_code=False):
        if host.uniform_delay:
            code = f'''
def new_{func_name}(delay_num_step, delay_in_idx, delay_out_idx):
    delay_in_idx = (delay_in_idx + 1) % delay_num_step
    delay_out_idx = (delay_out_idx + 1) % delay_num_step
    return delay_in_idx, delay_out_idx
            '''
            code = code.strip()
            code_scope = {host.name: host}
            calls = [f'{host.name}.delay_num_step', f'{host.name}.delay_in_idx', f'{host.name}.delay_out_idx']
            assigns = [f'{host.name}.delay_in_idx', f'{host.name}.delay_out_idx']

            if show_code:
                print(code)
                print(code_scope)
                print()

            # compile
            exec(compile(code, '', 'exec'), code_scope)
            func = code_scope[f'new_{func_name}']
            func = numba.njit(func)

        else:
            code = f'''
def new_{func_name}(delay_num_step, delay_in_idx, delay_out_idx):
    thread_id = cuda.grid(1)
    if thread_id < delay_num_step.shape[0]:
        step = delay_num_step[thread_id]
        delay_in_idx[thread_id] = (delay_in_idx[thread_id] + 1) % step
        delay_out_idx[thread_id] = (delay_out_idx[thread_id] + 1) % step
        '''
            code = code.strip()
            self.transfer_data_cpu2gpu(host, key='delay_num_step_cuda', data=cuda.to_device(host.delay_num_step))
            self.transfer_data_cpu2gpu(host, key='delay_in_idx_cuda', data=cuda.to_device(host.delay_in_idx))
            self.transfer_data_cpu2gpu(host, key='delay_out_idx_cuda', data=cuda.to_device(host.delay_out_idx))

            code_scope = {host.name: host, 'cuda': cuda}
            calls = [f'{host.name}.delay_num_step_cuda',
                     f'{host.name}.delay_in_idx_cuda',
                     f'{host.name}.delay_out_idx_cuda']
            assigns = []

            if show_code:
                print(code)
                print(code_scope)
                print()

            # compile
            exec(compile(code, '', 'exec'), code_scope)
            func = code_scope[f'new_{func_name}']
            func = cuda.jit(func)
        return func, calls, assigns

    def get_steps_func(self, show_code=False):
        for func_name, step in self.steps.items():
            # the host
            if hasattr(step, '__self__'):
                host = step.__self__
            else:
                host = self.host
            if not hasattr(host, 'name'):
                raise errors.ModelDefError(f'Each host should have a unique name. But we '
                                           f'"name" attribute is not found in {host}.')

            # the function reprocessed
            if host == self.host:
                func, calls, assigns = self._reprocess_steps(step, func_name=func_name, show_code=show_code)
                synchronize = True
            elif isinstance(host, ConstantDelay):
                func, calls, assigns = self._reprocess_delays(host, f=step, func_name=func_name, show_code=show_code)
                if host.uniform_delay:
                    synchronize = False
                else:
                    synchronize = True
            else:
                raise NotImplementedError

            # set function
            setattr(host, f'new_{func_name}', func)

            # code scope
            code_scope = {host.name: host, }

            # functional call
            assign_line = ''
            if len(assigns):
                assign_line = f'{", ".join(assigns)} = '
            final_calls = [f'{assign_line}{host.name}.new_{func_name}({", ".join(calls)})']
            if synchronize:
                final_calls.append('cuda.synchronize()')
                code_scope['cuda'] = cuda

            # finally
            self.formatted_funcs[func_name] = {'func': func, 'scope': code_scope, 'call': final_calls}

    def get_input_func(self, formatted_inputs, show_code=False):
        need_rebuild = False
        # check whether the input is changed
        # --
        new_inputs = {}
        input_keep_same = True
        old_input_keys = list(self.last_inputs.keys())
        for key, val, ops_, data_type in formatted_inputs:
            # set data
            self.upload(self.input_data_name(key), val)
            # compare
            if key in old_input_keys:
                old_input_keys.remove(key)
                if ops.shape(self.last_inputs[key][0]) != ops.shape(val):
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input shape {ops.shape(val)} is different '
                              f'from the last input shape {ops.shape(self.last_inputs[key][0])}.')
                if self.last_inputs[key][1] != ops_:
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input operation "{ops_}" is different '
                              f'from the last operation "{self.last_inputs[key][1]}". ')
            else:
                input_keep_same = False
                if show_code:
                    print(f'The input to a new key "{key}" in {self.host}.')
            new_inputs[key] = (val, ops_, data_type)
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
            for key, val, ops_, data_type in formatted_inputs:
                if ops_ == '=':
                    line = f'  {host_name}.{key} = {host_name}.{self.input_data_name(key)}'
                else:
                    line = f'  {host_name}.{key} {ops_}= {host_name}.{self.input_data_name(key)}'
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
                shape = ops.shape(getattr(self.host, key))
                mon[key] = ops.zeros((mon_length,) + shape)

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


class NumbaCUDANetDriver(drivers.BaseNetDriver):
    pass
