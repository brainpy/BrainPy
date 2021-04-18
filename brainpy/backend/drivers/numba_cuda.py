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
from brainpy.simulation.monitors import Monitor
from . import utils
from .general import GeneralNetDriver
from .numba_cpu import NumbaCPUNodeDriver
from .numba_cpu import _CPUReader

try:
    import numba
    from numba import cuda
    from numba.cuda.compiler import DeviceFunctionTemplate, Dispatcher
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    if not cuda.is_available():
        raise errors.PackageMissingError('User choose the numba-cuda backend, '
                                         'while cuda is not available.')

except ModuleNotFoundError:
    raise errors.BackendNotInstalled('numba')


__all__ = [
    'NumbaCUDANodeDriver',
    'NumbaCudaDiffIntDriver',
    'set_num_thread_gpu',
    'get_num_thread_gpu',
    'set_monitor_done_in',
]

_num_thread_gpu = 1024

# Monitor can be done in :
# 1. 'cpu'
# 2. 'cuda'
_monitor_done_in = 'cuda'


def set_monitor_done_in(place):
    global _monitor_done_in
    if place not in ['cpu', 'cuda']:
        raise NotImplementedError
    _monitor_done_in = place


def cuda_name_of(data_name):
    return f'{data_name}_bpcuda'


def set_num_thread_gpu(num_thread):
    global _num_thread_gpu
    _num_thread_gpu = num_thread


def get_num_thread_gpu():
    return _num_thread_gpu


def get_cuda_size(num):
    if num <= get_num_thread_gpu():
        num_block, num_thread = 1, num
    else:
        num_thread = get_num_thread_gpu()
        num_block = math.ceil(num / num_thread)
    return num_block, num_thread


def get_categories(category):
    if category is None:
        category = [NeuGroup, SynConn, Monitor, ConstantDelay]
    else:
        str2target = {'mon': Monitor, 'neu': NeuGroup,
                      'syn': SynConn, 'delay': ConstantDelay}
        if isinstance(category, str):
            category = [str2target[category]]
        elif isinstance(category, (tuple, list)):
            category = [str2target[c] for c in category]
        else:
            raise ValueError
    return category

def data_shape(data):
    if isinstance(data, DeviceNDArray):
        return data.shape
    else:
        return ops.shape(data)


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

        # code
        code = '\n'.join(self.code_lines)
        if self.show_code:
            print(code)
            print()
            pprint(self.code_scope)
            print()

        # compile
        exec(compile(code, '', 'exec'), self.code_scope)

        # attribute assignment
        new_f = self.code_scope[self.func_name]
        for key, value in self.uploads.items():
            self.upload(host=new_f, key=key, value=value)
        if not has_jitted:
            new_f = cuda.jit(new_f, device=True)
        return new_f


class _CUDATransformer(ast.NodeTransformer):
    """Code Transformer in the Numba CUDA backend.

    The tasks of this transformer are:

    - correct the SDE integrators, for example:
      add "rng_states" and "thread_id" arguments.

    """
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
    """Code reader in the Numba CUDA backend.

    The tasks done in "CudaStepFuncReader" are:

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

            success = False  # whether found an place for automatic operation
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
                    raise ValueError(f'Cannot parse automic operation for this '
                                     f'expression: {target}[{slice_}] {op}= {expr}')
                self.lefts.append(target)
                self.rights.append(expr)
                self.lines.append(f'{prefix}cuda.atomic.add({target}, {slice_}, {expr})')
                self.need_add_cuda_to_scope = True

            self.visited_nodes.add(node)

        self.generic_visit(node)


class NumbaCUDANodeDriver(NumbaCPUNodeDriver):
    FOR_LOOP_TYPE = 'for_loop'
    CUSTOMIZE_TYPE = 'customize'

    def __init__(self, pop, steps=None):
        super(NumbaCUDANodeDriver, self).__init__(pop=pop, steps=steps)
        self.host_cpukey_gpukey = {}  # with the format of "{host: {cpu_key: gpu_key}}"

    def transfer_cpu_data_to_gpu(self, host, cpu_key, cpu_data):
        if not hasattr(host, 'stream'):
            host.stream = cuda.stream()

        gpu_key = cuda_name_of(cpu_key)
        cpu_data = self.cpu2gpu(host, cpu_data)
        setattr(host, gpu_key, cpu_data)

        # register the cpu and gpu key
        if host not in self.host_cpukey_gpukey:
            self.host_cpukey_gpukey[host] = {}
        self.host_cpukey_gpukey[host][cpu_key] = gpu_key
        return gpu_key

    def cpu2gpu(self, host, cpu_data):
        if not hasattr(host, 'stream'):
            host.stream = cuda.stream()
        return cuda.to_device(cpu_data, stream=host.stream)


    def _reprocess_steps(self, f, func_name=None, show_code=False):
        """Analyze the step functions in a DynamicSystem.
        """
        code_string = tools.deindent(inspect.getsource(f)).strip()
        tree = ast.parse(code_string)

        # arguments
        # ---------
        host_name = self.host.name
        func_name = f.__name__ if func_name is None else func_name
        _arg_ast = ast.fix_missing_locations(tree.body[0].args)
        args = [arg.strip() for arg in tools.ast2code(_arg_ast).split(',')]

        # judge step function type
        # ---------
        func_body = tree.body[0].body
        if len(func_body) == 1 and isinstance(func_body[0], ast.For):
            code_type = self.FOR_LOOP_TYPE
            iter_target = func_body[0].target.id
            iter_args = func_body[0].iter.args
            if len(iter_args) == 1:
                iter_seq = tools.ast2code(ast.fix_missing_locations(iter_args[0]))
            else:
                raise NotImplementedError
            splits = iter_seq.split('.')
            assert splits[0] in backend.CLASS_KEYWORDS
            obj = self.host
            for attr in splits[1:]:
                obj = getattr(obj, attr)
            num_block, num_thread = get_cuda_size(obj)

            # MAIN Task 1: add "rng_states" to sde integrals
            # ------
            transformer = _CUDATransformer(host=self.host)
            tree = transformer.visit(ast.parse(code_string))
            if transformer.need_add_rng_states:
                if not hasattr(self.host, 'num'):
                    raise errors.ModelDefError(
                        f'Must define "num" in {self.host} when user uses the Numba CUDA backend.')
                rng_state = create_xoroshiro128p_states(num_block * num_thread, seed=np.random.randint(100000))
                self.upload('rng_states', rng_state)

            tree_to_analyze = ast.Module(body=tree.body[0].body[0].body)
        else:
            if not hasattr(self.host, 'num'):
                raise errors.ModelDefError(f'Each host should have "num" attribute when using Numba CUDA backend. '
                                           f'But "num" is not found in {self.host}.')
            num_block, num_thread = get_cuda_size(self.host.num)
            iter_seq = ''
            code_type = self.CUSTOMIZE_TYPE
            tree_to_analyze = tree

        # AST reader
        # -------
        reader = _CUDAReader(host=self.host)
        reader.visit(tree_to_analyze)

        # data assigned by self.xx in line right
        self_data_in_right = []
        if args[0] in backend.CLASS_KEYWORDS:
            code = ', \n'.join(reader.rights) + ', ' + iter_seq
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

            # 1. check jitted integrators
            if isinstance(obj, Dispatcher):
                raise ValueError(f'Cannot call a cuda.jit function, please change it '
                                 f'to a numba.cuda.jit device function: {obj}')
            elif isinstance(obj, DeviceFunctionTemplate):
                code_scope[utils.attr_replace(data)] = obj
            else:
                if callable(obj):
                    code_scope[utils.attr_replace(data)] = obj
                    continue
                if isinstance(obj, np.ndarray):  # 2. transform the cpu data to cuda data
                    splits[-1] = self.transfer_cpu_data_to_gpu(host, cpu_key=splits[-1], cpu_data=obj)
                # 3. data need pass
                new_args.append(utils.attr_replace(data))
                calls.append('.'.join([host_name] + splits[1:]))

        # format final codes
        if code_type == self.FOR_LOOP_TYPE:
            replaces_later['_thread_id_'] = iter_target
            for_loop = f'{iter_target} = cuda.grid(1)\n' \
                       f'if {iter_target} < {iter_seq}:\n'
            main_code = for_loop + tools.indent(main_code, spaces_per_tab=2)
        elif code_type == self.CUSTOMIZE_TYPE:
            pass
        else:
            raise NotImplementedError(f'Unknown coding type: {code_type}')
        header = f'def new_{func_name}({", ".join(new_args)}):\n'
        main_code = header + tools.indent(main_code, spaces_per_tab=2)
        main_code = tools.word_replace(main_code, replaces_later)
        if show_code:
            print(main_code)
            pprint(code_scope)
            print()

        # recompile
        exec(compile(main_code, '', 'exec'), code_scope)
        func = code_scope[f'new_{func_name}']
        func = cuda.jit(func)

        call_lines = [f'{host_name}.new_{func_name}[{num_block}, {num_thread}, {host_name}.stream]({", ".join(calls)})',
                      f'{host_name}.stream.synchronize()']

        return func, call_lines

    def _reprocess_delays(self, host, f, func_name, show_code=False):
        if f.__name__ != 'update':
            raise NotImplementedError

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
            call_lines = [f'{", ".join(assigns)} = {host.name}.new_{func_name}({", ".join(calls)})']

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

            num_step_name = self.transfer_cpu_data_to_gpu(host, cpu_key="delay_num_step", cpu_data=host.delay_num_step)
            in_idx_name = self.transfer_cpu_data_to_gpu(host, cpu_key="delay_in_idx", cpu_data=host.delay_in_idx)
            out_idx_name = self.transfer_cpu_data_to_gpu(host, cpu_key="delay_out_idx", cpu_data=host.delay_out_idx)
            code_scope = {host.name: host, 'cuda': cuda}
            calls = [f'{host.name}.{num_step_name}',
                     f'{host.name}.{in_idx_name}',
                     f'{host.name}.{out_idx_name}']

            if show_code:
                print(code)
                pprint(code_scope)
                print()

            # compile
            exec(compile(code, '', 'exec'), code_scope)
            func = code_scope[f'new_{func_name}']
            func = cuda.jit(func)
            num_block, num_thread = get_cuda_size(host.num)
            call_lines = [f'{host.name}.new_{func_name}[{num_block}, {num_thread}, {host.name}.stream]({", ".join(calls)})',
                          f'{host.name}.stream.synchronize()']

        return func, call_lines

    def get_steps_func(self, show_code=False):
        for func_name, step in self.steps.items():
            # the host
            if hasattr(step, '__self__'):
                host = step.__self__
            else:
                host = self.host
            if not hasattr(host, 'name'):
                raise errors.ModelDefError(f'Each host should have a unique name when using Numba CUDA backend. '
                                           f'But "name" attribute is not found in {host}.')

            # the function reprocessed
            if host == self.host:
                # if not hasattr(host, 'num'):
                #     raise errors.ModelDefError(f'Each host should have "num" attribute when using Numba CUDA backend. '
                #                                f'But "num" is not found in {host}.')
                func, call_lines = self._reprocess_steps(step, func_name=func_name, show_code=show_code)
            elif isinstance(host, ConstantDelay):
                func, call_lines = self._reprocess_delays(host, f=step, func_name=func_name, show_code=show_code)
            else:
                raise NotImplementedError

            # set function
            setattr(host, f'new_{func_name}', func)

            # code scope
            code_scope = {host.name: host, 'cuda': cuda}

            # finally
            self.formatted_funcs[func_name] = {'func': func, 'scope': code_scope, 'call': call_lines}

    def _check_inputs_change(self, formatted_inputs, show_code):
        # check whether the input is changed
        # ---
        new_inputs = {}
        input_keep_same = True
        old_input_keys = list(self.last_inputs.keys())
        for key, val, op, data_type in formatted_inputs:
            # set data, and transfer cpu data to gpu
            if isinstance(val, DeviceNDArray) or isinstance(val, (int, float)):
                pass
            elif hasattr(val, '__cuda_array_interface__'):
                val = cuda.as_cuda_array(val)
            else:
                val = self.cpu2gpu(host=self.host, cpu_data=val)
            self.upload(self.input_data_name_of(key), val)

            # compare
            if key in old_input_keys:
                # shape
                val_shape = data_shape(val)
                last_shape = data_shape(self.last_inputs[key][0])
                # check
                old_input_keys.remove(key)
                if last_shape != val_shape:
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input shape {val_shape} is different '
                              f'from the last input shape {last_shape}.')
                if self.last_inputs[key][1] != op:
                    input_keep_same = False
                    if show_code:
                        print(f'The current "{key}" input operation "{op}" is different '
                              f'from the last operation "{self.last_inputs[key][1]}". ')
            else:
                input_keep_same = False
                if show_code:
                    print(f'The input to a new key "{key}" in {self.host}.')
            new_inputs[key] = (val, op, data_type)
        self.last_inputs = new_inputs
        if len(old_input_keys):
            input_keep_same = False
            if show_code:
                print(f'The inputs of {old_input_keys} in {self.host} are not provided.')

        return input_keep_same

    def _format_inputs_func(self, formatted_inputs, show_code):
        """For the function for inputs

        The ability of this automatic inputs function:

        1. Only supports inputs to the 1D vector data.
        2. Do not support inputs to the int/float etc. scalar data.
        """

        # function and host name
        input_func_name = 'input_step'
        host_name = self.host.name

        if len(formatted_inputs) > 0:
            # code scope
            args2calls = {}
            code_scope = {host_name: self.host, 'cuda': cuda}

            # task 1: group data according to the data shape
            # task 2: transfer data from cpu to gpu
            new_formatted_inputs = {}
            for key, val, op, data_type in formatted_inputs:
                key_val = getattr(self.host, key)
                if isinstance(key_val, (int, float)):
                    raise errors.ModelUseError(f'BrainPy Numba CUDA backend does not support inputs '
                                               f'for scalar value: {host_name}.{key} = {key_val}')
                if isinstance(key_val, np.ndarray):
                    if key_val.ndim != 1:
                        raise NotImplementedError(f'BrainPy Numba CUDA backend only supports inputs for 1D vector data'
                                                  f', not {key_val.ndim}-dimensional data of {host_name}.{key}')
                    gpu_key = self.transfer_cpu_data_to_gpu(self.host, cpu_key=key, cpu_data=key_val)
                    args2calls[f'{host_name}_{key}'] = f'{host_name}.{gpu_key}'
                    size = key_val.size
                elif isinstance(key_val, DeviceNDArray):
                    if key_val.ndim != 1:
                        raise NotImplementedError(f'BrainPy Numba CUDA backend only supports inputs for 1D vector data'
                                                  f', not {key_val.ndim}-dimensional data of {host_name}.{key}')
                    size = key_val.size
                    args2calls[f'{host_name}_{key}'] = f'{host_name}.{key}'
                else:
                    raise NotImplementedError
                if size not in new_formatted_inputs:
                    new_formatted_inputs[size] = []
                new_formatted_inputs[size].append((key, val, op, data_type))

            # function arguments and code lines
            code_lines = []
            for size, inputs in new_formatted_inputs.items():
                code_lines = [f'  if thread_i < {size}:']
                for key, val, op, data_type in inputs:
                    key_name_in_host = self.input_data_name_of(key)
                    if not isinstance(val, (int, float)):
                        if data_type == 'iter':
                            if val.ndim == 1:
                                postfix = '[_i]'
                            elif val.ndim == 2:
                                postfix = '[_i, thread_i]'
                            else:
                                raise NotImplementedError
                        else:
                            postfix = '[thread_i]'
                    else:
                        postfix = ''
                    args2calls[f'{host_name}_{key_name_in_host}'] = f'{host_name}.{key_name_in_host}'
                    if data_type == 'iter':
                        args2calls['_i'] = '_i'
                    if op == '=':
                        line = f'    {host_name}_{key}[thread_i] = {host_name}_{key_name_in_host}{postfix}'
                    else:
                        line = f'    {host_name}_{key}[thread_i] {op}= {host_name}_{key_name_in_host}{postfix}'
                    code_lines.append(line)

            # arguments
            args2calls = sorted(args2calls.items())
            args = [arg for arg, _ in args2calls]
            calls = [call for _, call in args2calls]
            code_lines.insert(0, f'def {input_func_name}({", ".join(args)}):')
            code_lines.insert(1, f'  thread_i = cuda.grid(1)')

            # function code
            code = '\n'.join(code_lines)
            if show_code:
                print(code)
                pprint(code_scope)
                print()
            exec(compile(code, '', 'exec'), code_scope)
            func = cuda.jit(code_scope[input_func_name])

            # results
            self.upload(input_func_name, func)
            num_block, num_thread = get_cuda_size(max(list(new_formatted_inputs.keys())))
            self.formatted_funcs['input'] = {
                'func': func,
                'scope': {host_name: self.host, 'cuda': cuda},
                'call': [f'{host_name}.{input_func_name}[{num_block}, {num_thread}, {host_name}.stream]({", ".join(calls)})',
                         f'{host_name}.stream.synchronize()'],
            }
        else:
            func = lambda: None
            self.upload(input_func_name, func)
            self.formatted_funcs['input'] = {'func': func,
                                             'scope': {host_name: self.host},
                                             'call': [f'{host_name}.{input_func_name}()']}

    def reshape_mon_items(self, run_length):
        for var, data in self.host.mon.item_contents.items():
            shape = ops.shape(data)
            if run_length < shape[0]:
                data = data[:run_length]
                setattr(self.host.mon, var, data)
                if _monitor_done_in == 'cuda':
                    setattr(self.host.mon, cuda_name_of(var), self.cpu2gpu(self.host, cpu_data=data))

            elif run_length > shape[0]:
                append = ops.zeros((run_length - shape[0],) + shape[1:])
                data = ops.vstack([data, append])
                setattr(self.host.mon, var, data)
                if _monitor_done_in == 'cuda':
                    setattr(self.host.mon, cuda_name_of(var), self.cpu2gpu(self.host, cpu_data=data))

    def get_monitor_func(self, mon_length, show_code=False):
        """Get monitor function.

        There are two kinds of ways to form monitor function.

        1. First, we can transfer GPU data to CPU, then utilize
           previous 'get_monitor_func()'.

        2. Second, we can form a GPU version function, at the same
           time, we monitor the data in the GPU backend.

        """
        mon = self.host.mon
        host_name = self.host.name
        monitor_func_name = 'monitor_step'
        if mon.num_item > 0:

            if _monitor_done_in == 'cpu':

                code_scope = {host_name: self.host}
                code_lines = [f'def {monitor_func_name}(_i):']

                # check monitor keys
                mon_keys = []
                for key in mon.item_names:
                    if not hasattr(self.host, key):
                        raise errors.ModelUseError(f'{self.host} do not have {key}, thus it cannot be monitored.')
                    key_val = getattr(self.host, key)
                    if isinstance(key_val, np.ndarray):
                        code_lines.append(f'  {host_name}.{cuda_name_of(key)}.copy_to_host(ary={host_name}.{key}, stream={host_name}.stream)')
                        mon_keys.append([key, f'{host_name}.{key}'])
                    elif isinstance(key_val, DeviceNDArray):
                        code_lines.append(f'  {key} = {host_name}.{cuda_name_of(key)}.copy_to_host(stream={host_name}.stream)')
                        mon_keys.append([key, key])
                    else:
                        mon_keys.append([key, f'{host_name}.{key}'])

                # add monitors
                code_lines.append(f'  {host_name}.stream.synchronize()')
                for key, transfer_key in mon_keys:
                    line = f'  {host_name}.mon.{key}[_i] = {transfer_key}'
                    code_lines.append(line)

                # function
                code = '\n'.join(code_lines)
                if show_code:
                    print(code)
                    pprint(code_scope)
                    print()
                exec(compile(code, '', 'exec'), code_scope)
                func = code_scope[monitor_func_name]
                self.upload(monitor_func_name, func)

                # results
                self.formatted_funcs['monitor'] = {
                    'func': func,
                    'scope': {host_name: self.host},
                    'call': [f'{host_name}.{monitor_func_name}(_i)'],
                }

            elif _monitor_done_in == 'cuda':
                args2calls = {'_i': '_i'}

                # code scope
                code_scope = {host_name: self.host, 'cuda': cuda}

                # task 1: group data according to the data shape
                # task 2: transfer data from cpu to gpu
                new_formatted_monitors = {}
                for key in mon.item_names:
                    if not hasattr(self.host, key):
                        raise errors.ModelUseError(f'{self.host} do not have {key}, thus it cannot be monitored.')
                    key_val = getattr(self.host, key)
                    if isinstance(key_val, (int, float)):
                        raise errors.ModelUseError(f'BrainPy Numba CUDA backend does not support monitor '
                                                   f'scalar value: {host_name}.{key} = {key_val}')
                    if isinstance(key_val, np.ndarray):
                        if key_val.ndim != 1:
                            raise NotImplementedError(f'BrainPy Numba CUDA backend only supports monitors for 1D vector '
                                                      f'data, not {key_val.ndim}-dimensional data of {host_name}.{key}')
                        key_gpu_name = self.transfer_cpu_data_to_gpu(self.host, cpu_key=key, cpu_data=key_val)
                        args2calls[f'{host_name}_{key}'] = f'{host_name}.{key_gpu_name}'
                        size = key_val.size
                    elif isinstance(key_val, DeviceNDArray):
                        if key_val.ndim != 1:
                            raise NotImplementedError(f'BrainPy Numba CUDA backend only supports inputs for 1D vector '
                                                      f'data, not {key_val.ndim}-dimensional data of {host_name}.{key}')
                        size = key_val.size
                        args2calls[f'{host_name}_{key}'] = f'{host_name}.{key}'
                    else:
                        raise NotImplementedError
                    if size not in new_formatted_monitors:
                        new_formatted_monitors[size] = []
                    new_formatted_monitors[size].append(key)

                # format code lines
                code_lines = []
                for size, keys in new_formatted_monitors.items():
                    code_lines = [f'  if thread_i < {size}:']
                    for key in keys:
                        # # initialize monitor array #
                        # mon[key] = np.zeros((mon_length, size), dtype=getattr(self.host, key).dtype)
                        # transfer data from GPU to CPU
                        mon_gpu_name = self.transfer_cpu_data_to_gpu(self.host.mon, cpu_key=key, cpu_data=getattr(mon, key))
                        args2calls[f'{host_name}_mon_{key}'] = f'{host_name}.mon.{mon_gpu_name}'
                        # add line #
                        line = f'    {host_name}_mon_{key}[_i, thread_i] = {host_name}_{key}[thread_i]'
                        code_lines.append(line)

                # arguments
                args2calls = sorted(args2calls.items())
                args = [arg for arg, _ in args2calls]
                calls = [call for _, call in args2calls]
                code_lines.insert(0, f'def {monitor_func_name}({", ".join(args)}):')
                code_lines.insert(1, f'  thread_i = cuda.grid(1)')

                # function
                code = '\n'.join(code_lines)
                if show_code:
                    print(code)
                    print(code_scope)
                    print()
                exec(compile(code, '', 'exec'), code_scope)

                func = cuda.jit(code_scope[monitor_func_name])
                self.upload(monitor_func_name, func)

                # results
                num_block, num_thread = get_cuda_size(max(list(new_formatted_monitors.keys())))
                self.formatted_funcs['monitor'] = {
                    'func': func,
                    'scope': {host_name: self.host, 'cuda': cuda},
                    'call': [f'{host_name}.{monitor_func_name}[{num_block}, {num_thread}, {host_name}.stream]({", ".join(calls)})',
                             f'{host_name}.stream.synchronize()'],
                }

            else:
                raise ValueError(f'Monitor is set to an unknown place by "_monitor_done_in = '
                                 f'{_monitor_done_in}".')

    def to_host(self, category=None):
        categories = get_categories(category)
        for host, keys in self.host_cpukey_gpukey.items():
            if type(host) in categories:
                for cpukey, gpukey in keys.items():
                    setattr(host, cpukey, getattr(host, gpukey).copy_to_host(stream=host.stream))

    def to_device(self, category=None):
        categories = get_categories(category)
        for host, keys in self.host_cpukey_gpukey.items():
            if type(host) in categories:
                for cpukey, gpukey in keys.items():
                    setattr(host, gpukey, cuda.to_device(getattr(host, cpukey),
                                                         stream=host.stream))


class NumbaCUDANetDriver(GeneralNetDriver):
    def to_host(self, category=None):
        categories = get_categories(category)

        host_cpukey_gpukey = {}
        for node in self.host.all_nodes.values():
            for host, keys in node.driver.host_cpukey_gpukey.items():
                if type(host) in categories:
                    if host not in host_cpukey_gpukey:
                        host_cpukey_gpukey[host] = {}
                    for cpukey, gpukey in keys.items():
                        if cpukey in host_cpukey_gpukey[host]:
                            continue
                        host_cpukey_gpukey[host][cpukey] = gpukey
                        setattr(host, cpukey, getattr(host, gpukey).copy_to_host(stream=host.stream))

    def to_device(self, category=None):
        categories = get_categories(category)

        host_cpukey_gpukey = {}
        for node in self.host.all_nodes.values():
            for host, keys in node.driver.host_cpukey_gpukey.items():
                if type(host) in categories:
                    if host not in host_cpukey_gpukey:
                        host_cpukey_gpukey[host] = {}
                    for cpukey, gpukey in keys.items():
                        if cpukey in host_cpukey_gpukey[host]:
                            continue
                        host_cpukey_gpukey[host][cpukey] = gpukey
                        setattr(host, gpukey, cuda.to_device(getattr(host, cpukey), stream=host.stream))
