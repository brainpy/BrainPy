# -*- coding: utf-8 -*-

from types import ModuleType

from brainpy import errors
from .operators.bk_numpy import *

_backend = 'numpy'  # default backend is NumPy
_node_runner = None
_net_runner = None
NEEDED_OPS = ['normal', 'reshape', 'shape', 'exp', 'sum', 'zeros',
              'eye', 'matmul', 'vstack', 'arange']
SUPPORTED_BACKEND = ['numba', 'numba-parallel', 'numba-cuda', 'jax',
                     'numpy', 'pytorch', 'tensorflow', ]
SYSTEM_KEYWORDS = ['_dt', '_t', '_i']


def set(backend, module_or_operations=None, node_runner=None, net_runner=None):
    if _backend == backend:
        return

    if backend == 'numpy':
        from .operators import bk_numpy
        from .runners.general_runner import GeneralNodeRunner, GeneralNetRunner

        node_runner = GeneralNodeRunner if node_runner is None else node_runner
        net_runner = GeneralNetRunner if net_runner is None else net_runner
        module_or_operations = bk_numpy if module_or_operations is None else module_or_operations

    elif backend == 'pytorch':
        from .operators import bk_pytorch
        from .runners.general_runner import GeneralNodeRunner, GeneralNetRunner

        node_runner = GeneralNodeRunner if node_runner is None else node_runner
        net_runner = GeneralNetRunner if net_runner is None else net_runner
        module_or_operations = bk_pytorch if module_or_operations is None else module_or_operations

    elif backend == 'tensorflow':
        from .operators import bk_tensorflow
        from .runners.general_runner import GeneralNodeRunner, GeneralNetRunner

        node_runner = GeneralNodeRunner if node_runner is None else node_runner
        net_runner = GeneralNetRunner if net_runner is None else net_runner
        module_or_operations = bk_tensorflow if module_or_operations is None else module_or_operations

    elif backend == 'numba':
        from .operators import bk_numba_cpu
        from .runners.numba_cpu_runner import NumbaCPUNodeRunner, set_numba_profile

        node_runner = NumbaCPUNodeRunner if node_runner is None else node_runner
        module_or_operations = bk_numba_cpu if module_or_operations is None else module_or_operations
        set_numba_profile(parallel=False)

    elif backend == 'numba-parallel':
        from .operators import bk_numba_cpu
        from .runners.numba_cpu_runner import NumbaCPUNodeRunner, set_numba_profile

        node_runner = NumbaCPUNodeRunner if node_runner is None else node_runner
        module_or_operations = bk_numba_cpu if module_or_operations is None else module_or_operations
        set_numba_profile(parallel=True)

    elif backend == 'numba-cuda':
        from .operators import bk_numba_cuda
        from .runners.numba_cuda_runner import NumbaCudaNodeRunner

        node_runner = NumbaCudaNodeRunner if node_runner is None else node_runner
        module_or_operations = bk_numba_cuda if module_or_operations is None else module_or_operations

    elif backend == 'jax':
        from .operators import bk_jax
        from .runners.jax_runner import JaxRunner

        node_runner = JaxRunner if node_runner is None else node_runner
        module_or_operations = bk_jax if module_or_operations is None else module_or_operations

    else:
        if module_or_operations is None:
            raise errors.ModelUseError(f'Backend "{backend}" is unknown, '
                                       f'please provide the "module_or_operations" '
                                       f'to specify the necessary computation units.')
        from .runners.general_runner import GeneralNodeRunner
        node_runner = GeneralNodeRunner if node_runner is None else node_runner

    global_vars = globals()
    global_vars['_backend'] = backend
    global_vars['_node_runner'] = node_runner
    global_vars['_net_runner'] = net_runner
    if isinstance(module_or_operations, ModuleType):
        set_ops_from_module(module_or_operations)
    elif isinstance(module_or_operations, dict):
        set_ops(**module_or_operations)
    else:
        raise errors.ModelUseError('"module_or_operations" must be a module '
                                   'or a dict of operations.')


def set_ops_from_module(module):
    global_vars = globals()
    for ops in NEEDED_OPS:
        if not hasattr(module, ops):
            raise ValueError(f'Operation "{ops}" is needed, but is not '
                             f'defined in module "{module}".')
        global_vars[ops] = getattr(module, ops)


def set_ops(**kwargs):
    global_vars = globals()
    for key in global_vars.keys():
        if (not key.startswith('__')) and (key in kwargs):
            global_vars[key] = kwargs.pop(key)

    if len(kwargs):
        raise ValueError(f'Unknown operations: {list(kwargs.keys())}')


def get_backend():
    return _backend


def get_node_runner():
    global _node_runner
    if _node_runner is None:
        from .runners.general_runner import GeneralNodeRunner
        _node_runner = GeneralNodeRunner
    return _node_runner


def get_net_runner():
    global _net_runner
    if _net_runner is None:
        from .runners.general_runner import GeneralNetRunner
        _net_runner = GeneralNetRunner
    return _net_runner
