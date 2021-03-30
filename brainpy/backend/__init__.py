# -*- coding: utf-8 -*-

from types import ModuleType

from brainpy import errors
from .operators.bk_numpy import *
from .runners.general_runner import GeneralNodeRunner, GeneralNetRunner

_backend = 'numpy'  # default backend is NumPy
_node_runner = None
_net_runner = None
_dt = 0.1

CLASS_KEYWORDS = ['self', 'cls']

OPS_FOR_SOLVER = ['normal', 'sum', 'exp', 'matmul', 'shape', ]
OPS_FOR_SIMULATION = ['as_tensor', 'zeros', 'ones', 'arange',
                      'vstack', 'where', 'unsqueeze', 'squeeze']

SUPPORTED_BACKEND = {
    'numba', 'numba-parallel', 'numba-cuda', 'jax',  # JIT framework
    'numpy', 'pytorch', 'tensorflow',
}

SYSTEM_KEYWORDS = ['_dt', '_t', '_i']


def set(backend=None, module_or_operations=None, node_runner=None, net_runner=None, dt=None):
    """Basic backend setting function.

    Using this function, users can set the backend they prefer. For backend
    which is unknown, users can provide `module_or_operations` to specify
    the operations needed. Also, users can customize the node runner, or the
    network runner, by providing the `node_runner` or `net_runner` keywords.
    The default numerical precision `dt` can also be set by this function.

    Parameters
    ----------
    backend : str
        The backend name.
    module_or_operations : module, dict, optional
        The module or the a dict containing necessary operations.
    node_runner : GeneralNodeRunner
        An instance of node runner.
    net_runner : GeneralNetRunner
        An instance of network runner.
    dt : float
        The numerical precision.
    """
    if dt is not None:
        set_dt(dt)

    if (backend is None) or (_backend == backend):
        return

    global_vars = globals()
    if backend == 'numpy':
        from .operators import bk_numpy

        node_runner = GeneralNodeRunner if node_runner is None else node_runner
        net_runner = GeneralNetRunner if net_runner is None else net_runner
        module_or_operations = bk_numpy if module_or_operations is None else module_or_operations

    elif backend == 'pytorch':
        from .operators import bk_pytorch

        node_runner = GeneralNodeRunner if node_runner is None else node_runner
        net_runner = GeneralNetRunner if net_runner is None else net_runner
        module_or_operations = bk_pytorch if module_or_operations is None else module_or_operations

    elif backend == 'tensorflow':
        from .operators import bk_tensorflow

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
        raise NotImplementedError
        from .operators import bk_numba_cuda
        from .runners.numba_cuda_runner import NumbaCudaNodeRunner

        node_runner = NumbaCudaNodeRunner if node_runner is None else node_runner
        module_or_operations = bk_numba_cuda if module_or_operations is None else module_or_operations

    elif backend == 'jax':
        raise NotImplementedError
        from .operators import bk_jax
        from .runners.jax_runner import JaxRunner

        node_runner = JaxRunner if node_runner is None else node_runner
        module_or_operations = bk_jax if module_or_operations is None else module_or_operations

    else:
        if module_or_operations is None:
            raise errors.ModelUseError(f'Backend "{backend}" is unknown, '
                                       f'please provide the "module_or_operations" '
                                       f'to specify the necessary computation units.')
        node_runner = GeneralNodeRunner if node_runner is None else node_runner

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


def set_class_keywords(*args):
    """Set the keywords for class specification.

    For example:

    >>> class A(object):
    >>>    def __init__(cls):
    >>>        pass
    >>>    def f(self, ):
    >>>        pass

    In this case, I use "cls" to denote the "self". So, I can set this by

    >>> set_class_keywords('cls', 'self')

    """
    global CLASS_KEYWORDS
    CLASS_KEYWORDS = list(args)


def set_dt(dt):
    """Set the numerical integrator precision.

    Parameters
    ----------
    dt : float
        Numerical integration precision.
    """
    assert isinstance(dt, float)
    global _dt
    _dt = dt


def get_dt():
    """Get the numerical integrator precision.

    Returns
    -------
    dt : float
        Numerical integration precision.
    """
    return _dt


def set_ops_from_module(module):
    """Set operations from a module.

    Parameters
    ----------
    module :
    """
    global_vars = globals()
    for ops in OPS_FOR_SOLVER:
        if not hasattr(module, ops):
            raise ValueError(f'Operation "{ops}" is needed, but is not '
                             f'defined in module "{module}".')
        global_vars[ops] = getattr(module, ops)
    for ops in OPS_FOR_SIMULATION:
        if hasattr(module, ops):
            global_vars[ops] = getattr(module, ops)
        else:
            del global_vars[ops]


def set_ops(**kwargs):
    """Set operations.

    Parameters
    ----------
    kwargs :
        The key=operation setting.
    """
    global_vars = globals()
    for key, value in kwargs.items():
        global_vars[key] = value


def get_backend():
    """Get the current backend name.

    Returns
    -------
    backend : str
        The name of the current backend name.
    """
    return _backend


def get_node_runner():
    """Get the current node runner.

    Returns
    -------
    node_runner
        The node runner class.
    """
    global _node_runner
    if _node_runner is None:
        from .runners.general_runner import GeneralNodeRunner
        _node_runner = GeneralNodeRunner
    return _node_runner


def get_net_runner():
    """Get the current network runner.

    Returns
    -------
    net_runner
        The network runner.
    """
    global _net_runner
    if _net_runner is None:
        from .runners.general_runner import GeneralNetRunner
        _net_runner = GeneralNetRunner
    return _net_runner
