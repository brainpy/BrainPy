# -*- coding: utf-8 -*-

from brainpy.simulation.drivers import AbstractNodeDriver, AbstractNetDriver
from .general import GeneralNodeDriver, GeneralNetDriver

__all__ = [
    'switch_to',
    'set_buffer',
    'get_buffer',
    'get_node_driver',
    'get_net_driver',

    'BUFFER',
]

NODE_DRIVER = GeneralNodeDriver
NET_DRIVER = GeneralNetDriver
BUFFER = {}


def switch_to(backend):
    buffer = get_buffer(backend)

    global NODE_DRIVER, NET_DRIVER
    if backend in ['numpy', 'pytorch', 'tensorflow']:
        from . import general
        NODE_DRIVER = buffer.get('node', None) or GeneralNodeDriver
        NET_DRIVER = buffer.get('net', None) or GeneralNetDriver

    elif backend in ['numba', 'numba-parallel']:
        from . import numba_cpu
        NODE_DRIVER = buffer.get('node', None) or numba_cpu.NumbaCPUNodeDriver
        NET_DRIVER = buffer.get('net', None) or GeneralNetDriver

    elif backend == 'numba-cuda':
        from . import numba_cuda
        NODE_DRIVER = buffer.get('node', None) or numba_cuda.NumbaCudaNodeDriver
        NET_DRIVER = buffer.get('net', None) or numba_cuda.NumbaCudaNetDriver

    else:
        if 'node' not in buffer:
            raise ValueError(f'"{backend}" is an unknown backend, should set node buffer '
                             f'by "brainpy.drivers.set_buffer(backend, node=SomeNodeDriver)"')
        if 'net' not in buffer:
            raise ValueError(f'"{backend}" is an unknown backend, should set node buffer '
                             f'by "brainpy.drivers.set_buffer(backend, net=SomeNeteDriver)"')
        NODE_DRIVER = buffer.get('node')
        NET_DRIVER = buffer.get('net')



def set_buffer(backend, node=None, net=None):
    global BUFFER
    if backend not in BUFFER:
        BUFFER[backend] = dict()

    if node is not None:
        assert isinstance(node, AbstractNodeDriver)
        BUFFER[backend]['node'] = node
    if net is not None:
        assert isinstance(node, AbstractNetDriver)
        BUFFER[backend]['net'] = net


def get_buffer(backend):
    return BUFFER.get(backend, dict())


def get_node_driver():
    """Get the current node runner.

    Returns
    -------
    node_runner
        The node runner class.
    """
    return NODE_DRIVER


def get_net_driver():
    """Get the current network runner.

    Returns
    -------
    net_runner
        The network runner.
    """
    return NET_DRIVER
