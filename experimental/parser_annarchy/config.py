# -*- coding: utf-8 -*-

config = dict(
    {
        'dt': 1.0,
        'verbose': False,
        'debug': False,
        'show_time': False,
        'suppress_warnings': False,
        'num_threads': 1,
        'paradigm': "openmp",
        'method': "explicit",
        'precision': "double",
        'seed': -1,
        'structural_plasticity': False,
        'profiling': False,
        'profile_out': None
    }
)

_objects = {
    'functions': [],
    'neurons': [],
    'synapses': [],
    'constants': [],
}

# Authorized keywork for attributes
authorized_keywords = [
    # Init
    'init',
    # Bounds
    'min',
    'max',
    # Locality
    'population',
    'postsynaptic',
    'projection',
    # Numerical methods
    'explicit',
    'implicit',
    'semiimplicit',
    'exponential',
    'midpoint',
    'exact',
    'event-driven',
    # Refractory
    'unless_refractory',
    # Type
    'int',
    'bool',
    'float',
    # Event-based
    'unless_post',
]


def list_constants(net_id=0):
    """
    Returns a list of all constants declared with ``Constant(name, value)``.
    """
    l = []
    for obj in _objects['constants']:
        l.append(obj.name)
    return l


def get_constant(name, net_id=0):
    """
    Returns the ``Constant`` object with the given name, ``None`` otherwise.
    """
    for obj in _objects['constants']:
        if obj.name == name:
            return obj
    return None
