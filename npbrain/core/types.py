# -*- coding: utf-8 -*-

from .. import _numpy as bnp
from .. import profile

try:
    from numba import typed

    List = typed.List
    Dict = typed.Dict
except ImportError as e:
    if profile.is_numba_bk():
        raise e

    List = list
    Dict = dict

__all__ = [
    'ObjState',
    'ObjPars',
    'ConnList',
    'ConnMat',
    'ArrayType',
]


class TypeChecker(object):
    pass


class ObjState(dict):
    """Object State. """
    def __init__(self, *args, **kwargs):
        variables = dict()
        for k in args:
            if isinstance(k, str):
                variables[k] = 0.
            elif isinstance(k, (tuple, list)):
                variables.update({v: 0. for v in k})
            elif isinstance(k, dict):
                variables.update(k)
            else:
                assert ValueError(f'"variables" only supports tuple/list/dict, not {type(variables)}.')
        variables.update(kwargs)
        self.variables = variables

    def __call__(self, size):
        # check size
        # -----------
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise ValueError(f'Unknown size type: {type(size)}.')

        # initialize state
        # -----------------
        if profile.is_numpy_bk():
            state = {k: bnp.zeros(size, dtype=bnp.float_) * v
                     for k, v in self.variables.items()}

        elif profile.is_numba_bk():
            data = bnp.zeros((len(self.variables),) + size, dtype=bnp.float_)
            var2idx = dict()
            state = dict()
            for i, k in enumerate(self.variables.keys()):
                state[k] = data[i]
                var2idx[k] = i
            state['_data'] = data
            state['_var2idx'] = var2idx

        else:
            raise ValueError

        super(ObjState, self).__init__(state)

    @staticmethod
    def check_consistent(cls):
        return isinstance(cls, ObjState)


class ObjPars(Dict):
    """Object Parameters."""
    def __new__(cls, *args, **kwargs):
        if profile.is_numba_bk():
            return object.__new__(typed.Dict)
        else:
            return dict.__new__(dict)


class ConnList(List):
    """Synaptic connection with list type."""
    def __new__(cls, *args, **kwargs):
        if profile.is_numba_bk():
            return object.__new__(List)
        else:
            return list.__new__(list)

    @staticmethod
    def check_consistent(cls):
        if profile.is_numba_bk():
            return isinstance(cls, typed.List)
        else:
            return isinstance(cls, list)


class ConnMat(bnp.ndarray):
    """Synaptic connection with matrix (2d array) type."""
    @staticmethod
    def check_consistent(cls):
        return isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == 2


class ArrayType(object):
    """NumPy ndarray."""
    def __init__(self, dim):
        self.dim = dim

    def check_consistent(self, cls):
        return isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == self.dim
