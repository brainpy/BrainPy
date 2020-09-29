# -*- coding: utf-8 -*-

from npbrain import _numpy as bnp
from npbrain import profile

try:
    import numba as nb
except ImportError as e:
    if profile.is_numba_bk():
        raise e

    nb = None

__all__ = [
    'TypeChecker',
    'ObjState',
    'ListConn',
    'MatConn',
    'ijConn',
    'Array',
    'Int',
    'Float',
]


class TypeChecker(object):
    def check(self, cls):
        raise NotImplementedError


class ObjState(dict, TypeChecker):
    """Object State. """

    def __init__(self, fields):
        variables = dict()
        if isinstance(fields, (tuple, list)):
            variables.update({v: 0. for v in fields})
        elif isinstance(fields, dict):
            variables.update(fields)
        else:
            assert ValueError(f'"fields" only supports tuple/list/dict, not {type(variables)}.')
        self._keys = list(variables.keys())
        self._values = list(variables.values())
        self._vars = variables

    def __call__(self, size=None):
        if size is None:  # single-neuron level
            assert profile.is_numpy_bk(), '"size" cannot be None.'
            state = {k: v for k, v in self._vars.items()}

        else:  # neuron-group level
            if isinstance(size, int):
                size = (size,)
            elif isinstance(size, (tuple, list)):
                size = tuple(size)
            else:
                raise ValueError(f'Unknown size type: {type(size)}.')

            if profile.is_numpy_bk():
                state = {k: bnp.ones(size, dtype=bnp.float_) * v for k, v in self._vars.items()}

            elif profile.is_numba_bk():
                data = bnp.zeros((len(self._vars),) + size, dtype=bnp.float_)
                var2idx = dict()
                idx2var = dict()
                state = dict()
                for i, k in enumerate(self._keys):
                    state[k] = data[i]
                    var2idx[k] = i
                    idx2var[i] = k
                state['_data'] = data
                state['_var2idx'] = var2idx
                state['_idx2var'] = idx2var

            else:
                raise NotImplementedError

        super(ObjState, self).__init__(state)

        return self

    def __setitem__(self, key, val):
        if key in self._vars:
            key_val = self.__getitem__(key)
            if isinstance(key_val, bnp.ndarray):
                key_val[:] = val
            else:
                super(ObjState, self).__setitem__(key, val)
        elif key in ['_data', '_var2idx', '_idx2var']:
            raise KeyError(f'"{key}" cannot be modified.')
        else:
            raise KeyError(f'"{key}" is not defined in "{str(self._keys)}".')

    def check(self, cls):
        if not isinstance(cls, ObjState):
            return False
        for k in self._keys:
            if k not in cls:
                return False
        return True

    def __str__(self):
        return type(self).__name__ + f' ({str(self._keys)})'


class _ListConn(TypeChecker):
    """Synaptic connection with list type."""

    def check(self, cls):
        if profile.is_numba_bk():
            return isinstance(cls, nb.typed.List) and isinstance(cls[0], (nb.typed.List, bnp.ndarray))
        else:
            return isinstance(cls, list) and isinstance(cls[0], (list, bnp.ndarray))

    def __str__(self):
        return 'ListConn'


ListConn = _ListConn()


class _MatConn(TypeChecker):
    """Synaptic connection with matrix (2d array) type."""

    def check(self, cls):
        return isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == 2

    def __str__(self):
        return 'MatConn'


MatConn = _MatConn()


class ijConn(TypeChecker):
    def __str__(self):
        return 'ijConn'


class Array(TypeChecker):
    """NumPy ndarray."""

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, size):
        if isinstance(size, int):
            assert self.dim == 1
        else:
            assert len(size) == self.dim
        return bnp.zeros(size, dtype=bnp.float_)

    def check(self, cls):
        return isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == self.dim

    def __str__(self):
        return type(self).__name__ + f' (dim={self.dim})'


class _StringType(TypeChecker):
    def check(self, cls):
        return isinstance(cls, str)

    def __str__(self):
        return 'StringType'


String = _StringType


class _IntType(TypeChecker):
    def check(self, cls):
        return isinstance(cls, int)

    def __str__(self):
        return 'IntType'


Int = _IntType()


class _FloatType(TypeChecker):
    def check(self, cls):
        return isinstance(cls, float)

    def __str__(self):
        return 'Floatype'


Float = _FloatType()


class List(TypeChecker):
    def __init__(self, item_type=None):
        if item_type is None:
            self.item_type = None
        else:
            assert isinstance(item_type, TypeChecker), 'Must be a TypeChecker.'
            self.item_type = item_type

    def check(self, cls):
        if profile.is_numba_bk():
            if not isinstance(cls, nb.typed.List):
                return False
        else:
            if not isinstance(cls, list):
                return False

        if self.item_type is not None:
            return self.item_type.check(cls[0])

        return True

    def __str__(self):
        return type(self).__name__ + f'(item_type={str(self.item_type)})'


class Dict(TypeChecker):
    def __init__(self, key_type=String, item_type=None):
        if key_type is not None:
            assert isinstance(key_type, TypeChecker), 'Must be a TypeChecker.'
        self.key_type = key_type
        if item_type is not None:
            assert isinstance(item_type, TypeChecker), 'Must be a TypeChecker.'
        self.item_type = item_type

    def check(self, cls):
        if profile.is_numba_bk():
            if not isinstance(cls, nb.typed.Dict):
                return False
        else:
            if not isinstance(cls, dict):
                return False

        if self.key_type is not None:
            for key in cls.keys():
                if not self.key_type.check(key):
                    return False
        if self.item_type is not None:
            for item in cls.items():
                if not self.item_type.check(item):
                    return False

        return True

    def __str__(self):
        return type(self).__name__ + f'(key_type={str(self.key_type)}, item_type={str(self.item_type)})'
