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
    'NeuState',
    'SynState',
    'ListConn',
    'MatConn',
    'ijConn',
    'Array',
    'Int',
    'Float',
    'List',
    'Dict',
]


class TypeChecker(object):
    def __init__(self, help):
        self.help = help

    def check(self, cls):
        raise NotImplementedError


class NeuState(dict, TypeChecker):
    """Neuron State. """

    def __init__(self, fields, help=''):
        TypeChecker.__init__(self, help=help)
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

    def __call__(self, size):
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise ValueError(f'Unknown size type: {type(size)}.')

        data = bnp.zeros((len(self._vars),) + size, dtype=bnp.float_)
        var2idx = dict()
        idx2var = dict()
        state = dict()
        for i, (k, v) in enumerate(self._vars.items()):
            state[k] = data[i]
            data[i] = v
            var2idx[k] = i
            idx2var[i] = k
        state['_data'] = data
        state['_var2idx'] = var2idx
        state['_idx2var'] = idx2var

        dict.__init__(self, state)

        return self

    def __setitem__(self, key, val):
        if key in self._vars:
            data = self.__getitem__('_data')
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[key]] = val
        elif key in ['_data', '_var2idx', '_idx2var']:
            raise KeyError(f'"{key}" cannot be modified.')
        else:
            raise KeyError(f'"{key}" is not defined in "{str(self._keys)}".')

    def extract_by_index(self, idx):
        return {v: self.__getitem__(v)[idx] for v in self._vars}

    def update_by_index(self, idx, val):
        data = self.__getitem__('_data')
        for k, v in val:
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[k], idx] = v

    def check(self, cls):
        if not isinstance(cls, NeuState):
            return False
        for k in self._keys:
            if k not in cls:
                return False
        return True

    def __str__(self):
        return f'NeuState ({str(self._keys)})'


class SynState(dict, TypeChecker):
    """Synapse State. """

    def __init__(self, fields, help=''):
        TypeChecker.__init__(self, help=help)
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
        self._delay_len = 1
        self._delay_in = 0
        self._delay_out = 0

    def __call__(self, size, delay=None):
        # check size
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise ValueError(f'Unknown size type: {type(size)}.')

        # check delay
        delay = 1 if delay is None else delay
        assert isinstance(delay, int), '"delay" must be a int to specify the delay length.'
        self._delay_len = delay

        # initialize data
        data = bnp.zeros((self._delay_len + len(self._vars),) + size, dtype=bnp.float_)
        var2idx = dict()
        idx2var = dict()
        state = dict()
        for i, (k, v) in enumerate(self._vars.items()):
            idx = i + self._delay_len
            data[idx] = v
            state[k] = data[idx]
            var2idx[k] = idx
            idx2var[i] = k
        state['_cond_delay'] = data[:self._delay_len]
        state['_data'] = data
        state['_var2idx'] = var2idx
        state['_idx2var'] = idx2var

        dict.__init__(self, state)

        return self

    def push_cond(self, g):
        data = self.__getitem__('_data')
        data[self._delay_in] = g

    def pull_cond(self):
        data = self.__getitem__('_data')
        return data[self._delay_out]

    def _update_delay_indices(self):
        self._delay_in = (self._delay_in + 1) % self._delay_len
        self._delay_in = (self._delay_in + 1) % self._delay_len

    def __setitem__(self, key, val):
        if key in self._vars:
            data = self.__getitem__('_data')
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[key]] = val
        elif key in ['_data', '_var2idx', '_idx2var', '_cond_delay']:
            raise KeyError(f'"{key}" cannot be modified.')
        else:
            raise KeyError(f'"{key}" is not defined in "{str(self._keys)}".')

    def extract_by_index(self, idx):
        return {v: self.__getitem__(v)[idx] for v in self._vars}

    def update_by_index(self, idx, val):
        data = self.__getitem__('_data')
        for k, v in val:
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[k], idx] = v

    def check(self, cls):
        if not isinstance(cls, SynState):
            return False
        for k in self._keys:
            if k not in cls:
                return False
        return True

    def __str__(self):
        return f'SynState ({str(self._keys)})'



class ListConn(TypeChecker):
    """Synaptic connection with list type."""

    def __init__(self, help=''):
        super(ListConn, self).__init__(help=help)

    def check(self, cls):
        if profile.is_numba_bk():
            return isinstance(cls, nb.typed.List) and isinstance(cls[0], (nb.typed.List, bnp.ndarray))
        else:
            return isinstance(cls, list) and isinstance(cls[0], (list, bnp.ndarray))

    def __str__(self):
        return 'ListConn'



class MatConn(TypeChecker):
    """Synaptic connection with matrix (2d array) type."""

    def __init__(self, help=''):
        super(MatConn, self).__init__(help=help)

    def check(self, cls):
        return isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == 2

    def __str__(self):
        return 'MatConn'


class ijConn(TypeChecker):
    def __init__(self, help=''):
        super(ijConn, self).__init__(help=help)

    def __str__(self):
        return 'ijConn'


class Array(TypeChecker):
    """NumPy ndarray."""

    def __init__(self, dim, help=''):
        self.dim = dim
        super(Array, self).__init__(help=help)

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


class String(TypeChecker):
    def __init__(self, help=''):
        super(String, self).__init__(help=help)

    def check(self, cls):
        return isinstance(cls, str)

    def __str__(self):
        return 'StringType'




class Int(TypeChecker):
    def __init__(self, help=''):
        super(Int, self).__init__(help=help)

    def check(self, cls):
        return isinstance(cls, int)

    def __str__(self):
        return 'IntType'




class Float(TypeChecker):
    def __init__(self, help=''):
        super(Float, self).__init__(help=help)

    def check(self, cls):
        return isinstance(cls, float)

    def __str__(self):
        return 'Floatype'


class List(TypeChecker):
    def __init__(self, item_type=None, help=''):
        if item_type is None:
            self.item_type = None
        else:
            assert isinstance(item_type, TypeChecker), 'Must be a TypeChecker.'
            self.item_type = item_type

        super(List, self).__init__(help=help)

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
    def __init__(self, key_type=String, item_type=None, help=''):
        if key_type is not None:
            assert isinstance(key_type, TypeChecker), 'Must be a TypeChecker.'
        self.key_type = key_type
        if item_type is not None:
            assert isinstance(item_type, TypeChecker), 'Must be a TypeChecker.'
        self.item_type = item_type
        super(Dict, self).__init__(help=help)

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

