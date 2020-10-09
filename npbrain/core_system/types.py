# -*- coding: utf-8 -*-

from collections import OrderedDict

from .. import _numpy as bnp
from .. import profile

try:
    import numba as nb
except ImportError as e:
    if profile.is_numba_bk():
        raise e

    nb = None

__all__ = [
    'TypeChecker',
    'TypeMismatchError',
    'ObjState',
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

    @classmethod
    def copy_to(cls, *args, **kwargs):
        raise NotImplementedError


class TypeMismatchError(Exception):
    pass


class ObjState(dict, TypeChecker):
    def __init__(self, fields, help=''):
        TypeChecker.__init__(self, help=help)
        variables = OrderedDict()
        if isinstance(fields, (tuple, list)):
            variables.update({v: 0. for v in fields})
        elif isinstance(fields, dict):
            variables.update(fields)
        else:
            assert ValueError(f'"fields" only supports tuple/list/dict, not {type(variables)}.')
        self._keys = list(variables.keys())
        self._values = list(variables.values())
        self._vars = variables

    def extract_by_index(self, idx):
        return {k: self.__getitem__(k)[idx] for k in self._keys}

    def update_by_index(self, idx, val):
        data = self.__getitem__('_data')
        for k, v in val.items():
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[k], idx] = v

    def check(self, cls):
        if not isinstance(cls, type(self)):
            raise TypeMismatchError(f'Must be an instance of "{type(self)}", but got "{type(cls)}".')
        for k in self._keys:
            if k not in cls:
                raise TypeMismatchError(f'Key "{k}" is not found in "cls".')

    def __str__(self):
        return f'{self.__class__.__name__} ({str(self._keys)})'


class NeuState(ObjState):
    """Neuron State. """

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

    def copy_to(self, size):
        obj = NeuState(self._vars)
        return obj(size=size)


class SynState(ObjState):
    """Synapse State. """

    def __init__(self, fields, help=''):
        super(SynState, self).__init__(fields=fields, help=help)
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
        delay = 0 if (delay is None) or (delay < 1) else delay
        assert isinstance(delay, int), '"delay" must be a int to specify the delay length.'
        self._delay_len = delay
        self._delay_in = delay - 1

        # initialize data
        length = self._delay_len + len(self._vars)
        data = bnp.zeros((length,) + size, dtype=bnp.float_)
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

    def __setitem__(self, key, val):
        if key in self._vars:
            data = self.__getitem__('_data')
            _var2idx = self.__getitem__('_var2idx')
            data[_var2idx[key]] = val
        elif key in ['_data', '_var2idx', '_idx2var', '_cond_delay']:
            raise KeyError(f'"{key}" cannot be modified.')
        else:
            raise KeyError(f'"{key}" is not defined in "{str(self._keys)}".')

    def copy_to(self, size, delay=None):
        obj = SynState(self._vars)
        return obj(size=size, delay=delay)

    def push_cond(self, g):
        data = self.__getitem__('_data')
        data[self._delay_in] = g

    def pull_cond(self):
        data = self.__getitem__('_data')
        return data[self._delay_out]

    def _update_delay_indices(self):
        if self._delay_len > 0:
            self._delay_in = (self._delay_in + 1) % self._delay_len
            self._delay_out = (self._delay_out + 1) % self._delay_len


class _SynStateForNbSingleMode(SynState):
    def __init__(self, fields, help=''):
        super(_SynStateForNbSingleMode, self).__init__(fields=fields, help=help)
        self._delay_offset = {}

    def __call__(self, num, delay, delay_var):
        # check size
        assert isinstance(num, int)

        # check delay
        delay = 1 if (delay is None) or (delay < 1) else delay
        assert isinstance(delay, int), '"delay" must be a int to specify the delay length.'
        self._delay_len = delay
        self._delay_in = delay - 1
        self._delay_in2 = (self._delay_in - 1) % delay

        # initialize data
        non_delay_var = [k for k in self._keys if k not in delay_var]
        length = delay * len(delay_var) + len(non_delay_var)
        data = bnp.zeros((length, num), dtype=bnp.float_)
        var2idx = dict()
        idx2var = dict()
        state = dict()
        offset = 0
        for k, v in self._vars.items():
            var2idx[k] = offset
            idx2var[offset] = k
            if k in delay_var:
                data[offset: offset + delay] = 0.
                data[offset + delay - 1] = v
                state[k] = data[offset: offset + delay]
                self._delay_offset[k] = offset
                offset += delay
            else:
                data[offset] = v
                state[k] = data[offset]
                offset += 1
        state['_data'] = data
        state['_var2idx'] = var2idx
        state['_idx2var'] = idx2var

        dict.__init__(self, state)

        return self

    def _update_delay_indices(self):
        if self._delay_len > 0:
            self._delay_in2 = self._delay_in
            self._delay_in = (self._delay_in + 1) % self._delay_len
            self._delay_out = (self._delay_out + 1) % self._delay_len


class ListConn(TypeChecker):
    """Synaptic connection with list type."""

    def __init__(self, help=''):
        super(ListConn, self).__init__(help=help)

    def check(self, cls):
        if profile.is_numba_bk():
            if not isinstance(cls, nb.typed.List):
                raise TypeMismatchError(f'In numba mode, "cls" must be an instance of {type(nb.typed.List)}, '
                                        f'but got {type(cls)}. Hint: you can use "ListConn.create()" method.')
            if not isinstance(cls[0], (nb.typed.List, bnp.ndarray)):
                raise TypeMismatchError(f'In numba mode, elements in "cls" must be an instance of '
                                        f'{type(nb.typed.List)} or ndarray, but got {type(cls[0])}. '
                                        f'Hint: you can use "ListConn.create()" method.')
        else:
            if not isinstance(cls, list):
                raise TypeMismatchError(f'ListConn requires a list, but got {type(cls)}.')
            if not isinstance(cls[0], (list, bnp.ndarray)):
                raise TypeMismatchError(f'ListConn requires the elements of the list must be list or '
                                        f'ndarray, but got {type(cls)}.')

    @classmethod
    def copy_to(cls, conn):
        assert isinstance(conn, (list, tuple)), '"conn" must be a tuple/list.'
        assert isinstance(conn[0], (list, tuple)), 'Elements of "conn" must be tuple/list.'
        if profile.is_numba_bk():
            a_list = nb.typed.List()
            for l in conn:
                a_list.append(bnp.uint64(l))
        else:
            a_list = conn
        return a_list

    def __str__(self):
        return 'ListConn'


class MatConn(TypeChecker):
    """Synaptic connection with matrix (2d array) type."""

    def __init__(self, help=''):
        super(MatConn, self).__init__(help=help)

    def check(self, cls):
        if not (isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == 2):
            raise TypeMismatchError(f'MatConn requires a two-dimensional ndarray.')

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
        if not (isinstance(cls, bnp.ndarray) and bnp.ndim(cls) == self.dim):
            raise TypeMismatchError(f'MatConn requires a {self.dim}-D ndarray.')

    def __str__(self):
        return type(self).__name__ + f' (dim={self.dim})'


class String(TypeChecker):
    def __init__(self, help=''):
        super(String, self).__init__(help=help)

    def check(self, cls):
        if not isinstance(cls, str):
            raise TypeMismatchError(f'Require a string, got {type(cls)}.')

    def __str__(self):
        return 'StringType'


class Int(TypeChecker):
    def __init__(self, help=''):
        super(Int, self).__init__(help=help)

    def check(self, cls):
        if not isinstance(cls, int):
            raise TypeMismatchError(f'Require an int, got {type(cls)}.')

    def __str__(self):
        return 'IntType'


class Float(TypeChecker):
    def __init__(self, help=''):
        super(Float, self).__init__(help=help)

    def check(self, cls):
        if not isinstance(cls, float):
            raise TypeMismatchError(f'Require a float, got {type(cls)}.')

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
                raise TypeMismatchError(f'In numba, "List" requires an instance of {type(nb.typed.List)}, '
                                        f'but got {type(cls)}.')
        else:
            if not isinstance(cls, list):
                raise TypeMismatchError(f'"List" requires an instance of list, '
                                        f'but got {type(cls)}.')

        if self.item_type is not None:
            self.item_type.check(cls[0])

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
                raise TypeMismatchError(f'In numba, "Dict" requires an instance of {type(nb.typed.Dict)}, '
                                        f'but got {type(cls)}.')
        else:
            if not isinstance(cls, dict):
                raise TypeMismatchError(f'"Dict" requires an instance of dict, '
                                        f'but got {type(cls)}.')

        if self.key_type is not None:
            for key in cls.keys():
                self.key_type.check(key)
        if self.item_type is not None:
            for item in cls.items():
                self.item_type.check(item)

    def __str__(self):
        return type(self).__name__ + f'(key_type={str(self.key_type)}, item_type={str(self.item_type)})'
