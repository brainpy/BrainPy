# -*- coding: utf-8 -*-

from collections import OrderedDict

import numba as nb
import numpy as np

from .. import profile
from ..errors import TypeMismatchError

__all__ = [
    'TypeChecker',
    'ObjState',
    'NeuState',
    'SynState',
    'ListConn',
    'MatConn',
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
    def make_copy(cls, *args, **kwargs):
        raise NotImplementedError


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

    def __repr__(self):
        return self.__str__()


class NeuState(ObjState):
    """Neuron State Management. """

    def __call__(self, size):
        if isinstance(size, int):
            size = (size,)
        elif isinstance(size, (tuple, list)):
            size = tuple(size)
        else:
            raise ValueError(f'Unknown size type: {type(size)}.')

        data = np.zeros((len(self._vars),) + size, dtype=np.float_)
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

    def make_copy(self, size):
        obj = NeuState(self._vars)
        return obj(size=size)


class SynState(ObjState):
    """Synapse State Management. """

    def __init__(self, fields, help=''):
        super(SynState, self).__init__(fields=fields, help=help)
        self._delay_len = 1
        self._delay_in = 0
        self._delay_out = 0

    def __call__(self, size, delay=None, delay_vars=()):
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

        # check delay_vars
        if isinstance(delay_vars, str):
            delay_vars = (delay_vars,)
        elif isinstance(delay_vars, (tuple, list)):
            delay_vars = tuple(delay_vars)
        else:
            raise ValueError(f'Unknown delay_vars type: {type(delay_vars)}.')

        # initialize data
        length = len(self._vars) + delay * len(delay_vars)
        data = np.zeros((length,) + size, dtype=np.float_)
        var2idx = dict()
        idx2var = dict()
        state = dict()
        for i, (k, v) in enumerate(self._vars.items()):
            data[i] = v
            state[k] = data[i]
            var2idx[k] = i
            idx2var[i] = k
        index_offset = len(self._vars)
        for i, v in enumerate(delay_vars):
            var2idx[f'_{v}_offset'] = i * delay + index_offset
            state[f'_{v}_delay'] = data[i * delay + index_offset: (i + 1) * delay + index_offset]
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
            raise KeyError(f'"{key}" is not defined in {type(self).__name__}, '
                           f'only finds "{str(self._keys)}".')

    def extract_by_index(self, idx, delay_pull=False):
        if delay_pull:
            res = {}
            for k in self._keys:
                if f'_{k}_delay' in self:
                    res[k] = self.delay_pull(k)[idx]
                else:
                    res[k] = self.__getitem__(k)[idx]
            return res
        else:
            return {k: self.__getitem__(k)[idx] for k in self._keys}

    def make_copy(self, size, delay=None, delay_vars=('cond',)):
        obj = SynState(self._vars)
        return obj(size=size, delay=delay, delay_vars=delay_vars)

    def delay_push(self, g, var='cond'):
        if self._delay_len > 0:
            data = self.__getitem__('_data')
            offset = self.__getitem__('_var2idx')[f'_{var}_offset']
            data[self._delay_in + offset] = g

    def delay_pull(self, var='cond'):
        if self._delay_len > 0:
            data = self.__getitem__('_data')
            offset = self.__getitem__('_var2idx')[f'_{var}_offset']
            return data[self._delay_out + offset]
        else:
            data = self.__getitem__('_data')
            var2idx = self.__getitem__('_var2idx')
            return data[var2idx[var]]

    def _update_delay_indices(self):
        if self._delay_len > 0:
            self._delay_in = (self._delay_in + 1) % self._delay_len
            self._delay_out = (self._delay_out + 1) % self._delay_len


class ListConn(TypeChecker):
    """Synaptic connection with list type."""

    def __init__(self, help=''):
        super(ListConn, self).__init__(help=help)

    def check(self, cls):
        if profile.is_jit_backend():
            if not isinstance(cls, nb.typed.List):
                raise TypeMismatchError(f'In numba mode, "cls" must be an instance of {type(nb.typed.List)}, '
                                        f'but got {type(cls)}. Hint: you can use "ListConn.create()" method.')
            if not isinstance(cls[0], (nb.typed.List, np.ndarray)):
                raise TypeMismatchError(f'In numba mode, elements in "cls" must be an instance of '
                                        f'{type(nb.typed.List)} or ndarray, but got {type(cls[0])}. '
                                        f'Hint: you can use "ListConn.create()" method.')
        else:
            if not isinstance(cls, list):
                raise TypeMismatchError(f'ListConn requires a list, but got {type(cls)}.')
            if not isinstance(cls[0], (list, np.ndarray)):
                raise TypeMismatchError(f'ListConn requires the elements of the list must be list or '
                                        f'ndarray, but got {type(cls)}.')

    @classmethod
    def make_copy(cls, conn):
        assert isinstance(conn, (list, tuple)), '"conn" must be a tuple/list.'
        assert isinstance(conn[0], (list, tuple)), 'Elements of "conn" must be tuple/list.'
        if profile.is_jit_backend():
            a_list = nb.typed.List()
            for l in conn:
                a_list.append(np.uint64(l))
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
        if not (isinstance(cls, np.ndarray) and np.ndim(cls) == 2):
            raise TypeMismatchError(f'MatConn requires a two-dimensional ndarray.')

    def __str__(self):
        return 'MatConn'


class SliceConn(TypeChecker):
    def __init__(self, help=''):
        super(SliceConn, self).__init__(help=help)

    def check(self, cls):
        if not (isinstance(cls, np.ndarray) and np.shape[1] == 2):
            raise TypeMismatchError(f'')

    def __str__(self):
        return 'SliceConn'



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
        return np.zeros(size, dtype=np.float_)

    def check(self, cls):
        if not (isinstance(cls, np.ndarray) and np.ndim(cls) == self.dim):
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
        if profile.is_jit_backend():
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
        if profile.is_jit_backend():
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
