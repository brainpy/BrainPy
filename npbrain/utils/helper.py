# -*- coding: utf-8 -*-

import copy
import functools
import types

import numba as nb
import numpy as np
from numba.core.dispatcher import Dispatcher

from .. import profile

__all__ = [
    # parameter helpers
    'check_params',

    # function helpers
    'jit_function',
    'autojit',
    'func_copy',

    # data structure
    'Dict',
    'default_dict',
    'ddict',

    # 'others'
    'clip',
    'get_clip'
]


##############################
# parameter helpers
##############################


def check_params(kwargs):
    """Check the ``kwargs`` parameters.

    If there are more parameters left, it means the users give the
    wrong parameters.

    Parameters
    ----------
    kwargs : dict
        Parameters.
    """
    assert len(kwargs) == 0, "No arguments about: {}".format(list(kwargs.keys()))


##############################
# function helpers
##############################

def jit_function(f):
    """Generate ``numba`` JIT functions.

    Parameters
    ----------
    f : callable
        The function.

    Returns
    -------
    callable
        JIT function.
    """
    op = profile.get_numba_profile()
    return nb.jit(f, **op)


def autojit(signature_or_func=None):
    """Format user defined functions.

    Parameters
    ----------
    signature_or_func : callable, a_list, str

    Returns
    -------
    callable
        function.
    """
    if callable(signature_or_func):  # function

        if profile.is_numba_bk():
            if not isinstance(signature_or_func, Dispatcher):
                op = profile.get_numba_profile()
                signature_or_func = nb.jit(signature_or_func, **op)
        return signature_or_func

    else:  # signature

        if signature_or_func is None:
            pass
        else:
            if isinstance(signature_or_func, str):
                signature_or_func = [signature_or_func]
            else:
                assert isinstance(signature_or_func, (list, tuple))
                signature_or_func = list(signature_or_func)
            for i in range(len(signature_or_func)):
                if ('float' in signature_or_func[i]) or ('int' in signature_or_func[i]):
                    pass
                elif '{' in signature_or_func[i] and '}' in signature_or_func[i]:
                    signature_or_func[i] = signature_or_func[i].format(f=profile.ftype, i=profile.itype)
                else:
                    s = signature_or_func[i].replace(' ', '')
                    s = s.replace('f[', profile.ftype + '[')
                    s = s.replace('f,', profile.ftype + ',')
                    s = s.replace('f)', profile.ftype + ')')
                    s = s.replace('i[', profile.itype + '[')
                    s = s.replace('i,', profile.itype + ',')
                    s = s.replace('i)', profile.itype + ')')
                    signature_or_func[i] = s

        def wrapper(f):
            if profile.is_numba_bk() and not isinstance(f, Dispatcher):
                op = profile.get_numba_profile()
                if profile.define_signature:
                    j = nb.jit(signature_or_func, **op)
                else:
                    j = nb.jit(**op)
                f = j(f)
            return f

        return wrapper


def is_lambda_function(func):
    """Check whether the function is a ``lambda`` function. Comes from
    https://stackoverflow.com/questions/23852423/how-to-check-that-variable-is-a-lambda-function

    Parameters
    ----------
    func : callable function
        The function.

    Returns
    -------
    bool
        True of False.
    """
    return isinstance(func, types.LambdaType) and func.__name__ == "<lambda>"


def func_copy(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__,
                           name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


##############################
# data structure
##############################

class Dict(dict):
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(self, '__key', kwargs.pop('__key', None))
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                self[arg[0]] = self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    self[key] = self._hook(val)

        for key, val in kwargs.items():
            self[key] = self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError(f"Attribute '{name}' is read-only in 'Dict' object.")
        else:
            self[name] = value

    def __setitem__(self, name, value):
        super(Dict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __delattr__(self, name):
        del self[name]

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(item.to_dict() if isinstance(item, type(self)) else item
                                        for item in value)
            else:
                base[key] = value
        return base

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if (k not in self) or (not isinstance(self[k], dict)) or (not isinstance(v, dict)):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def unique_add(self, key, val):
        if key in self:
            raise ValueError(f'Key "{key}" has already exists.')
        else:
            self[key] = val

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default


def default_dict(args_update, *args, **kwargs):
    args = Dict(*args, **kwargs)
    args.update(args_update)
    return args


ddict = default_dict


##############################
# other helpers
##############################


def clip(a, a_min, a_max):
    """Reproduce the ``clip`` function in NumPy.

    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to but faster than ``bnp.maximum(a_min, bnp.minimum(a, a_max))``.
    No check is performed to ensure ``a_min < a_max``.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like or None
        Minimum value. If None, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        None.
    a_max : scalar or array_like or None
        Maximum value. If None, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        None. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.
    """
    a = np.maximum(a, a_min)
    a = np.minimum(a, a_max)
    return a


def get_clip():
    @autojit(['f[:](f[:], f, f)'])
    def f(a, a_min, a_max):
        a = np.maximum(a, a_min)
        a = np.minimum(a, a_max)
        return a

    return f
