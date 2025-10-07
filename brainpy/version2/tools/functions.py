# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import inspect
from functools import partial
from operator import attrgetter
from types import MethodType

__all__ = [
    'compose', 'pipe'
]


def identity(x):
    """ Identity function. Return x

    >>> identity(3)
    3
    """
    return x


def instanceproperty(fget=None, fset=None, fdel=None, doc=None, classval=None):
    """ Like @property, but returns ``classval`` when used as a class attribute

    >>> class MyClass(object):
    ...     '''The class docstring'''
    ...     @instanceproperty(classval=__doc__)
    ...     def __doc__(self):
    ...         return 'An object docstring'
    ...     @instanceproperty
    ...     def val(self):
    ...         return 42
    ...
    >>> MyClass.__doc__
    'The class docstring'
    >>> MyClass.val is None
    True
    >>> obj = MyClass()
    >>> obj.__doc__
    'An object docstring'
    >>> obj.val
    42
    """
    if fget is None:
        return partial(instanceproperty, fset=fset, fdel=fdel, doc=doc,
                       classval=classval)
    return InstanceProperty(fget=fget, fset=fset, fdel=fdel, doc=doc,
                            classval=classval)


class InstanceProperty(property):
    """ Like @property, but returns ``classval`` when used as a class attribute

    Should not be used directly.  Use ``instanceproperty`` instead.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 classval=None):
        self.classval = classval
        property.__init__(self, fget=fget, fset=fset, fdel=fdel, doc=doc)

    def __get__(self, obj, type=None):
        if obj is None:
            return self.classval
        return property.__get__(self, obj, type)

    def __reduce__(self):
        state = (self.fget, self.fset, self.fdel, self.__doc__, self.classval)
        return InstanceProperty, state


class Compose(object):
    """ A composition of functions

    See Also:
        compose
    """
    __slots__ = 'first', 'funcs'

    def __init__(self, funcs):
        funcs = tuple(reversed(funcs))
        self.first = funcs[0]
        self.funcs = funcs[1:]

    def __call__(self, *args, **kwargs):
        ret = self.first(*args, **kwargs)
        for f in self.funcs:
            ret = f(ret)
        return ret

    def __getstate__(self):
        return self.first, self.funcs

    def __setstate__(self, state):
        self.first, self.funcs = state

    @instanceproperty(classval=__doc__)
    def __doc__(self):
        def composed_doc(*fs):
            """Generate a docstring for the composition of fs.
            """
            if not fs:
                # Argument name for the docstring.
                return '*args, **kwargs'

            return '{f}({g})'.format(f=fs[0].__name__, g=composed_doc(*fs[1:]))

        try:
            return (
                'lambda *args, **kwargs: ' +
                composed_doc(*reversed((self.first,) + self.funcs))
            )
        except AttributeError:
            # One of our callables does not have a `__name__`, whatever.
            return 'A composition of functions'

    @property
    def __name__(self):
        try:
            return '_of_'.join(
                (f.__name__ for f in reversed((self.first,) + self.funcs))
            )
        except AttributeError:
            return type(self).__name__

    def __repr__(self):
        return '{.__class__.__name__}{!r}'.format(
            self, tuple(reversed((self.first,) + self.funcs)))

    def __eq__(self, other):
        if isinstance(other, Compose):
            return other.first == self.first and other.funcs == self.funcs
        return NotImplemented

    def __ne__(self, other):
        equality = self.__eq__(other)
        return NotImplemented if equality is NotImplemented else not equality

    def __hash__(self):
        return hash(self.first) ^ hash(self.funcs)

    # Mimic the descriptor behavior of python functions.
    # i.e. let Compose be called as a method when bound to a class.
    # adapted from
    # docs.python.org/3/howto/descriptor.html#functions-and-methods
    def __get__(self, obj, objtype=None):
        return self if obj is None else MethodType(self, obj)

    # introspection with Signature is only possible from py3.3+
    @instanceproperty
    def __signature__(self):
        base = inspect.signature(self.first)
        last = inspect.signature(self.funcs[-1])
        return base.replace(return_annotation=last.return_annotation)

    __wrapped__ = instanceproperty(attrgetter('first'))


def compose(*funcs):
    """ Compose functions to operate in series.

    Returns a function that applies other functions in sequence.

    Functions are applied from right to left so that
    ``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

    If no arguments are provided, the identity function (f(x) = x) is returned.

    >>> inc = lambda i: i + 1
    >>> compose(str, inc)(3)
    '4'
    """
    if not funcs:
        return identity
    if len(funcs) == 1:
        return funcs[0]
    else:
        return Compose(funcs)


def pipe(*funcs):
    """ Pipe a value through a sequence of functions

    I.e. ``pipe(f, g, h)(data)`` is equivalent to ``h(g(f(data)))``

    We think of the value as progressing through a pipe of several
    transformations, much like pipes in UNIX


    >>> double = lambda i: 2 * i
    >>> pipe(double, str)(3)
    '6'
    """
    return compose(*reversed(funcs))
