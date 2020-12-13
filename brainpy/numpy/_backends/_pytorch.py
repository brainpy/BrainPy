# -*- coding: utf-8 -*-

from importlib import import_module
from typing import Iterable
from typing import Tuple, Any, Optional
from typing import Union

import numpy as np

torch = import_module('torch')

Axes = Tuple[int, ...]
AxisAxes = Union[int, Axes]
Shape = Tuple[int, ...]
ShapeOrScalar = Union[Shape, int]
ArrayOrScalar = Union["ndarray", int, float]


class ndarray:
    __slots__ = ('data',)

    def byteswap(self, inplace: bool = False):
        pass

    def choose(self, choices, mode: str = 'raise'):
        pass

    def compress(self, condition, axis=None):
        pass

    def conj(self):
        pass

    def conjugate(self):
        pass

    def copy(self, order='C'):
        pass

    def cumprod(self, axis=None, dtype=None):
        pass

    def diagonal(self, offset=0, axis1=0, axis2=1):
        pass

    def dot(self, b):
        pass

    def dump(self, file):
        pass

    def dumps(self):
        pass

    def fill(self, value):
        pass

    def getfield(self, dtype, offset=0):
        pass

    def itemset(self, *args):
        pass

    def newbyteorder(self, new_order='S'):
        pass

    def nonzero(self):
        pass

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        pass

    def ptp(self, axis=None, keepdims=False):
        pass

    def put(self, indices, values, mode='raise'):
        pass

    def ravel(self, order=None):
        pass

    def repeat(self, repeats, axis=None):
        pass

    def resize(self, new_shape, refcheck=True):
        pass

    def round(self, decimals=0):
        pass

    def searchsorted(self, v, side='left', sorter=None):
        pass

    def setfield(self, val, dtype, offset=0):
        pass

    def setflags(self, write=None, align=None, uic=None):
        pass

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False):
        pass

    def swapaxes(self, axis1, axis2):
        pass

    def take(self, indices, axis=None, mode='raise'):
        pass

    def tobytes(self, order='C'):
        pass

    def tofile(self, fid, sep="", format="%s"):
        pass

    def tolist(self):
        pass

    def tostring(self, order='C'):
        pass

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        pass

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False):
        pass

    def view(self, dtype=None, type=None):
        pass

    def __init__(self, data):
        self.data = data
        super(ndarray, self).__init__()

    #####################
    # basic operations
    #####################

    def __repr__(self) -> str:
        lines = repr(self.data).split("\n")
        prefix = self.__class__.__name__ + "("
        lines[0] = prefix + lines[0]
        prefix = " " * len(prefix)
        for i in range(1, len(lines)):
            lines[i] = prefix + lines[i]
        lines[-1] = lines[-1] + ")"
        return "\n".join(lines)

    def __format__(self, format_spec: str) -> str:
        return format(self.data, format_spec)

    def __lt__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__lt__(other))

    def __le__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__le__(other))

    def __eq__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__eq__(other))

    def __ne__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__ne__(other))

    def __gt__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__gt__(other))

    def __ge__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__ge__(other))

    def __getitem__(self, index: Any):
        if isinstance(index, tuple):
            index = tuple(x.data if isinstance(x, ndarray) else x for x in index)
        elif isinstance(index, ndarray):
            index = index.data
        return ndarray(self.data[index])

    def __bool__(self):
        return ndarray(self.data.astype(torch.bool))

    def __len__(self) -> int:
        return len(self.data)

    def __abs__(self):
        return ndarray(torch.abs(self.data))

    def __neg__(self):
        return ndarray(-self.data)

    def __add__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__add__(other))

    def __radd__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__radd__(other))

    def __sub__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__sub__(other))

    def __rsub__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__rsub__(other))

    def __mul__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__mul__(other))

    def __rmul__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__rmul__(other))

    def __truediv__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__truediv__(other))

    def __rtruediv__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__rtruediv__(other))

    def __floordiv__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__floordiv__(other))

    def __rfloordiv__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__rfloordiv__(other))

    def __mod__(self, other: ArrayOrScalar):
        other = other.data if isinstance(other, ndarray) else other
        return ndarray(self.data.__mod__(other))

    def __pow__(self, exponent: ArrayOrScalar):
        exponent = exponent.data if isinstance(exponent, ndarray) else exponent
        return ndarray(self.data.__pow__(exponent))

    # ----------
    # property
    # ----------

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.nelement()

    @property
    def strides(self) -> int:
        return self.data.stride()

    # ----------
    # transform
    # ----------

    def numpy(self) -> Any:
        return self.data.copy()

    def from_numpy(self, a: Any):
        return ndarray(torch.as_tensor(a, device=self.data.device))

    # ---------
    # methods
    # ---------

    def all(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.all())
        else:
            if axis is None:
                axis = tuple(range(self.ndim))
            elif not isinstance(axis, Iterable):
                axis = (axis,)
            x = self.data
            for i in sorted(axis, reverse=True):
                x = x.all(i, keepdim=keepdims)
            return ndarray(x)

    def any(self,
            axis: Optional[AxisAxes] = None,
            keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.any())
        else:
            if axis is None:
                axis = tuple(range(self.ndim))
            elif not isinstance(axis, Iterable):
                axis = (axis,)
            x = self.data
            for i in sorted(axis, reverse=True):
                x = x.any(i, keepdim=keepdims)
            return ndarray(x)

    def argmin(self, axis: Optional[int] = None):
        return ndarray(self.data.argmin(dim=axis))

    def argmax(self, axis: Optional[int] = None):
        return ndarray(self.data.argmax(dim=axis))

    def argsort(self, axis: int = -1):
        return ndarray(self.data.argsort(dim=axis))

    def astype(self, dtype: Any):
        return ndarray(self.data.astype(dtype))

    def cumsum(self, axis: Optional[int] = None):
        if axis is None:
            return ndarray(self.data.reshape(-1).cumsum(dim=0))
        else:
            return ndarray(self.data.cumsum(dim=axis))

    def clip(self, min_: float, max_: float):
        return ndarray(self.data.clamp(min_, max_))

    def flatten(self):
        return self.reshape((-1,))

    def mean(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.mean())
        else:
            axis = tuple(range(self.ndim)) if axis is None else axis
            return ndarray(self.data.mean(axis=axis, keepdims=keepdims))

    def min(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.min())
        else:
            if axis is None:
                axis = tuple(range(self.ndim))
            elif not isinstance(axis, Iterable):
                axis = (axis,)
            x = self.data
            for i in sorted(axis, reverse=True):
                x, _ = x.min(i, keepdim=keepdims)
            return ndarray(x)

    def max(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.max())
        else:
            if axis is None:
                axis = tuple(range(self.ndim))
            elif not isinstance(axis, Iterable):
                axis = (axis,)
            x = self.data
            for i in sorted(axis, reverse=True):
                x, _ = x.max(i, keepdim=keepdims)
            return ndarray(x)

    def prod(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.prod())
        else:
            if axis is None:
                axis = tuple(range(self.ndim))
            elif not isinstance(axis, Iterable):
                axis = (axis,)
            x = self.data
            for i in sorted(axis, reverse=True):
                x = x.prod(i, keepdim=keepdims)
            return ndarray(x)

    def reshape(self, shape: Shape):
        return ndarray(self.data.reshape(shape))

    def squeeze(self, axis: Optional[AxisAxes] = None):
        if axis is None:
            return ndarray(self.data.squeeze())
        else:
            axis = (axis,) if not isinstance(axis, Iterable) else axis
            x = self.data
            for i in sorted(axis, reverse=True):
                if x.shape[i] != 1:
                    raise ValueError("Cannot select an axis to squeeze out which has size "
                                     "not equal to one.")
                x = x.squeeze(dim=i)
            return ndarray(x)

    def sort(self, axis: int = -1):
        return ndarray(self.data.sort(dim=axis).values)

    def sum(self, axis: Optional[AxisAxes] = None, keepdims: bool = False):
        if axis is None and not keepdims:
            return ndarray(self.data.sum())
        else:
            axis = tuple(range(self.ndim)) if axis is None else axis
            return ndarray(self.data.sum(axis=axis, keepdims=keepdims))

    def transpose(self, axes: Optional[Axes] = None):
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        return ndarray(self.data.permute(*axes))


# --------------------
# math operations
# --------------------


def exp(arr):
    return ndarray(torch.exp(arr.data))


def log(arr):
    return ndarray(torch.log(arr.data))


def log2(arr):
    return ndarray(torch.log2(arr.data))


def log10(arr):
    return ndarray(torch.log10(arr.data))


def log1p(arr):
    return ndarray(torch.log1p(arr.data))


def matmul(arr1, arr2):
    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError(f"matmul requires both arrays to be 2D, got {arr1.ndim}D and {arr2.ndim}D")
    return ndarray(torch.matmul(arr1.data, arr2.data))


def power(arr, exponent: ArrayOrScalar):
    exponent = exponent.data if isinstance(exponent, ndarray) else exponent
    return ndarray(torch.pow(arr.data, exponent))


def sign(arr):
    return ndarray(torch.sign(arr.data))


def sqrt(arr):
    return ndarray(torch.sqrt(arr.data))


def square(arr):
    return ndarray(arr.data ** 2)


# -------------------------
# trigonometric functions
# -------------------------


def arccos(arr):
    return ndarray(torch.acos(arr.data))


def arccosh(arr):
    return ndarray(torch.acosh(arr.data))


def arcsin(arr):
    return ndarray(torch.asin(arr.data))


def arcsinh(arr):
    return ndarray(torch.asinh(arr.data))


def arctan(arr):
    return ndarray(torch.atan(arr.data))


def arctan2(arr):
    return ndarray(torch.atan2(arr.data))


def arctanh(arr):
    return ndarray(torch.atanh(arr.data))


def cos(arr):
    return ndarray(torch.cos(arr.data))


def cosh(arr):
    return ndarray(torch.cosh(arr.data))


def deg2rad(arr):
    return ndarray(torch.deg2rad(arr.data))


def radians(arr):
    return ndarray(torch.deg2rad(arr.data))


def rad2deg(arr):
    return ndarray(torch.rad2deg(arr.data))


def degrees(arr):
    return ndarray(torch.rad2deg(arr.data))


def hypot(arr):
    return ndarray(torch.hypot(arr.data))


def sin(arr):
    return ndarray(torch.sin(arr.data))


def sinc(arr):
    return ndarray(torch.sin(arr.data) / arr.data)


def sinh(arr):
    return ndarray(torch.sinh(arr.data))


def tan(arr):
    return ndarray(torch.tan(arr.data))


def tanh(arr):
    return ndarray(torch.tanh(arr.data))


# -------------------------
# bitwise functions
# -------------------------


def bitwise_and(arr):
    return ndarray(torch.bitwise_and(arr.data))


def bitwise_not(arr):
    return ndarray(torch.bitwise_not(arr.data))


def invert(arr):
    return ndarray(torch.bitwise_not(arr.data))


def bitwise_or(arr):
    return ndarray(torch.bitwise_or(arr.data))


def bitwise_xor(arr):
    return ndarray(torch.bitwise_xor(arr.data))


# -------------------------
# comparison functions
# -------------------------


def equal(arr1, arr2):
    return ndarray(torch.eq(arr1.data, arr2.data))


def not_equal(arr1, arr2):
    return ndarray(torch.ne(arr1.data, arr2.data))


def logical_and(arr1, arr2):
    return ndarray(torch.logical_and(arr1.data, arr2.data))


def logical_or(arr1, arr2):
    return ndarray(torch.logical_or(arr1.data, arr2.data))


def logical_not(arr):
    return ndarray(torch.logical_not(arr.data))


def logical_xor(arr1, arr2):
    return ndarray(torch.logical_xor(arr1.data, arr2.data))


def greater(arr1, arr2):
    return ndarray(torch.gt(arr1.data, arr2.data))


def greater_equal(arr1, arr2):
    return ndarray(torch.ge(arr1.data, arr2.data))


def less(arr1, arr2):
    return ndarray(torch.lt(arr1.data, arr2.data))


def less_equal(arr1, arr2):
    return ndarray(torch.le(arr1.data, arr2.data))


def minimum(arr1, arr2: ArrayOrScalar):
    if isinstance(arr2, ndarray):
        arr2 = arr2.data
    elif isinstance(arr2, int) or isinstance(arr2, float):
        arr2 = torch.full_like(arr1.data, arr2)
    else:
        raise TypeError("Expected 'x' to be an ndarray, int or float.")
    return ndarray(torch.min(arr1.data, arr2))


def maximum(arr1, arr2: ArrayOrScalar):
    if isinstance(arr2, ndarray):
        arr2 = arr2.data
    elif isinstance(arr2, int) or isinstance(arr2, float):
        arr2 = torch.full_like(arr1.data, arr2)
    else:
        raise TypeError("Expected 'x' to be an ndarray, int or float.")
    return ndarray(torch.max(arr1.data, arr2))


# -------------------------
# floating functions
# -------------------------

def ceil(arr):
    return ndarray(torch.ceil(arr.data))


def floor(arr):
    return ndarray(torch.floor(arr.data))


def fmod(arr):
    return ndarray(torch.fmod(arr.data))


def isfinite(arr):
    return ndarray(torch.isfinite(arr.data))


def isinf(arr):
    return ndarray(torch.isinf(arr.data))


def isnan(arr):
    return ndarray(torch.isnan(arr.data))


def trunc(arr):
    return ndarray(torch.trunc(arr.data))


# -------------------------
# array manipulation
# -------------------------


def shape(arr):
    return arr.data.shape


def reshape(arr, shape: Shape):
    return ndarray(arr.data.reshape(shape))


def ravel(arr):
    # https://github.com/pytorch/pytorch/issues/1582
    return ndarray(arr.data.view(-1))


def flatten(arr, start: int = 0, end: int = -1):
    return ndarray(arr.flatten(start, end))


def moveaxis(arr, source: int, destination: int):
    # https://github.com/pytorch/pytorch/issues/36048
    return ndarray(torch.movedim(arr.data, source, destination))


def transpose(arr, axes: Optional[Axes] = None):
    return arr.transpose(axes)


def concatenate(arrays: Iterable[ndarray], axis: int = 0):
    arrays = tuple(arr.data if isinstance(arr, ndarray) else arr for arr in arrays)
    return ndarray(torch.cat(arrays, dim=axis))


def stack(arrays: Iterable[ndarray], axis: int = 0):
    arrays = tuple(arr.data if isinstance(arr, ndarray) else arr for arr in arrays)
    return ndarray(torch.stack(arrays, dim=axis))


def split(arr, indices_or_sections: ArrayOrScalar, axis: int = 0) -> Tuple:
    indices_or_sections = indices_or_sections.data if isinstance(indices_or_sections, ndarray) \
        else indices_or_sections
    outputs = torch.split(arr.data, split_size_or_sections=indices_or_sections, dim=axis)
    return tuple(ndarray(out) for out in outputs)


def tile(arr, multiples: Axes):
    return ndarray(arr.data.repeat(multiples))


def repeat(arr, repeats: ArrayOrScalar, axis: int = None):
    repeats = repeats.data if isinstance(repeats, ndarray) else repeats
    return ndarray(torch.repeat_interleave(arr.data, repeats=repeats, dim=axis))


def flip(arr, axis: Optional[AxisAxes] = None):
    if axis is None:
        axis = tuple(range(arr.ndim))
    if not isinstance(axis, Iterable):
        axis = (axis,)
    return ndarray(arr.data.flip(dims=axis))


# -------------------------
# array creation
# -------------------------


def copy(arr: ArrayOrScalar):
    return ndarray(arr.data.clone().detach())


def empty(shape: ShapeOrScalar, dtype: Any = None, **kwargs: Any):
    shape = shape if isinstance(shape, Iterable) else (shape,)
    dtype = torch.float64 if dtype is None else dtype
    return ndarray(torch.empty(*shape, dtype=dtype, **kwargs))


def empty_like(arr):
    return ndarray(torch.empty_like(arr.data))


def ones(shape: ShapeOrScalar, dtype: Any = None, **kwargs: Any):
    shape = shape if isinstance(shape, Iterable) else (shape,)
    dtype = torch.float64 if dtype is None else dtype
    return ndarray(torch.ones(shape, dtype=dtype, **kwargs))


def ones_like(arr):
    return ndarray(torch.ones_like(arr.data))


def zeros(shape: ShapeOrScalar, dtype: Any = None, **kwargs: Any):
    shape = shape if isinstance(shape, Iterable) else (shape,)
    dtype = torch.float64 if dtype is None else dtype
    return ndarray(torch.zeros(shape, dtype=dtype, **kwargs))


def zeros_like(arr):
    return ndarray(torch.zeros_like(arr.data))


def full(shape: ShapeOrScalar, value: float, dtype: Any = None, **kwargs: Any):
    shape = shape if isinstance(shape, Iterable) else (shape,)
    return ndarray(torch.full(shape, value, dtype=dtype, **kwargs))


def full_like(arr, fill_value: float):
    return ndarray(torch.full_like(arr.data, fill_value))


def arange(start: int, stop: Optional[int] = None, step: Optional[int] = None,
           dtype: Any = None, **kwargs: Any):
    step = 1 if step is None else step
    stop, start = (start, 0) if stop is None else (stop, start)
    return ndarray(torch.arange(start=start, end=stop, step=step, dtype=dtype, **kwargs))


def linspace(start: int, stop: int, num: int = 50, dtype: Any = None, **kwargs: Any):
    return ndarray(torch.linspace(start, stop, num=num, dtype=dtype, **kwargs))


def logspace(start: int, stop: int, num: int = 50, base: float = 10.0,
             dtype: Any = None, **kwargs: Any):
    return ndarray(torch.logspace(start, stop, num=num, base=base, dtype=dtype, **kwargs))


def eye(N: int, M: int = None, dtype: Any = None, **kwargs: Any):
    return ndarray(torch.eye(n=N, m=M, dtype=dtype, **kwargs))


def identity(N: int, dtype: Any = None, **kwargs: Any):
    return ndarray(torch.eye(n=N, m=None, dtype=dtype, **kwargs))


def meshgrid(arr, *arrays, indexing: str = "xy"):
    arrays = tuple(arr.data if isinstance(arr, ndarray) else arr for arr in arrays)
    if indexing == "ij" or len(arrays) == 0:
        outputs = torch.meshgrid(arr.data, *arrays)
    elif indexing == "xy":
        outputs = torch.meshgrid(arrays[0], arr.data, *arrays[1:])
        if len(outputs) >= 2:
            outputs[0], outputs[1] = outputs[1], outputs[0]
    else:
        raise ValueError(f"Valid values for indexing are 'xy' and 'ij', got {indexing}")
    return tuple(ndarray(out) for out in outputs)


def array(data: Any, dtype: Any = None, device: Any = None):
    data = np.array(data)
    return ndarray(torch.as_tensor(data, dtype=dtype, device=device))


def asarray(data: Any, dtype: Any = None, device: Any = None):
    data = np.asarray(data)
    return ndarray(torch.as_tensor(data, dtype=dtype, device=device))


# -------------------------
# random distribution
# -------------------------


def seed(seed=0):
    torch.manual_seed(seed)


def uniform(low: float = 0.0, high: float = 1.0, size: ShapeOrScalar = 1, **kwargs: Any):
    size = (size,) if isinstance(size, Iterable) else size
    return ndarray(torch.rand(*size, **kwargs) * (high - low) + low)


def rand(*size, **kwargs: Any):
    return ndarray(torch.rand(*size, **kwargs))


def randint(low, high=None, size=None):
    if high is None:
        low, high = 0, low
    if size is None:
        size = (1,)
    return ndarray(torch.randint(low=low, high=high, size=size))


def randn(*size):
    return ndarray(torch.randn(*size))


def random(size=None):
    size = (1,) if size is None else size
    return uniform(low=0., high=1., size=size)


# -------------------------
# others
# -------------------------


def expand_dims(arr, axis: int):
    return ndarray(arr.data.unsqueeze(dim=axis))


def pad(arr, paddings: Tuple[Tuple[int, int], ...],
        mode: str = "constant", value: float = 0, ):
    if len(paddings) != arr.ndim:
        raise ValueError("pad requires a tuple for each dimension.")
    for p in paddings:
        if len(p) != 2:
            raise ValueError("pad requires a tuple for each dimension.")
    if not (mode == "constant" or mode == "reflect"):
        raise ValueError("pad requires mode 'constant' or 'reflect'.")
    if mode == "reflect":
        if arr.ndim != 3 and arr.ndim != 4:
            raise NotImplementedError
        k = arr.ndim - 2
        if paddings[:k] != ((0, 0),) * k:
            raise NotImplementedError
        paddings = paddings[k:]
    paddings_ = list(x for p in reversed(paddings) for x in p)
    return ndarray(torch.nn.functional.pad(arr.data, paddings_, mode=mode, value=value))


def where(arr, x: ArrayOrScalar, y: ArrayOrScalar):
    if isinstance(x, ndarray):
        x = x.data
    elif isinstance(x, int) or isinstance(x, float):
        x = torch.full_like(arr.data, x, dtype=arr.data.dtype)
    else:
        raise TypeError("expected x to be an Array, int or float.")

    if isinstance(y, ndarray):
        y = y.data
    elif isinstance(y, int) or isinstance(y, float):
        y = torch.full_like(arr.data, y, dtype=arr.data.dtype)
    else:
        raise TypeError("expected y to be an Array, int or float.")
    return ndarray(torch.where(arr.data, x, y))


def take_along_axis(arr, index, axis: int):
    if axis % arr.ndim != arr.ndim - 1:
        raise NotImplementedError("take_along_axis is currently only supported for the last axis")
    return ndarray(torch.gather(arr.data, axis, index.data))


def all(t, axis: Optional[AxisAxes] = None, keepdims: bool = False):
    return t.all(axis=axis, keepdims=keepdims)


def any(t, axis: Optional[AxisAxes] = None, keepdims: bool = False):
    return t.any(axis=axis, keepdims=keepdims)


def argmin(t, axis: Optional[int] = None):
    return t.argmin(axis=axis)


def argmax(t, axis: Optional[int] = None):
    return t.argmax(axis=axis)


def argsort(t, axis: int = -1):
    return t.argsort(axis=axis)


def cumsum(t, axis: Optional[int] = None):
    return t.cumsum(axis=axis)


def clip(t, min_: float, max_: float):
    return t.clip(min_, max_)


def from_numpy(t, a: Any):
    return t.from_numpy(a)


def prod(t, axis: Optional[AxisAxes] = None, keepdims: bool = False):
    return t.prod(axis=axis, keepdims=keepdims)


def pow(t, exponent: ArrayOrScalar):
    return t.pow(exponent)


def squeeze(t, axis: Optional[AxisAxes] = None):
    return t.squeeze(axis=axis)


def sum(t, axis: Optional[AxisAxes] = None, keepdims: bool = False):
    return t.sum(axis=axis, keepdims=keepdims)


def sort(t, axis: int = -1):
    return t.sort(axis=axis)
