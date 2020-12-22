# -*- coding: utf-8 -*-

from importlib import import_module

import numpy

from . import linalg, _numba_cpu
from . import random

# https://numpy.org/doc/stable/reference/routines.math.html
_math_funcs = [
    # Basics
    # --------
    'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',  # 'angle',

    # Arithmetic operations
    # ----------------------
    'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
    'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
    'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs',

    # Exponents and logarithms
    # -------------------------
    'exp', 'exp2', 'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',

    # Rational routines
    # -------------------------
    'lcm', 'gcd',

    # trigonometric functions
    # --------------------------
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'cos', 'cosh', 'sin',
    'sinc', 'sinh', 'tan', 'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians',  # 'unwrap'

    # Rounding
    # --------
    'around', 'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix',

    # Sums, products, differences, Reductions
    # --------------------------------------------
    'prod', 'sum', 'diff', 'median', 'nancumprod', 'nancumsum', 'nanprod', 'nansum',
    'cumprod', 'cumsum', 'ediff1d', 'cross', 'trapz',  # 'gradient',

    # floating_functions
    # -------------------
    'isfinite', 'isinf', 'isnan', 'signbit', 'copysign', 'nextafter',
    'ldexp', 'frexp', 'spacing',

    # Miscellaneous
    # --------------
    'convolve', 'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign',
    'heaviside', 'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip',
]

isreal = numpy.isreal
isscalar = numpy.isscalar
real = numpy.real
imag = numpy.imag
conj = numpy.conj
conjugate = numpy.conjugate
ndim = numpy.ndim

add = numpy.add
reciprocal = numpy.reciprocal
negative = numpy.negative
positive = numpy.positive
multiply = numpy.multiply
divide = numpy.divide
power = numpy.power
subtract = numpy.subtract
true_divide = numpy.true_divide
floor_divide = numpy.floor_divide
float_power = numpy.float_power
fmod = numpy.fmod
mod = numpy.mod
modf = numpy.modf
divmod = numpy.divmod
remainder = numpy.remainder
abs = numpy.absolute

exp = numpy.exp
exp2 = numpy.exp2
expm1 = numpy.expm1
log = numpy.log
log10 = numpy.log10
log1p = numpy.log1p
log2 = numpy.log2
logaddexp = numpy.logaddexp
logaddexp2 = numpy.logaddexp2

lcm = numpy.lcm
gcd = numpy.gcd

arccos = numpy.arccos
arccosh = numpy.arccosh
arcsin = numpy.arcsin
arcsinh = numpy.arcsinh
arctan = numpy.arctan
arctan2 = numpy.arctan2
arctanh = numpy.arctanh
cos = numpy.cos
cosh = numpy.cosh
sin = numpy.sin
sinc = numpy.sinc
sinh = numpy.sinh
tan = numpy.tan
tanh = numpy.tanh
deg2rad = numpy.deg2rad
hypot = numpy.hypot
rad2deg = numpy.rad2deg
degrees = numpy.degrees
radians = numpy.radians

around = numpy.around
round_ = numpy.round_
rint = numpy.rint
floor = numpy.floor
ceil = numpy.ceil
trunc = numpy.trunc
fix = numpy.fix

prod = numpy.prod
sum = numpy.sum
diff = numpy.diff
median = numpy.median
nancumprod = numpy.nancumprod
nancumsum = numpy.nancumsum
nanprod = numpy.nanprod
nansum = numpy.nansum
cumprod = numpy.cumprod
cumsum = numpy.cumsum
ediff1d = numpy.ediff1d
cross = numpy.cross
trapz = numpy.trapz

isfinite = numpy.isfinite
isinf = numpy.isinf
isnan = numpy.isnan
signbit = numpy.signbit
copysign = numpy.copysign
nextafter = numpy.nextafter
ldexp = numpy.ldexp
frexp = numpy.frexp
spacing = numpy.spacing

convolve = numpy.convolve
sqrt = numpy.sqrt
cbrt = numpy.cbrt
square = numpy.square
absolute = numpy.absolute
fabs = numpy.fabs
sign = numpy.sign
heaviside = numpy.heaviside
maximum = numpy.maximum
minimum = numpy.minimum
fmax = numpy.fmax
fmin = numpy.fmin
interp = numpy.interp
clip = numpy.clip

# https://numpy.org/doc/stable/reference/routines.bitwise.html
_binary_funcs = [
    # Elementwise bit operations
    # ----------------------------
    'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
    'invert', 'left_shift', 'right_shift',
]

bitwise_and = numpy.bitwise_and
bitwise_not = numpy.bitwise_not
bitwise_or = numpy.bitwise_or
bitwise_xor = numpy.bitwise_xor
invert = numpy.invert
left_shift = numpy.left_shift
right_shift = numpy.right_shift

# https://numpy.org/doc/stable/reference/routines.logic.html
_logic_funcs = [
    # Comparison
    # --------------
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'array_equal', 'isclose', 'allclose',

    # Logical operations
    # ---------------------
    'logical_and', 'logical_not', 'logical_or', 'logical_xor',

    # Truth value testing
    # ----------------------
    'all', 'any',
]

equal = numpy.equal
not_equal = numpy.not_equal
greater = numpy.greater
greater_equal = numpy.greater_equal
less = numpy.less
less_equal = numpy.less_equal
array_equal = numpy.array_equal
isclose = numpy.isclose
allclose = numpy.allclose

logical_and = numpy.logical_and
logical_not = numpy.logical_not
logical_or = numpy.logical_or
logical_xor = numpy.logical_xor

all = numpy.all
any = numpy.any

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html
# https://numpy.org/doc/stable/reference/routines.sort.html
_array_manipulation = [
    # Changing array shape
    # ---------------------
    'shape', 'size', 'reshape', 'ravel',

    # Transpose-like operations
    # --------------------------
    'moveaxis', 'transpose', 'swapaxes',

    # Joining arrays
    # ---------------
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',

    # Splitting arrays
    # -----------------
    'split', 'dsplit', 'hsplit', 'vsplit',

    # Tiling arrays
    # ---------------
    'tile', 'repeat',

    # Adding and removing elements
    # ------------------------------
    'unique', 'delete', 'append',

    # Rearranging elements
    # ---------------------
    'flip', 'fliplr', 'flipud', 'roll',

    # Changing number of dimensions
    # -------------------------------
    'atleast_1d', 'atleast_2d', 'atleast_3d', 'expand_dims', 'squeeze',

    # Sorting
    # -----------
    'sort', 'argsort', 'partition',

    # searching
    # ------------
    'argmax', 'argmin', 'argwhere', 'nonzero', 'flatnonzero', 'where',
    'searchsorted', 'extract',  # 'nanargmax', 'nanargmin',

    # counting
    # ------------
    'count_nonzero',

    # array intrinsic methods
    # -------------------------
    'max', 'min',

    # padding
    # ---------
    # 'pad'
]

shape = numpy.shape
size = numpy.size
reshape = numpy.reshape
ravel = numpy.ravel
moveaxis = numpy.moveaxis
transpose = numpy.transpose
swapaxes = numpy.swapaxes
concatenate = numpy.concatenate
stack = numpy.stack
vstack = numpy.vstack
hstack = numpy.hstack
dstack = numpy.dstack
column_stack = numpy.column_stack
split = numpy.split
dsplit = numpy.dsplit
hsplit = numpy.hsplit
vsplit = numpy.vsplit
tile = numpy.tile
repeat = numpy.repeat
unique = numpy.unique
delete = numpy.delete
append = numpy.append
flip = numpy.flip
fliplr = numpy.fliplr
flipud = numpy.flipud
roll = numpy.roll
atleast_1d = numpy.atleast_1d
atleast_2d = numpy.atleast_2d
atleast_3d = numpy.atleast_3d
expand_dims = numpy.expand_dims
squeeze = numpy.squeeze
sort = numpy.sort
argsort = numpy.argsort
partition = numpy.partition
argmax = numpy.argmax
argmin = numpy.argmin
argwhere = numpy.argwhere
nonzero = numpy.nonzero
flatnonzero = numpy.flatnonzero
where = numpy.where
searchsorted = numpy.searchsorted
extract = numpy.extract
count_nonzero = numpy.count_nonzero
max = numpy.max
min = numpy.min

# https://numpy.org/doc/stable/reference/routines.array-creation.html
_array_creation = [
    'ndarray',

    # Ones and zeros
    # ---------------
    'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
    'full', 'full_like', 'eye', 'identity',

    # From existing data
    # --------------------
    'array', 'asarray',

    # Numerical ranges
    # ------------------
    'arange', 'linspace', 'logspace', 'meshgrid', 'copy',

    # Building matrices
    # -------------------
    'diag', 'tri', 'tril', 'triu', 'vander',
]

ndarray = numpy.ndarray
empty = numpy.empty
empty_like = numpy.empty_like
ones = numpy.ones
ones_like = numpy.ones_like
zeros = numpy.zeros
zeros_like = numpy.zeros_like
full = numpy.full
full_like = numpy.full_like
eye = numpy.eye
identity = numpy.identity

array = numpy.array
asarray = numpy.asarray

arange = numpy.arange
linspace = numpy.linspace
logspace = numpy.logspace
meshgrid = numpy.meshgrid
copy = numpy.copy

diag = numpy.diag
tri = numpy.tri
tril = numpy.tril
triu = numpy.triu
vander = numpy.vander

# https://numpy.org/doc/stable/reference/routines.indexing.html
_indexing_funcs = [
    # Generating index arrays
    # -------------------------
    'nonzero', 'where', 'tril_indices', 'tril_indices_from',
    'triu_indices', 'triu_indices_from',

    # Indexing-like operations
    # --------------------------
    'take', 'diag', 'select',

    # Iterating over arrays
    # -----------------------
    'nditer', 'ndenumerate', 'ndindex',
]

# nonzero = numpy.nonzero
# where = numpy.where
tril_indices = numpy.tril_indices
tril_indices_from = numpy.tril_indices_from
triu_indices = numpy.triu_indices
triu_indices_from = numpy.triu_indices_from

take = numpy.take
# diag = numpy.diag
select = numpy.select

nditer = numpy.nditer
ndenumerate = numpy.ndenumerate
ndindex = numpy.ndindex

# https://numpy.org/doc/stable/reference/routines.statistics.html
_statistic_funcs = [
    # Order statistics
    # ------------------
    'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile', 'quantile', 'nanquantile',
    # 'amin', 'amax',

    # Averages and variances
    # ----------------------
    'median', 'average', 'mean', 'std', 'var', 'nanmedian', 'nanmean', 'nanstd', 'nanvar',

    # Correlating
    # ----------------------
    'corrcoef', 'correlate', 'cov',

    # Histograms
    # ----------------------
    'histogram', 'bincount', 'digitize',  # 'histogram2d', 'histogramdd', 'histogram_bin_edges'
]

nanmin = numpy.nanmin
nanmax = numpy.nanmax
ptp = numpy.ptp
percentile = numpy.percentile
nanpercentile = numpy.nanpercentile
quantile = numpy.quantile
nanquantile = numpy.nanquantile

# median = numpy.median
average = numpy.average
mean = numpy.mean
std = numpy.std
var = numpy.var
nanmedian = numpy.nanmedian
nanmean = numpy.nanmean
nanstd = numpy.nanstd
nanvar = numpy.nanvar

corrcoef = numpy.corrcoef
correlate = numpy.correlate
cov = numpy.cov

histogram = numpy.histogram
bincount = numpy.bincount
digitize = numpy.digitize

# https://numpy.org/doc/stable/reference/routines.window.html
_window_funcs = ['bartlett', 'blackman', 'hamming', 'hanning', 'kaiser']

bartlett = numpy.bartlett
blackman = numpy.blackman
hamming = numpy.hamming
hanning = numpy.hanning
kaiser = numpy.kaiser

# https://numpy.org/doc/stable/reference/constants.html
_constants = ['e', 'pi', 'inf']

e = numpy.e
pi = numpy.pi
inf = numpy.inf

# https://numpy.org/doc/stable/reference/routines.linalg.html
_linear_algebra = [
    'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',
    # 'tensordot', 'einsum', 'einsum_path',
]

dot = numpy.dot
vdot = numpy.vdot
inner = numpy.inner
outer = numpy.outer
kron = numpy.kron
matmul = numpy.matmul
trace = numpy.trace

# https://numpy.org/doc/stable/reference/routines.dtype.html
_data_types = [
    # functions
    # ---------
    'dtype', 'finfo', 'iinfo', 'MachAr',

    # objects
    # --------
    'bool_',
    'uint8', 'uint16', 'uint32', 'uint64',
    'int_', 'int8', 'int16', 'int32', 'int64',
    'float_', 'float16', 'float32', 'float64',
    'complex_', 'complex64', 'complex128',
]

dtype = numpy.dtype
finfo = numpy.finfo
iinfo = numpy.iinfo
MachAr = numpy.MachAr

bool_ = numpy.bool_
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64
int_ = numpy.int_
int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
float_ = numpy.float_
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
complex_ = numpy.complex_
complex64 = numpy.complex64
complex128 = numpy.complex128

_all = _math_funcs + _binary_funcs + _logic_funcs + _array_manipulation + \
       _array_creation + _indexing_funcs + _statistic_funcs + _window_funcs + \
       _constants + _linear_algebra + _data_types


def _reload(backend):
    random._reload(backend)
    linalg._reload(backend)

    global_vars = globals()

    if backend == 'numpy':
        for __ops in _all:
            global_vars[__ops] = getattr(numpy, __ops)

    elif backend == 'numba':

        for __ops in _all:
            if hasattr(_numba_cpu, __ops):
                global_vars[__ops] = getattr(_numba_cpu, __ops)
            else:
                global_vars[__ops] = getattr(numpy, __ops)

    elif backend == 'tensorflow':
        tf_numpy = import_module('tensorflow.experimental.numpy')
        from ._backends import _tensorflow

        for __ops in _all:
            if hasattr(tf_numpy, __ops):
                global_vars[__ops] = getattr(tf_numpy, __ops)
            else:
                global_vars[__ops] = getattr(_tensorflow, __ops)

    else:
        raise ValueError(f'Unknown backend device: {backend}')


def _set_default_int(itype):
    """Set default int type.

    Parameters
    ----------
    itype : str, numpy.generic
        Int type.
    """
    global int_
    if isinstance(itype, str):
        int_ = func_by_name(itype)
    else:
        int_ = itype


def _set_default_float(ftype):
    """Set default float type.

    Parameters
    ----------
    ftype : str, numpy.generic
        float type.
    """
    global float_

    if isinstance(ftype, str):
        float_ = func_by_name(ftype)
    else:
        float_ = ftype


def func_by_name(name):
    """Get numpy function by its name.

    Parameters
    ----------
    name : str
        Function name.

    Returns
    -------
    func : callable
        Numpy function.
    """
    if name in globals():
        return globals()[name]
    elif hasattr(random, name):
        return getattr(random, name)
    elif hasattr(linalg, name):
        return getattr(linalg, name)
    else:
        return None
