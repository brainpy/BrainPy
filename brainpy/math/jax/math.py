# -*- coding: utf-8 -*-

import jax.numpy as jnp
from brainpy.math.jax.ndarray import _wrap, ndarray


__all__ = [
  # math funcs
  'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',
  'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
  'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
  'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs', 'exp', 'exp2',
  'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',
  'lcm', 'gcd', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctan2', 'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians', 'around',
  'round_', 'rint', 'floor', 'ceil', 'trunc', 'fix', 'prod', 'sum', 'diff',
  'median', 'nancumprod', 'nancumsum', 'nanprod', 'nansum', 'cumprod',
  'cumsum', 'ediff1d', 'cross', 'trapz', 'isfinite', 'isinf', 'isnan',
  'signbit', 'copysign', 'nextafter', 'ldexp', 'frexp', 'convolve',
  'sqrt', 'cbrt', 'square', 'absolute', 'fabs', 'sign', 'heaviside',
  'maximum', 'minimum', 'fmax', 'fmin', 'interp', 'clip',

  # Elementwise bit operations
  'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
  'invert', 'left_shift', 'right_shift',

  # logic funcs
  'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
  'array_equal', 'isclose', 'allclose', 'logical_and', 'logical_not',
  'logical_or', 'logical_xor', 'all', 'any',

  # array manipulation
  'shape', 'size', 'reshape', 'ravel', 'moveaxis', 'transpose', 'swapaxes',
  'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
  'split', 'dsplit', 'hsplit', 'vsplit', 'tile', 'repeat', 'unique',
  'append', 'flip', 'fliplr', 'flipud', 'roll', 'atleast_1d', 'atleast_2d',
  'atleast_3d', 'expand_dims', 'squeeze', 'sort', 'argsort', 'argmax', 'argmin',
  'argwhere', 'nonzero', 'flatnonzero', 'where', 'searchsorted', 'extract',
  'count_nonzero', 'max', 'min',

  # array creation
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like', 'full',
  'full_like', 'eye', 'identity', 'array', 'asarray', 'arange', 'linspace',
  'logspace', 'meshgrid', 'diag', 'tri', 'tril', 'triu', 'vander',

  # indexing funcs
  'nonzero', 'where', 'tril_indices', 'tril_indices_from', 'triu_indices',
  'triu_indices_from', 'take', 'diag', 'select',

  # statistic funcs
  'nanmin', 'nanmax', 'ptp', 'percentile', 'nanpercentile', 'quantile', 'nanquantile',
  'median', 'average', 'mean', 'std', 'var', 'nanmedian', 'nanmean', 'nanstd', 'nanvar',
  'corrcoef', 'correlate', 'cov', 'histogram', 'bincount', 'digitize',

  # window funcs
  'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',

  # constants
  'e', 'pi', 'inf',

  # linear algebra
  'dot', 'vdot', 'inner', 'outer', 'kron', 'matmul', 'trace',

  # data types
  'dtype', 'finfo', 'iinfo',
  'bool_', 'uint8', 'uint16', 'uint32', 'uint64',
  'int_', 'int8', 'int16', 'int32', 'int64',
  'float_', 'float16', 'float32', 'float64',
  'complex_', 'complex64', 'complex128',
]

# math funcs
# ----------

# 1. Basics

isreal = _wrap(jnp.isreal)
isscalar = _wrap(jnp.isscalar)
real = _wrap(jnp.real)
imag = _wrap(jnp.imag)
conj = _wrap(jnp.conj)
conjugate = _wrap(jnp.conjugate)
ndim = _wrap(jnp.ndim)

# 2. Arithmetic operations
add = _wrap(jnp.add)
reciprocal = _wrap(jnp.reciprocal)
negative = _wrap(jnp.negative)
positive = _wrap(jnp.positive)
multiply = _wrap(jnp.multiply)
divide = _wrap(jnp.divide)
power = _wrap(jnp.power)
subtract = _wrap(jnp.subtract)
true_divide = _wrap(jnp.true_divide)
floor_divide = _wrap(jnp.floor_divide)
float_power = _wrap(jnp.float_power)
fmod = _wrap(jnp.fmod)
mod = _wrap(jnp.mod)
modf = _wrap(jnp.modf)
divmod = _wrap(jnp.divmod)
remainder = _wrap(jnp.remainder)
abs = _wrap(jnp.absolute)

# 3. Exponents and logarithms
exp = _wrap(jnp.exp)
exp2 = _wrap(jnp.exp2)
expm1 = _wrap(jnp.expm1)
log = _wrap(jnp.log)
log10 = _wrap(jnp.log10)
log1p = _wrap(jnp.log1p)
log2 = _wrap(jnp.log2)
logaddexp = _wrap(jnp.logaddexp)
logaddexp2 = _wrap(jnp.logaddexp2)

# 4. Rational routines
lcm = _wrap(jnp.lcm)
gcd = _wrap(jnp.gcd)

# 5. trigonometric functions
arccos = _wrap(jnp.arccos)
arccosh = _wrap(jnp.arccosh)
arcsin = _wrap(jnp.arcsin)
arcsinh = _wrap(jnp.arcsinh)
arctan = _wrap(jnp.arctan)
arctan2 = _wrap(jnp.arctan2)
arctanh = _wrap(jnp.arctanh)
cos = _wrap(jnp.cos)
cosh = _wrap(jnp.cosh)
sin = _wrap(jnp.sin)
sinc = _wrap(jnp.sinc)
sinh = _wrap(jnp.sinh)
tan = _wrap(jnp.tan)
tanh = _wrap(jnp.tanh)
deg2rad = _wrap(jnp.deg2rad)
hypot = _wrap(jnp.hypot)
rad2deg = _wrap(jnp.rad2deg)
degrees = _wrap(jnp.degrees)
radians = _wrap(jnp.radians)

# 6. Rounding
around = _wrap(jnp.around)
round_ = _wrap(jnp.round_)
rint = _wrap(jnp.rint)
floor = _wrap(jnp.floor)
ceil = _wrap(jnp.ceil)
trunc = _wrap(jnp.trunc)
fix = _wrap(jnp.fix)

# 7. Sums, products, differences, Reductions
prod = _wrap(jnp.prod)
sum = _wrap(jnp.sum)
diff = _wrap(jnp.diff)
median = _wrap(jnp.median)
nancumprod = _wrap(jnp.nancumprod)
nancumsum = _wrap(jnp.nancumsum)
nanprod = _wrap(jnp.nanprod)
nansum = _wrap(jnp.nansum)
cumprod = _wrap(jnp.cumprod)
cumsum = _wrap(jnp.cumsum)
ediff1d = _wrap(jnp.ediff1d)
cross = _wrap(jnp.cross)
trapz = _wrap(jnp.trapz)

# 8. floating_functions
isfinite = _wrap(jnp.isfinite)
isinf = _wrap(jnp.isinf)
isnan = _wrap(jnp.isnan)
signbit = _wrap(jnp.signbit)
copysign = _wrap(jnp.copysign)
nextafter = _wrap(jnp.nextafter)
ldexp = _wrap(jnp.ldexp)
frexp = _wrap(jnp.frexp)

# 9. Miscellaneous
convolve = _wrap(jnp.convolve)
sqrt = _wrap(jnp.sqrt)
cbrt = _wrap(jnp.cbrt)
square = _wrap(jnp.square)
absolute = _wrap(jnp.absolute)
fabs = _wrap(jnp.fabs)
sign = _wrap(jnp.sign)
heaviside = _wrap(jnp.heaviside)
maximum = _wrap(jnp.maximum)
minimum = _wrap(jnp.minimum)
fmax = _wrap(jnp.fmax)
fmin = _wrap(jnp.fmin)
interp = _wrap(jnp.interp)
clip = _wrap(jnp.clip)

# binary funcs
# -------------

bitwise_and = _wrap(jnp.bitwise_and)
bitwise_not = _wrap(jnp.bitwise_not)
bitwise_or = _wrap(jnp.bitwise_or)
bitwise_xor = _wrap(jnp.bitwise_xor)
invert = _wrap(jnp.invert)
left_shift = _wrap(jnp.left_shift)
right_shift = _wrap(jnp.right_shift)

# logic funcs
# -----------

# 1. Comparison
equal = _wrap(jnp.equal)
not_equal = _wrap(jnp.not_equal)
greater = _wrap(jnp.greater)
greater_equal = _wrap(jnp.greater_equal)
less = _wrap(jnp.less)
less_equal = _wrap(jnp.less_equal)
array_equal = _wrap(jnp.array_equal)
isclose = _wrap(jnp.isclose)
allclose = _wrap(jnp.allclose)

# 2. Logical operations
logical_and = _wrap(jnp.logical_and)
logical_not = _wrap(jnp.logical_not)
logical_or = _wrap(jnp.logical_or)
logical_xor = _wrap(jnp.logical_xor)

# 3. Truth value testing
all = _wrap(jnp.all)
any = _wrap(jnp.any)

# array manipulation
# ------------------

shape = _wrap(jnp.shape)
size = _wrap(jnp.size)
reshape = _wrap(jnp.reshape)
ravel = _wrap(jnp.ravel)
moveaxis = _wrap(jnp.moveaxis)
transpose = _wrap(jnp.transpose)
swapaxes = _wrap(jnp.swapaxes)


def concatenate(arrays, axis: int = 0):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.concatenate(arrays, axis))


def stack(arrays, axis: int = 0):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.stack(arrays, axis))


def vstack(arrays):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.vstack(arrays))


def hstack(arrays):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.hstack(arrays))


def dstack(arrays):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.dstack(arrays))


def column_stack(arrays):
  arrays = [a.value if isinstance(a, ndarray) else a for a in arrays]
  return ndarray(jnp.column_stack(arrays))


split = _wrap(jnp.split)
dsplit = _wrap(jnp.dsplit)
hsplit = _wrap(jnp.hsplit)
vsplit = _wrap(jnp.vsplit)
tile = _wrap(jnp.tile)
repeat = _wrap(jnp.repeat)
unique = _wrap(jnp.unique)
append = _wrap(jnp.append)
flip = _wrap(jnp.flip)
fliplr = _wrap(jnp.fliplr)
flipud = _wrap(jnp.flipud)
roll = _wrap(jnp.roll)
atleast_1d = _wrap(jnp.atleast_1d)
atleast_2d = _wrap(jnp.atleast_2d)
atleast_3d = _wrap(jnp.atleast_3d)
expand_dims = _wrap(jnp.expand_dims)
squeeze = _wrap(jnp.squeeze)
sort = _wrap(jnp.sort)
argsort = _wrap(jnp.argsort)
argmax = _wrap(jnp.argmax)
argmin = _wrap(jnp.argmin)
argwhere = _wrap(jnp.argwhere)
nonzero = _wrap(jnp.nonzero)
flatnonzero = _wrap(jnp.flatnonzero)
where = _wrap(jnp.where)
searchsorted = _wrap(jnp.searchsorted)
extract = _wrap(jnp.extract)
count_nonzero = _wrap(jnp.count_nonzero)
max = _wrap(jnp.max)
min = _wrap(jnp.min)

# array creation
# --------------

empty = _wrap(jnp.empty)
empty_like = _wrap(jnp.empty_like)
ones = _wrap(jnp.ones)
ones_like = _wrap(jnp.ones_like)
zeros = _wrap(jnp.zeros)
zeros_like = _wrap(jnp.zeros_like)
full = _wrap(jnp.full)
full_like = _wrap(jnp.full_like)
eye = _wrap(jnp.eye)
identity = _wrap(jnp.identity)

array = _wrap(jnp.array)
asarray = _wrap(jnp.asarray)

arange = _wrap(jnp.arange)
linspace = _wrap(jnp.linspace)
logspace = _wrap(jnp.logspace)
meshgrid = _wrap(jnp.meshgrid)

diag = _wrap(jnp.diag)
tri = _wrap(jnp.tri)
tril = _wrap(jnp.tril)
triu = _wrap(jnp.triu)
vander = _wrap(jnp.vander)

# indexing funcs
# --------------

tril_indices = _wrap(jnp.tril_indices)
tril_indices_from = _wrap(jnp.tril_indices_from)
triu_indices = _wrap(jnp.triu_indices)
triu_indices_from = _wrap(jnp.triu_indices_from)
take = _wrap(jnp.take)
select = _wrap(jnp.select)

# statistic funcs
# ---------------
nanmin = _wrap(jnp.nanmin)
nanmax = _wrap(jnp.nanmax)
ptp = _wrap(jnp.ptp)
percentile = _wrap(jnp.percentile)
nanpercentile = _wrap(jnp.nanpercentile)
quantile = _wrap(jnp.quantile)
nanquantile = _wrap(jnp.nanquantile)

average = _wrap(jnp.average)
mean = _wrap(jnp.mean)
std = _wrap(jnp.std)
var = _wrap(jnp.var)
nanmedian = _wrap(jnp.nanmedian)
nanmean = _wrap(jnp.nanmean)
nanstd = _wrap(jnp.nanstd)
nanvar = _wrap(jnp.nanvar)

corrcoef = _wrap(jnp.corrcoef)
correlate = _wrap(jnp.correlate)
cov = _wrap(jnp.cov)

histogram = _wrap(jnp.histogram)
bincount = _wrap(jnp.bincount)
digitize = _wrap(jnp.digitize)

bartlett = _wrap(jnp.bartlett)
blackman = _wrap(jnp.blackman)
hamming = _wrap(jnp.hamming)
hanning = _wrap(jnp.hanning)
kaiser = _wrap(jnp.kaiser)

# constants
# ---------

e = jnp.e
pi = jnp.pi
inf = jnp.inf

# linear algebra
# --------------

dot = _wrap(jnp.dot)
vdot = _wrap(jnp.vdot)
inner = _wrap(jnp.inner)
outer = _wrap(jnp.outer)
kron = _wrap(jnp.kron)
matmul = _wrap(jnp.matmul)
trace = _wrap(jnp.trace)

# data types
# ----------

dtype = jnp.dtype
finfo = jnp.finfo
iinfo = jnp.iinfo

bool_ = jnp.bool_
uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int_ = jnp.int_
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
float_ = jnp.float_
float16 = jnp.float16
float32 = jnp.float32
float64 = jnp.float64
complex_ = jnp.complex_
complex64 = jnp.complex64
complex128 = jnp.complex128
