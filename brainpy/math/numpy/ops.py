# -*- coding: utf-8 -*-

import numpy

from brainpy.tools import numba_jit

__all__ = [
  # math funcs
  'real', 'imag', 'conj', 'conjugate', 'ndim', 'isreal', 'isscalar',
  'add', 'reciprocal', 'negative', 'positive', 'multiply', 'divide',
  'power', 'subtract', 'true_divide', 'floor_divide', 'float_power',
  'fmod', 'mod', 'modf', 'divmod', 'remainder', 'abs', 'exp', 'exp2',
  'expm1', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2',
  'lcm', 'gcd', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
  'arctan2', 'arctanh', 'cos', 'cosh', 'sin', 'sinc', 'sinh', 'tan',
  'tanh', 'deg2rad', 'hypot', 'rad2deg', 'degrees', 'radians', 'round', 'around',
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

  # others
  'take_along_axis', 'clip_by_norm',
]

# math funcs
# ----------

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

round = numpy.round
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

# binary funcs
# -------------

bitwise_and = numpy.bitwise_and
bitwise_not = numpy.bitwise_not
bitwise_or = numpy.bitwise_or
bitwise_xor = numpy.bitwise_xor
invert = numpy.invert
left_shift = numpy.left_shift
right_shift = numpy.right_shift

# logic funcs
# -----------

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

# array manipulation
# ------------------

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

# array creation
# --------------

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

diag = numpy.diag
tri = numpy.tri
tril = numpy.tril
triu = numpy.triu
vander = numpy.vander

# indexing funcs
# --------------

tril_indices = numpy.tril_indices
tril_indices_from = numpy.tril_indices_from
triu_indices = numpy.triu_indices
triu_indices_from = numpy.triu_indices_from
take = numpy.take
select = numpy.select

# statistic funcs
# ---------------
nanmin = numpy.nanmin
nanmax = numpy.nanmax
ptp = numpy.ptp
percentile = numpy.percentile
nanpercentile = numpy.nanpercentile
quantile = numpy.quantile
nanquantile = numpy.nanquantile

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

bartlett = numpy.bartlett
blackman = numpy.blackman
hamming = numpy.hamming
hanning = numpy.hanning
kaiser = numpy.kaiser

# constants
# ---------

e = numpy.e
pi = numpy.pi
inf = numpy.inf

# linear algebra
# --------------

dot = numpy.dot
vdot = numpy.vdot
inner = numpy.inner
outer = numpy.outer
kron = numpy.kron
matmul = numpy.matmul
trace = numpy.trace

# data types
# ----------

dtype = numpy.dtype
finfo = numpy.finfo
iinfo = numpy.iinfo

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

# others
# -------

take_along_axis = numpy.take_along_axis


@numba_jit
def clip_by_norm(t, clip_norm, axis=None):
  """Clips values to a maximum L2-norm.

  Given a tensor ``t``, and a maximum clip value ``clip_norm``, this operation
  normalizes ``t`` so that its L2-norm is less than or equal to ``clip_norm``,
  along the dimensions given in ``axis``. Specifically, in the default case
  where all dimensions are used for calculation, if the L2-norm of ``t`` is
  already less than or equal to ``clip_norm``, then ``t`` is not modified. If
  the L2-norm is greater than ``clip_norm``, then this operation returns a
  tensor of the same type and shape as ``t`` with its values set to:

  .. math::
    t * clip_norm / l2norm(t)

  In this case, the L2-norm of the output tensor is `clip_norm`.

  As another example, if ``t`` is a matrix and ``axis=1``, then each row
  of the output will have L2-norm less than or equal to ``clip_norm``. If
  ``axis=0`` instead, each column of the output will be clipped.

  This operation is typically used to clip gradients before applying them with
  an optimizer.
  """
  l2norm = sqrt(sum(t * t, axis=axis, keepdims=True))
  clip_values = t * clip_norm / maximum(l2norm, clip_norm)
  return clip_values
