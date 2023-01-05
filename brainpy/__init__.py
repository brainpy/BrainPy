# -*- coding: utf-8 -*-

__version__ = "2.3.2"


# fundamental supporting modules
from brainpy import errors, check, tools


#  Part 1: Math Foundation  #
# ------------------------- #

# math foundation
from brainpy import math
from .math import BrainPyObject


#  Part 2: Toolbox  #
# ----------------- #

# modules of toolbox
from brainpy import (
  connect,  # synaptic connection
  initialize,  # weight initialization
  optim,  # gradient descent optimizers
  losses,  # loss functions
  measure,  # methods for data analysis
  inputs,  # methods for generating input currents
  encoding,  # encoding schema
  checkpoints,  # checkpoints
  check,  # error checking
)
from . import algorithms  # online or offline training algorithms

# convenient alias
conn = connect
init = initialize
optimizers = optim

# numerical integrators
from brainpy import integrators
from brainpy.integrators import ode, sde, fde
from brainpy._src.integrators.joint_eq import (JointEq as JointEq)
from brainpy._src.integrators.runner import (IntegratorRunner as IntegratorRunner)
from brainpy._src.integrators.ode.generic import (odeint as odeint)
from brainpy._src.integrators.sde.generic import (sdeint as sdeint)
from brainpy._src.integrators.fde.generic import (fdeint as fdeint)


#  Part 3: Models  #
# ---------------- #

from brainpy import (channels,  # channel models
                     layers,  # ANN layers
                     neurons,  # neuron groups
                     rates,  # rate models
                     synapses,  # synaptic dynamics
                     synouts,  # synaptic output
                     synplast,  # synaptic plasticity
                     )
from brainpy._src.dyn.base import (DynamicalSystem as DynamicalSystem,
                                   Container as Container,
                                   Sequential as Sequential,
                                   Network as Network,
                                   NeuGroup as NeuGroup,
                                   SynConn as SynConn,
                                   SynOut as SynOut,
                                   SynSTP as SynSTP,
                                   SynLTP as SynLTP,
                                   TwoEndConn as TwoEndConn,
                                   CondNeuGroup as CondNeuGroup,
                                   Channel as Channel)
from brainpy._src.dyn.base import (DSPartial as DSPartial)
from brainpy._src.dyn.transform import (NoSharedArg as NoSharedArg,  # transformations
                                        LoopOverTime as LoopOverTime,)
from brainpy._src.dyn.runners import (DSRunner as DSRunner)  # runner


#  Part 4: Training  #
# ------------------ #

from . import train
from ._src.train.base import (DSTrainer as DSTrainer)
from ._src.train.back_propagation import (BPTT as BPTT,
                                          BPFF as BPFF,)
from ._src.train.online import (OnlineTrainer as OnlineTrainer,
                                ForceTrainer as ForceTrainer,)
from ._src.train.offline import (OfflineTrainer as OfflineTrainer,
                                 RidgeTrainer as RidgeTrainer,)


#  Part 5: Analysis  #
# ------------------ #

from . import analysis


#  Part 6: Others    #
# ------------------ #

from . import running
from ._src.visualization import (visualize as visualize)
from ._src.running.runner import (Runner as Runner)


#  Part 7: Deprecations  #
# ---------------------- #


# deprecated
from brainpy._src.math.object_transform.base import (Base as Base,
                                                     ArrayCollector as ArrayCollector,
                                                     Collector as Collector,
                                                     TensorCollector as TensorCollector)

train.__dict__['DSTrainer'] = DSTrainer
train.__dict__['BPTT'] = BPTT
train.__dict__['BPFF'] = BPFF
train.__dict__['OnlineTrainer'] = OnlineTrainer
train.__dict__['ForceTrainer'] = ForceTrainer
train.__dict__['OfflineTrainer'] = OfflineTrainer
train.__dict__['RidgeTrainer'] = RidgeTrainer


from . import base
base.base.__dict__['BrainPyObject'] = BrainPyObject
base.base.__dict__['Base'] = Base
base.collector.__dict__['Collector'] = Collector
base.collector.__dict__['ArrayCollector'] = ArrayCollector
base.collector.__dict__['TensorCollector'] = TensorCollector
base.function.__dict__['FunAsObject'] = math.FunAsObject
base.function.__dict__['Function'] = math.FunAsObject
base.io.__dict__['save_as_h5'] = checkpoints.io.save_as_h5
base.io.__dict__['save_as_npz'] = checkpoints.io.save_as_npz
base.io.__dict__['save_as_pkl'] = checkpoints.io.save_as_pkl
base.io.__dict__['save_as_mat'] = checkpoints.io.save_as_mat
base.io.__dict__['load_by_h5'] = checkpoints.io.load_by_h5
base.io.__dict__['load_by_npz'] = checkpoints.io.load_by_npz
base.io.__dict__['load_by_pkl'] = checkpoints.io.load_by_pkl
base.io.__dict__['load_by_mat'] = checkpoints.io.load_by_mat
base.naming.__dict__['check_name_uniqueness'] = tools.check_name_uniqueness
base.naming.__dict__['clear_name_cache'] = tools.clear_name_cache
base.naming.__dict__['get_unique_name'] = tools.get_unique_name
base.__dict__['BrainPyObject'] = BrainPyObject
base.__dict__['Base'] = Base
base.__dict__['Collector'] = Collector
base.__dict__['ArrayCollector'] = ArrayCollector
base.__dict__['TensorCollector'] = TensorCollector
base.__dict__['FunAsObject'] = math.FunAsObject
base.__dict__['Function'] = math.FunAsObject
base.__dict__['save_as_h5'] = checkpoints.io.save_as_h5
base.__dict__['save_as_npz'] = checkpoints.io.save_as_npz
base.__dict__['save_as_pkl'] = checkpoints.io.save_as_pkl
base.__dict__['save_as_mat'] = checkpoints.io.save_as_mat
base.__dict__['load_by_h5'] = checkpoints.io.load_by_h5
base.__dict__['load_by_npz'] = checkpoints.io.load_by_npz
base.__dict__['load_by_pkl'] = checkpoints.io.load_by_pkl
base.__dict__['load_by_mat'] = checkpoints.io.load_by_mat
base.__dict__['check_name_uniqueness'] = tools.check_name_uniqueness
base.__dict__['clear_name_cache'] = tools.clear_name_cache
base.__dict__['get_unique_name'] = tools.get_unique_name


from . import modes
modes.__dict__['Mode'] = math.Mode
modes.__dict__['NormalMode'] = math.NonBatchingMode
modes.__dict__['BatchingMode'] = math.BatchingMode
modes.__dict__['TrainingMode'] = math.TrainingMode
modes.__dict__['normal'] = math.nonbatching_mode
modes.__dict__['batching'] = math.batching_mode
modes.__dict__['training'] = math.training_mode
modes.__dict__['check_mode'] = check.is_subclass


from brainpy import dyn
dyn.__dict__['channels'] = channels
dyn.__dict__['neurons'] = neurons
dyn.__dict__['rates'] = rates
dyn.__dict__['synapses'] = synapses
dyn.__dict__['synouts'] = synouts
dyn.__dict__['synplast'] = synplast
dyn.__dict__['DynamicalSystem'] = DynamicalSystem
dyn.__dict__['Container'] = Container
dyn.__dict__['Sequential'] = Sequential
dyn.__dict__['Network'] = Network
dyn.__dict__['NeuGroup'] = NeuGroup
dyn.__dict__['SynConn'] = SynConn
dyn.__dict__['SynOut'] = SynOut
dyn.__dict__['SynSTP'] = SynSTP
dyn.__dict__['SynLTP'] = SynLTP
dyn.__dict__['TwoEndConn'] = TwoEndConn
dyn.__dict__['CondNeuGroup'] = CondNeuGroup
dyn.__dict__['Channel'] = Channel
dyn.__dict__['NoSharedArg'] = NoSharedArg
dyn.__dict__['LoopOverTime'] = LoopOverTime
dyn.__dict__['DSRunner'] = DSRunner
integrators.__dict__['odeint'] = odeint
integrators.__dict__['sdeint'] = sdeint
integrators.__dict__['fdeint'] = fdeint
integrators.__dict__['IntegratorRunner'] = IntegratorRunner
integrators.__dict__['JointEq'] = JointEq


import brainpy._src.math.arraycompatible as bm
math.__dict__['full'] = bm.full
math.__dict__['full_like'] = bm.full_like
math.__dict__['eye'] = bm.eye
math.__dict__['identity'] = bm.identity
math.__dict__['diag'] = bm.diag
math.__dict__['tri'] = bm.tri
math.__dict__['tril'] = bm.tril
math.__dict__['triu'] = bm.triu
math.__dict__['real'] = bm.real
math.__dict__['imag'] = bm.imag
math.__dict__['conj'] = bm.conj
math.__dict__['conjugate'] = bm.conjugate
math.__dict__['ndim'] = bm.ndim
math.__dict__['isreal'] = bm.isreal
math.__dict__['isscalar'] = bm.isscalar
math.__dict__['add'] = bm.add
math.__dict__['reciprocal'] = bm.reciprocal
math.__dict__['negative'] = bm.negative
math.__dict__['positive'] = bm.positive
math.__dict__['multiply'] = bm.multiply
math.__dict__['divide'] = bm.divide
math.__dict__['power'] = bm.power
math.__dict__['subtract'] = bm.subtract
math.__dict__['true_divide'] = bm.true_divide
math.__dict__['floor_divide'] = bm.floor_divide
math.__dict__['float_power'] = bm.float_power
math.__dict__['fmod'] = bm.fmod
math.__dict__['mod'] = bm.mod
math.__dict__['modf'] = bm.modf
math.__dict__['divmod'] = bm.divmod
math.__dict__['remainder'] = bm.remainder
math.__dict__['abs'] = bm.abs
math.__dict__['exp'] = bm.exp
math.__dict__['exp2'] = bm.exp2
math.__dict__['expm1'] = bm.expm1
math.__dict__['log'] = bm.log
math.__dict__['log10'] = bm.log10
math.__dict__['log1p'] = bm.log1p
math.__dict__['log2'] = bm.log2
math.__dict__['logaddexp'] = bm.logaddexp
math.__dict__['logaddexp2'] = bm.logaddexp2
math.__dict__['lcm'] = bm.lcm
math.__dict__['gcd'] = bm.gcd
math.__dict__['arccos'] = bm.arccos
math.__dict__['arccosh'] = bm.arccosh
math.__dict__['arcsin'] = bm.arcsin
math.__dict__['arcsinh'] = bm.arcsinh
math.__dict__['arctan'] = bm.arctan
math.__dict__['arctan2'] = bm.arctan2
math.__dict__['arctanh'] = bm.arctanh
math.__dict__['cos'] = bm.cos
math.__dict__['cosh'] = bm.cosh
math.__dict__['sin'] = bm.sin
math.__dict__['sinc'] = bm.sinc
math.__dict__['sinh'] = bm.sinh
math.__dict__['tan'] = bm.tan
math.__dict__['tanh'] = bm.tanh
math.__dict__['deg2rad'] = bm.deg2rad
math.__dict__['hypot'] = bm.hypot
math.__dict__['rad2deg'] = bm.rad2deg
math.__dict__['degrees'] = bm.degrees
math.__dict__['radians'] = bm.radians
math.__dict__['round'] = bm.round
math.__dict__['around'] = bm.around
math.__dict__['round_'] = bm.round_
math.__dict__['rint'] = bm.rint
math.__dict__['floor'] = bm.floor
math.__dict__['ceil'] = bm.ceil
math.__dict__['trunc'] = bm.trunc
math.__dict__['fix'] = bm.fix
math.__dict__['prod'] = bm.prod
math.__dict__['sum'] = bm.sum
math.__dict__['diff'] = bm.diff
math.__dict__['median'] = bm.median
math.__dict__['nancumprod'] = bm.nancumprod
math.__dict__['nancumsum'] = bm.nancumsum
math.__dict__['nanprod'] = bm.nanprod
math.__dict__['nansum'] = bm.nansum
math.__dict__['cumprod'] = bm.cumprod
math.__dict__['cumsum'] = bm.cumsum
math.__dict__['ediff1d'] = bm.ediff1d
math.__dict__['cross'] = bm.cross
math.__dict__['trapz'] = bm.trapz
math.__dict__['isfinite'] = bm.isfinite
math.__dict__['isinf'] = bm.isinf
math.__dict__['isnan'] = bm.isnan
math.__dict__['signbit'] = bm.signbit
math.__dict__['copysign'] = bm.copysign
math.__dict__['nextafter'] = bm.nextafter
math.__dict__['ldexp'] = bm.ldexp
math.__dict__['frexp'] = bm.frexp
math.__dict__['convolve'] = bm.convolve
math.__dict__['sqrt'] = bm.sqrt
math.__dict__['cbrt'] = bm.cbrt
math.__dict__['square'] = bm.square
math.__dict__['absolute'] = bm.absolute
math.__dict__['fabs'] = bm.fabs
math.__dict__['sign'] = bm.sign
math.__dict__['heaviside'] = bm.heaviside
math.__dict__['maximum'] = bm.maximum
math.__dict__['minimum'] = bm.minimum
math.__dict__['fmax'] = bm.fmax
math.__dict__['fmin'] = bm.fmin
math.__dict__['interp'] = bm.interp
math.__dict__['clip'] = bm.clip
math.__dict__['angle'] = bm.angle
math.__dict__['bitwise_and'] = bm.bitwise_and
math.__dict__['bitwise_not'] = bm.bitwise_not
math.__dict__['bitwise_or'] = bm.bitwise_or
math.__dict__['bitwise_xor'] = bm.bitwise_xor
math.__dict__['invert'] = bm.invert
math.__dict__['left_shift'] = bm.left_shift
math.__dict__['right_shift'] = bm.right_shift
math.__dict__['equal'] = bm.equal
math.__dict__['not_equal'] = bm.not_equal
math.__dict__['greater'] = bm.greater
math.__dict__['greater_equal'] = bm.greater_equal
math.__dict__['less'] = bm.less
math.__dict__['less_equal'] = bm.less_equal
math.__dict__['array_equal'] = bm.array_equal
math.__dict__['isclose'] = bm.isclose
math.__dict__['allclose'] = bm.allclose
math.__dict__['logical_and'] = bm.logical_and
math.__dict__['logical_not'] = bm.logical_not
math.__dict__['logical_or'] = bm.logical_or
math.__dict__['logical_xor'] = bm.logical_xor
math.__dict__['all'] = bm.all
math.__dict__['any'] = bm.any
math.__dict__['alltrue'] = bm.alltrue
math.__dict__['sometrue'] = bm.sometrue
math.__dict__['shape'] = bm.shape
math.__dict__['size'] = bm.size
math.__dict__['reshape'] = bm.reshape
math.__dict__['ravel'] = bm.ravel
math.__dict__['moveaxis'] = bm.moveaxis
math.__dict__['transpose'] = bm.transpose
math.__dict__['swapaxes'] = bm.swapaxes
math.__dict__['concatenate'] = bm.concatenate
math.__dict__['stack'] = bm.stack
math.__dict__['vstack'] = bm.vstack
math.__dict__['hstack'] = bm.hstack
math.__dict__['dstack'] = bm.dstack
math.__dict__['column_stack'] = bm.column_stack
math.__dict__['split'] = bm.split
math.__dict__['dsplit'] = bm.dsplit
math.__dict__['hsplit'] = bm.hsplit
math.__dict__['vsplit'] = bm.vsplit
math.__dict__['tile'] = bm.tile
math.__dict__['repeat'] = bm.repeat
math.__dict__['unique'] = bm.unique
math.__dict__['append'] = bm.append
math.__dict__['flip'] = bm.flip
math.__dict__['fliplr'] = bm.fliplr
math.__dict__['flipud'] = bm.flipud
math.__dict__['roll'] = bm.roll
math.__dict__['atleast_1d'] = bm.atleast_1d
math.__dict__['atleast_2d'] = bm.atleast_2d
math.__dict__['atleast_3d'] = bm.atleast_3d
math.__dict__['expand_dims'] = bm.expand_dims
math.__dict__['squeeze'] = bm.squeeze
math.__dict__['sort'] = bm.sort
math.__dict__['argsort'] = bm.argsort
math.__dict__['argmax'] = bm.argmax
math.__dict__['argmin'] = bm.argmin
math.__dict__['argwhere'] = bm.argwhere
math.__dict__['nonzero'] = bm.nonzero
math.__dict__['flatnonzero'] = bm.flatnonzero
math.__dict__['where'] = bm.where
math.__dict__['searchsorted'] = bm.searchsorted
math.__dict__['extract'] = bm.extract
math.__dict__['count_nonzero'] = bm.count_nonzero
math.__dict__['max'] = bm.max
math.__dict__['min'] = bm.min
math.__dict__['amax'] = bm.amax
math.__dict__['amin'] = bm.amin
math.__dict__['array_split'] = bm.array_split
math.__dict__['meshgrid'] = bm.meshgrid
math.__dict__['vander'] = bm.vander
math.__dict__['nonzero'] = bm.nonzero
math.__dict__['where'] = bm.where
math.__dict__['tril_indices'] = bm.tril_indices
math.__dict__['tril_indices_from'] = bm.tril_indices_from
math.__dict__['triu_indices'] = bm.triu_indices
math.__dict__['triu_indices_from'] = bm.triu_indices_from
math.__dict__['take'] = bm.take
math.__dict__['select'] = bm.select
math.__dict__['nanmin'] = bm.nanmin
math.__dict__['nanmax'] = bm.nanmax
math.__dict__['ptp'] = bm.ptp
math.__dict__['percentile'] = bm.percentile
math.__dict__['nanpercentile'] = bm.nanpercentile
math.__dict__['quantile'] = bm.quantile
math.__dict__['nanquantile'] = bm.nanquantile
math.__dict__['median'] = bm.median
math.__dict__['average'] = bm.average
math.__dict__['mean'] = bm.mean
math.__dict__['std'] = bm.std
math.__dict__['var'] = bm.var
math.__dict__['nanmedian'] = bm.nanmedian
math.__dict__['nanmean'] = bm.nanmean
math.__dict__['nanstd'] = bm.nanstd
math.__dict__['nanvar'] = bm.nanvar
math.__dict__['corrcoef'] = bm.corrcoef
math.__dict__['correlate'] = bm.correlate
math.__dict__['cov'] = bm.cov
math.__dict__['histogram'] = bm.histogram
math.__dict__['bincount'] = bm.bincount
math.__dict__['digitize'] = bm.digitize
math.__dict__['bartlett'] = bm.bartlett
math.__dict__['blackman'] = bm.blackman
math.__dict__['hamming'] = bm.hamming
math.__dict__['hanning'] = bm.hanning
math.__dict__['kaiser'] = bm.kaiser
math.__dict__['e'] = bm.e
math.__dict__['pi'] = bm.pi
math.__dict__['inf'] = bm.inf
math.__dict__['dot'] = bm.dot
math.__dict__['vdot'] = bm.vdot
math.__dict__['inner'] = bm.inner
math.__dict__['outer'] = bm.outer
math.__dict__['kron'] = bm.kron
math.__dict__['matmul'] = bm.matmul
math.__dict__['trace'] = bm.trace
math.__dict__['dtype'] = bm.dtype
math.__dict__['finfo'] = bm.finfo
math.__dict__['iinfo'] = bm.iinfo
math.__dict__['uint8'] = bm.uint8
math.__dict__['uint16'] = bm.uint16
math.__dict__['uint32'] = bm.uint32
math.__dict__['uint64'] = bm.uint64
math.__dict__['int8'] = bm.int8
math.__dict__['int16'] = bm.int16
math.__dict__['int32'] = bm.int32
math.__dict__['int64'] = bm.int64
math.__dict__['float16'] = bm.float16
math.__dict__['float32'] = bm.float32
math.__dict__['float64'] = bm.float64
math.__dict__['complex64'] = bm.complex64
math.__dict__['complex128'] = bm.complex128
math.__dict__['product'] = bm.product
math.__dict__['row_stack'] = bm.row_stack
math.__dict__['apply_over_axes'] = bm.apply_over_axes
math.__dict__['apply_along_axis'] = bm.apply_along_axis
math.__dict__['array_equiv'] = bm.array_equiv
math.__dict__['array_repr'] = bm.array_repr
math.__dict__['array_str'] = bm.array_str
math.__dict__['block'] = bm.block
math.__dict__['broadcast_arrays'] = bm.broadcast_arrays
math.__dict__['broadcast_shapes'] = bm.broadcast_shapes
math.__dict__['broadcast_to'] = bm.broadcast_to
math.__dict__['compress'] = bm.compress
math.__dict__['cumproduct'] = bm.cumproduct
math.__dict__['diag_indices'] = bm.diag_indices
math.__dict__['diag_indices_from'] = bm.diag_indices_from
math.__dict__['diagflat'] = bm.diagflat
math.__dict__['diagonal'] = bm.diagonal
math.__dict__['einsum'] = bm.einsum
math.__dict__['einsum_path'] = bm.einsum_path
math.__dict__['geomspace'] = bm.geomspace
math.__dict__['gradient'] = bm.gradient
math.__dict__['histogram2d'] = bm.histogram2d
math.__dict__['histogram_bin_edges'] = bm.histogram_bin_edges
math.__dict__['histogramdd'] = bm.histogramdd
math.__dict__['i0'] = bm.i0
math.__dict__['in1d'] = bm.in1d
math.__dict__['indices'] = bm.indices
math.__dict__['insert'] = bm.insert
math.__dict__['intersect1d'] = bm.intersect1d
math.__dict__['iscomplex'] = bm.iscomplex
math.__dict__['isin'] = bm.isin
math.__dict__['ix_'] = bm.ix_
math.__dict__['lexsort'] = bm.lexsort
math.__dict__['load'] = bm.load
math.__dict__['save'] = bm.save
math.__dict__['savez'] = bm.savez
math.__dict__['mask_indices'] = bm.mask_indices
math.__dict__['msort'] = bm.msort
math.__dict__['nan_to_num'] = bm.nan_to_num
math.__dict__['nanargmax'] = bm.nanargmax
math.__dict__['setdiff1d'] = bm.setdiff1d
math.__dict__['nanargmin'] = bm.nanargmin
math.__dict__['pad'] = bm.pad
math.__dict__['poly'] = bm.poly
math.__dict__['polyadd'] = bm.polyadd
math.__dict__['polyder'] = bm.polyder
math.__dict__['polyfit'] = bm.polyfit
math.__dict__['polyint'] = bm.polyint
math.__dict__['polymul'] = bm.polymul
math.__dict__['polysub'] = bm.polysub
math.__dict__['polyval'] = bm.polyval
math.__dict__['resize'] = bm.resize
math.__dict__['rollaxis'] = bm.rollaxis
math.__dict__['roots'] = bm.roots
math.__dict__['rot90'] = bm.rot90
math.__dict__['setxor1d'] = bm.setxor1d
math.__dict__['tensordot'] = bm.tensordot
math.__dict__['trim_zeros'] = bm.trim_zeros
math.__dict__['union1d'] = bm.union1d
math.__dict__['unravel_index'] = bm.unravel_index
math.__dict__['unwrap'] = bm.unwrap
math.__dict__['take_along_axis'] = bm.take_along_axis
math.__dict__['can_cast'] = bm.can_cast
math.__dict__['choose'] = bm.choose
math.__dict__['copy'] = bm.copy
math.__dict__['frombuffer'] = bm.frombuffer
math.__dict__['fromfile'] = bm.fromfile
math.__dict__['fromfunction'] = bm.fromfunction
math.__dict__['fromiter'] = bm.fromiter
math.__dict__['fromstring'] = bm.fromstring
math.__dict__['get_printoptions'] = bm.get_printoptions
math.__dict__['iscomplexobj'] = bm.iscomplexobj
math.__dict__['isneginf'] = bm.isneginf
math.__dict__['isposinf'] = bm.isposinf
math.__dict__['isrealobj'] = bm.isrealobj
math.__dict__['issubdtype'] = bm.issubdtype
math.__dict__['issubsctype'] = bm.issubsctype
math.__dict__['iterable'] = bm.iterable
math.__dict__['packbits'] = bm.packbits
math.__dict__['piecewise'] = bm.piecewise
math.__dict__['printoptions'] = bm.printoptions
math.__dict__['set_printoptions'] = bm.set_printoptions
math.__dict__['promote_types'] = bm.promote_types
math.__dict__['ravel_multi_index'] = bm.ravel_multi_index
math.__dict__['result_type'] = bm.result_type
math.__dict__['sort_complex'] = bm.sort_complex
math.__dict__['unpackbits'] = bm.unpackbits
math.__dict__['delete'] = bm.delete
math.__dict__['add_docstring'] = bm.add_docstring
math.__dict__['add_newdoc'] = bm.add_newdoc
math.__dict__['add_newdoc_ufunc'] = bm.add_newdoc_ufunc
math.__dict__['array2string'] = bm.array2string
math.__dict__['asanyarray'] = bm.asanyarray
math.__dict__['ascontiguousarray'] = bm.ascontiguousarray
math.__dict__['asfarray'] = bm.asfarray
math.__dict__['asscalar'] = bm.asscalar
math.__dict__['common_type'] = bm.common_type
math.__dict__['disp'] = bm.disp
math.__dict__['genfromtxt'] = bm.genfromtxt
math.__dict__['loadtxt'] = bm.loadtxt
math.__dict__['info'] = bm.info
math.__dict__['issubclass_'] = bm.issubclass_
math.__dict__['place'] = bm.place
math.__dict__['polydiv'] = bm.polydiv
math.__dict__['put'] = bm.put
math.__dict__['putmask'] = bm.putmask
math.__dict__['safe_eval'] = bm.safe_eval
math.__dict__['savetxt'] = bm.savetxt
math.__dict__['savez_compressed'] = bm.savez_compressed
math.__dict__['show_config'] = bm.show_config
math.__dict__['typename'] = bm.typename
math.__dict__['copyto'] = bm.copyto
math.__dict__['matrix'] = bm.matrix
math.__dict__['asmatrix'] = bm.asmatrix
math.__dict__['mat'] = bm.mat
del bm
