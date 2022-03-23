# -*- coding: utf-8 -*-

"""
The parallel compilation tools for JAX backend.

1. Vectorize compilation is implemented by the 'vmap()' function
2. Parallel compilation is implemented by the 'pmap()' function

"""


import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.interpreters.partial_eval import JaxprTracer
from jax.interpreters.pxla import ShardedDeviceArray

try:
  from jax.errors import UnexpectedTracerError
except ImportError:
  from jax.core import UnexpectedTracerError

from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import TensorCollector
from brainpy.math.random import RandomState
from brainpy.math.jaxarray import JaxArray
from brainpy.tools.codes import change_func_name

__all__ = [
  'vmap',
  'pmap',
]


def _make_vmap(func, nonbatched_vars, batched_vars, in_axes, out_axes,
               batch_idx, axis_name, f_name=None):
  @functools.partial(jax.vmap, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)
  def vmapped_func(nonbatched_data, batched_data, *args, **kwargs):
    nonbatched_vars.assign(nonbatched_data)
    batched_vars.assign(batched_data)
    out = func(*args, **kwargs)
    nonbatched_changes = nonbatched_vars.dict()
    batched_changes = batched_vars.dict()
    return nonbatched_changes, batched_changes, out

  def call(*args, **kwargs):
    n = args[batch_idx[0]].shape[batch_idx[1]]
    nonbatched_data = nonbatched_vars.dict()
    batched_data = {key: val.split_keys(n) for key, val in batched_vars.items()}
    try:
      out, dyn_changes, rand_changes = vmapped_func(nonbatched_data, batched_data, *args, **kwargs)
    except UnexpectedTracerError as e:
      nonbatched_vars.assign(nonbatched_data)
      batched_vars.assign(batched_data)
      raise errors.JaxTracerError() from e
    # for key, v in dyn_changes.items():
    #   dyn_vars[key] = reduce_func(v)
    # for key, v in rand_changes.items():
    #   rand_vars[key] = reduce_func(v)
    return out

  return change_func_name(name=f_name, f=call) if f_name else call


def vmap(func, dyn_vars=None, batched_vars=None,
         in_axes=0, out_axes=0, axis_name=None,
         reduce_func=None, auto_infer=False):
  """Vectorization compilation for class objects.

  Vectorized compile a function or a module to run in parallel on a single device.

  Examples
  --------

  Parameters
  ----------
  func : Base, function, callable
    The function or the module to compile.
  dyn_vars : dict, sequence
  batched_vars : dict
  in_axes : optional, int, sequence of int
    Specify which input array axes to map over. If each positional argument to
    ``obj_or_func`` is an array, then ``in_axes`` can be an integer, a None,
    or a tuple of integers and Nones with length equal to the number of
    positional arguments to ``obj_or_func``. An integer or ``None``
    indicates which array axis to map over for all arguments (with ``None``
    indicating not to map any axis), and a tuple indicates which axis to map
    for each corresponding positional argument. Axis integers must be in the
    range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
    dimensions (axes) of the corresponding input array.

    If the positional arguments to ``obj_or_func`` are container types, the
    corresponding element of ``in_axes`` can itself be a matching container,
    so that distinct array axes can be mapped for different container
    elements. ``in_axes`` must be a container tree prefix of the positional
    argument tuple passed to ``obj_or_func``.

    At least one positional argument must have ``in_axes`` not None. The sizes
    of the mapped input axes for all mapped positional arguments must all be
    equal.

    Arguments passed as keywords are always mapped over their leading axis
    (i.e. axis index 0).
  out_axes : optional, int, tuple/list/dict
    Indicate where the mapped axis should appear in the output. All outputs
    with a mapped axis must have a non-None ``out_axes`` specification. Axis
    integers must be in the range ``[-ndim, ndim)`` for each output array,
    where ``ndim`` is the number of dimensions (axes) of the array returned
    by the :func:`vmap`-ed function, which is one more than the number of
    dimensions (axes) of the corresponding array returned by ``obj_or_func``.
  axis_name : optional

  Returns
  -------
  obj_or_func : Any
    Batched/vectorized version of ``obj_or_func`` with arguments that correspond to
    those of ``obj_or_func``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``obj_or_func``, but
    with extra array axes at positions indicated by ``out_axes``.

  """
  # if isinstance(func, DynamicalSystem):
  #   if len(func.steps):  # DynamicalSystem has step functions
  #
  #     # dynamical variables
  #     dyn_vars = (dyn_vars or func.vars().unique())
  #     dyn_vars, rand_vars = TensorCollector(), TensorCollector()
  #     for key, val in dyn_vars.items():
  #       if isinstance(val, RandomState):
  #         rand_vars[key] = val
  #       else:
  #         dyn_vars[key] = val
  #
  #     # in axes
  #     if in_axes is None:
  #       in_axes = {key: (None, 0) for key in func.steps.keys()}
  #     elif isinstance(in_axes, int):
  #       in_axes = {key: (None, 0, in_axes) for key in func.steps.keys()}
  #     elif isinstance(in_axes, (tuple, list)):
  #       in_axes = {key: (None, 0) + tuple(in_axes) for key in func.steps.keys()}
  #     elif isinstance(in_axes, dict):
  #       keys = list(func.steps.keys())
  #       if keys[0] not in in_axes:
  #         in_axes = {key: (None, 0, in_axes) for key in keys}
  #       else:
  #         in_axes = {key: (None, 0) + tuple(in_axes[key]) for key in keys}
  #     assert isinstance(in_axes, dict)
  #
  #     # batch size index
  #     batch_idx = {}
  #     for key, axes in in_axes.items():
  #       for i, axis in enumerate(axes[2:]):
  #         if axis is not None:
  #           batch_idx[key] = (i, axis)
  #           break
  #       else:
  #         raise ValueError(f'Found no batch axis: {axes}.')
  #
  #     # out axes
  #     if out_axes is None:
  #       out_axes = {key: 0 for key in func.steps.keys()}
  #     elif isinstance(out_axes, int):
  #       out_axes = {key: out_axes for key in func.steps.keys()}
  #     elif isinstance(out_axes, (tuple, list)):
  #       out_axes = {key: tuple(out_axes) + (0, 0) for key in func.steps.keys()}
  #     elif isinstance(out_axes, dict):
  #       keys = list(func.steps.keys())
  #       if keys[0] not in out_axes:
  #         out_axes = {key: (out_axes, 0, 0) for key in keys}
  #       else:
  #         out_axes = {key: tuple(out_axes[key]) + (0, 0) for key in keys}
  #     assert isinstance(out_axes, dict)
  #
  #     # reduce_func
  #     if reduce_func is None:
  #       reduce_func = lambda x: x.mean(axis=0)
  #
  #     # vectorized map functions
  #     for key in func.steps.keys():
  #       func.steps[key] = _make_vmap(func=func.steps[key],
  #                                    dyn_vars=dyn_vars,
  #                                    rand_vars=rand_vars,
  #                                    in_axes=in_axes[key],
  #                                    out_axes=out_axes[key],
  #                                    axis_name=axis_name,
  #                                    batch_idx=batch_idx[key],
  #                                    reduce_func=reduce_func,
  #                                    f_name=key)
  #
  #     return func

  if callable(func):
    if auto_infer:
      if dyn_vars is not None:
        dyn_vars = dyn_vars
      elif isinstance(func, Base):  # Base has '__call__()' implementation
        dyn_vars = func.vars().unique()
      elif hasattr(func, '__self__'):
        if isinstance(func.__self__, Base):
          dyn_vars = func.__self__.vars().unique()

    if dyn_vars is None:
      return jax.vmap(func,
                      in_axes=in_axes,
                      out_axes=out_axes,
                      axis_name=axis_name)

    else:
      if isinstance(dyn_vars, JaxArray):
        dyn_vars = [dyn_vars]
      if isinstance(dyn_vars, (tuple, list)):
        dyn_vars = {f'_vmap_v{i}': v for i, v in enumerate(dyn_vars)}
      assert isinstance(dyn_vars, dict)

      # dynamical variables
      _dyn_vars, _rand_vars = TensorCollector(), TensorCollector()
      for key, val in dyn_vars.items():
        if isinstance(val, RandomState):
          _rand_vars[key] = val
        else:
          _dyn_vars[key] = val

      # in axes
      if in_axes is None:
        in_axes = (None, 0)
      elif isinstance(in_axes, (int, dict)):
        in_axes = (None, 0, in_axes)
      elif isinstance(in_axes, (tuple, list)):
        in_axes = (None, 0) + tuple(in_axes)
      assert isinstance(in_axes, (tuple, list))

      # batch size index
      batch_idx = {}
      for key, axes in batch_idx.items():
        for i, axis in enumerate(axes[2:]):
          if axis is not None:
            batch_idx[key] = (i, axis)
            break
        else:
          raise ValueError(f'Found no batch axis: {axes}.')

      # out axes
      if out_axes is None:
        out_axes = 0
      elif isinstance(out_axes, (int, dict)):
        out_axes = (out_axes, 0, 0)
      elif isinstance(out_axes, (tuple, list)):
        out_axes = tuple(out_axes) + (0, 0)
      assert isinstance(out_axes, (list, tuple))

      # reduce_func
      if reduce_func is None:
        reduce_func = lambda x: x.mean(axis=0)

      # jit function
      return _make_vmap(func=func,
                        nonbatched_vars=_dyn_vars,
                        batched_vars=_rand_vars,
                        in_axes=in_axes,
                        out_axes=out_axes,
                        axis_name=axis_name,
                        batch_idx=batch_idx)

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable '
                              f'function, but we got {type(func)}.')


def _device_reshape(x):
  """Reshape an input array in order to broadcast to multiple devices."""
  num_device = jax.local_device_count()

  if not hasattr(x, 'ndim'):
    raise errors.BrainPyError(f'Expected JaxArray, got {type(x)}. If you are trying to pass a scalar to '
                              f'parallel, first convert it to a JaxArray, for example np.float(0.5)')
  if x.ndim == 0:
    return np.broadcast_to(x, [num_device])
  if x.shape[0] % num_device != 0:
    raise errors.BrainPyError(f'Must be able to equally divide batch {x.shape} among '
                              f'{num_device} devices, but does not go equally.')
  return x.reshape((num_device, x.shape[0] // num_device) + x.shape[1:])


def _make_pmap(func, dyn_vars, rand_vars, reduce_func, axis_name=None, in_axes=0,
               out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None,
               axis_size=None, donate_argnums=(), global_arg_shapes=None, f_name=None):
  @functools.partial(jax.pmap, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name,
                     static_broadcasted_argnums=static_broadcasted_argnums, devices=devices,
                     backend=backend, axis_size=axis_size, donate_argnums=donate_argnums,
                     global_arg_shapes=global_arg_shapes)
  def pmapped_func(dyn_data, rand_data, *args, **kwargs):
    dyn_vars.assign(dyn_data)
    rand_vars.assign(rand_data)
    out = func(*args, **kwargs)
    dyn_changes = dyn_vars.dict()
    rand_changes = rand_vars.dict()
    return out, dyn_changes, rand_changes

  def call(*args):
    un_replicated = [k for k, v in dyn_vars.items()
                     if not isinstance(v.value, (ShardedDeviceArray, JaxprTracer, DynamicJaxprTracer))]
    if len(un_replicated):
      raise errors.BrainPyError(f'Some variables were not replicated: {un_replicated}.'
                                f'did you forget to call xx.replicate() on them?')
    _args = []
    for i, x in enumerate(args):
      if i + 2 in static_broadcasted_argnums:
        _args.append(x)
      else:
        _args.append(jax.tree_map(_device_reshape, [x])[0])
    dyn_data = dyn_vars.dict()
    rand_data = rand_vars.dict()
    output, dyn_changes, rand_changes = pmapped_func(dyn_data, rand_data, *_args)
    dyn_vars.assign(dyn_changes)
    rand_vars.assign(rand_changes)
    return jax.tree_map(reduce_func, output)

  return change_func_name(name=f_name, f=call) if f_name else call


def pmap(func, dyn_vars=None, axis_name=None, in_axes=0, out_axes=0, static_broadcasted_argnums=(),
         devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None,
         reduce_func=None):
  """Parallel compilation for class objects.

  Parallel compile a function or a module to run on multiple devices in parallel.

  Parameters
  ----------
  func
  axis_name
  in_axes
  out_axes
  static_broadcasted_argnums
  devices
  backend
  axis_size
  donate_argnums
  global_arg_shapes

  Returns
  -------


  Examples
  --------


  """

  # if isinstance(func, DynamicalSystem):
  #   if len(func.steps):  # DynamicalSystem has step functions
  #
  #     # dynamical variables
  #     all_vars = (dyn_vars or func.vars().unique())
  #     dyn_vars = TensorCollector()
  #     rand_vars = TensorCollector()
  #     for key, val in all_vars.items():
  #       if isinstance(val, RandomState):
  #         rand_vars[key] = val
  #       else:
  #         dyn_vars[key] = val
  #
  #     # reduce function
  #     if reduce_func is None:
  #       reduce_func = jnp.concatenate
  #
  #     # static broadcast-ed arguments
  #     if static_broadcasted_argnums is None:
  #       static_broadcasted_argnums = ()
  #     elif isinstance(static_broadcasted_argnums, int):
  #       static_broadcasted_argnums = (static_broadcasted_argnums + 2,)
  #     elif isinstance(static_broadcasted_argnums, (tuple, list)):
  #       static_broadcasted_argnums = tuple(argnum + 2 for argnum in static_broadcasted_argnums)
  #     assert isinstance(static_broadcasted_argnums, (tuple, list))
  #
  #     # jit functions
  #     for key in func.steps.keys():
  #       step = func.steps[key]
  #       func.steps[key] = _make_pmap(dyn_vars=dyn_vars,
  #                                    rand_vars=rand_vars,
  #                                    func=step,
  #                                    axis_name=axis_name,
  #                                    in_axes=in_axes,
  #                                    out_axes=out_axes,
  #                                    static_broadcasted_argnums=static_broadcasted_argnums,
  #                                    devices=devices,
  #                                    backend=backend,
  #                                    axis_size=axis_size,
  #                                    donate_argnums=donate_argnums,
  #                                    global_arg_shapes=global_arg_shapes,
  #                                    reduce_func=reduce_func,
  #                                    f_name=key)
  #     return func

  if callable(func):
    if dyn_vars is not None:
      dyn_vars = dyn_vars
    elif isinstance(func, Base):  # Base has '__call__()' implementation
      dyn_vars = func.vars().unique()
    elif hasattr(func, '__self__'):
      if isinstance(func.__self__, Base):
        dyn_vars = func.__self__.vars().unique()

    if dyn_vars is None:
      return jax.pmap(func,
                      axis_name=axis_name,
                      in_axes=in_axes,
                      out_axes=out_axes,
                      static_broadcasted_argnums=static_broadcasted_argnums,
                      devices=devices,
                      backend=backend,
                      axis_size=axis_size,
                      donate_argnums=donate_argnums,
                      global_arg_shapes=global_arg_shapes)
    else:
      # dynamical variables
      dyn_vars = TensorCollector()
      rand_vars = TensorCollector()
      for key, val in dyn_vars.items():
        if isinstance(val, RandomState):
          rand_vars[key] = val
        else:
          dyn_vars[key] = val

      # static broadcast-ed arguments
      if static_broadcasted_argnums is None:
        static_broadcasted_argnums = ()
      elif isinstance(static_broadcasted_argnums, int):
        static_broadcasted_argnums = (static_broadcasted_argnums + 2,)
      elif isinstance(static_broadcasted_argnums, (tuple, list)):
        static_broadcasted_argnums = tuple(argnum + 2 for argnum in static_broadcasted_argnums)
      assert isinstance(static_broadcasted_argnums, (tuple, list))

      # reduce function
      if reduce_func is None:
        reduce_func = jnp.concatenate

      # jit function
      func.__call__ = _make_pmap(dyn_vars=dyn_vars,
                                 rand_vars=rand_vars,
                                 func=func,
                                 axis_name=axis_name,
                                 in_axes=in_axes,
                                 out_axes=out_axes,
                                 static_broadcasted_argnums=static_broadcasted_argnums,
                                 devices=devices,
                                 backend=backend,
                                 axis_size=axis_size,
                                 donate_argnums=donate_argnums,
                                 global_arg_shapes=global_arg_shapes,
                                 reduce_func=reduce_func)
      return func

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable function, '
                              f'but we got {type(func)}.')
