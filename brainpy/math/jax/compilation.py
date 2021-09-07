# -*- coding: utf-8 -*-

"""
The compilation tools for JAX backend.

1. Just-In-Time compilation is implemented by the 'jit()' function
2. Vectorize compilation is implemented by the 'vmap()' function
3. Parallel compilation is implemented by the 'pmap()' function

"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.interpreters.partial_eval import JaxprTracer
from jax.interpreters.pxla import ShardedDeviceArray

from brainpy import errors
from brainpy.base.base import Base
from brainpy.base.collector import ArrayCollector
from brainpy.math.jax.random import RandomState
from brainpy.tools.codes import change_func_name

__all__ = [
  'jit',
  'vmap',
  'pmap',
]

logger = logging.getLogger('brainpy.math.jax.compilation')


def _make_jit(func, vars_to_change, vars_needed,
              static_argnums, static_argnames=None, device=None,
              backend=None, donate_argnums=(), inline=False, f_name=None):
  @functools.partial(jax.jit, static_argnums=static_argnums, static_argnames=static_argnames,
                     device=device, backend=backend, donate_argnums=donate_argnums, inline=inline)
  def jitted_func(data_to_change, data_needed, *args, **kwargs):
    vars_to_change.assign(data_to_change)
    vars_needed.assign(data_needed)
    out = func(*args, **kwargs)
    changes = vars_to_change.dict()
    return out, changes

  def call(*args, **kwargs):
    data_to_change = vars_to_change.dict()
    data_needed = vars_needed.dict()
    out, changes = jitted_func(data_to_change, data_needed, *args, **kwargs)
    vars_to_change.assign(changes)
    return out

  return change_func_name(name=f_name, f=call) if f_name else call


def jit(obj_or_func, vars_to_change=None, vars_needed=None,
        static_argnums=None, static_argnames=None, device=None,
        backend=None, donate_argnums=(), inline=False, **kwargs):
  """JIT (Just-In-Time) Compilation for JAX backend.

  This function has the same ability to Just-In-Time compile a pure function,
  but it can also JIT compile a :py:class:`brainpy.DynamicSystem`, or a
  :py:class:`brainpy.Base` object, or a bounded method of a
  :py:class:`brainpy.Base` object.

  If you are using "numpy", please refer to the JIT compilation
  in NumPy backend `bp.math.numpy.jit() <brainpy.math.numpy.jit.rst>`_.

  Examples
  --------

  You can JIT a :py:class:`brainpy.DynamicSystem`

  >>> import brainpy as bp
  >>>
  >>> class LIF(bp.NeuGroup):
  >>>   pass
  >>> lif = bp.math.jit(LIF(10))

  You can JIT a :py:class:`brainpy.Base` object with ``__call__()`` implementation.

  >>> mlp = bp.dnn.MLP((10, 100, 10))
  >>> jit_mlp = bp.math.jit(mlp)

  You can also JIT a bounded method of a :py:class:`brainpy.Base` object.

  >>> class Hello(bp.dnn.Module):
  >>>   def __init__(self):
  >>>     super(Hello, self).__init__()
  >>>     self.a = 10.
  >>>     self.b = 2
  >>>   def transform(self):
  >>>     return self.a ** self.b
  >>>
  >>> test = Hello()
  >>> bp.math.jit(test.transform)

  Further, you can JIT a normal function, just used like in JAX.

  >>> @bp.math.jit
  >>> def selu(x, alpha=1.67, lmbda=1.05):
  >>>   return lmbda * bp.math.where(x > 0, x, alpha * bp.math.exp(x) - alpha)

  Parameters
  ----------
  obj_or_func : Base, function
    The instance of Base or a function.
  vars_to_change : optional, dict
    These variables will be changed in the function. If ``obj_or_func`` is an instance of Base,
    and ``vars_to_change`` is not provided, then all variables ``obj_or_func.vars()`` will be
    assumed to ``vars_to_change``.
  vars_needed : optional, dict
    The variables are needed to do computations, and will not be changed during the computation.
  static_argnums : optional, int, list, tuple, dict
    An optional int or collection of ints that specify which positional arguments to treat
    as static (compile-time constant).
  static_argnames : optional, str, list, tuple, dict
    An optional string or collection of strings specifying which named arguments to treat
    as static (compile-time constant). See the comment on ``static_argnums`` for details.
    If not provided but ``static_argnums`` is set, the default is based on calling
    ``inspect.signature(fun)`` to find corresponding named arguments.
  device: optional, Any
    This is an experimental feature and the API is likely to change.
    Optional, the Device the jitted function will run on. (Available devices
    can be retrieved via :py:func:`jax.devices`.) The default is inherited
    from XLA's DeviceAssignment logic and is usually to use
    ``jax.devices()[0]``.
  backend: optional, str, dict
    This is an experimental feature and the API is likely to change. Optional,
    a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or ``'tpu'``.
  donate_argnums: optional, int, dict, tuple, list
    Specify which arguments are "donated" to the computation.
    It is safe to donate arguments if you no longer need them once the
    computation has finished. In some cases XLA can make use of donated
    buffers to reduce the amount of memory needed to perform a computation,
    for example recycling one of your input buffers to store a result. You
    should not reuse buffers that you donate to a computation, JAX will raise
    an error if you try to. By default, no arguments are donated.
  inline: bool
    Specify whether this function should be inlined into enclosing
    jaxprs (rather than being represented as an application of the xla_call
    primitive with its own subjaxpr). Default False.

  Returns
  -------
  ds_of_func : Base, function
    A wrapped version of Base object or function,
    set up for just-in-time compilation.
  """
  from brainpy.simulation.brainobjects.base import DynamicSystem

  if isinstance(obj_or_func, DynamicSystem):
    if len(obj_or_func.steps):  # DynamicSystem has step functions

      # dynamical variables
      vars_to_change = (vars_to_change or obj_or_func.vars().unique())
      vars_needed = (vars_needed or ArrayCollector())

      # static arguments by num
      if static_argnums is None:
        static_argnums = {key: () for key in obj_or_func.steps.keys()}
      elif isinstance(static_argnums, int):
        static_argnums = {key: (static_argnums + 2,) for key in obj_or_func.steps.keys()}
      elif isinstance(static_argnums, (tuple, list)) and isinstance(static_argnums[0], int):
        static_argnums = {key: tuple(x + 2 for x in static_argnums) for key in obj_or_func.steps.keys()}
      assert isinstance(static_argnums, dict)

      # static arguments by name
      if static_argnames is None:
        static_argnames = {key: None for key in obj_or_func.steps.keys()}
      elif isinstance(static_argnames, str):
        static_argnames = {key: (static_argnames,) for key in obj_or_func.steps.keys()}
      elif isinstance(static_argnames, (tuple, list)) and isinstance(static_argnames[0], str):
        static_argnames = {key: static_argnames for key in obj_or_func.steps.keys()}
      assert isinstance(static_argnames, dict)

      # donate arguments by num
      if donate_argnums is None:
        donate_argnums = {key: () for key in obj_or_func.steps.keys()}
      elif isinstance(donate_argnums, int):
        donate_argnums = {key: (donate_argnums + 2,) for key in obj_or_func.steps.keys()}
      elif isinstance(donate_argnums, (tuple, list)):
        donate_argnums = {key: tuple(x + 2 for x in donate_argnums) for key in obj_or_func.steps.keys()}
      assert isinstance(donate_argnums, dict)

      # inline
      if not isinstance(inline, dict):
        inline = {key: inline for key in obj_or_func.steps.keys()}
      assert isinstance(inline, dict)

      # jit functions
      for key in obj_or_func.steps.keys():
        obj_or_func.steps[key] = _make_jit(vars_to_change=vars_to_change,
                                           vars_needed=vars_needed,
                                           func=obj_or_func.steps[key],
                                           static_argnums=static_argnums[key],
                                           static_argnames=static_argnames[key],
                                           device=device,
                                           backend=backend,
                                           donate_argnums=donate_argnums[key],
                                           inline=inline[key],
                                           f_name=key)
      return obj_or_func

  if callable(obj_or_func):
    if vars_to_change is not None:
      vars_to_change = vars_to_change
    elif isinstance(obj_or_func, Base):
      vars_to_change = obj_or_func.vars().unique()
    elif hasattr(obj_or_func, '__self__'):
      if isinstance(obj_or_func.__self__, Base):
        vars_to_change = obj_or_func.__self__.vars().unique()
    else:
      vars_to_change = ArrayCollector()

    if vars_needed is not None:
      vars_needed = vars_needed
    else:
      vars_needed = ArrayCollector()

    if len(vars_to_change) == 0 and len(vars_needed) == 0:  # pure function
      return jax.jit(obj_or_func,
                     static_argnums=static_argnums,
                     static_argnames=static_argnames,
                     device=device,
                     backend=backend,
                     donate_argnums=donate_argnums,
                     inline=inline)

    else:  # Base object which implements __call__, or bounded method of Base object

      static_argnums = tuple(x + 2 for x in sorted(static_argnums or ()))
      return _make_jit(vars_to_change=vars_to_change,
                       vars_needed=vars_needed,
                       func=obj_or_func,
                       static_argnums=static_argnums,
                       static_argnames=static_argnames,
                       device=device,
                       backend=backend,
                       donate_argnums=donate_argnums,
                       inline=inline)

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable '
                              f'function, but we got {type(obj_or_func)}.')


def _make_vmap(func, dyn_vars, rand_vars, in_axes, out_axes,
               batch_idx, axis_name, reduce_func, f_name=None):
  @functools.partial(jax.vmap, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)
  def vmapped_func(dyn_data, rand_data, *args, **kwargs):
    dyn_vars.assign(dyn_data)
    rand_vars.assign(rand_data)
    out = func(*args, **kwargs)
    dyn_changes = dyn_vars.dict()
    rand_changes = rand_vars.dict()
    return out, dyn_changes, rand_changes

  def call(*args, **kwargs):
    dyn_data = dyn_vars.dict()
    n = args[batch_idx[0]].shape[batch_idx[1]]
    rand_data = {key: val.split_keys(n) for key, val in rand_vars.items()}
    out, dyn_changes, rand_changes = vmapped_func(dyn_data, rand_data, *args, **kwargs)
    for key, v in dyn_changes.items():
      dyn_vars[key] = reduce_func(v)
    for key, v in rand_changes.items():
      rand_vars[key] = reduce_func(v)
    return out

  return change_func_name(name=f_name, f=call) if f_name else call


def vmap(obj_or_func, vars=None, vars_batched=None,
         in_axes=0, out_axes=0, axis_name=None, reduce_func=None):
  """Vectorization compilation in JAX backend.

  Vectorized compile a function or a module to run in parallel on a single device.

  Examples
  --------

  Parameters
  ----------
  obj_or_func : Base, function
    The function or the module to compile.
  vars_needed : dict
  vars_batched : dict
  in_axes : optional, int, tuple/list/dict
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
  obj_or_func : Base, function
    Batched/vectorized version of ``obj_or_func`` with arguments that correspond to
    those of ``obj_or_func``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``obj_or_func``, but
    with extra array axes at positions indicated by ``out_axes``.

  """
  from brainpy.simulation.brainobjects.base import DynamicSystem

  if isinstance(obj_or_func, DynamicSystem):
    if len(obj_or_func.steps):  # DynamicSystem has step functions

      # dynamical variables
      vars = (vars or obj_or_func.vars().unique())
      dyn_vars, rand_vars = ArrayCollector(), ArrayCollector()
      for key, val in vars.items():
        if isinstance(val, RandomState):
          rand_vars[key] = val
        else:
          dyn_vars[key] = val

      # in axes
      if in_axes is None:
        in_axes = {key: (None, 0) for key in obj_or_func.steps.keys()}
      elif isinstance(in_axes, int):
        in_axes = {key: (None, 0, in_axes) for key in obj_or_func.steps.keys()}
      elif isinstance(in_axes, (tuple, list)):
        in_axes = {key: (None, 0) + tuple(in_axes) for key in obj_or_func.steps.keys()}
      elif isinstance(in_axes, dict):
        keys = list(obj_or_func.steps.keys())
        if keys[0] not in in_axes:
          in_axes = {key: (None, 0, in_axes) for key in keys}
        else:
          in_axes = {key: (None, 0) + tuple(in_axes[key]) for key in keys}
      assert isinstance(in_axes, dict)

      # batch size index
      batch_idx = {}
      for key, axes in in_axes.items():
        for i, axis in enumerate(axes[2:]):
          if axis is not None:
            batch_idx[key] = (i, axis)
            break
        else:
          raise ValueError(f'Found no batch axis: {axes}.')

      # out axes
      if out_axes is None:
        out_axes = {key: 0 for key in obj_or_func.steps.keys()}
      elif isinstance(out_axes, int):
        out_axes = {key: out_axes for key in obj_or_func.steps.keys()}
      elif isinstance(out_axes, (tuple, list)):
        out_axes = {key: tuple(out_axes) + (0, 0) for key in obj_or_func.steps.keys()}
      elif isinstance(out_axes, dict):
        keys = list(obj_or_func.steps.keys())
        if keys[0] not in out_axes:
          out_axes = {key: (out_axes, 0, 0) for key in keys}
        else:
          out_axes = {key: tuple(out_axes[key]) + (0, 0) for key in keys}
      assert isinstance(out_axes, dict)

      # reduce_func
      if reduce_func is None:
        reduce_func = lambda x: x.mean(axis=0)

      # vectorized map functions
      for key in obj_or_func.steps.keys():
        obj_or_func.steps[key] = _make_vmap(func=obj_or_func.steps[key],
                                            dyn_vars=dyn_vars,
                                            rand_vars=rand_vars,
                                            in_axes=in_axes[key],
                                            out_axes=out_axes[key],
                                            axis_name=axis_name,
                                            batch_idx=batch_idx[key],
                                            reduce_func=reduce_func,
                                            f_name=key)

      return obj_or_func

  if callable(obj_or_func):
    if vars is not None:
      vars = vars
    elif isinstance(obj_or_func, Base):  # Base has '__call__()' implementation
      vars = obj_or_func.vars().unique()
    elif hasattr(obj_or_func, '__self__'):
      if isinstance(obj_or_func.__self__, Base):
        vars = obj_or_func.__self__.vars().unique()

    if vars is None:
      return jax.vmap(obj_or_func,
                      in_axes=in_axes,
                      out_axes=out_axes,
                      axis_name=axis_name)

    else:
      # dynamical variables
      dyn_vars, rand_vars = ArrayCollector(), ArrayCollector()
      for key, val in vars.items():
        if isinstance(val, RandomState):
          rand_vars[key] = val
        else:
          dyn_vars[key] = val

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
      return _make_vmap(func=obj_or_func,
                        dyn_vars=dyn_vars,
                        rand_vars=rand_vars,
                        in_axes=in_axes,
                        out_axes=out_axes,
                        axis_name=axis_name,
                        batch_idx=batch_idx,
                        reduce_func=reduce_func)

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable '
                              f'function, but we got {type(obj_or_func)}.')


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


def pmap(obj_or_func, vars=None, axis_name=None, in_axes=0, out_axes=0, static_broadcasted_argnums=(),
         devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None,
         reduce_func=None):
  """Parallel compilation in JAX backend.

  Parallel compile a function or a module to run on multiple devices in parallel.

  Parameters
  ----------
  obj_or_func
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
  from brainpy.simulation.brainobjects.base import DynamicSystem

  if isinstance(obj_or_func, DynamicSystem):
    if len(obj_or_func.steps):  # DynamicSystem has step functions

      # dynamical variables
      all_vars = (vars or obj_or_func.vars().unique())
      dyn_vars = ArrayCollector()
      rand_vars = ArrayCollector()
      for key, val in all_vars.items():
        if isinstance(val, RandomState):
          rand_vars[key] = val
        else:
          dyn_vars[key] = val

      # reduce function
      if reduce_func is None:
        reduce_func = jnp.concatenate

      # static broadcast-ed arguments
      if static_broadcasted_argnums is None:
        static_broadcasted_argnums = ()
      elif isinstance(static_broadcasted_argnums, int):
        static_broadcasted_argnums = (static_broadcasted_argnums + 2,)
      elif isinstance(static_broadcasted_argnums, (tuple, list)):
        static_broadcasted_argnums = tuple(argnum + 2 for argnum in static_broadcasted_argnums)
      assert isinstance(static_broadcasted_argnums, (tuple, list))

      # jit functions
      for key in obj_or_func.steps.keys():
        step = obj_or_func.steps[key]
        obj_or_func.steps[key] = _make_pmap(dyn_vars=dyn_vars,
                                            rand_vars=rand_vars,
                                            func=step,
                                            axis_name=axis_name,
                                            in_axes=in_axes,
                                            out_axes=out_axes,
                                            static_broadcasted_argnums=static_broadcasted_argnums,
                                            devices=devices,
                                            backend=backend,
                                            axis_size=axis_size,
                                            donate_argnums=donate_argnums,
                                            global_arg_shapes=global_arg_shapes,
                                            reduce_func=reduce_func,
                                            f_name=key)
      return obj_or_func

  if callable(obj_or_func):
    if vars is not None:
      vars = vars
    elif isinstance(obj_or_func, Base):  # Base has '__call__()' implementation
      vars = obj_or_func.vars().unique()
    elif hasattr(obj_or_func, '__self__'):
      if isinstance(obj_or_func.__self__, Base):
        vars = obj_or_func.__self__.vars().unique()

    if vars is None:
      return jax.pmap(obj_or_func,
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
      dyn_vars = ArrayCollector()
      rand_vars = ArrayCollector()
      for key, val in vars.items():
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
      obj_or_func.__call__ = _make_pmap(dyn_vars=dyn_vars,
                                        rand_vars=rand_vars,
                                        func=obj_or_func,
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
      return obj_or_func

  else:
    raise errors.BrainPyError(f'Only support instance of {Base.__name__}, or a callable function, '
                              f'but we got {type(obj_or_func)}.')
