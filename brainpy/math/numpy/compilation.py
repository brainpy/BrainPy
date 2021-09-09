# -*- coding: utf-8 -*-

import logging

from brainpy import errors

try:
  import numba
  from brainpy.math.numpy import ast2numba
except ModuleNotFoundError:
  ast2numba = None
  numba = None


__all__ = [
  'jit',
  'vmap',
  'pmap',
]

logger = logging.getLogger('brainpy.math.numpy.compilation')


def jit(obj_or_fun, nopython=True, fastmath=True, parallel=False, nogil=False,
        forceobj=False, looplift=True, error_model='python', inline='never',
        boundscheck=None, show_code=False, **kwargs):
  """Just-In-Time (JIT) Compilation in NumPy backend.

  JIT compilation in NumPy backend relies on `Numba <http://numba.pydata.org/>`_. However,
  in BrainPy, `bp.math.numpy.jit()` can apply to class objects, especially the instance
  of :py:class:`brainpy.DynamicalSystem`.

  If you are using JAX backend, please refer to the JIT compilation in
  JAX backend `bp.math.jax.jit() <brainpy.math.jax.jit.rst>`_.

  Parameters
  ----------
  obj_or_fun : callable, Base
    The function or the base model to jit compile.

  nopython : bool
    Set to True to disable the use of PyObjects and Python API
    calls. Default value is True.

  fastmath : bool
    In certain classes of applications strict IEEE 754 compliance
    is less important. As a result it is possible to relax some
    numerical rigour with view of gaining additional performance.
    The way to achieve this behaviour in Numba is through the use
    of the ``fastmath`` keyword argument.

  parallel : bool
    Enables automatic parallelization (and related optimizations) for
    those operations in the function known to have parallel semantics.

  nogil : bool
    Whenever Numba optimizes Python code to native code that only
    works on native types and variables (rather than Python objects),
    it is not necessary anymore to hold Python’s global interpreter
    lock (GIL). Numba will release the GIL when entering such a
    compiled function if you passed ``nogil=True``.

  forceobj: bool
    Set to True to force the use of PyObjects for every value.
    Default value is False.

  looplift: bool
    Set to True to enable jitting loops in nopython mode while
    leaving surrounding code in object mode. This allows functions
    to allocate NumPy arrays and use Python objects, while the
    tight loops in the function can still be compiled in nopython
    mode. Any arrays that the tight loop uses should be created
    before the loop is entered. Default value is True.

  error_model: str
    The error-model affects divide-by-zero behavior.
    Valid values are 'python' and 'numpy'. The 'python' model
    raises exception.  The 'numpy' model sets the result to
    *+/-inf* or *nan*. Default value is 'python'.

  inline: str or callable
    The inline option will determine whether a function is inlined
    at into its caller if called. String options are 'never'
    (default) which will never inline, and 'always', which will
    always inline. If a callable is provided it will be called with
    the call expression node that is requesting inlining, the
    caller's IR and callee's IR as arguments, it is expected to
    return Truthy as to whether to inline.
    NOTE: This inlining is performed at the Numba IR level and is in
    no way related to LLVM inlining.

  boundscheck: bool or None
    Set to True to enable bounds checking for array indices. Out
    of bounds accesses will raise IndexError. The default is to
    not do bounds checking. If False, bounds checking is disabled,
    out of bounds accesses can produce garbage results or segfaults.
    However, enabling bounds checking will slow down typical
    functions, so it is recommended to only use this flag for
    debugging. You can also set the NUMBA_BOUNDSCHECK environment
    variable to 0 or 1 to globally override this flag. The default
    value is None, which under normal execution equates to False,
    but if debug is set to True then bounds checking will be
    enabled.

  show_code : bool
    Debugging.

  kwargs

  Returns
  -------
  res : callable, Base
    The jitted objects.
  """

  # checking
  if ast2numba is None or numba is None:
    raise errors.PackageMissingError('JIT compilation in numpy backend need Numba. '
                                     'Please install numba via: \n\n'
                                     '>>> pip install numba\n'
                                     '>>> # or \n'
                                     '>>> conda install numba')
  return ast2numba.jit(obj_or_fun, show_code=show_code,
                       nopython=nopython, fastmath=fastmath, parallel=parallel, nogil=nogil,
                       forceobj=forceobj, looplift=looplift, error_model=error_model,
                       inline=inline, boundscheck=boundscheck)


def vmap(obj_or_func, *args, **kwargs):
  """Vectorization Compilation in NumPy backend.

  Vectorization compilation is not implemented in NumPy backend.
  Please refer to the vectorization compilation in JAX backend
  `bp.math.jax.vmap() <brainpy.math.jax.vmap.rst>`_.

  Parameters
  ----------
  obj_or_func
  args
  kwargs

  Returns
  -------

  """
  _msg = 'Vectorize compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)


def pmap(obj_or_func, *args, **kwargs):
  """Parallel Compilation in NumPy backend.

  Parallel compilation is not implemented in NumPy backend.
  Please refer to the parallel compilation in JAX backend
  `bp.math.jax.pmap() <brainpy.math.jax.pmap.rst>`_.

  Parameters
  ----------
  obj_or_func
  args
  kwargs

  Returns
  -------

  """
  _msg = 'Parallel compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)
