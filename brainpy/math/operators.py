# -*- coding: utf-8 -*-

from typing import Union, Sequence, Callable

import jax.numpy as jnp
from jax import jit, vmap
from jax import ops as jops
from jax.abstract_arrays import ShapedArray

from brainpy.errors import PackageMissingError, MathError
from brainpy.math import setting
from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import as_device_array, _remove_jaxarray
from brainpy.types import Shape

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  # pre-to-post
  'pre2post_sum',
  'pre2post_prod',
  'pre2post_max',
  'pre2post_min',
  'pre2post_mean',

  # pre-to-syn
  'pre2syn',

  # syn-to-post
  'syn2post_sum', 'syn2post',
  'syn2post_prod',
  'syn2post_max',
  'syn2post_min',
  'syn2post_mean',
  'syn2post_softmax',

  # pre-to-post event operator
  'pre2post_event_sum',
  'pre2post_event_prod',

  # others
  'sparse_matmul',

  # numba operators
  'register_op'
]

_BRAINPYLIB_MINIMAL_VERSION = '0.0.4'

_pre2post = vmap(lambda pre_ids, pre_vs: pre_vs[pre_ids].sum(), in_axes=(0, None))
_pre2syn = vmap(lambda pre_id, pre_vs: pre_vs[pre_id], in_axes=(0, None))
_jit_seg_sum = jit(jops.segment_sum, static_argnums=(2, 3))
_jit_seg_prod = jit(jops.segment_prod, static_argnums=(2, 3))
_jit_seg_max = jit(jops.segment_max, static_argnums=(2, 3))
_jit_seg_min = jit(jops.segment_min, static_argnums=(2, 3))


def _check_brainpylib(ops_name):
  if brainpylib is not None:
    if brainpylib.__version__ < _BRAINPYLIB_MINIMAL_VERSION:
      raise PackageMissingError(
        f'"{ops_name}" operator need "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}". \n'
        f'Please install it through:\n\n'
        f'>>> pip install brainpylib>={_BRAINPYLIB_MINIMAL_VERSION} -U'
      )
  else:
    raise PackageMissingError(
      f'"brainpylib" must be installed when the user '
      f'wants to use "{ops_name}" operator. \n'
      f'Please install "brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}" through:\n\n'
      f'>>> pip install brainpylib>={_BRAINPYLIB_MINIMAL_VERSION}'
    )


def register_op(
    op_name: str,
    cpu_func: Callable,
    gpu_func: Callable = None,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]] = None,
    apply_cpu_func_to_gpu: bool = False
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  op_name: str
    Name of the operators.
  cpu_func: Callble
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.
  gpu_func: Callable, default = None
    A callable cuda-jitted kernel running on GPU.
  out_shapes: Callable, ShapedArray, Sequence[ShapedArray], default = None
    Outputs shapes of target function. `out_shapes` can be a `ShapedArray` or
    a sequence of `ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `ShapedArray`.
  apply_cpu_func_to_gpu: bool, default = False
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.

  Returns
  -------
  A jitable JAX function.
  """
  _check_brainpylib(register_op.__name__)
  f = brainpylib.register_op(op_name, cpu_func, gpu_func, out_shapes, apply_cpu_func_to_gpu)

  def fixed_op(*inputs):
    inputs = tuple([i.value if isinstance(i, JaxArray) else i for i in inputs])
    return f(*inputs)

  return fixed_op


def pre2post_event_sum(events, pre2post, post_num, values=1.):
  """The pre-to-post synaptic computation with event-driven summation.

  When ``values`` is a scalar, this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] += values

  When ``values`` is a vector (with the length of ``len(post_ids)``),
  this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)

    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] += values[j]


  Parameters
  ----------
  events: JaxArray, jax.numpy.ndarray, Variable
    The events, must be bool.
  pre2post: tuple of JaxArray, tuple of jax.numpy.ndarray
    A tuple contains the connection information of pre-to-post.
  post_num: int
    The number of post-synaptic group.
  values: float, JaxArray, jax.numpy.ndarray
    The value to make summation.

  Returns
  -------
  out: JaxArray, jax.numpy.ndarray
    A tensor with the shape of ``post_num``.
  """
  _check_brainpylib(pre2post_event_sum.__name__)
  indices, idnptr = pre2post
  events = as_device_array(events)
  indices = as_device_array(indices)
  idnptr = as_device_array(idnptr)
  values = as_device_array(values)
  return brainpylib.event_sum(events, (indices, idnptr), post_num, values)


def pre2post_event_prod(events, pre2post, post_num, values=1.):
  """The pre-to-post synaptic computation with event-driven production.

  When ``values`` is a scalar, this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.ones(post_num)
    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] *= values

  When ``values`` is a vector (with the length of ``len(post_ids)``),
  this function is equivalent to

  .. highlight:: python
  .. code-block:: python

    post_val = np.ones(post_num)

    post_ids, idnptr = pre2post
    for i in range(pre_num):
      if events[i]:
        for j in range(idnptr[i], idnptr[i+1]):
          post_val[post_ids[i]] *= values[j]


  Parameters
  ----------
  events: JaxArray, jax.numpy.ndarray, Variable
    The events, must be bool.
  pre2post: tuple of JaxArray, tuple of jax.numpy.ndarray
    A tuple contains the connection information of pre-to-post.
  post_num: int
    The number of post-synaptic group.
  values: float, JaxArray, jax.numpy.ndarray
    The value to make summation.

  Returns
  -------
  out: JaxArray, jax.numpy.ndarray
    A tensor with the shape of ``post_num``.
  """
  _check_brainpylib(pre2post_event_prod.__name__)
  indices, idnptr = pre2post
  events = as_device_array(events)
  indices = as_device_array(indices)
  idnptr = as_device_array(idnptr)
  values = as_device_array(values)
  return brainpylib.event_prod(events, (indices, idnptr), post_num, values)


def _raise_pre_ids_is_none(pre_ids):
  if pre_ids is None:
    raise MathError(f'pre2post synaptic computation needs "pre_ids" '
                    f'when providing heterogeneous "pre_values" '
                    f'(brainpy.math.ndim(pre_values) != 0).')


def pre2post_sum(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic summation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] += pre_values[pre_ids[i]]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.
  pre_ids: optional, jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num, dtype=setting.float_)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].add(pre_values)


def pre2post_prod(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic production.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] *= pre_values[pre_ids[i]]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num, dtype=setting.float_)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].multiply(pre_values)


def pre2post_min(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic minimization.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] = np.minimum(post_val[j], pre_values[pre_ids[i]])

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num, dtype=setting.float_)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].min(pre_values)


def pre2post_max(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic maximization.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

     post_val = np.zeros(post_num)
     for i, j in zip(pre_ids, post_ids):
       post_val[j] = np.maximum(post_val[j], pre_values[pre_ids[i]])

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num, dtype=setting.float_)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) != 0:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre_values[pre_ids]
  return out.at[post_ids].max(pre_values)


def pre2post_mean(pre_values, post_num, post_ids, pre_ids=None):
  """The pre-to-post synaptic mean computation.

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic values.
  pre_ids: jax.numpy.ndarray, JaxArray
    The connected pre-synaptic neuron ids.
  post_ids: jax.numpy.ndarray, JaxArray
    The connected post-synaptic neuron ids.
  post_num: int
    Output dimension. The number of post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The value with the size of post-synaptic neurons.
  """
  out = jnp.zeros(post_num, dtype=setting.float_)
  pre_values = as_device_array(pre_values)
  post_ids = as_device_array(post_ids)
  if jnp.ndim(pre_values) == 0:
    # return out.at[post_ids].set(pre_values)
    return out.at[jnp.unique(post_ids)].set(pre_values)
  else:
    _raise_pre_ids_is_none(pre_ids)
    pre_ids = as_device_array(pre_ids)
    pre_values = pre2syn(pre_values, pre_ids)
    return syn2post_mean(pre_values, post_ids, post_num)


def pre2syn(pre_values, pre_ids):
  """The pre-to-syn computation.

  Change the pre-synaptic data to the data with the dimension of synapses.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    syn_val = np.zeros(len(pre_ids))
    for syn_i, pre_i in enumerate(pre_ids):
      syn_val[i] = pre_values[pre_i]

  Parameters
  ----------
  pre_values: float, jax.numpy.ndarray, JaxArray, Variable
    The pre-synaptic value.
  pre_ids: jax.numpy.ndarray, JaxArray
    The pre-synaptic neuron index.

  Returns
  -------
  syn_val: jax.numpy.ndarray, JaxArray
    The synaptic value.
  """
  pre_values = as_device_array(pre_values)
  pre_ids = as_device_array(pre_ids)
  if jnp.ndim(pre_values) == 0:
    return jnp.ones(len(pre_ids), dtype=pre_values.dtype) * pre_values
  else:
    return _pre2syn(pre_ids, pre_values)


def syn2post_sum(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post summation computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] += syn_values[syn_i]

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids.
  post_num: int
    The number of the post-synaptic neurons.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)


syn2post = syn2post_sum


def syn2post_prod(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post product computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] *= syn_values[syn_i]

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_prod(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_max(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post maximum computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] = np.maximum(post_val[post_i], syn_values[syn_i])

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_max(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_min(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post minimization computation.

  This function is equivalent to:

  .. highlight:: python
  .. code-block:: python

    post_val = np.zeros(post_num)
    for syn_i, post_i in enumerate(post_ids):
      post_val[post_i] = np.minimum(post_val[post_i], syn_values[syn_i])

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  return _jit_seg_min(syn_values, post_ids, post_num, indices_are_sorted)


def syn2post_mean(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post mean computation.

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  nominator = _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)
  denominator = _jit_seg_sum(jnp.ones_like(syn_values), post_ids, post_num, indices_are_sorted)
  return jnp.nan_to_num(nominator / denominator)


def syn2post_softmax(syn_values, post_ids, post_num: int, indices_are_sorted=True):
  """The syn-to-post softmax computation.

  Parameters
  ----------
  syn_values: jax.numpy.ndarray, JaxArray, Variable
    The synaptic values.
  post_ids: jax.numpy.ndarray, JaxArray
    The post-synaptic neuron ids. If ``post_ids`` is generated by
    ``brainpy.conn.TwoEndConnector``, then it has sorted indices.
    Otherwise, this function cannot guarantee indices are sorted.
    You's better set ``indices_are_sorted=False``.
  post_num: int
    The number of the post-synaptic neurons.
  indices_are_sorted: whether ``post_ids`` is known to be sorted.

  Returns
  -------
  post_val: jax.numpy.ndarray, JaxArray
    The post-synaptic value.
  """
  post_ids = as_device_array(post_ids)
  syn_values = as_device_array(syn_values)
  if syn_values.dtype == jnp.bool_:
    syn_values = jnp.asarray(syn_values, dtype=jnp.int32)
  syn_maxs = _jit_seg_max(syn_values, post_ids, post_num, indices_are_sorted)
  syn_values = syn_values - syn_maxs[post_ids]
  syn_values = jnp.exp(syn_values)
  normalizers = _jit_seg_sum(syn_values, post_ids, post_num, indices_are_sorted)
  softmax = syn_values / normalizers[post_ids]
  return jnp.nan_to_num(softmax)


def _matmul_with_left_sparse(sparse: Sequence,
                             dense: Union[JaxArray, jnp.ndarray],
                             shape: int):
  r"""Matrix multiplication with sparse matrix on the left.

  .. math::

    Y = M_{\mathrm{sparse}} @ M_{\mathrm{dense}}

  Parameters
  ----------
  sparse: tuple, list
    The sparse matrix with shape of :math:`(N, M)`.
  dense: JaxArray, jnp.ndarray
    The dense matrix with the shape of :math:`(M, K)`.
  shape: int
    The dimension of :math:`N`.

  Returns
  -------
  matrix
    A tensor the the shape of :math:`(N, K)`.
  """
  assert dense.ndim in [1, 2], 'Dense matrix must be a one- or two-dimensional matrix.'
  values, (rows, cols) = sparse
  values = _remove_jaxarray(values)
  rows = _remove_jaxarray(rows)
  cols = _remove_jaxarray(cols)
  dense = _remove_jaxarray(dense)
  B = dense.take(cols, axis=0)
  if B.ndim == 2:
    prod = B * jnp.reshape(values, (-1, 1))
  else:
    prod = B * values
  return jops.segment_sum(prod, rows, shape)


def _matmul_with_right_sparse(dense, sparse, shape):
  r"""Matrix multiplication with sparse matrix on the left.

  .. math::

    Y = M_{\mathrm{dense}} @ M_{\mathrm{sparse}}

  Parameters
  ----------
  dense: JaxArray, jnp.ndarray
    The dense matrix with the shape of :math:`(N, M)`.
  sparse: tuple, list
    The sparse matrix with shape of :math:`(M, K)`.
  shape: int
    The dimension of :math:`K`.

  Returns
  -------
  matrix
    A tensor the the shape of :math:`(N, K)`.
  """
  assert dense.ndim in [1, 2], 'Dense matrix must be a one- or two-dimensional matrix.'
  values, (rows, cols) = sparse
  values = _remove_jaxarray(values)
  rows = _remove_jaxarray(rows)
  cols = _remove_jaxarray(cols)
  dense = _remove_jaxarray(dense)
  if dense.ndim == 2:
    A = dense[:, rows]
    prod = (A * values).T
    res = jops.segment_sum(prod, cols, shape).T
  else:
    prod = dense[rows] * values
    res = jops.segment_sum(prod, cols, shape)
  return res


def sparse_matmul(A, B, shape: Shape):
  r"""Sparse matrix multiplication.

  .. math::

     y = A @ B

  where :math:`A` or :math:`B` is a sparse matrix.
  :math:`A` and :math:`B` cannot be both sparse.

  Examples
  --------

  >>> import brainpy.math as bm

  1. when the left matrix :math:`A` is a sparse matrix with the shape of :math:`(N, M)`,
     we should provide :math:`N` as the ``shape`` in the ``brainpy.math.sparse_matmul``
     function.

  >>> # A is a sparse matrix (3, 4):
  >>> #   [[0, 2, 0, 4],
  >>> #    [1, 0, 0, 0],
  >>> #    [0, 3, 0, 2]]
  >>> values = bm.asarray([2, 4, 1, 3, 2])
  >>> rows = bm.asarray([0, 0, 1, 2, 2])
  >>> cols = bm.asarray([1, 3, 0, 1, 3])
  >>> B = bm.arange(4)
  >>> bm.sparse_matmul([values, (rows, cols)], B, 3)
  JaxArray([14,  0,  9], dtype=int32)
  >>> B = bm.random.rand(4, 3)
  >>> bm.sparse_matmul([values, (rows, cols)], B, 3)
  JaxArray([[3.8331761 , 1.3708692 , 4.510223  ],
            [0.9960836 , 0.37550318, 0.7370341 ],
            [2.3700516 , 0.7574289 , 4.1124535 ]], dtype=float32)

  2. when the right matrix :math:`B` is a sparse matrix with the shape of :math:`(M, K)`,
     we should provide :math:`K` as the ``shape`` in the ``brainpy.math.sparse_matmul``
     function.

  >>> A = bm.arange(3)
  >>> bm.sparse_matmul(A, [values, (rows, cols)], 4)
  JaxArray([1, 6, 0, 4], dtype=int32)
  >>> A = bm.random.rand(2, 3)
  JaxArray([[0.438388  , 1.4346815 , 0.        , 2.361964  ],
            [0.9171978 , 1.1214957 , 0.        , 0.90534496]],  dtype=float32)

  Parameters
  ----------
  A: tensor, sequence
    The dense or sparse matrix with the shape of :math:`(N, M)`.
  B: tensor, sequence
    The dense or sparse matrix with the shape of :math:`(M, K)`.
  shape: int
    The dimension of :math:`N` when ``A`` is sparse, or
    the dimension of :math:`K` when ``B`` is sparse.

  Returns
  -------
  results: JaxArray, jnp.ndarray
    The tensor with the shape of :math:`(N, K)`.
  """
  if isinstance(A, (tuple, list)):
    assert isinstance(B, (JaxArray, jnp.ndarray)), ('A and B cannot be both sparse. \n'
                                                    f'A:\n{A}\n'
                                                    f'B:\n{B}')
    return _matmul_with_left_sparse(A, B, shape)
  else:
    assert isinstance(B, (tuple, list)), ('A and B cannot be both dense. \n'
                                          f'A:\n{A}\n'
                                          f'B:\n{B}')
    return _matmul_with_right_sparse(A, B, shape)
