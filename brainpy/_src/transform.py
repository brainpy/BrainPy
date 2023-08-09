# -*- coding: utf-8 -*-

import functools
from typing import Union, Optional, Dict, Sequence

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from brainpy import tools, math as bm
from brainpy._src.context import share
from brainpy._src.dynsys import DynamicalSystem
from brainpy.check import is_float, is_integer
from brainpy.types import PyTree

__all__ = [
  'LoopOverTime',
]


class LoopOverTime(DynamicalSystem):
  """Transform a single step :py:class:`~.DynamicalSystem`
  into a multiple-step forward propagation :py:class:`~.BrainPyObject`.

  .. note::

     This object transforms a :py:class:`~.DynamicalSystem` into a :py:class:`~.BrainPyObject`.

     If the `target` has a batching mode, before sending the data into the wrapped object,
     reset the state (``.reset_state(batch_size)``) with the same batch size as in the given data.


  For more flexible customization, we recommend users to use :py:func:`~.for_loop`,
  or :py:class:`~.DSRunner`.

  Examples
  --------

  This model can be used for network training:

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>>
  >>> n_time, n_batch, n_in = 30, 128, 100
  >>> model = bp.Sequential(l1=bp.layers.RNNCell(n_in, 20),
  >>>                       l2=bm.relu,
  >>>                       l3=bp.layers.RNNCell(20, 2))
  >>> over_time = bp.LoopOverTime(model, data_first_axis='T')
  >>> over_time.reset_state(n_batch)
  (30, 128, 2)
  >>>
  >>> hist_l3 = over_time(bm.random.rand(n_time, n_batch, n_in))
  >>> print(hist_l3.shape)
  >>>
  >>> # monitor the "l1" layer state
  >>> over_time = bp.LoopOverTime(model, out_vars=model['l1'].state, data_first_axis='T')
  >>> over_time.reset_state(n_batch)
  >>> hist_l3, hist_l1 = over_time(bm.random.rand(n_time, n_batch, n_in))
  >>> print(hist_l3.shape)
  (30, 128, 2)
  >>> print(hist_l1.shape)
  (30, 128, 20)

  It is also able to used in brain simulation models:

  .. plot::
     :include-source: True

     >>> import brainpy as bp
     >>> import brainpy.math as bm
     >>> import matplotlib.pyplot as plt
     >>>
     >>> hh = bp.neurons.HH(1)
     >>> over_time = bp.LoopOverTime(hh, out_vars=hh.V)
     >>>
     >>> # running with a given duration
     >>> _, potentials = over_time(100.)
     >>> plt.plot(bm.as_numpy(potentials), label='with given duration')
     >>>
     >>> # running with the given inputs
     >>> _, potentials = over_time(bm.ones(1000) * 5)
     >>> plt.plot(bm.as_numpy(potentials), label='with given inputs')
     >>> plt.legend()
     >>> plt.show()


  Parameters
  ----------
  target: DynamicalSystem
    The target to transform.
  no_state: bool
    Denoting whether the `target` has the shared argument or not.

    - For ANN layers which are no_state, like :py:class:`~.Dense` or :py:class:`~.Conv2d`,
      set `no_state=True` is high efficiently. This is because :math:`Y[t]` only relies on
      :math:`X[t]`, and it is not necessary to calculate :math:`Y[t]` step-bt-step.
      For this case, we reshape the input from `shape = [T, N, *]` to `shape = [TN, *]`,
      send data to the object, and reshape output to `shape = [T, N, *]`.
      In this way, the calculation over different time is parralelized.

  out_vars: PyTree
    The variables to monitor over the time loop.
  t0: float, optional
    The start time to run the system. If None, ``t`` will be no longer generated in the loop.
  i0: int, optional
    The start index to run the system. If None, ``i`` will be no longer generated in the loop.
  dt: float
    The time step.
  shared_arg: dict
    The shared arguments across the nodes.
    For instance, `shared_arg={'fit': False}` for the prediction phase.
  data_first_axis: str
    Denoting the type of the first axis of input data.
    If ``'T'``, we treat the data as `(time, ...)`.
    If ``'B'``, we treat the data as `(batch, time, ...)` when the `target` is in Batching mode.
    Default is ``'T'``.
  name: str
    The transformed object name.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      out_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      no_state: bool = False,
      t0: Optional[float] = 0.,
      i0: Optional[int] = 0,
      dt: Optional[float] = None,
      shared_arg: Optional[Dict] = None,
      data_first_axis: str = 'T',
      name: str = None,
      jit: bool = True,
      remat: bool = False,
  ):
    super().__init__(name=name)
    assert data_first_axis in ['B', 'T']
    is_integer(i0, 'i0', allow_none=True)
    is_float(t0, 't0', allow_none=True)
    is_float(dt, 'dt', allow_none=True)
    dt = share.dt if dt is None else dt
    if shared_arg is None:
      shared_arg = dict(dt=dt)
    else:
      assert isinstance(shared_arg, dict)
      shared_arg['dt'] = dt
    self.dt = dt
    self._t0 = t0
    self._i0 = i0
    self.t0 = None if t0 is None else bm.Variable(bm.as_jax(t0))
    self.i0 = None if i0 is None else bm.Variable(bm.as_jax(i0))

    self.jit = jit
    self.remat = remat
    self.shared_arg = shared_arg
    self.data_first_axis = data_first_axis
    self.target = target
    if not isinstance(target, DynamicalSystem):
      raise TypeError(f'Must be instance of {DynamicalSystem.__name__}, '
                      f'but we got {type(target)}')
    self.no_state = no_state
    self.out_vars = out_vars
    if out_vars is not None:
      out_vars, _ = tree_flatten(out_vars, is_leaf=lambda s: isinstance(s, bm.Variable))
      for v in out_vars:
        if not isinstance(v, bm.Variable):
          raise TypeError('out_vars must be a PyTree of Variable.')

  def __call__(
      self,
      duration_or_xs: Union[float, PyTree],
  ):
    """Forward propagation along the time or inputs.

    Parameters
    ----------
    duration_or_xs: float, PyTree
      If `float`, it indicates a running duration.
      If a PyTree, it is the given inputs.

    Returns
    -------
    out: PyTree
      The accumulated outputs over time.
    """
    # inputs
    if isinstance(duration_or_xs, float):
      shared = tools.DotDict()
      if self.t0 is not None:
        shared['t'] = jnp.arange(0, duration_or_xs, self.dt) + self.t0.value
      if self.i0 is not None:
        shared['i'] = jnp.arange(0, shared['t'].shape[0]) + self.i0.value
      xs = None
      if self.no_state:
        raise ValueError('Under the `no_state=True` setting, input cannot be a duration.')
      length = shared['t'].shape

    else:
      inp_err_msg = ('\n'
                     'Input should be a Array PyTree with the shape '
                     'of (B, T, ...) or (T, B, ...) with `data_first_axis="T"`, '
                     'where B the batch size and T the time length.')
      xs, tree = tree_flatten(duration_or_xs, lambda a: isinstance(a, bm.Array))
      if self.target.mode.is_child_of(bm.BatchingMode):
        b_idx, t_idx = (1, 0) if self.data_first_axis == 'T' else (0, 1)

        try:
          batch = tuple(set([x.shape[b_idx] for x in xs]))
        except (AttributeError, IndexError) as e:
          raise ValueError(inp_err_msg) from e
        if len(batch) != 1:
          raise ValueError('\n'
                           'Input should be a Array PyTree with the same batch dimension. '
                           f'but we got {tree_unflatten(tree, batch)}.')
        try:
          length = tuple(set([x.shape[t_idx] for x in xs]))
        except (AttributeError, IndexError) as e:
          raise ValueError(inp_err_msg) from e
        if len(batch) != 1:
          raise ValueError('\n'
                           'Input should be a Array PyTree with the same batch size. '
                           f'but we got {tree_unflatten(tree, batch)}.')
        if len(length) != 1:
          raise ValueError('\n'
                           'Input should be a Array PyTree with the same time length. '
                           f'but we got {tree_unflatten(tree, length)}.')

        if self.no_state:
          xs = [bm.reshape(x, (length[0] * batch[0],) + x.shape[2:]) for x in xs]
        else:
          if self.data_first_axis == 'B':
            xs = [jnp.moveaxis(x, 0, 1) for x in xs]
        xs = tree_unflatten(tree, xs)
        origin_shape = (length[0], batch[0]) if self.data_first_axis == 'T' else (batch[0], length[0])

      else:

        try:
          length = tuple(set([x.shape[0] for x in xs]))
        except (AttributeError, IndexError) as e:
          raise ValueError(inp_err_msg) from e
        if len(length) != 1:
          raise ValueError('\n'
                           'Input should be a Array PyTree with the same time length. '
                           f'but we got {tree_unflatten(tree, length)}.')
        xs = tree_unflatten(tree, xs)
        origin_shape = (length[0],)

      # computation
      if self.no_state:
        share.save(**self.shared_arg)
        outputs = self._run(self.shared_arg, dict(), xs)
        results = tree_map(lambda a: jnp.reshape(a, origin_shape + a.shape[1:]), outputs)
        if self.i0 is not None:
          self.i0 += length[0]
        if self.t0 is not None:
          self.t0 += length[0] * self.dt
        return results

      else:
        shared = tools.DotDict()
        if self.t0 is not None:
          shared['t'] = jnp.arange(0, self.dt * length[0], self.dt) + self.t0.value
        if self.i0 is not None:
          shared['i'] = jnp.arange(0, length[0]) + self.i0.value

    assert not self.no_state
    results = bm.for_loop(functools.partial(self._run, self.shared_arg),
                          (shared, xs),
                          jit=self.jit,
                          remat=self.remat)
    if self.i0 is not None:
      self.i0 += length[0]
    if self.t0 is not None:
      self.t0 += length[0] * self.dt
    return results

  def reset_state(self, batch_size=None):
    self.target.reset_state(batch_size)
    if self.i0 is not None:
      self.i0.value = bm.as_jax(self._i0)
    if self.t0 is not None:
      self.t0.value = bm.as_jax(self._t0)

  def _run(self, static_sh, dyn_sh, x):
    share.save(**static_sh, **dyn_sh)
    outs = self.target(x)
    if self.out_vars is not None:
      outs = (outs, tree_map(bm.as_jax, self.out_vars))
    self.target.clear_input()
    return outs

