# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, Sequence

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from brainpy import tools, math as bm
from brainpy.check import is_float
from brainpy.types import PyTree
from .base import DynamicalSystem, Sequential

__all__ = [
  'LoopOverTime',
  'NoSharedArg',
]


class DynSysToBPObj(bm.BrainPyObject):
  """Transform a :py:class:`DynamicalSystem` to a :py:class:`BrainPyObject`.

  Parameters
  ----------
  target: DynamicalSystem
    The target to transform.
  name: str
    The transformed object name.

  """

  def __init__(self, target: DynamicalSystem, name: str = None):
    super().__init__(name=name)
    self.target = target
    if not isinstance(target, DynamicalSystem):
      raise TypeError(f'Must be instance of {DynamicalSystem.__name__}, '
                      f'but we got {type(target)}')

  def __repr__(self):
    name = self.__class__.__name__
    return f"{name}({tools.repr_context(str(self.target), ' ' * len(name))})"


class LoopOverTime(DynSysToBPObj):
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
  >>> over_time = bp.LoopOverTime(model)
  >>> over_time.reset_state(n_batch)
  (30, 128, 2)
  >>>
  >>> hist_l3 = over_time(bm.random.rand(n_time, n_batch, n_in), data_first_axis='T')
  >>> print(hist_l3.shape)
  >>>
  >>> # monitor the "l1" layer state
  >>> over_time = bp.LoopOverTime(model, out_vars=model['l1'].state)
  >>> over_time.reset_state(n_batch)
  >>> hist_l3, hist_l1 = over_time(bm.random.rand(n_time, n_batch, n_in), data_first_axis='T')
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

    - For ANN layers which are no_state, like :py:class:`~.Dense` or :py:class:`~.Conv2D`,
      set `no_state=True` is high efficiently. This is because :math:`Y[t]` only relies on
      :math:`X[t]`, and it is not necessary to calculate :math:`Y[t]` step-bt-step.
      For this case, we reshape the input from `shape = [T, N, *]` to `shape = [TN, *]`,
      send data to the object, and reshape output to `shape = [T, N, *]`.
      In this way, the calculation over different time is parralelized.

  out_vars: PyTree
    The variables to monitor over the time loop.
  name: str
    The transformed object name.
  """

  def __init__(
      self,
      target: DynamicalSystem,
      out_vars: Union[bm.Variable, Sequence[bm.Variable], Dict[str, bm.Variable]] = None,
      no_state: bool = False,
      name: str = None
  ):
    super().__init__(target=target, name=name)
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
      t0: float = 0.,
      dt: Optional[float] = None,
      shared_arg: Optional[Dict] = None,
      data_first_axis: str = 'T'
  ):
    """Forward propagation along the time or inputs.

    Parameters
    ----------
    duration_or_xs: float, PyTree
      If `float`, it indicates a running duration.
      If a PyTree, it is the given inputs.
    t0: float
      The start time to run the system.
    dt: float
      The time step.
    shared_arg: dict
      The shared arguments across the nodes.
      For instance, `shared_arg={'fit': False}` for the prediction phase.
    data_first_axis: str
      Denote whether the input data is time major.
      If so, we treat the data as `(time, batch, ...)` when the `target` is in Batching mode.
      Default is True.

    Returns
    -------
    out: PyTree
      The accumulated outputs over time.
    """
    assert data_first_axis in ['B', 'T']

    is_float(t0, 't0')
    is_float(dt, 'dt', allow_none=True)
    dt = bm.get_dt() if dt is None else dt
    if shared_arg is None:
      shared_arg = dict(dt=dt)
    else:
      assert isinstance(shared_arg, dict)
      shared_arg['dt'] = dt

    # inputs
    if isinstance(duration_or_xs, float):
      shared = tools.DotDict(t=jnp.arange(t0, duration_or_xs, dt))
      shared['i'] = jnp.arange(0, shared['t'].shape[0])
      xs = None
      if self.no_state:
        raise ValueError('Under the `no_state=True` setting, input cannot be a duration.')

    else:
      inp_err_msg = ('\n'
                     'Input should be a Array PyTree with the shape '
                     'of (B, T, ...) or (T, B, ...) with `data_first_axis="T"`, '
                     'where B the batch size and T the time length.')
      xs, tree = tree_flatten(duration_or_xs, lambda a: isinstance(a, bm.Array))
      if isinstance(self.target.mode, bm.BatchingMode):
        b_idx, t_idx = (1, 0) if data_first_axis == 'T' else (0, 1)

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
          xs = [jnp.reshape(x, (length[0] * batch[0],) + x.shape[2:]) for x in xs]
        else:
          if data_first_axis == 'B':
            xs = [jnp.moveaxis(x, 0, 1) for x in xs]
        xs = tree_unflatten(tree, xs)
        origin_shape = (length[0], batch[0]) if data_first_axis == 'T' else (batch[0], length[0])

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
        outputs = self.target(tools.DotDict(shared_arg), xs)
        return tree_map(lambda a: jnp.reshape(a, origin_shape + a.shape[1:]), outputs)

      else:
        shared = tools.DotDict(t=jnp.arange(t0, dt * length[0], dt),
                               i=jnp.arange(0, length[0]))

    assert not self.no_state

    # function
    @bm.to_object(child_objs=self.target)
    def f(sha, x):
      sha['dt'] = dt
      sha.update(shared_arg)
      outs = self.target(sha, x)
      if self.out_vars is not None:
        outs = (outs, tree_map(bm.as_jax, self.out_vars))
      self.target.clear_input()
      return outs

    return bm.for_loop(f, (shared, xs))

  def reset(self, batch_size=None):
    """Reset function which reset the whole variables in the model.
    """
    self.target.reset(batch_size)

  def reset_state(self, batch_size=None):
    self.target.reset_state(batch_size)


class NoSharedArg(DynSysToBPObj):
  """Transform an instance of :py:class:`~.DynamicalSystem` into a callable
  :py:class:`~.BrainPyObject` :math:`y=f(x)`.

  .. note::

     This object transforms a :py:class:`~.DynamicalSystem` into a :py:class:`~.BrainPyObject`.

     If some children nodes need shared arguments, like :py:class:`~.Dropout` or
     :py:class:`~.LIF` models, using ``NoSharedArg`` will cause errors.

  Examples
  --------

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>> l = bp.Sequential(bp.layers.Dense(100, 10),
  >>>                   bm.relu,
  >>>                   bp.layers.Dense(10, 2))
  >>> l = bp.NoSharedArg(l)
  >>> l(bm.random.random(256, 100))

  Parameters
  ----------
  target: DynamicalSystem
    The target to transform.
  name: str
    The transformed object name.
  """

  def __init__(self, target: DynamicalSystem, name: str = None):
    super().__init__(target=target, name=name)
    if isinstance(target, Sequential) and target.no_shared_arg:
      raise ValueError(f'It is a {Sequential.__name__} object with `no_shared_arg=True`, '
                       f'which has already able to be called with `f(x)`. ')

  def __call__(self, *args, **kwargs):
    return self.target(tools.DotDict(), *args, **kwargs)

  def reset(self, batch_size=None):
    """Reset function which reset the whole variables in the model.
    """
    self.target.reset(batch_size)

  def reset_state(self, batch_size=None):
    self.target.reset_state(batch_size)
