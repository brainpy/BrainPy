# -*- coding: utf-8 -*-

import numbers
from typing import Optional, Union, Sequence, Tuple, Callable

import jax.numpy as jnp
from jax import vmap

import brainpy.math as bm
from brainpy._src.dynsys import Projection
from brainpy._src.initialize import Initializer
from brainpy.check import is_sequence
from brainpy.types import ArrayType

__all__ = [
  'DelayCoupling',
  'DiffusiveCoupling',
  'AdditiveCoupling',
]


class DelayCoupling(Projection):
  """Delay coupling.

  Parameters
  ----------
  delay_var: Variable
    The delay variable.
  var_to_output: Variable, sequence of Variable
    The target variables to output.
  conn_mat: ArrayType
    The connection matrix.
  required_shape: sequence of int
    The required shape of `(pre, post)`.
  delay_steps: int, ArrayType
    The matrix of delay time steps. Must be int.
  initial_delay_data: Initializer, Callable
    The initializer of the initial delay data.
  """

  def __init__(
      self,
      delay_var: bm.Variable,
      var_to_output: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: ArrayType,
      required_shape: Tuple[int, ...],
      delay_steps: Optional[Union[int, ArrayType, Callable]] = None,
      initial_delay_data: Union[Callable, ArrayType, numbers.Number] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # delay variable
    if not isinstance(delay_var, bm.Variable):
      raise ValueError(f'"delay_var" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(delay_var)}')
    self.delay_var = delay_var

    # output variables
    if isinstance(var_to_output, bm.Variable):
      var_to_output = [var_to_output]
    is_sequence(var_to_output, 'output_var', elem_type=bm.Variable, allow_none=False)
    self.output_var = var_to_output

    # Connection matrix
    self.conn_mat = bm.asarray(conn_mat)
    if self.conn_mat.shape != required_shape:
      raise ValueError(f'we expect the structural connection matrix has the shape of '
                       f'(pre.num, post.num), i.e., {required_shape}, '
                       f'while we got {self.conn_mat.shape}.')

    # Delay matrix
    if delay_steps is None:
      self.delay_steps = None
      self.delay_type = 'none'
      num_delay_step = None
    elif callable(delay_steps):
      delay_steps = delay_steps(required_shape)
      if delay_steps.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError(f'"delay_steps" must be integer typed. But we got {delay_steps.dtype}')
      self.delay_steps = delay_steps
      self.delay_type = 'array'
      num_delay_step = self.delay_steps.max()
    elif isinstance(delay_steps, (bm.Array, jnp.ndarray)):
      if delay_steps.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
        raise ValueError(f'"delay_steps" must be integer typed. But we got {delay_steps.dtype}')
      if delay_steps.ndim == 0:
        self.delay_type = 'int'
      else:
        self.delay_type = 'array'
        if delay_steps.shape != required_shape:
          raise ValueError(f'we expect the delay matrix has the shape of '
                           f'(pre.num, post.num), i.e., {required_shape}. '
                           f'While we got {delay_steps.shape}.')
      self.delay_steps = delay_steps
      num_delay_step = self.delay_steps.max()
    elif isinstance(delay_steps, int):
      self.delay_steps = delay_steps
      num_delay_step = delay_steps
      self.delay_type = 'int'
    else:
      raise ValueError(f'Unknown type of delay steps: {type(delay_steps)}')

    # delay variables
    _ = self.register_delay(f'delay_{id(delay_var)}',
                            delay_step=num_delay_step,
                            delay_target=delay_var,
                            initial_delay_data=initial_delay_data)

  def reset_state(self, batch_size=None):
    pass


class DiffusiveCoupling(DelayCoupling):
  """Diffusive coupling.

  This class simulates the model of::

     coupling = g * (delayed_coupling_var1 - coupling_var2)
     target_var += coupling


  Examples
  --------

  >>> import brainpy as bp
  >>> from brainpy import rates
  >>> areas = bp.rates.FHN(80, x_ou_sigma=0.01, y_ou_sigma=0.01, name='fhn')
  >>> conn = bp.synapses.DiffusiveCoupling(areas.x, areas.x, areas.input,
  >>>                                      conn_mat=Cmat, delay_steps=Dmat,
  >>>                                      initial_delay_data=bp.init.Uniform(0, 0.05))
  >>> net = bp.Network(areas, conn)

  Parameters
  ----------
  coupling_var1: Variable
    The first coupling variable, used for delay.
  coupling_var2: Variable
    Another coupling variable.
  var_to_output: Variable, sequence of Variable
    The target variables to output.
  conn_mat: ArrayType
    The connection matrix.
  delay_steps: int, ArrayType
    The matrix of delay time steps. Must be int.
  initial_delay_data: Initializer, Callable
    The initializer of the initial delay data.
  name: str
    The name of the model.
  """

  def __init__(
      self,
      coupling_var1: bm.Variable,
      coupling_var2: bm.Variable,
      var_to_output: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: ArrayType,
      delay_steps: Optional[Union[int, ArrayType, Initializer, Callable]] = None,
      initial_delay_data: Union[Initializer, Callable, ArrayType, float, int, bool] = None,
      name: str = None,
      mode: bm.Mode = None,
  ):
    if not isinstance(coupling_var1, bm.Variable):
      raise ValueError(f'"coupling_var1" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var1)}')
    if not isinstance(coupling_var2, bm.Variable):
      raise ValueError(f'"coupling_var2" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var2)}')
    if jnp.ndim(coupling_var1) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {jnp.ndim(coupling_var1)}')
    if jnp.ndim(coupling_var2) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {jnp.ndim(coupling_var2)}')

    super().__init__(
      delay_var=coupling_var1,
      var_to_output=var_to_output,
      conn_mat=conn_mat,
      required_shape=(coupling_var1.size, coupling_var2.size),
      delay_steps=delay_steps,
      initial_delay_data=initial_delay_data,
      name=name,
      mode=mode,
    )

    self.coupling_var1 = coupling_var1
    self.coupling_var2 = coupling_var2

  def update(self):
    # delays
    axis = self.coupling_var1.ndim
    delay_var: bm.LengthDelay = self.get_delay_var(f'delay_{id(self.delay_var)}')[0]
    if self.delay_steps is None:
      diffusive = (jnp.expand_dims(self.coupling_var1.value, axis=axis) -
                   jnp.expand_dims(self.coupling_var2.value, axis=axis - 1))
      diffusive = (self.conn_mat * diffusive).sum(axis=axis - 1)
    elif self.delay_type == 'array':
      if isinstance(self.mode, bm.TrainingMode):
        indices = (slice(None, None, None), jnp.arange(self.coupling_var1.size),)
      else:
        indices = (jnp.arange(self.coupling_var1.size),)
      f = vmap(lambda steps: delay_var(steps, *indices), in_axes=1)  # (..., pre.num)
      delays = f(self.delay_steps)  # (..., post.num, pre.num)
      diffusive = (jnp.moveaxis(bm.as_jax(delays), axis - 1, axis) -
                   jnp.expand_dims(self.coupling_var2.value, axis=axis - 1))  # (..., pre.num, post.num)
      diffusive = (self.conn_mat * diffusive).sum(axis=axis - 1)
    elif self.delay_type == 'int':
      delayed_data = delay_var(self.delay_steps)  # (..., pre.num)
      diffusive = (jnp.expand_dims(delayed_data, axis=axis) -
                   jnp.expand_dims(self.coupling_var2.value, axis=axis - 1))  # (..., pre.num, post.num)
      diffusive = (self.conn_mat * diffusive).sum(axis=axis - 1)
    else:
      raise ValueError(f'Unknown delay type {self.delay_type}')

    # output to target variable
    for target in self.output_var:
      target.value += diffusive


class AdditiveCoupling(DelayCoupling):
  """Additive coupling.

  This class simulates the model of::

     coupling = g * delayed_coupling_var
     target_var += coupling

  Parameters
  ----------
  coupling_var: Variable
    The coupling variable, used for delay.
  var_to_output: Variable, sequence of Variable
    The target variables to output.
  conn_mat: ArrayType
    The connection matrix.
  delay_steps: int, ArrayType
    The matrix of delay time steps. Must be int.
  initial_delay_data: Initializer, Callable
    The initializer of the initial delay data.
  name: str
    The name of the model.
  """

  def __init__(
      self,
      coupling_var: bm.Variable,
      var_to_output: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: ArrayType,
      delay_steps: Optional[Union[int, ArrayType, Initializer, Callable]] = None,
      initial_delay_data: Union[Initializer, Callable, ArrayType, float, int, bool] = None,
      name: str = None,
      mode: bm.Mode = None,
  ):
    if not isinstance(coupling_var, bm.Variable):
      raise ValueError(f'"coupling_var" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var)}')
    if jnp.ndim(coupling_var) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {jnp.ndim(coupling_var)}')

    super().__init__(
      delay_var=coupling_var,
      var_to_output=var_to_output,
      conn_mat=conn_mat,
      required_shape=(coupling_var.size, coupling_var.size),
      delay_steps=delay_steps,
      initial_delay_data=initial_delay_data,
      name=name,
      mode=mode,
    )

    self.coupling_var = coupling_var

  def update(self):
    # delay function
    axis = self.coupling_var.ndim
    delay_var: bm.LengthDelay = self.get_delay_var(f'delay_{id(self.delay_var)}')[0]
    if self.delay_steps is None:
      additive = self.coupling_var @ self.conn_mat
    elif self.delay_type == 'array':
      if isinstance(self.mode, bm.TrainingMode):
        indices = (slice(None, None, None), jnp.arange(self.coupling_var.size),)
      else:
        indices = (jnp.arange(self.coupling_var.size),)
      f = vmap(lambda steps: delay_var(steps, *indices), in_axes=1)  # (.., pre.num,)
      delays = f(self.delay_steps)  # (..., post.num, pre.num)
      additive = (self.conn_mat * jnp.moveaxis(delays, axis - 1, axis)).sum(axis=axis - 1)
    elif self.delay_type == 'int':
      delayed_var = delay_var(self.delay_steps)  # (..., pre.num)
      additive = delayed_var @ self.conn_mat
    else:
      raise ValueError

    # output to target variable
    for target in self.output_var:
      target.value += additive
