# -*- coding: utf-8 -*-

from typing import Optional, Union, Sequence, Tuple, Callable

import jax.numpy as jnp
from jax import vmap

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import Initializer
from brainpy.tools.checking import check_sequence, check_integer
from brainpy.types import Tensor

__all__ = [
  'DelayCoupling',
  'DiffusiveCoupling',
  'AdditiveCoupling',
]


class DelayCoupling(DynamicalSystem):
  """Delay coupling.

  Parameters
  ----------
  delay_var: Variable
    The delay variable.
  output_var: Variable, sequence of Variable
    The target variables to output.
  conn_mat: JaxArray, ndarray
    The connection matrix.
  required_shape: sequence of int
    The required shape of `(pre, post)`.
  delay_steps: int, JaxArray, ndarray
    The matrix of delay time steps. Must be int.
  initial_delay_data: Initializer, Callable
    The initializer of the initial delay data.
  """

  def __init__(
      self,
      delay_var: bm.Variable,
      output_var: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: Tensor,
      required_shape: Tuple[int, ...],
      delay_steps: Optional[Union[int, Tensor, Initializer, Callable]] = None,
      initial_delay_data: Union[Initializer, Callable, Tensor, float, int, bool] = None,
      name: str = None
  ):
    super(DelayCoupling, self).__init__(name=name)

    # delay variable
    if not isinstance(delay_var, bm.Variable):
      raise ValueError(f'"delay_var" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(delay_var)}')
    self.delay_var = delay_var

    # output variables
    if isinstance(output_var, bm.Variable):
      output_var = [output_var]
    check_sequence(output_var, 'output_var', elem_type=bm.Variable, allow_none=False)
    self.output_var = output_var

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
      num_delay_step = 0
    elif isinstance(delay_steps, int):
      self.delay_steps = delay_steps
      num_delay_step = delay_steps
      check_integer(delay_steps, 'delay_steps', min_bound=0, allow_none=False)
      self.delay_type = 'int'
    elif callable(delay_steps):
      delay_steps = delay_steps(required_shape)
      if delay_steps.dtype not in [bm.int32, bm.int64, bm.uint32, bm.uint64]:
        raise ValueError(f'"delay_steps" must be integer typed. But we got {delay_steps.dtype}')
      self.delay_steps = delay_steps
      self.delay_type = 'array'
      num_delay_step = int(self.delay_steps.max())
    elif isinstance(delay_steps, (bm.JaxArray, jnp.ndarray)):
      if delay_steps.dtype not in [bm.int32, bm.int64, bm.uint32, bm.uint64]:
        raise ValueError(f'"delay_steps" must be integer typed. But we got {delay_steps.dtype}')
      if delay_steps.shape != required_shape:
        raise ValueError(f'we expect the delay matrix has the shape of {required_shape}. '
                         f'While we got {delay_steps.shape}.')
      self.delay_steps = delay_steps
      self.delay_type = 'array'
      num_delay_step = int(self.delay_steps.max())
    else:
      raise ValueError(f'Unknown type of delay steps: {type(delay_steps)}')

    # delay variables
    if self.delay_type != 'none':
      self.register_delay(f'delay_{id(delay_var)}',
                          delay_step=num_delay_step,
                          delay_target=delay_var,
                          initial_delay_data=initial_delay_data)

  def reset(self):
    if self.delay_steps is not None:
      self.reset_delay(f'delay_{id(self.delay_var)}', self.delay_var)


class DiffusiveCoupling(DelayCoupling):
  """Diffusive coupling.

  This class simulates the model of::

     coupling = g * (delayed_coupling_var1 - coupling_var2)
     output_var += coupling


  Examples
  --------

  >>> import brainpy as bp
  >>> areas = bp.dyn.RateFHN(80, x_ou_sigma=0.01, y_ou_sigma=0.01, name='fhn')
  >>> conn = bp.dyn.DiffusiveCoupling(areas.x, areas.x, areas.input,
  >>>                                 conn_mat=Cmat, delay_steps=Dmat,
  >>>                                 initial_delay_data=bp.init.Uniform(0, 0.05))
  >>> net = bp.dyn.Network(areas, conn)

  Parameters
  ----------
  coupling_var1: Variable
    The first coupling variable, used for delay.
  coupling_var2: Variable
    Another coupling variable.
  output_var: Variable, sequence of Variable
    The target variables to output.
  conn_mat: JaxArray, ndarray
    The connection matrix.
  delay_steps: int, JaxArray, ndarray
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
      output_var: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: Tensor,
      delay_steps: Optional[Union[int, Tensor, Initializer, Callable]] = None,
      initial_delay_data: Union[Initializer, Callable, Tensor, float, int, bool] = None,
      name: str = None
  ):
    if not isinstance(coupling_var1, bm.Variable):
      raise ValueError(f'"coupling_var1" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var1)}')
    if not isinstance(coupling_var2, bm.Variable):
      raise ValueError(f'"coupling_var2" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var2)}')
    if bm.ndim(coupling_var1) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {bm.ndim(coupling_var1)}')
    if bm.ndim(coupling_var2) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {bm.ndim(coupling_var2)}')

    super(DiffusiveCoupling, self).__init__(
      delay_var=coupling_var1,
      output_var=output_var,
      conn_mat=conn_mat,
      required_shape=(coupling_var1.size, coupling_var2.size),
      delay_steps=delay_steps,
      initial_delay_data=initial_delay_data,
      name=name
    )

    self.coupling_var1 = coupling_var1
    self.coupling_var2 = coupling_var2

  def update(self, _t, _dt):
    # delay variable
    if self.delay_type != 'none':
      delay_var: bm.LengthDelay = self.global_delay_vars[f'delay_{id(self.delay_var)}']

    # delays
    if self.delay_type == 'none':
      diffusive = bm.expand_dims(self.coupling_var1, axis=1) - self.coupling_var2
    elif self.delay_type == 'array':
      f = vmap(lambda i: delay_var(self.delay_steps[i], bm.arange(self.coupling_var1.size)))  # (pre.num,)
      delays = f(bm.arange(self.coupling_var2.size).value)
      diffusive = delays.T - self.coupling_var2  # (post.num, pre.num)
    elif self.delay_type == 'int':
      diffusive = bm.expand_dims(delay_var(self.delay_steps), axis=1) - self.coupling_var2
    else:
      raise ValueError
    diffusive = (self.conn_mat * diffusive).sum(axis=0)

    # output to target variable
    for target in self.output_var:
      target.value += diffusive

    # update
    if self.delay_type != 'none':
      delay_var.update(self.delay_var)


class AdditiveCoupling(DelayCoupling):
  """Additive coupling.

  This class simulates the model of::

     coupling = g * delayed_coupling_var1
     output_var += coupling

  Parameters
  ----------
  coupling_var: Variable
    The coupling variable, used for delay.
  output_var: Variable, sequence of Variable
    The target variables to output.
  conn_mat: JaxArray, ndarray
    The connection matrix.
  delay_steps: int, JaxArray, ndarray
    The matrix of delay time steps. Must be int.
  initial_delay_data: Initializer, Callable
    The initializer of the initial delay data.
  name: str
    The name of the model.
  """

  def __init__(
      self,
      coupling_var: bm.Variable,
      output_var: Union[bm.Variable, Sequence[bm.Variable]],
      conn_mat: Tensor,
      delay_steps: Optional[Union[int, Tensor, Initializer, Callable]] = None,
      initial_delay_data: Union[Initializer, Callable, Tensor, float, int, bool] = None,
      name: str = None
  ):
    if not isinstance(coupling_var, bm.Variable):
      raise ValueError(f'"coupling_var" must be an instance of brainpy.math.Variable. '
                       f'But we got {type(coupling_var)}')
    if bm.ndim(coupling_var) != 1:
      raise ValueError(f'Only support 1d vector of coupling variable. '
                       f'But we got {bm.ndim(coupling_var)}')

    super(AdditiveCoupling, self).__init__(
      delay_var=coupling_var,
      output_var=output_var,
      conn_mat=conn_mat,
      required_shape=(coupling_var.size, coupling_var.size),
      delay_steps=delay_steps,
      initial_delay_data=initial_delay_data,
      name=name
    )

    self.coupling_var = coupling_var

  def update(self, _t, _dt):
    # delay variable
    delay_var: bm.LengthDelay = self.global_delay_vars[f'delay_{id(self.delay_var)}']

    # delay function
    if self.delay_steps is None:
      additive = self.conn_mat * bm.expand_dims(self.coupling_var, axis=1)
    else:
      f = vmap(lambda i: delay_var(self.delay_steps[i], bm.arange(self.coupling_var.size)))  # (pre.num,)
      delays = f(bm.arange(self.coupling_var.size).value)  # (post.num, pre.num)
      additive = (self.conn_mat * delays.T).sum(axis=0)

    # output to target variable
    for target in self.output_var:
      target.value += additive

    # update
    delay_var.update(self.delay_var)
