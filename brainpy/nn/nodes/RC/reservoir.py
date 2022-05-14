# -*- coding: utf-8 -*-

from typing import Optional, Union, Callable

import brainpy.math as bm
from brainpy.initialize import Normal, ZeroInit, Initializer, init_param
from brainpy.nn.base import RecurrentNode
from brainpy.nn.datatypes import MultipleData
from brainpy.tools.checking import (check_shape_consistency,
                                    check_float,
                                    check_initializer,
                                    check_string)
from brainpy.types import Tensor

__all__ = [
  'Reservoir',
]


class Reservoir(RecurrentNode):
  r"""Reservoir node, a pool of leaky-integrator neurons
  with random recurrent connections [1]_.

  Parameters
  ----------
  num_unit: int
    The number of reservoir nodes.
  ff_initializer: Initializer
    The initialization method for the feedforward connections.
  rec_initializer: Initializer
    The initialization method for the recurrent connections.
  fb_initializer: optional, Tensor, Initializer
    The initialization method for the feedback connections.
  bias_initializer: optional, Tensor, Initializer
    The initialization method for the bias.
  leaky_rate: float
    A float between 0 and 1.
  activation : str, callable, optional
    Reservoir activation function.
    - If a str, should be a :py:mod:`brainpy.math.activations` function name.
    - If a callable, should be an element-wise operator on tensor.
  activation_type : str
    - If "internal" (default), then leaky integration happens on states transformed
      by the activation function:

    .. math::

        r[n+1] = (1 - \alpha) \cdot r[t] +
        \alpha \cdot f(W_{ff} \cdot u[n] + W_{fb} \cdot b[n] + W_{rec} \cdot r[t])

    - If "external", then leaky integration happens on internal states of
      each neuron, stored in an ``internal_state`` parameter (:math:`x` in
      the equation below).
      A neuron internal state is the value of its state before applying
      the activation function :math:`f`:

      .. math::

          x[n+1] &= (1 - \alpha) \cdot x[t] +
          \alpha \cdot f(W_{ff} \cdot u[n] + W_{rec} \cdot r[t] + W_{fb} \cdot b[n]) \\
          r[n+1] &= f(x[n+1])
  ff_connectivity : float, optional
    Connectivity of input neurons, i.e. ratio of input neurons connected
    to reservoir neurons. Must be in [0, 1], by default 0.1
  rec_connectivity : float, optional
    Connectivity of recurrent weights matrix, i.e. ratio of reservoir
    neurons connected to other reservoir neurons, including themselves.
    Must be in [0, 1], by default 0.1
  fb_connectivity : float, optional
    Connectivity of feedback neurons, i.e. ratio of feedabck neurons
    connected to reservoir neurons. Must be in [0, 1], by default 0.1
  conn_type: str
    The connectivity type, can be "dense" or "sparse".
  spectral_radius : float, optional
    Spectral radius of recurrent weight matrix, by default None
  noise_rec : float, optional
    Gain of noise applied to reservoir internal states, by default 0.0
  noise_in : float, optional
    Gain of noise applied to feedforward signals, by default 0.0
  noise_fb : float, optional
    Gain of noise applied to feedback signals, by default 0.0
  noise_type : optional, str, callable
    Distribution of noise. Must be a random variable generator
    distribution (see :py:class:`brainpy.math.random.RandomState`),
    by default "normal". 
  seed: optional, int
    The seed for random sampling in this node.

  References
  ----------
  .. [1] Lukoševičius, Mantas. "A practical guide to applying echo state networks."
         Neural networks: Tricks of the trade. Springer, Berlin, Heidelberg, 2012. 659-686.
  """
  data_pass = MultipleData('sequence')

  def __init__(
      self,
      num_unit: int,
      leaky_rate: float = 0.3,
      activation: Union[str, Callable] = 'tanh',
      activation_type: str = 'internal',
      ff_initializer: Union[Initializer, Callable, Tensor] = Normal(scale=0.1),
      rec_initializer: Union[Initializer, Callable, Tensor] = Normal(scale=0.1),
      fb_initializer: Optional[Union[Initializer, Callable, Tensor]] = Normal(scale=0.1),
      bias_initializer: Optional[Union[Initializer, Callable, Tensor]] = ZeroInit(),
      ff_connectivity: float = 0.1,
      rec_connectivity: float = 0.1,
      fb_connectivity: float = 0.1,
      conn_type='dense',
      spectral_radius: Optional[float] = None,
      noise_ff: float = 0.,
      noise_rec: float = 0.,
      noise_fb: float = 0.,
      noise_type: str = 'normal',
      seed: Optional[int] = None,
      trainable: bool = False,
      **kwargs
  ):
    super(Reservoir, self).__init__(trainable=trainable, **kwargs)

    # parameters
    self.num_unit = num_unit
    assert num_unit > 0, f'Must be a positive integer, but we got {num_unit}'
    self.leaky_rate = leaky_rate
    check_float(leaky_rate, 'leaky_rate', 0., 1.)
    self.activation = bm.activations.get(activation)
    self.activation_type = activation_type
    check_string(activation_type, 'activation_type', ['internal', 'external'])
    self.rng = bm.random.RandomState(seed)
    check_float(spectral_radius, 'spectral_radius', allow_none=True)
    self.spectral_radius = spectral_radius

    # initializations
    check_initializer(ff_initializer, 'ff_initializer', allow_none=False)
    check_initializer(rec_initializer, 'rec_initializer', allow_none=False)
    check_initializer(fb_initializer, 'fb_initializer', allow_none=True)
    check_initializer(bias_initializer, 'bias_initializer', allow_none=True)
    self.ff_initializer = ff_initializer
    self.fb_initializer = fb_initializer
    self.rec_initializer = rec_initializer
    self.bias_initializer = bias_initializer

    # connectivity
    check_float(ff_connectivity, 'ff_connectivity', 0., 1.)
    check_float(rec_connectivity, 'rec_connectivity', 0., 1.)
    check_float(fb_connectivity, 'fb_connectivity', 0., 1.)
    self.ff_connectivity = ff_connectivity
    self.rec_connectivity = rec_connectivity
    self.fb_connectivity = fb_connectivity
    check_string(conn_type, 'conn_type', ['dense', 'sparse'])
    self.conn_type = conn_type

    # noises
    check_float(noise_ff, 'noise_ff')
    check_float(noise_fb, 'noise_fb')
    check_float(noise_rec, 'noise_rec')
    self.noise_ff = noise_ff
    self.noise_fb = noise_fb
    self.noise_rec = noise_rec
    self.noise_type = noise_type
    check_string(noise_type, 'noise_type', ['normal', 'uniform'])

  def init_ff_conn(self):
    """Initialize feedforward connections, weights, and variables."""
    unique_shape, free_shapes = check_shape_consistency(self.feedforward_shapes, -1, True)
    self.set_output_shape(unique_shape + (self.num_unit,))

    # initialize feedforward weights
    weight_shape = (sum(free_shapes), self.num_unit)
    self.Wff_shape = weight_shape
    self.Wff = init_param(self.ff_initializer, weight_shape)
    if self.ff_connectivity < 1.:
      conn_mat = self.rng.random(weight_shape) > self.ff_connectivity
      self.Wff[conn_mat] = 0.
    if self.conn_type == 'sparse' and self.ff_connectivity < 1.:
      self.ff_pres, self.ff_posts = bm.where(bm.logical_not(conn_mat))
      self.Wff = self.Wff[self.ff_pres, self.ff_posts]
    if self.trainable:
      self.Wff = bm.TrainVar(self.Wff)

    # initialize recurrent weights
    recurrent_shape = (self.num_unit, self.num_unit)
    self.Wrec = init_param(self.rec_initializer, recurrent_shape)
    if self.rec_connectivity < 1.:
      conn_mat = self.rng.random(recurrent_shape) > self.rec_connectivity
      self.Wrec[conn_mat] = 0.
    if self.spectral_radius is not None:
      current_sr = max(abs(bm.linalg.eig(self.Wrec)[0]))
      self.Wrec *= self.spectral_radius / current_sr
    if self.conn_type == 'sparse' and self.rec_connectivity < 1.:
      self.rec_pres, self.rec_posts = bm.where(bm.logical_not(conn_mat))
      self.Wrec = self.Wrec[self.rec_pres, self.rec_posts]
    self.bias = init_param(self.bias_initializer, (self.num_unit,))
    if self.trainable:
      self.Wrec = bm.TrainVar(self.Wrec)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

    # initialize feedback weights
    self.Wfb = None

  def init_state(self, num_batch=1):
    # initialize internal state
    return bm.zeros((num_batch, self.num_unit), dtype=bm.float_)

  def init_fb_conn(self):
    """Initialize feedback connections, weights, and variables."""
    if self.feedback_shapes is not None:
      unique_shape, free_shapes = check_shape_consistency(self.feedback_shapes, -1, True)
      fb_shape = (sum(free_shapes), self.num_unit)
      self.Wfb_shape = fb_shape
      self.Wfb = init_param(self.fb_initializer, fb_shape)
      if self.fb_connectivity < 1.:
        conn_mat = self.rng.random(fb_shape) > self.fb_connectivity
        self.Wfb[conn_mat] = 0.
      if self.conn_type == 'sparse' and self.fb_connectivity < 1.:
        self.fb_pres, self.fb_posts = bm.where(bm.logical_not(conn_mat))
        self.Wfb = self.Wfb[self.fb_pres, self.fb_posts]
      if self.trainable:
        self.Wfb = bm.TrainVar(self.Wfb)

  def forward(self, ff, fb=None, **shared_kwargs):
    """Feedforward output."""
    # inputs
    x = bm.concatenate(ff, axis=-1)
    if self.noise_ff > 0: x += self.noise_ff * self.rng.uniform(-1, 1, x.shape)
    if self.conn_type == 'sparse' and self.ff_connectivity < 1.:
      sparse = {'data': self.Wff, 'index': (self.ff_pres, self.ff_posts), 'shape': self.Wff_shape}
      hidden = bm.sparse_matmul(x, sparse)
    else:
      hidden = bm.dot(x, self.Wff)
    # feedback
    if self.Wfb is not None:
      assert fb is not None, 'Should provide feedback signals, but we got None.'
      fb = bm.concatenate(fb, axis=-1)
      if self.noise_fb: fb += self.noise_fb * self.rng.uniform(-1, 1, fb.shape)
      if self.conn_type == 'sparse' and self.fb_connectivity < 1.:
        sparse = {'data': self.Wfb, 'index': (self.fb_pres, self.fb_posts), 'shape': self.Wfb_shape}
        hidden += bm.sparse_matmul(fb, sparse)
      else:
        hidden += bm.dot(fb, self.Wfb)
    # recurrent
    if self.conn_type == 'sparse' and self.rec_connectivity < 1.:
      sparse = {'data': self.Wrec, 'index': (self.rec_pres, self.rec_posts), 'shape': (self.num_unit, self.num_unit)}
      hidden += bm.sparse_matmul(self.state, sparse)
    else:
      hidden += bm.dot(self.state, self.Wrec)
    if self.activation_type == 'internal':
      hidden = self.activation(hidden)
    if self.noise_rec > 0.: hidden += self.noise_rec * self.rng.uniform(-1, -1, self.state.shape)
    # new state/output
    state = (1 - self.leaky_rate) * self.state + self.leaky_rate * hidden
    if self.activation_type == 'external':
      state = self.activation(state)
    self.state.value = state
    return state
