# -*- coding: utf-8 -*-

from typing import Optional, Union, Callable, Tuple

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem
from brainpy.initialize import Normal, ZeroInit, Initializer, parameter, variable
from brainpy.modes import Mode, TrainingMode, batching
from brainpy.tools.checking import check_float, check_initializer, check_string
from brainpy.tools.others import to_size
from brainpy.types import Array

__all__ = [
  'Reservoir',
]


class Reservoir(DynamicalSystem):
  r"""Reservoir node, a pool of leaky-integrator neurons
  with random recurrent connections [1]_.

  Parameters
  ----------
  input_shape: int, tuple of int
    The input shape.
  num_out: int
    The number of reservoir nodes.
  Win_initializer: Initializer
    The initialization method for the feedforward connections.
  Wrec_initializer: Initializer
    The initialization method for the recurrent connections.
  b_initializer: optional, Array, Initializer
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
  in_connectivity : float, optional
    Connectivity of input neurons, i.e. ratio of input neurons connected
    to reservoir neurons. Must be in [0, 1], by default 0.1
  rec_connectivity : float, optional
    Connectivity of recurrent weights matrix, i.e. ratio of reservoir
    neurons connected to other reservoir neurons, including themselves.
    Must be in [0, 1], by default 0.1
  comp_type: str
    The connectivity type, can be "dense" or "sparse".
  spectral_radius : float, optional
    Spectral radius of recurrent weight matrix, by default None
  noise_rec : float, optional
    Gain of noise applied to reservoir internal states, by default 0.0
  noise_in : float, optional
    Gain of noise applied to feedforward signals, by default 0.0
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

  def __init__(
      self,
      input_shape: Union[int, Tuple[int]],
      num_out: int,
      leaky_rate: float = 0.3,
      activation: Union[str, Callable] = 'tanh',
      activation_type: str = 'internal',
      Win_initializer: Union[Initializer, Callable, Array] = Normal(scale=0.1),
      Wrec_initializer: Union[Initializer, Callable, Array] = Normal(scale=0.1),
      b_initializer: Optional[Union[Initializer, Callable, Array]] = ZeroInit(),
      in_connectivity: float = 0.1,
      rec_connectivity: float = 0.1,
      comp_type='dense',
      spectral_radius: Optional[float] = None,
      noise_in: float = 0.,
      noise_rec: float = 0.,
      noise_type: str = 'normal',
      seed: Optional[int] = None,
      mode: Mode = batching,
      name: str = None
  ):
    super(Reservoir, self).__init__(mode=mode, name=name)

    # parameters
    input_shape = to_size(input_shape)
    if input_shape[0] is None:
      input_shape = input_shape[1:]
    self.input_shape = input_shape
    self.output_shape = input_shape[:-1] + (num_out,)
    self.num_unit = num_out
    assert num_out > 0, f'Must be a positive integer, but we got {num_out}'
    self.leaky_rate = leaky_rate
    check_float(leaky_rate, 'leaky_rate', 0., 1.)
    self.activation = bm.activations.get(activation)
    self.activation_type = activation_type
    check_string(activation_type, 'activation_type', ['internal', 'external'])
    self.rng = bm.random.RandomState(seed)
    check_float(spectral_radius, 'spectral_radius', allow_none=True)
    self.spectral_radius = spectral_radius

    # initializations
    check_initializer(Win_initializer, 'ff_initializer', allow_none=False)
    check_initializer(Wrec_initializer, 'rec_initializer', allow_none=False)
    check_initializer(b_initializer, 'bias_initializer', allow_none=True)
    self._Win_initializer = Win_initializer
    self._Wrec_initializer = Wrec_initializer
    self._b_initializer = b_initializer

    # connectivity
    check_float(in_connectivity, 'ff_connectivity', 0., 1.)
    check_float(rec_connectivity, 'rec_connectivity', 0., 1.)
    self.ff_connectivity = in_connectivity
    self.rec_connectivity = rec_connectivity
    check_string(comp_type, 'conn_type', ['dense', 'sparse'])
    self.comp_type = comp_type

    # noises
    check_float(noise_in, 'noise_ff')
    check_float(noise_rec, 'noise_rec')
    self.noise_ff = noise_in
    self.noise_rec = noise_rec
    self.noise_type = noise_type
    check_string(noise_type, 'noise_type', ['normal', 'uniform'])

    # initialize feedforward weights
    weight_shape = (input_shape[-1], self.num_unit)
    self.Wff_shape = weight_shape
    self.Win = parameter(self._Win_initializer, weight_shape)
    if self.ff_connectivity < 1.:
      conn_mat = self.rng.random(weight_shape) > self.ff_connectivity
      self.Win[conn_mat] = 0.
    if self.comp_type == 'sparse' and self.ff_connectivity < 1.:
      self.ff_pres, self.ff_posts = bm.where(bm.logical_not(conn_mat))
      self.Win = self.Win[self.ff_pres, self.ff_posts]
    if isinstance(self.mode, TrainingMode):
      self.Win = bm.TrainVar(self.Win)

    # initialize recurrent weights
    recurrent_shape = (self.num_unit, self.num_unit)
    self.Wrec = parameter(self._Wrec_initializer, recurrent_shape)
    if self.rec_connectivity < 1.:
      conn_mat = self.rng.random(recurrent_shape) > self.rec_connectivity
      self.Wrec[conn_mat] = 0.
    if self.spectral_radius is not None:
      current_sr = max(abs(bm.linalg.eig(self.Wrec)[0]))
      self.Wrec *= self.spectral_radius / current_sr
    if self.comp_type == 'sparse' and self.rec_connectivity < 1.:
      self.rec_pres, self.rec_posts = bm.where(bm.logical_not(conn_mat))
      self.Wrec = self.Wrec[self.rec_pres, self.rec_posts]
    self.bias = parameter(self._b_initializer, (self.num_unit,))
    if isinstance(self.mode, TrainingMode):
      self.Wrec = bm.TrainVar(self.Wrec)
      self.bias = None if (self.bias is None) else bm.TrainVar(self.bias)

    # initialize state
    self.state = variable(bm.zeros, mode, self.output_shape)

  def reset_state(self, batch_size=None):
    self.state.value = variable(bm.zeros, batch_size, self.output_shape)

  def update(self, sha, x):
    """Feedforward output."""
    # inputs
    x = bm.concatenate(x, axis=-1)
    if self.noise_ff > 0: x += self.noise_ff * self.rng.uniform(-1, 1, x.shape)
    if self.comp_type == 'sparse' and self.ff_connectivity < 1.:
      sparse = {'data': self.Win,
                'index': (self.ff_pres, self.ff_posts),
                'shape': self.Wff_shape}
      hidden = bm.sparse_matmul(x, sparse)
    else:
      hidden = bm.dot(x, self.Win)
    # recurrent
    if self.comp_type == 'sparse' and self.rec_connectivity < 1.:
      sparse = {'data': self.Wrec,
                'index': (self.rec_pres, self.rec_posts),
                'shape': (self.num_unit, self.num_unit)}
      hidden += bm.sparse_matmul(self.state, sparse)
    else:
      hidden += bm.dot(self.state, self.Wrec)
    if self.activation_type == 'internal':
      hidden = self.activation(hidden)
    if self.noise_rec > 0.:
      hidden += self.noise_rec * self.rng.uniform(-1, -1, self.state.shape)
    # new state/output
    state = (1 - self.leaky_rate) * self.state + self.leaky_rate * hidden
    if self.activation_type == 'external':
      state = self.activation(state)
    self.state.value = state
    return state
