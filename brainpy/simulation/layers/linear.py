# -*- coding: utf-8 -*-


from brainpy.simulation._imports import mjax
from brainpy.simulation.connectivity.base import TwoEndConnector
from brainpy.simulation.brainobjects.neuron import NeuGroup
from brainpy.simulation.brainobjects.synapse import TwoEndConn
from brainpy.simulation.initialize import XavierNormal, Initializer, ZeroInit

__all__ = [
  'Linear'
]


class Linear(TwoEndConn):
  """A fully connected layer implemented as the dot product of inputs and
  weights.

  Parameters
  ----------
  pre : NeuGroup
    The pre-synaptic neuron group.
  post : NeuGroup
    The post-synaptic neuron group.
  conn : optional, math.ndarray, TwoEndConnector
    The synaptic connectivity.
  train_mask : optional, math.ndarray
    The training mask.
  w_init : Initializer
    Initializer for the weights.
  b_init : Initializer
    Initializer for the bias.
  has_bias : bool
    Whether has the bias to compute.
  name : str, optional
    The name of the neuron group.
  """

  def __init__(self, pre, post, conn=None, w_init=XavierNormal(), b_init=ZeroInit(),
               train_mask=None, has_bias=True, name=None):
    super(Linear, self).__init__(pre, post, conn, name=name)

    # parameters
    self.pre = pre
    self.post = post
    self.has_bias = has_bias

    # variables
    self.conn_mask = self.conn.requires('conn_mat')
    self.w = mjax.TrainVar(w_init((pre.num, post.num)) * self.conn_mask)
    if has_bias: self.b = mjax.TrainVar(b_init(post.num))
    if train_mask is not None:
      assert train_mask
      self.train_mask = train_mask

    del self.conn

  def update(self, x, **kwargs):
    """Returns the results of applying the linear transformation to input x."""
    y = mjax.dot(x, self.w)
    if self.has_bias: y += self.b
    return y
