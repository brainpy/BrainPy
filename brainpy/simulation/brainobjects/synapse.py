# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.errors import ModelBuildError
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.brainobjects.neuron import NeuGroup
from brainpy.simulation.connect import TwoEndConnector, MatConn, IJConn

__all__ = [
  'TwoEndConn',
]


class TwoEndConn(DynamicalSystem):
  """Base class to model two-end synaptic connections.

  Parameters
  ----------
  pre : NeuGroup
      Pre-synaptic neuron group.
  post : NeuGroup
      Post-synaptic neuron group.
  conn : optional, math.ndarray, dict of (str, math.ndarray), TwoEndConnector
      The connection method between pre- and post-synaptic groups.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, pre, post, conn=None, name=None, steps=('update',)):
    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, NeuGroup):
      raise ModelBuildError('"pre" must be an instance of NeuGroup.')
    if not isinstance(post, NeuGroup):
      raise ModelBuildError('"post" must be an instance of NeuGroup.')
    self.pre = pre
    self.post = post

    # connectivity
    # ------------
    if isinstance(conn, TwoEndConnector):
      self.conn = conn(pre.size, post.size)
    elif isinstance(conn, math.ndarray):
      if (pre.num, post.num) != conn.shape:
        raise ModelBuildError(f'"conn" is provided as a matrix, and it is expected '
                              f'to be an array with shape of (pre.num, post.num) = '
                              f'{(pre.num, post.num)}, however we got {conn.shape}')
      self.conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise ModelBuildError(f'"conn" is provided as a dict, and it is expected to '
                              f'be a dictionary with "i" and "j" specification, '
                              f'however we got {conn}')
      self.conn = IJConn(i=conn['i'], j=conn['j'])
    elif conn is None:
      self.conn = conn
    else:
      raise ModelBuildError(f'Unknown "conn" type: {conn}')

    # initialize
    # ----------
    super(TwoEndConn, self).__init__(steps=steps, name=name)

  def check_pre_attrs(self, *attrs):
    """Check whether pre group satisfies the requirement."""
    if not hasattr(self, 'pre'):
      raise ModelBuildError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise ValueError(f'Must be string. But got {attr}.')
      if not hasattr(self.pre, attr):
        raise ModelBuildError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def check_post_attrs(self, *attrs):
    """Check whether post group satisfies the requirement."""
    if not hasattr(self, 'post'):
      raise ModelBuildError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise ValueError(f'Must be string. But got {attr}.')
      if not hasattr(self.post, attr):
        raise ModelBuildError(f'{self} need "pre" neuron group has attribute "{attr}".')

