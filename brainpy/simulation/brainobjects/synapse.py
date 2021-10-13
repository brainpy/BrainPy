# -*- coding: utf-8 -*-

from brainpy import errors, math
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
  steps : function, list of function, tuple of function, dict of (str, function), optional
      The step functions.
  pre : NeuGroup
      Pre-synaptic neuron group.
  post : NeuGroup
      Post-synaptic neuron group.
  conn : math.ndarray, dict of (str, math.ndarray), TwoEndConnector
      The connection method between pre- and post-synaptic groups.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  show_code : bool
      Whether show the formatted code.
  """

  def __init__(self, pre, post, conn=None, name=None, steps=('update',), **kwargs):
    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, NeuGroup):
      raise errors.BrainPyError('"pre" must be an instance of NeuGroup.')
    if not isinstance(post, NeuGroup):
      raise errors.BrainPyError('"post" must be an instance of NeuGroup.')
    if pre.num_batch != post.num_batch:
      raise errors.BrainPyError('pre.num_batch != post.num_batch')
    self.pre = pre
    self.post = post

    # connectivity
    # ------------
    if isinstance(conn, TwoEndConnector):
      self.conn = conn(pre.size, post.size)
    elif isinstance(conn, math.ndarray):
      if (pre.num, post.num) != conn.shape:
        raise errors.BrainPyError(f'"conn" is provided as a matrix, and it is expected '
                                  f'to be an array with shape of (pre.num, post.num) = '
                                  f'{(pre.num, post.num)}, however we got {conn.shape}')
      self.conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise errors.BrainPyError(f'"conn" is provided as a dict, and it is expected to '
                                  f'be a dictionary with "i" and "j" specification, '
                                  f'however we got {conn}')
      self.conn = IJConn(i=conn['i'], j=conn['j'])
    elif conn is None:
      self.conn = conn
    else:
      raise errors.BrainPyError(f'Unknown "conn" type: {conn}')

    # initialize
    # ----------
    super(TwoEndConn, self).__init__(steps=steps, name=name, **kwargs)
