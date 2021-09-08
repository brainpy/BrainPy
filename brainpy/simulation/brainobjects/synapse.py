# -*- coding: utf-8 -*-

from brainpy import errors, math
from brainpy.simulation.connectivity import TwoEndConnector, MatConn, IJConn
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation.brainobjects.delays import ConstantDelay
from brainpy.simulation.brainobjects.neuron import NeuGroup

__all__ = [
  'TwoEndConn',
]


class TwoEndConn(DynamicalSystem):
  """Two End Synaptic Connections.

  Parameters
  ----------
  steps : function, list of function, tuple of function, dict of (str, function), optional
      The step functions.
  pre : NeuGroup
      Pre-synaptic neuron group.
  post : NeuGroup
      Post-synaptic neuron group.
  conn : math.ndarray, dict, TwoEndConnector
  monitors : list of str, tuple of str
      Variables to monitor.
  name : str
      The name of the neuron group.
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

  def register_constant_delay(self, key, size, delay, dtype=None):
    """Register a constant delay.

    Parameters
    ----------
    key : str
        The delay name.
    size : int, list of int, tuple of int
        The delay data size.
    delay : int, float, ndarray
        The delay time, with the unit same with `brainpy.math.get_dt()`.

    Returns
    -------
    delay : ConstantDelay
        An instance of ConstantDelay.
    """

    if not hasattr(self, 'steps'):
      raise errors.BrainPyError('Please initialize the super class first before '
                                'registering constant_delay. \n\n'
                                'super(YourClassName, self).__init__(**kwargs)')
    if not key.isidentifier():
      raise ValueError(f'{key} is not a valid identifier.')

    cdelay = ConstantDelay(size, delay, name=f'{self.name}_delay_{key}', dtype=dtype)
    self.steps[f'{key}_update'] = cdelay.update

    return cdelay

  def update(self, _t, _dt):
    raise NotImplementedError
