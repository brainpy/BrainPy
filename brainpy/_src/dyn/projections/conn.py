from typing import Union, Dict, Optional

import jax
import numpy as np

from brainpy import math as bm
from brainpy._src.connect import TwoEndConnector, MatConn, IJConn
from brainpy._src.dynsys import Projection, DynamicalSystem
from brainpy.types import ArrayType

__all__ = [
  'SynConn',
]


class SynConn(Projection):
  """Base class to model two-end synaptic connections.

  Parameters
  ----------
  pre : NeuGroup
    Pre-synaptic neuron group.
  post : NeuGroup
    Post-synaptic neuron group.
  conn : optional, ndarray, ArrayType, dict, TwoEndConnector
    The connection method between pre- and post-synaptic groups.
  name : str, optional
    The name of the dynamic system.
  """

  def __init__(
      self,
      pre: DynamicalSystem,
      post: DynamicalSystem,
      conn: Union[TwoEndConnector, ArrayType, Dict[str, ArrayType]] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(name=name, mode=mode)

    # pre or post neuron group
    # ------------------------
    if not isinstance(pre, DynamicalSystem):
      raise TypeError('"pre" must be an instance of DynamicalSystem.')
    if not isinstance(post, DynamicalSystem):
      raise TypeError('"post" must be an instance of DynamicalSystem.')
    self.pre = pre
    self.post = post

    # connectivity
    # ------------
    if isinstance(conn, TwoEndConnector):
      self.conn = conn(pre.size, post.size)
    elif isinstance(conn, (bm.Array, np.ndarray, jax.Array)):
      if (pre.num, post.num) != conn.shape:
        raise ValueError(f'"conn" is provided as a matrix, and it is expected '
                         f'to be an array with shape of (pre.num, post.num) = '
                         f'{(pre.num, post.num)}, however we got {conn.shape}')
      self.conn = MatConn(conn_mat=conn)
    elif isinstance(conn, dict):
      if not ('i' in conn and 'j' in conn):
        raise ValueError(f'"conn" is provided as a dict, and it is expected to '
                         f'be a dictionary with "i" and "j" specification, '
                         f'however we got {conn}')
      self.conn = IJConn(i=conn['i'], j=conn['j'])
    elif isinstance(conn, str):
      self.conn = conn
    elif conn is None:
      self.conn = None
    else:
      raise ValueError(f'Unknown "conn" type: {conn}')

  def __repr__(self):
    names = self.__class__.__name__
    return (f'{names}(name={self.name}, mode={self.mode}, \n'
            f'{" " * len(names)} pre={self.pre}, \n'
            f'{" " * len(names)} post={self.post})')

  def check_pre_attrs(self, *attrs):
    """Check whether pre group satisfies the requirement."""
    if not hasattr(self, 'pre'):
      raise ValueError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise TypeError(f'Must be string. But got {attr}.')
      if not hasattr(self.pre, attr):
        raise ValueError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def check_post_attrs(self, *attrs):
    """Check whether post group satisfies the requirement."""
    if not hasattr(self, 'post'):
      raise ValueError('Please call __init__ function first.')
    for attr in attrs:
      if not isinstance(attr, str):
        raise TypeError(f'Must be string. But got {attr}.')
      if not hasattr(self.post, attr):
        raise ValueError(f'{self} need "pre" neuron group has attribute "{attr}".')

  def update(self, *args, **kwargs):
    """The function to specify the updating rule.

    Assume any dynamical system depends on the shared variables (`sha`),
    like time variable ``t``, the step precision ``dt``, and the time step `i`.
    """
    raise NotImplementedError('Must implement "update" function by subclass self.')

