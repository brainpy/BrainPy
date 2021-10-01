# -*- coding: utf-8 -*-

import abc
from typing import Union, List, Tuple

from brainpy import errors
from brainpy.simulation import utils
from brainpy.simulation.connectivity import formatter

__all__ = [
  'CONN_MAT',
  'PRE_IDS', 'POST_IDS',
  'PRE2POST', 'POST2PRE',
  'PRE2SYN', 'POST2SYN',
  'PRE_SLICE', 'POST_SLICE',
  'SUPPORTED_SYN_STRUCTURE',

  'PROVIDE_MAT', 'PROVIDE_IJ',

  'Connector', 'TwoEndConnector',
]

CONN_MAT = 'mat'
PRE_IDS = 'pre_ids'
POST_IDS = 'post_ids'
PRE2POST = 'pre2post'
POST2PRE = 'post2pre'
PRE2SYN = 'pre2syn'
POST2SYN = 'post2syn'
PRE_SLICE = 'pre_slice'
POST_SLICE = 'post_slice'

SUPPORTED_SYN_STRUCTURE = [CONN_MAT,
                           PRE_IDS, POST_IDS,
                           PRE2POST, POST2PRE,
                           PRE2SYN, POST2SYN,
                           PRE_SLICE, POST_SLICE]

PROVIDE_MAT = 'mat'
PROVIDE_IJ = 'ij'


class Connector(abc.ABC):
  pass


class TwoEndConnector(Connector):
  """Abstract connector class for two end connections."""

  def __init__(self):
    # synaptic structures
    self.pre_ids = None
    self.post_ids = None
    self.conn_mat = None
    self.pre2post = None
    self.post2pre = None
    self.pre2syn = None
    self.post2syn = None
    self.pre_slice = None
    self.post_slice = None

    # synaptic weights
    self.weights = None

  def __call__(self, pre_size, post_size):
    """Create the concrete connections between two end objects.

    Parameters
    ----------
    pre_size : int, tuple of int, list of int
        The size of the pre-synaptic group.
    post_size : int, tuple of int, list of int
        The size of the post-synaptic group.

    Returns
    -------
    conn : TwoEndConnector
        Return the self.
    """
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = utils.size2len(self.pre_size)
    self.post_num = utils.size2len(self.post_size)
    return self

  def check(self, structures: Union[Tuple, List, str]):
    if (not hasattr(self, 'pre_size')) or (not hasattr(self, 'post_size')):
      raise errors.BrainPyError(f'Please call "__call__" first to gather the size of the '
                                f'pre-synaptic and post-synaptic neuron groups for: {str(self)}')

    # check synaptic structures
    if isinstance(structures, str):
      structures = (structures,)
    if 'pre_slice' in structures and 'post_slice' in structures:
      raise errors.BrainPyError('Cannot use "pre_slice" and "post_slice" simultaneously. \n'
                                'We recommend you use "pre_slice + post2syn" or "post_slice + pre2syn".')
    self.structures = structures

    # get synaptic structures
    for n in structures:
      if n not in SUPPORTED_SYN_STRUCTURE:
        raise ValueError(f'Unknown synapse structure {n}. We only '
                         f'support {SUPPORTED_SYN_STRUCTURE}.')

    # provide what synaptic structure?
    if CONN_MAT in structures:
      return PROVIDE_MAT
    else:
      return PROVIDE_IJ

  def returns(self, mat=None, ij=None):
    if not hasattr(self, 'structures'):
      raise errors.BrainPyError(f'Please call "self.check" first to get synaptic '
                                f'structures required. Error in {str(self)}')

    if len(self.structures) == 1 and self.structures[0] == CONN_MAT:
      if mat is None:
        mat = formatter.ij2mat(i=ij[0], j=ij[1], num_pre=self.pre_num, num_post=self.post_num)
      return mat
    else:
      all_data = dict()

      # check 'ij'
      if ij is None:
        if mat is None:
          raise errors.BrainPyError(f'"mat" and "ij" are both none, please provide at least one of them.')
        ij = formatter.mat2ij(mat)
      all_data[PRE_IDS] = ij[0]
      all_data[POST_IDS] = ij[1]

      # check 'mat'
      if CONN_MAT in self.structures:
        if mat is None:
          mat = formatter.ij2mat(i=ij[0], j=ij[1], num_pre=self.pre_num, num_post=self.post_num)
      all_data[CONN_MAT] = mat

      # names of the needed structures
      if PRE_SLICE in self.structures:
        all_data[PRE_SLICE] = formatter.pre_slice(i=ij[0], j=ij[1], num_pre=self.pre_num)
      elif POST_SLICE in self.structures:
        all_data[POST_SLICE] = formatter.post_slice(i=ij[0], j=ij[1], num_post=self.post_num)
      for n in self.structures:
        if n in [PRE_SLICE, POST_SLICE, PRE_IDS, POST_IDS, CONN_MAT]: continue
        elif n == PRE2POST:
          all_data[PRE2POST] = formatter.pre2post(i=ij[0], j=ij[1], num_pre=self.pre_num)
        elif n == PRE2SYN:
          all_data[PRE2SYN] = formatter.pre2syn(i=ij[0], num_pre=self.pre_num)
        elif n == POST2PRE:
          all_data[POST2PRE] = formatter.post2pre(i=ij[0], j=ij[1], num_post=self.post_num)
        elif n == POST2SYN:
          all_data[POST2SYN] = formatter.post2syn(j=ij[1], num_post=self.post_num)
        else:
          raise errors.BrainPyError

      # data of the needed structures
      if len(self.structures) == 1:
        return all_data[self.structures[0]]
      else:
        return tuple([all_data[n] for n in self.structures])

  def require(self, *structures):
    raise NotImplementedError

  def requires(self, *structures):
    return self.require(*structures)
