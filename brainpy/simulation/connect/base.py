# -*- coding: utf-8 -*-

import abc
from typing import Union, List, Tuple

import numpy as np

from brainpy import errors, tools, math
from brainpy.simulation.connect import formatter

__all__ = [
  'CONN_MAT',
  'PRE_IDS', 'POST_IDS',
  'PRE2POST', 'POST2PRE',
  'PRE2SYN', 'POST2SYN',
  'PRE_SLICE', 'POST_SLICE',
  'SUPPORTED_SYN_STRUCTURE',

  'PROVIDE_MAT', 'PROVIDE_IJ',

  'Connector', 'TwoEndConnector', 'OneEndConnector',
]

CONN_MAT = 'conn_mat'
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
  """Base Synaptical Connector Class."""
  pass


class TwoEndConnector(Connector):
  """Synaptical connector to build synapse connections between two neuron groups."""

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
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
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
    if len(structures) == 0 or CONN_MAT in structures:
      return PROVIDE_MAT
    else:
      return PROVIDE_IJ

  def returns(self, mat=None, ij=None):
    if not hasattr(self, 'structures'):
      raise errors.BrainPyError(f'Please call "self.check" first to get synaptic '
                                f'structures required. Error in {str(self)}')

    if mat is not None:
      assert isinstance(mat, np.ndarray), f'"mat" must be a numpy.ndarray, but got {type(mat)}'
    if ij is not None:
      assert isinstance(ij[0], np.ndarray), f'"ij[0]" must be a numpy.ndarray, but got {type(ij[0])}'
      assert isinstance(ij[1], np.ndarray), f'"ij[1]" must be a numpy.ndarray, but got {type(ij[1])}'

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
      all_data[PRE_IDS] = math.asarray(ij[0], dtype=math.int_)
      all_data[POST_IDS] = math.asarray(ij[1], dtype=math.int_)

      # check 'mat'
      if CONN_MAT in self.structures:
        if mat is None:
          mat = formatter.ij2mat(i=ij[0], j=ij[1], num_pre=self.pre_num, num_post=self.post_num)
      all_data[CONN_MAT] = math.asarray(mat, dtype=math.bool_)

      # names of the needed structures
      if PRE_SLICE in self.structures:
        r = formatter.pre_slice(i=ij[0], j=ij[1], num_pre=self.pre_num)
        all_data[PRE_IDS] = math.asarray(r[0], dtype=math.int_)
        all_data[POST_IDS] = math.asarray(r[1], dtype=math.int_)
        all_data[PRE_SLICE] = math.asarray(r[2], dtype=math.int_)

      elif POST_SLICE in self.structures:
        r = formatter.post_slice(i=ij[0], j=ij[1], num_post=self.post_num)
        all_data[PRE_IDS] = math.asarray(r[0], dtype=math.int_)
        all_data[POST_IDS] = math.asarray(r[1], dtype=math.int_)
        all_data[PRE_SLICE] = math.asarray(r[2], dtype=math.int_)

      for n in self.structures:
        if n in [PRE_SLICE, POST_SLICE, PRE_IDS, POST_IDS, CONN_MAT]:
          continue
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

  def requires(self, *structures):
    return self.require(*structures)

  def require(self, *structures):
    raise NotImplementedError


class OneEndConnector(TwoEndConnector):
  """Synaptical connector to build synapse connections within a population of neurons."""

  def __call__(self, pre_size, post_size=None):
    if post_size is None: post_size = pre_size
    else: assert pre_size == post_size
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    self.pre_num = tools.size2num(self.pre_size)
    self.post_num = tools.size2num(self.post_size)
    return self

