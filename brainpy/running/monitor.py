# -*- coding: utf-8 -*-

import numpy as np

from brainpy import math as bm
from brainpy.errors import MonitorError

__all__ = [
  'Monitor'
]


class Monitor(object):
  """The basic Monitor class to store the past variable trajectories.

  Currently, :py:class:`brainpy.simulation.Monitor` support to specify:

  - variable key by `strings`.
  - variable index by `None`, `int`, `list`, `tuple`, `1D array/tensor`
    (==> all will be transformed into a 1D array/tensor)
  - variable monitor interval by `None`, `int`, `float`

  Users can instance a monitor object by multiple ways:

  1. list of strings.

  >>> Monitor(variables=['a', 'b', 'c'])

  1.1. list of strings and list of intervals

  >>> Monitor(variables=['a', 'b', 'c'],
  >>>         intervals=[None, 1, 2] # ms
  >>>        )

  2. list of strings and string + indices

  >>> Monitor(variables=['a', ('b', bm.array([1,2,3])), 'c'])

  2.1. list of string (+ indices) and list of intervals

  >>> Monitor(variables=['a', ('b', bm.array([1,2,3])), 'c'],
  >>>         intervals=[None, 2, 3])

  3. a dictionary with the format of {key: indices}

  >>> Monitor(variables={'a': None, 'b': bm.array([1,2,3])})

  3.1. a dictionary of variable and indexes, and a dictionary of time intervals

  >>> Monitor(variables={'a': None, 'b': bm.array([1,2,3])},
  >>>         intervals={'b': 2.})

  .. note::
      :py:class:`brainpy.simulation.Monitor` records any target variable with an
      two-dimensional array/tensor with the shape of `(num_time_step, variable_size)`.
      This means for any variable, no matter what's the shape of the data
      (int, float, vector, matrix, 3D array/tensor), will be reshaped into a
      one-dimensional vector.

  """

  _KEYWORDS = ['_KEYWORDS', 'target', 'vars', 'intervals', 'ts', 'num_item',
               'item_names', 'item_indices', 'item_intervals', 'item_contents',
               'has_build']

  def __init__(self, variables, intervals=None):
    if isinstance(variables, (list, tuple)):
      if intervals is not None:
        if not isinstance(intervals, (list, tuple)):
          raise MonitorError(f'"vars" and "intervals" must be the same type. '
                             f'While we got type(vars)={type(variables)}, '
                             f'type(intervals)={type(intervals)}.')
        if len(variables) != len(intervals):
          raise MonitorError(f'The length of "vars" and "every" are not equal.')

    elif isinstance(variables, dict):
      if intervals is not None:
        if not isinstance(intervals, dict):
          raise MonitorError(f'"vars" and "every" must be the same type. '
                             f'While we got type(vars)={type(variables)}, '
                             f'type(intervals)={type(intervals)}.')
        for key in intervals.keys():
          if key not in variables:
            raise MonitorError(f'"{key}" is not in "vars": {list(variables.keys())}')

    else:
      raise MonitorError(f'We only supports a format of list/tuple/dict of '
                         f'"vars", while we got {type(variables)}.')

    self.has_build = False
    self.ts = None
    self.vars = variables
    self.intervals = intervals
    self.item_names = []
    self.item_indices = []
    self.item_intervals = []
    self.item_contents = dict()
    self.num_item = len(variables)
    self.build()

  def __repr__(self):
    return (f'{self.__class__.__name__}(items={tuple(self.item_names)}, '
            f'indices={self.item_indices})')

  def build(self):
    if not self.has_build:
      item_names = []
      item_indices = []
      item_contents = dict()

      if isinstance(self.vars, (list, tuple)):
        if self.intervals is None:
          item_intervals = [None] * len(self.vars)
        else:
          item_intervals = list(self.intervals)

        for mon_var, interval in zip(self.vars, item_intervals):
          # users monitor a variable by a string
          if isinstance(mon_var, str):
            mon_key = mon_var
            mon_idx = None
          # users monitor a variable by a tuple: `('b', bm.array([1,2,3]))`
          elif isinstance(mon_var, (tuple, list)):
            mon_key = mon_var[0]
            mon_idx = mon_var[1]
          else:
            raise MonitorError(f'Unknown monitor item: {str(mon_var)}')

          # self.check(mon_key)
          item_names.append(mon_key)
          item_indices.append(mon_idx)
          item_contents[mon_key] = []
          if interval is not None:
            item_contents[f'{mon_key}.t'] = []

      elif isinstance(self.vars, dict):
        item_intervals = []
        # users monitor a variable by a dict: `{'a': None, 'b': bm.array([1,2,3])}`
        for mon_key, mon_idx in self.vars.items():
          item_names.append(mon_key)
          item_indices.append(mon_idx)
          item_contents[mon_key] = []
          if self.intervals is None:
            item_intervals.append(None)
          else:
            if mon_key in self.intervals:
              item_intervals.append(self.intervals[mon_key])
            if self.intervals[mon_key] is not None:
              item_contents[f'{mon_key}.t'] = []
      else:
        raise MonitorError(f'Unknown monitors type: {type(self.vars)}')

      self.item_names = item_names
      self.item_indices = item_indices
      self.item_intervals = item_intervals
      self.item_contents = item_contents
      self.num_item = len(item_contents)
      self.has_build = True

  def __getitem__(self, item: str):
    """Get item in the monitor values.

    Parameters
    ----------
    item : str

    Returns
    -------
    value : ndarray
      The monitored values.
    """
    item_contents = super(Monitor, self).__getattribute__('item_contents')
    if item not in item_contents:
      raise ValueError(f'Do not have "{item}". Available items are:\n'
                       f'{list(item_contents.keys())}')
    return item_contents[item]

  def __setitem__(self, key, value):
    """Get item value in the monitor.

    Parameters
    ----------
    key : str
      The item key.
    value : ndarray
      The item value.
    """
    item_contents = super(Monitor, self).__getattribute__('item_contents')
    if key not in item_contents:
      raise ValueError(f'Do not have "{key}". Available items are:\n'
                       f'{list(item_contents.keys())}')
    self.item_contents[key] = value

  def __getattr__(self, item):
    if item in self._KEYWORDS:
      return super(Monitor, self).__getattribute__(item)
    else:
      item_contents = super(Monitor, self).__getattribute__('item_contents')
      if item in item_contents:
        return item_contents[item]
      else:
        super(Monitor, self).__getattribute__(item)

  def __setattr__(self, key, value):
    if key in self._KEYWORDS:
      object.__setattr__(self, key, value)
    elif key in self.item_contents:
      self.item_contents[key] = value
    else:
      object.__setattr__(self, key, value)

  def numpy(self):
    for key, val in self.item_contents.items():
      self.item_contents[key] = np.asarray(val)
    if self.ts is not None:
      self.ts = np.asarray(self.ts)
