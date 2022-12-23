# -*- coding: utf-8 -*-


__all__ = [
  'CompMode',
  'NonBatchingMode',
  'BatchingMode',
  'TrainingMode',
  'nonbatching_mode',
  'batching_mode',
  'training_mode',
]


class CompMode(object):
  """Base class for computation Mode
  """

  def __repr__(self):
    return self.__class__.__name__


class NonBatchingMode(CompMode):
  """Normal non-batching mode."""
  pass


class BatchingMode(CompMode):
  """Batching mode."""
  pass


class TrainingMode(BatchingMode):
  """Training mode requires data batching."""
  pass


nonbatching_mode = NonBatchingMode()
'''Non-batching computation mode.'''

batching_mode = BatchingMode()
'''Batching computation mode.'''

training_mode = TrainingMode()
'''Training computation mode.'''

