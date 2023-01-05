# -*- coding: utf-8 -*-


__all__ = [
  'Mode',
  'NonBatchingMode',
  'BatchingMode',
  'TrainingMode',
  'nonbatching_mode',
  'batching_mode',
  'training_mode',
]


class Mode(object):
  """Base class for computation Mode
  """

  def __repr__(self):
    return self.__class__.__name__


class NonBatchingMode(Mode):
  """Normal non-batching mode.

  :py:class:`~.NonBatchingMode` is usually used in models of traditional
  computational neuroscience.
  """
  pass


class BatchingMode(Mode):
  """Batching mode.

  :py:class:`~.NonBatchingMode` is usually used in models of model trainings.
  """
  pass


class TrainingMode(BatchingMode):
  """Training mode requires data batching."""
  pass


nonbatching_mode = NonBatchingMode()
'''Default instance of the non-batching computation mode.'''

batching_mode = BatchingMode()
'''Default instance of the batching computation mode.'''

training_mode = TrainingMode()
'''Default instance of the training computation mode.'''

