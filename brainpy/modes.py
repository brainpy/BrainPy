# -*- coding: utf-8 -*-


__all__ = [
  'Mode',
  'NormalMode',
  'BatchingMode',
  'TrainingMode',

  'normal',
  'batching',
  'training',
]


class Mode(object):
  def __repr__(self):
    return self.__class__.__name__


class NormalMode(Mode):
  """Normal non-batching mode."""
  pass


class BatchingMode(Mode):
  """Batching mode."""
  pass


class TrainingMode(BatchingMode):
  """Training mode requires data batching."""
  pass


normal = NormalMode()
batching = BatchingMode()
training = TrainingMode()

