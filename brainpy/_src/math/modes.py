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

  def __eq__(self, other: 'Mode'):
    assert isinstance(other, Mode)
    return other.__class__ == self.__class__

  def is_a(self, mode: type):
    assert isinstance(mode, type)
    return self.__class__ == mode

  def is_parent_of(self, *modes):
    cls = self.__class__
    for smode in modes:
      if not isinstance(smode, type):
        raise TypeError(f'supported_types must be a tuple/list of type. But wwe got {smode}')
    if all([not issubclass(smode, cls) for smode in modes]):
      return False
    else:
      return True

  def is_child_of(self, *modes):
    for smode in modes:
      if not isinstance(smode, type):
        raise TypeError(f'supported_types must be a tuple/list of type. But wwe got {smode}')
    return isinstance(self, modes)


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

  def __init__(self, batch_size: int = 1):
    self.batch_size = batch_size

  def __repr__(self):
    return f'{self.__class__.__name__}(batch_size={self.batch_size})'


class TrainingMode(BatchingMode):
  """Training mode requires data batching."""
  pass


nonbatching_mode = NonBatchingMode()
'''Default instance of the non-batching computation mode.'''

batching_mode = BatchingMode()
'''Default instance of the batching computation mode.'''

training_mode = TrainingMode()
'''Default instance of the training computation mode.'''
