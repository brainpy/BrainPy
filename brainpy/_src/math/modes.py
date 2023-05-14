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

  def is_one_of(self, *modes):
    for m_ in modes:
      if not isinstance(m_, type):
        raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
    return self.__class__ in modes

  def is_a(self, mode: type):
    """Check whether the mode is exactly the desired mode."""
    assert isinstance(mode, type), 'Must be a type.'
    return self.__class__ == mode

  def is_parent_of(self, *modes):
    """Check whether the mode is a parent of the given modes."""
    cls = self.__class__
    for m_ in modes:
      if not isinstance(m_, type):
        raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
    if all([not issubclass(m_, cls) for m_ in modes]):
      return False
    else:
      return True

  def is_child_of(self, *modes):
    """Check whether the mode is a children of one of the given modes."""
    for m_ in modes:
      if not isinstance(m_, type):
        raise TypeError(f'The supported type must be a tuple/list of type. But we got {m_}')
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
