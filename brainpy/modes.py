# -*- coding: utf-8 -*-


__all__ = [
  'Mode',
  'NonBatching',
  'Batching',
  'Training',

  'nonbatching',
  'batching',
  'training',
]


class Mode(object):
  def __repr__(self):
    return self.__class__.__name__


class NonBatching(Mode):
  pass


class Batching(Mode):
  pass


class Training(Batching):
  pass


nonbatching = NonBatching()
batching = Batching()
training = Training()


