# -*- coding: utf-8 -*-


from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.brainobjects.container import Container

__all__ = [
  'BatchNorm',
]

class BatchNorm(DynamicSystem):
  pass


class BatchNorm0D(BatchNorm):
  pass


class BatchNorm1D(BatchNorm):
  pass


class BatchNorm2D(BatchNorm):
  pass


class SyncedBatchNorm(BatchNorm):
  pass


class SyncedBatchNorm0D(SyncedBatchNorm):
  pass


class SyncedBatchNorm1D(SyncedBatchNorm):
  pass


class SyncedBatchNorm2D(SyncedBatchNorm):
  pass


class Conv2D(DynamicSystem):
  pass


class Dropout(DynamicSystem):
  pass


class Linear(DynamicSystem):
  pass


class RNN(DynamicSystem):
  pass


class Sequential(Container):
  """Executes modules in the order they were passed to the constructor."""
  pass
