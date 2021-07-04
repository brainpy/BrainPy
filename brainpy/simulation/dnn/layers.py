# -*- coding: utf-8 -*-

from brainpy import errors, tools
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.brainobjects.container import Container

__all__ = [
  'Module', 'Sequential',
  'BatchNorm',
]


class Module(DynamicSystem):
  def __init__(self, name=None):
    super(Module, self).__init__(name=self.unique_name(name, 'Module'),
                                 steps=None,
                                 monitors=None)

  def update(self, _t, _i):  # deprecated
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Sequential(Container):
  def __init__(self, *args, name=None, **kwargs):
    all_objects = {}
    # check "args"
    for arg in args:
      if not isinstance(arg, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(arg)}.')
      all_objects[arg.name] = arg

    # check "kwargs"
    for key, arg in kwargs.items():
      if not isinstance(arg, Module):
        raise errors.ModelUseError(f'Only support {Module.__name__}, '
                                   f'but we got {type(arg)}.')
      all_objects[key] = arg

    # initialize base class
    super(Sequential, self).__init__(name=self.unique_name(name, 'Sequential'),
                                     steps=None,
                                     monitors=None,
                                     **all_objects)

  def update(self, _t, _i):  # deprecated
    raise NotImplementedError

  def __call__(self, *args, **kwargs):
    raise NotImplementedError


class Linear(Module):
  """A fully connected layer implemented as the dot product of inputs and
  weights.

  Parameters
  ----------
  n_out : (int, tuple)
      Desired size or shape of layer output
  n_in : (int, tuple) or None
      The layer input size feeding into this layer
  init : (Initializer, optional)
      Initializer object to use for initializing layer weights
  """

  def __init__(self, n_out, n_in=None, init='glorot_uniform', name=None):
    self.n_out = n_out
    self.n_in = n_in
    self.out_shape = (None, n_out)
    self.init = get_init(init)

    self.W = None
    self.b = None
    self.dW = None
    self.db = None
    self.last_input = None
    super(Linear, self).__init__(name=self.unique_name(name, 'Linear'))


class Conv2D(Module):
  pass


class Dropout(Module):
  pass


class BatchNorm(Module):
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


class RNN(Module):
  pass
