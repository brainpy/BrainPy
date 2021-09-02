# -*- coding: utf-8 -*-


from brainpy import errors


def normal(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.normal" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.normal(xx)')


def sum(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.sum" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.sum(xx)')


def exp(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.exp" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.exp(xx)')


def shape(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.shape" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.shape(xx)')


def as_tensor(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.as_tensor" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.as_tensor(xx)')


def zeros(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.zeros" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.zeros(xx)')


def ones(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.ones" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.ones(xx)')


def arange(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.arange" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.arange(xx)')


def concatenate(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.concatenate" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.concatenate(xx)')


def where(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.where" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.where(xx)')


def reshape(*args, **kwargs):
  raise errors.NoLongerSupportError('"brainpy.ops.reshape" is no longer supported. \n'
                                    'Please instead use: \n\n'
                                    '>>> brainpy.math.reshape(xx)')
