# -*- coding: utf-8 -*-

import numba
import numpy
from numba.extending import overload


@overload(numpy.shape)
def shape_func(x):
  if isinstance(x, (numba.types.Integer, numba.types.Float)):
    def shape(x):
      return (1,)

    return shape
  else:
    return numpy.shape


@overload(numpy.clip)
def clip_func(x, x_min, x_max):
  def clip(x, x_min, x_max):
    x = numpy.maximum(x, x_min)
    x = numpy.minimum(x, x_max)
    return x

  return clip


@overload(numpy.squeeze)
def squeeze_func(a, axis=None):
  if isinstance(axis, numba.types.NoneType):
    def squeeze(a, axis=None):
      shape = []
      for s in a.shape:
        if s != 1:
          shape.append(s)
      return numpy.reshape(a, shape)

    return squeeze

  elif isinstance(axis, numba.types.Integer):
    def squeeze(a, axis=None):
      shape = []
      for i, s in enumerate(a.shape):
        if s != 1 or i != axis:
          shape.append(s)
      return numpy.reshape(a, shape)

    return squeeze

  else:
    def squeeze(a, axis=None):
      shape = []
      for i, s in enumerate(a.shape):
        if s != 1 or i not in axis:
          shape.append(s)
      return numpy.reshape(a, shape)

    return squeeze


def dsplit(): pass


def hsplit():  pass


def vsplit():  pass


def isreal(): pass


def isscalar(): pass


def meshgrid(): pass


def moveaxis(): pass


def ndim(): pass


def size(): pass


def take_along_axis(): pass


def tile(): pass
