# -*- coding: utf-8 -*-


__all__ = [
  'Scale',
]


class Scale(object):
  def __init__(self, scale, bias):
    self.scale = scale
    self.bias = bias

  def scaling_offset(self, x, bias=None, scale=None):
    if bias is None:
      bias = self.bias
    if scale is None:
      scale = self.scale
    return (x + bias) / scale

  def scaling(self, x, scale=None):
    if scale is None:
      scale = self.scale
    return x / scale

  def scaling_inv(self, x, scale=None):
    if scale is None:
      scale = self.scale
    return x * scale

