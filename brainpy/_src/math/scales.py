# -*- coding: utf-8 -*-


__all__ = [
  'Scaling',
  'IdScaling',
]


class Scaling(object):
  def __init__(self, scale, bias):
    self.scale = scale
    self.bias = bias

  @classmethod
  def transform(cls, V_range:list, scaled_V_range:list):
    '''
    V_range: [V_min, V_max]
    scaled_V_range: [scaled_V_min, scaled_V_max]
    '''
    V_min, V_max = V_range
    scaled_V_min, scaled_V_max = scaled_V_range
    scale = (V_max - V_min) / (scaled_V_max - scaled_V_min)
    bias = V_min - scaled_V_min * scale
    return cls(scale=scale, bias=bias)

  def offset_scaling(self, x, bias=None, scale=None):
    if bias is None:
      bias = self.bias
    if scale is None:
      scale = self.scale
    return (x + bias) / scale

  def std_scaling(self, x, scale=None):
    if scale is None:
      scale = self.scale
    return x / scale

  def inv_scaling(self, x, scale=None):
    if scale is None:
      scale = self.scale
    return x * scale

  def clone(self, bias=None, scale=None):
    if bias is None:
      bias = self.bias
    if scale is None:
      scale = self.scale
    return Scaling(bias=bias, scale=scale)


class IdScaling(Scaling):
  def __init__(self):
    super().__init__(scale=1., bias=0.)

  def offset_scaling(self, x, bias=None, scale=None):
    return x

  def std_scaling(self, x, scale=None):
    return x

  def inv_scaling(self, x, scale=None):
    return x

  def clone(self, bias=None, scale=None):
    return IdScaling()
