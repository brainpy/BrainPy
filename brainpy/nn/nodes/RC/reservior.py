# -*- coding: utf-8 -*-

from brainpy.nn.base import Node

__all__ = [
  'Reservior',
]


class Reservior(Node):
  def __init__(self, name=None, in_size=None):
    super(Reservior, self).__init__(name=name, input_shape=in_size)

  def ff_init(self):
    pass

  def fb_init(self):
    pass

  def call(self, ff, fb=None, **kwargs):
    pass
