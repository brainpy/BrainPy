# -*- coding: utf-8 -*-


from brainpy.base.base import Base

__all__ = [
  'Integrator',
]


class Integrator(Base):
  """Basic Integrator Class."""

  # func_name
  # derivative
  # code_scope
  #
  def build(self, *args, **kwargs):
    raise NotImplementedError('Implement build method by yourself.')


