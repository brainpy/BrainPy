# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.integrators import odeint, sdeint, JointEq
from brainpy.types import Parameter, Shape
from .base import RateModel

__all__ = [
  ''
]

class JansenRitModel(RateModel):
  pass


class WilsonCowanModel(RateModel):
  pass


class StuartLandauOscillator(RateModel):
  pass


class KuramotoOscillator(RateModel):
  pass

