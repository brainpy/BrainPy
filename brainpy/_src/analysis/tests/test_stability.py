# -*- coding: utf-8 -*-

from brainpy.analysis.stability import *


def test_d1():
  assert stability_analysis(1.) == UNSTABLE_POINT_1D
  assert stability_analysis(-1.) == STABLE_POINT_1D
  assert stability_analysis(0.) == SADDLE_NODE


