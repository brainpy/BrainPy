# -*- coding: utf-8 -*-


import brainpy as bp


def test_Normal1():
  init = bp.initialize.Normal()
  init((100, 300))
