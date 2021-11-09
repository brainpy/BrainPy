# -*- coding: utf-8 -*-

import brainpy.math.jax as bm


def test_segment_sum():
  data = bm.arange(5)
  segment_ids = bm.array([0, 0, 1, 1, 2])
  print(bm.segment_sum(data, segment_ids, 3))

