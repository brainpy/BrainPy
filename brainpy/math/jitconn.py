from brainpy._src.dependency_check import import_taichi_else_None
if import_taichi_else_None() is not None:
  from brainpy._src.math.jitconn import (
    event_mv_prob_homo as event_mv_prob_homo,
    event_mv_prob_uniform as event_mv_prob_uniform,
    event_mv_prob_normal as event_mv_prob_normal,

    mv_prob_homo as mv_prob_homo,
    mv_prob_uniform as mv_prob_uniform,
    mv_prob_normal as mv_prob_normal,
  )

