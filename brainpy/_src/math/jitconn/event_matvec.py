# -*- coding: utf-8 -*-

from typing import Tuple, Optional

import jax

from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found
from brainpy._src.math.jitconn.matvec import (mv_prob_homo,
                                              mv_prob_uniform,
                                              mv_prob_normal)

bti = import_braintaichi(error_if_not_found=False)

__all__ = [
    'event_mv_prob_homo',
    'event_mv_prob_uniform',
    'event_mv_prob_normal',
]


def event_mv_prob_homo(
    events: jax.Array,
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    if bti is None:
        raise_braintaichi_not_found()
    return bti.jitc_event_mv_prob_homo(events, weight, conn_prob, seed,
                                       shape=shape,
                                       transpose=transpose,
                                       outdim_parallel=outdim_parallel)


event_mv_prob_homo.__doc__ = mv_prob_homo.__doc__


def event_mv_prob_uniform(
    events: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    if bti is None:
        raise_braintaichi_not_found()
    return bti.jitc_event_mv_prob_uniform(events, w_low, w_high, conn_prob, seed, shape=shape,
                                          transpose=transpose, outdim_parallel=outdim_parallel)


event_mv_prob_uniform.__doc__ = mv_prob_uniform.__doc__


def event_mv_prob_normal(
    events: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    if bti is None:
        raise_braintaichi_not_found()
    return bti.jitc_event_mv_prob_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)


event_mv_prob_normal.__doc__ = mv_prob_normal.__doc__
