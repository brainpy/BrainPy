# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Tuple, Optional

import brainevent
import jax
import numpy as np

from brainpy.math.jitconn.matvec import (mv_prob_homo,
                                         mv_prob_uniform,
                                         mv_prob_normal)
from brainpy.math.ndarray import Array as Array

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
    if seed is None:
        seed = np.random.randint(0, 1000000000)

    if isinstance(events, Array):
        events = events.value
    if isinstance(weight, Array):
        weight = weight.value

    events = brainevent.EventArray(events)
    csr = brainevent.JITCHomoR((weight, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return events @ csr
    else:
        return csr @ events


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
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(events, Array):
        events = events.value
    events = brainevent.EventArray(events)
    if isinstance(w_low, Array):
        w_low = w_low.value
    if isinstance(w_high, Array):
        w_high = w_high.value

    csr = brainevent.JITCUniformR((w_low, w_high, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return events @ csr
    else:
        return csr @ events


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
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(events, Array):
        events = events.value
    events = brainevent.EventArray(events)
    if isinstance(w_mu, Array):
        w_mu = w_mu.value
    if isinstance(w_sigma, Array):
        w_sigma = w_sigma.value

    csr = brainevent.JITCNormalR((w_mu, w_sigma, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return events @ csr
    else:
        return csr @ events


event_mv_prob_normal.__doc__ = mv_prob_normal.__doc__
