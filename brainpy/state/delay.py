# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate
import jax.numpy as jnp
import brainunit as u
from brainstate.typing import ArrayLike, Size

__all__ = [
    'InputDelay',
]


class InputDelay(brainstate.nn.Module):
    r"""Axonal delay line over a projection's pre-synaptic input.

    Buffers the signal that enters the communication module each step and reads
    it back delayed. A single ``brainstate.nn.Delay`` buffer over the
    pre-synaptic dimension supports two granularities depending on ``delay``:

    - ``delay is None`` — identity pass-through; no buffer is allocated, so a
      projection without a delay keeps its original behaviour at zero cost.
    - **scalar** ``delay`` (e.g. ``1.5 * u.ms``) — *global / homogeneous*: every
      pre-synaptic element is read at the same offset (a whole-frame slice).
    - **vector** ``delay`` of shape ``(N_pre,)`` with ``indices=None`` — *axonal*:
      each pre-synaptic element is read at its own offset via the diagonal gather
      ``retrieve_at_step(steps, arange(N_pre))``.
    - **vector** ``delay`` of shape ``(N_syn,)`` with ``indices`` of the same
      length — *heterogeneous / per-connection*: connection ``k`` reads
      pre-synaptic element ``indices[k]`` at its own offset, via the diagonal
      gather ``retrieve_at_step(steps, indices)``. The output is the per-connection
      delayed signal of shape ``(N_syn,)``. No extra buffer memory over the axonal
      case — only the gather index changes.

    The buffer is sized once at :meth:`init_state` from ``ceil(max(delay) / dt)``,
    so its depth is a static Python integer at trace time. Fractional (sub-``dt``)
    delays are honoured by linearly interpolating between the two bracketing
    frames, so ``delay`` need not be an integer multiple of ``dt``; an integer
    multiple reduces to an exact single-frame read.

    Parameters
    ----------
    in_size : Size
        Shape of the pre-synaptic signal (the communication module's input
        size). The last axis is the pre-synaptic neuron dimension.
    delay : ArrayLike or Quantity, optional
        ``None`` for no delay, a scalar time for a global delay, a ``(N_pre,)``
        array for an axonal (per-pre-neuron) delay, or a ``(N_syn,)`` array for a
        per-connection delay (requires ``indices``).
    indices : ArrayLike, optional
        Per-connection pre-synaptic indices (``pre_ids``) for a heterogeneous
        delay. When given, ``delay`` must be a 1-D array of the same length;
        connection ``k`` reads pre element ``indices[k]``.
    """
    __module__ = 'brainpy.state'

    def __init__(
        self,
        in_size: Size,
        delay: ArrayLike | u.Quantity | None = None,
        indices: ArrayLike | None = None,
    ):
        super().__init__()
        self.in_size = (in_size,) if isinstance(in_size, int) else tuple(in_size)
        self.delay = delay
        self.indices = indices
        self._buffer = None
        self._read_idx = None

    def init_state(self, *args, **kwargs):
        if self.delay is None:
            return
        dt = brainstate.environ.get_dt()
        delay = u.math.asarray(self.delay)
        # Buffer depth must be a static int at trace time -> derive from max delay.
        max_steps = int(u.math.ceil(u.math.max(delay) / dt))
        example = jnp.zeros(self.in_size, dtype=brainstate.environ.dftype())
        self._buffer = brainstate.nn.Delay(example, time=max_steps * dt,
                                           interp_method='linear_interp')
        brainstate.nn.init_all_states(self._buffer)
        # Resolve the per-step read index once (a bare retrieve_at_step(vector)
        # would be the outer product; the diagonal gather needs the index arg):
        #   scalar delay              -> None        (whole-population slice)
        #   (N_syn,) + indices given  -> indices     (per-connection diagonal)
        #   (N_pre,) axonal           -> arange(N_pre)
        if self.indices is not None:
            read_idx = jnp.asarray(self.indices)
            if u.math.ndim(delay) != 1 or delay.shape[0] != read_idx.shape[0]:
                raise ValueError(
                    f'Per-connection delay must be a 1-D array of length '
                    f'len(indices)={read_idx.shape[0]}, but got delay of shape {delay.shape}.'
                )
            self._read_idx = read_idx
        elif u.math.ndim(delay) == 0:
            self._read_idx = None
        else:
            self._read_idx = jnp.arange(self.in_size[-1])

    def update(self, x):
        if self.delay is None:
            return x
        self._buffer.update(x)
        # Per-element delay in (fractional) steps; delay/dt is dimensionless.
        steps = u.maybe_decimal(u.math.asarray(self.delay) / brainstate.environ.get_dt())
        lo = jnp.floor(steps).astype(int)
        hi = jnp.ceil(steps).astype(int)
        frac = steps - jnp.floor(steps)
        idx = () if self._read_idx is None else (self._read_idx,)
        # Linear interpolation between the two bracketing frames (exact when
        # ``steps`` is integral: frac == 0 so only the floor frame contributes).
        data_lo = self._buffer.retrieve_at_step(lo, *idx)
        data_hi = self._buffer.retrieve_at_step(hi, *idx)
        return data_lo * (1. - frac) + data_hi * frac
