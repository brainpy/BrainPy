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
from typing import Callable, Optional

import numpy as np

import brainpy.math as bm
from brainpy import check
from brainpy.types import ArrayType
from .base import Encoder

__all__ = [
    'LatencyEncoder',
    'WeightedPhaseEncoder',
]


class WeightedPhaseEncoder(Encoder):
    r"""Encode the rate input into the spike train according to [1]_.

    The main idea of the weighted spikes is assigning different weights
    to different phases (or to spikes in those phases) in order to pack
    more information into the spikes. This is the major difference from
    a conventional rate coding scheme that assigns the same weight to every spike [1]_.

    Parameters::

    min_val: float
      The minimal value in the given data `x`, used to the data normalization.
    max_val: float
      The maximum value in the given data `x`, used to the data normalization.
    num_phase: int
      The number of the encoding period.
    weight_fun: Callable
      The function to generate weight at the phase :math:`i`.

    References::

    .. [1] Kim, Jaehyun et al. “Deep neural networks with weighted spikes.” Neurocomputing 311 (2018): 373-386.
    """

    def __init__(self,
                 min_val: float,
                 max_val: float,
                 num_phase: int,
                 weight_fun: Callable = None):
        super().__init__()

        check.is_callable(weight_fun, 'weight_fun', allow_none=True)
        self.num_phase = check.is_integer(num_phase, 'num_phase', min_bound=1)
        self.min_val = check.is_float(min_val, 'min_val')
        self.max_val = check.is_float(max_val, 'max_val')
        self.weight_fun = (lambda i: 2 ** (-(i % num_phase + 1))) if weight_fun is None else weight_fun
        self.scale = (1 - self.weight_fun(self.num_phase - 1)) / (self.max_val - self.min_val)

    def __call__(self, x: ArrayType, num_step: int):
        """Encoding function.

        Parameters::

        x: ArrayType
          The input rate value.
        num_step: int
          The number of time steps.

        Returns::

        out: ArrayType
          The encoded spike train.
        """
        # normalize all input signals to fit into the range [1, 1-2^K]
        x = (x - self.min_val) * self.scale

        # run
        inputs = bm.Variable(x)

        def f(i):
            w = self.weight_fun(i)
            spike = inputs >= w
            inputs.value -= w * spike
            return spike

        return bm.for_loop(f, bm.arange(num_step).value)


class LatencyEncoder(Encoder):
    r"""Encode the rate input as the spike train using the latency encoding.

    Use input features to determine time-to-first spike.

    Expected inputs should be between 0 and 1. If not, the latency encoder will encode ``x``
    (normalized into ``[0, 1]`` according to
    :math:`x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}`)
    to spikes whose firing time is :math:`0 \le t_f \le \text{num_period}-1`.
    A larger ``x`` will cause the earlier firing time.


    Example::

      >>> a = bm.array([0.02, 0.5, 1])
      >>> encoder = LatencyEncoder(method='linear', normalize=True)
      >>> encoder.multi_steps(a, n_time=5)
      Array([[0., 0., 1.],
             [0., 0., 0.],
             [0., 1., 0.],
             [0., 0., 0.],
             [1., 0., 0.]])


    Args:
      min_val: float. The minimal value in the given data `x`, used to the data normalization.
      max_val: float. The maximum value in the given data `x`, used to the data normalization.
      method: str. How to convert intensity to firing time. Currently, we support `linear` or `log`.
        - If ``method='linear'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1)(1 - x)`.
        - If ``method='log'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1) - ln(\alpha * x + 1)`,
          where :math:`\alpha` satisfies :math:`t_f(1) = \text{num_period} - 1`.
      threshold: float. Input features below the threhold will fire at the
        final time step unless ``clip=True`` in which case they will not
        fire at all, defaults to ``0.01``.
      clip: bool. Option to remove spikes from features that fall
          below the threshold, defaults to ``False``.
      tau: float. RC Time constant for LIF model used to calculate
        firing time, defaults to ``1``.
      normalize: bool. Option to normalize the latency code such that
        the final spike(s) occur within num_steps, defaults to ``False``.
      epsilon: float. A tiny positive value to avoid rounding errors when
        using torch.arange, defaults to ``1e-7``.
    """

    def __init__(
        self,
        min_val: float = None,
        max_val: float = None,
        method: str = 'log',
        threshold: float = 0.01,
        clip: bool = False,
        tau: float = 1.,
        normalize: bool = False,
        first_spk_time: float = 0.,
        epsilon: float = 1e-7,
    ):
        super().__init__()

        if method not in ['linear', 'log']:
            raise ValueError('The conversion method can only be "linear" and "log".')
        self.method = method
        self.min_val = check.is_float(min_val, 'min_val', allow_none=True)
        self.max_val = check.is_float(max_val, 'max_val', allow_none=True)
        if threshold < 0 or threshold > 1:
            raise ValueError(f"``threshold`` [{threshold}] must be between [0, 1]")
        self.threshold = threshold
        self.clip = clip
        self.tau = tau
        self.normalize = normalize
        self.first_spk_time = check.is_float(first_spk_time)
        self.first_spk_step = int(first_spk_time / bm.get_dt())
        self.epsilon = epsilon

    def single_step(self, x, i_step: int = None):
        raise NotImplementedError

    def multi_steps(self, data, n_time: Optional[float] = None):
        """Generate latency spikes according to the given input data.

        Ensuring x in [0., 1.].

        Args:
          data: The rate-based input.
          n_time: float. The total time to generate data. If None, use ``tau`` instead.

        Returns:
          out: array. The output spiking trains.
        """
        if n_time is None:
            n_time = self.tau
        tau = n_time if self.normalize else self.tau
        x = data
        if self.min_val is not None and self.max_val is not None:
            x = (x - self.min_val) / (self.max_val - self.min_val)
        if self.method == 'linear':
            spike_time = (tau - self.first_spk_time - bm.dt) * (1 - x) + self.first_spk_time

        elif self.method == 'log':
            x = bm.maximum(x, self.threshold + self.epsilon)  # saturates all values below threshold.
            spike_time = (tau - self.first_spk_time - bm.dt) * bm.log(x / (x - self.threshold)) + self.first_spk_time

        else:
            raise ValueError(f'Unsupported method: {self.method}. Only support "log" and "linear".')

        if self.clip:
            spike_time = bm.where(data < self.threshold, np.inf, spike_time)
        spike_steps = bm.round(spike_time / bm.get_dt()).astype(int)
        return bm.one_hot(spike_steps, num_classes=int(n_time / bm.get_dt()), axis=0, dtype=x.dtype)
