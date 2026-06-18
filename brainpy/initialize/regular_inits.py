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
import braintools.init as _bt_init

from brainpy import math as bm, tools
from .base import _InterLayerInitializer

__all__ = [
    'ZeroInit',
    'Constant',
    'OneInit',
    'Identity',
]


class ZeroInit(_InterLayerInitializer):
    """Zero initializer.

    Initialize the weights with zeros.
    """

    def __call__(self, shape, dtype=None):
        shape = [tools.size2num(d) for d in shape]
        return bm.asarray(_bt_init.ZeroInit()(shape), dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__


class Constant(_InterLayerInitializer):
    """Constant initializer.

    Initialize the weights with the given values.

    Parameters::

    value : float, int, bm.ndarray
      The value to specify.
    """

    def __init__(self, value=1.):
        super(Constant, self).__init__()
        self.value = value

    def __call__(self, shape, dtype=None):
        shape = [tools.size2num(d) for d in shape]
        return bm.asarray(_bt_init.Constant(self.value)(shape), dtype=dtype)

    def __repr__(self):
        return f'{self.__class__.__name__}(value={self.value})'


class OneInit(Constant):
    """One initializer.
    """
    pass


class Identity(_InterLayerInitializer):
    """Returns the identity matrix.

    This initializer was proposed in (Le, et al., 2015) [1]_.

    Parameters::

    value : float
      The optional scaling factor.

    Returns::

    shape: tuple of int
      The weight shape/size.

    References::

    .. [1] Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. "A simple way to
           initialize recurrent networks of rectified linear units." arXiv preprint
           arXiv:1504.00941 (2015).
    """

    def __init__(self, value=1.):
        super(Identity, self).__init__()
        self.value = value

    def __call__(self, shape, dtype=None):
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, (tuple, list)):
            if len(shape) > 2:
                raise ValueError(f'Only support initialize 2D weights for {self.__class__.__name__}.')
        else:
            raise ValueError(f'Only support shape of int, or tuple/list of int '
                             f'in {self.__class__.__name__}, but we got {shape}.')
        shape = [tools.size2num(d) for d in shape]
        # brainpy treats a 1D shape ``(n,)`` as a square ``(n, n)`` identity
        # matrix; expand before delegating so braintools builds the same matrix.
        if len(shape) == 1:
            shape = [shape[0], shape[0]]
        return bm.asarray(_bt_init.Identity(scale=self.value)(tuple(shape)), dtype=dtype)

    def __repr__(self):
        return f'{self.__class__.__name__}(value={self.value})'
