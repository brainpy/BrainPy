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
from typing import Callable, Optional, Sequence

import brainpy.math as bm
from brainpy.dnn.base import Layer

__all__ = [
    'Activation',
    'Flatten',
    'Unflatten',
    'FunAsLayer',
]


class Activation(Layer):
    r"""Applies an activation function to the inputs

    Parameters:
    ----------
    activate_fun: Callable, function
      The function of Activation
    name: str, Optional
      The name of the object
    mode: Mode
      Enable training this node or not. (default True).
    """
    update_style = 'x'

    def __init__(
        self,
        activate_fun: Callable,
        name: Optional[str] = None,
        mode: bm.Mode = None,
        **kwargs,
    ):
        super().__init__(name, mode)
        self.activate_fun = activate_fun
        self.kwargs = kwargs

    def update(self, *args, **kwargs):
        return self.activate_fun(*args, **kwargs, **self.kwargs)


class Flatten(Layer):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
        name: str, Optional. The name of the object.
        mode: Mode. Enable training this node or not. (default True).

    Examples::
        >>> import brainpy.math as bm
        >>> inp = bm.random.randn(32, 1, 5, 5)
        >>> # With default parameters
        >>> m = Flatten()
        >>> output = m(inp)
        >>> output.shape
        (32, 25)
        >>> # With non-default parameters
        >>> m = Flatten(0, 2)
        >>> output = m(inp)
        >>> output.shape
        (160, 5)
    """

    def __init__(
        self,
        start_dim: int = 0,
        end_dim: int = -1,
        name: Optional[str] = None,
        mode: bm.Mode = None,
    ):
        super().__init__(name, mode)

        self.start_dim = start_dim
        self.end_dim = end_dim

    def update(self, x):
        if self.mode.is_child_of(bm.BatchingMode):
            start_dim = (self.start_dim + 1) if self.start_dim >= 0 else (x.ndim + self.start_dim + 1)
        else:
            start_dim = self.start_dim if self.start_dim >= 0 else x.ndim + self.start_dim
        return bm.flatten(x, start_dim, self.end_dim)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(start_dim={self.start_dim}, end_dim={self.end_dim})'


class Unflatten(Layer):
    r"""
    Unflattens a tensor dim expanding it to a desired shape. For use with :class:`~nn.Sequential`.

    * :attr:`dim` specifies the dimension of the input tensor to be unflattened, and it can
      be either `int` or `str` when `Tensor` or `NamedTensor` is used, respectively.

    * :attr:`unflattened_size` is the new shape of the unflattened dimension of the tensor and it can be
      a `tuple` of ints or a `list` of ints or `torch.Size` for `Tensor` input;  a `NamedShape`
      (tuple of `(name, size)` tuples) for `NamedTensor` input.

    Shape:
        - Input: :math:`(*, S_{\text{dim}}, *)`, where :math:`S_{\text{dim}}` is the size at
          dimension :attr:`dim` and :math:`*` means any number of dimensions including none.
        - Output: :math:`(*, U_1, ..., U_n, *)`, where :math:`U` = :attr:`unflattened_size` and
          :math:`\prod_{i=1}^n U_i = S_{\text{dim}}`.

    Args:
        dim: int, Dimension to be unflattened.
        sizes: Sequence of int. New shape of the unflattened dimension.

    Examples:
        >>> import brainpy as bp
        >>> import brainpy.math as bm
        >>> input = bm.random.randn(2, 50)
        >>> # With tuple of ints
        >>> m = bp.Sequential(
        >>>     bp.dnn.Linear(50, 50),
        >>>     Unflatten(1, (2, 5, 5))
        >>> )
        >>> output = m(input)
        >>> output.shape
        (2, 2, 5, 5)
        >>> # With torch.Size
        >>> m = bp.Sequential(
        >>>     bp.dnn.Linear(50, 50),
        >>>     Unflatten(1, [2, 5, 5])
        >>> )
        >>> output = m(input)
        >>> output.shape
        (2, 2, 5, 5)
    """

    def __init__(self, dim: int, sizes: Sequence[int], mode: bm.Mode = None, name: str = None) -> None:
        super().__init__(mode=mode, name=name)

        self.dim = dim
        self.sizes = sizes
        if isinstance(sizes, (tuple, list)):
            for idx, elem in enumerate(sizes):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " +
                                    "but found element of type {} at pos {}".format(type(elem).__name__, idx))
        else:
            raise TypeError("unflattened_size must be tuple or list, but found type {}".format(type(sizes).__name__))

    def update(self, x):
        dim = self.dim + 1 if self.mode.is_batch_mode() else self.dim
        return bm.unflatten(x, dim, self.sizes)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim}, sizes={self.sizes})'


class FunAsLayer(Layer):
    def __init__(
        self,
        fun: Callable,
        name: Optional[str] = None,
        mode: bm.Mode = None,
        **kwargs,
    ):
        super().__init__(name, mode)
        self._fun = fun
        self.kwargs = kwargs

    def update(self, *args, **kwargs):
        return self._fun(*args, **kwargs, **self.kwargs)
