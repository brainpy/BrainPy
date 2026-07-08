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
import dataclasses
from typing import Dict, Optional, Tuple

import jax
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

from brainpy import math as bm
from brainpy.context import share
from brainpy.dnn.base import Layer
from brainpy.dynsys import DynamicalSystem

try:
    import flax  # noqa
    from flax.linen.recurrent import RNNCellBase
except (ImportError, ModuleNotFoundError):
    flax = None  # type: ignore[assignment]  # optional dependency fallback
    RNNCellBase = object  # type: ignore[assignment, misc]  # optional dependency fallback base class

__all__ = [
    'FromFlax',
    'ToFlaxRNNCell',
    'ToFlax',
]


def _as_jax(a):
    if isinstance(a, bm.Array):
        return a.value
    else:
        return a


def _is_bp(a):
    return isinstance(a, bm.Array)


class FromFlax(Layer):
    """
    Transform a Flax module as a BrainPy :py:class:`~.DynamicalSystem`.

    Parameters
    ----------
    flax_module : Any
        The flax Module.
    module_args : Any
        The module arguments, used to initialize model parameters.
    module_kwargs : Any
        The module arguments, used to initialize model parameters.
    """

    def __init__(self, flax_module, *module_args, **module_kwargs):
        super().__init__()
        self.flax_module = flax_module
        params = self.flax_module.init(bm.random.split_key(),
                                       *tree_map(_as_jax, module_args, is_leaf=_is_bp),
                                       **tree_map(_as_jax, module_kwargs, is_leaf=_is_bp))
        leaves, self._tree = tree_flatten(params)
        self.variables = bm.VarList(tree_map(bm.TrainVar, leaves))

    def update(self, *args, **kwargs):
        params = tree_unflatten(self._tree, [v.value for v in self.variables])
        return self.flax_module.apply(params,
                                      *tree_map(_as_jax, args, is_leaf=_is_bp),
                                      **tree_map(_as_jax, kwargs, is_leaf=_is_bp))

    def reset_state(self, *args, **kwargs):
        pass


to_flax_doc = """Transform a BrainPy :py:class:`~.DynamicalSystem` into a Flax recurrent module."""

if flax is not None:
    class ToFlaxRNNCell(RNNCellBase):
        __doc__ = to_flax_doc

        model: DynamicalSystem
        train_params: Dict[str, jax.Array] = dataclasses.field(init=False)

        def initialize_carry(self, rng, input_shape: Tuple[int, ...]):
            batch_shape = input_shape[:-1]
            batch_dims: Optional[int]
            if len(batch_shape) == 1:
                batch_dims = 1
            elif len(batch_shape) == 0:
                batch_dims = None
            else:
                raise ValueError(f'Invalid input shape: {input_shape}')
            _state_vars = self.model.vars().unique().not_subset(bm.TrainVar)
            self.model.reset(batch_dims)
            return [_state_vars.dict(), 0, 0.]

        def setup(self):
            _vars = self.model.vars().unique()
            _train_vars = _vars.subset(bm.TrainVar)
            self.train_params = self.param(self.model.name, lambda rng, a: a.dict(), _train_vars)

        def __call__(self, carry, *inputs):
            """A recurrent cell that transformed from a BrainPy :py:class:`~.DynamicalSystem`.

            Parameters
            ----------
            carry
                the hidden state of the transformed recurrent cell, initialized using
                `.initialize_carry()` function in which the original `.reset_state()` is called.
            inputs
                an ndarray with the input for the current time step. All
                dimensions except the final are considered batch dimensions.

            Returns
            -------
                A tuple with the new carry and the output.
            """
            # shared arguments
            i, t = carry[1], carry[2]
            old_i = share.load('i', i)
            old_t = share.load('t', t)
            share.save(i=i, t=t)

            # carry
            _vars = self.model.vars().unique()
            _state_vars = _vars.not_subset(bm.TrainVar)
            for k, v in carry[0].items():
                _state_vars[k].value = v

            # train parameters
            _train_vars = _vars.subset(bm.TrainVar)
            for k, v in self.train_params.items():
                _train_vars[k].value = v

            # recurrent cell
            out = self.model(*inputs)

            # shared arguments
            share.save(i=old_i, t=old_t)
            # carray and output
            return [_state_vars.dict(), i + 1, t + share.dt], out

        @property
        def num_feature_axes(self) -> int:
            return 1

else:
    class ToFlaxRNNCell(object):  # type: ignore[no-redef]  # conditional stub when flax is unavailable
        __doc__ = to_flax_doc

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError('"flax" is not installed, or importing "flax" has errors. Please check.')

ToFlax = ToFlaxRNNCell
