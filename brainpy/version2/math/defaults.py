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
import jax.numpy as jnp
from jax import config

import brainstate
from .modes import NonBatchingMode
from .scales import IdScaling


class setting:
    def __init__(self):
        brainstate.environ.set(
            # Default computation mode.
            mode=NonBatchingMode(),
            # Default computation mode.
            membrane_scaling=IdScaling(),
            # Default time step.
            dt=0.1,
            # Default bool data type.
            bool_=jnp.bool_,
            # Default integer data type.
            int_=jnp.int64 if config.read('jax_enable_x64') else jnp.int32,
            # Default float data type.
            float_=jnp.float64 if config.read('jax_enable_x64') else jnp.float32,
            # Default complex data type.
            complex_=jnp.complex128 if config.read('jax_enable_x64') else jnp.complex64,
            # register brainpy object as pytree
            bp_object_as_pytree=False,
            # default return array type
            # numpy_func_return='jax_array',  # 'bp_array','jax_array'
            numpy_func_return='bp_array',  # 'bp_array','jax_array'
        )

    @property
    def mode(self):
        return brainstate.environ.get('mode')

    @property
    def membrane_scaling(self):
        return brainstate.environ.get('membrane_scaling')

    @property
    def dt(self):
        return brainstate.environ.get('dt')

    @property
    def bool_(self):
        return brainstate.environ.get('bool_')

    @property
    def int_(self):
        return brainstate.environ.get('int_')

    @property
    def float_(self):
        return brainstate.environ.get('float_')

    @property
    def complex_(self):
        return brainstate.environ.get('complex_')

    @property
    def bp_object_as_pytree(self):
        return brainstate.environ.get('bp_object_as_pytree')

    @property
    def numpy_func_return(self):
        return brainstate.environ.get('numpy_func_return')

    @mode.setter
    def mode(self, value):
        brainstate.environ.set(mode=value)

    @membrane_scaling.setter
    def membrane_scaling(self, value):
        brainstate.environ.set(membrane_scaling=value)

    @dt.setter
    def dt(self, value):
        brainstate.environ.set(dt=value)

    @bool_.setter
    def bool_(self, value):
        brainstate.environ.set(bool_=value)

    @int_.setter
    def int_(self, value):
        brainstate.environ.set(int_=value)

    @float_.setter
    def float_(self, value):
        brainstate.environ.set(float_=value)

    @complex_.setter
    def complex_(self, value):
        brainstate.environ.set(complex_=value)

    @bp_object_as_pytree.setter
    def bp_object_as_pytree(self, value):
        brainstate.environ.set(bp_object_as_pytree=value)

    @numpy_func_return.setter
    def numpy_func_return(self, value):
        brainstate.environ.set(numpy_func_return=value)


defaults = setting()
