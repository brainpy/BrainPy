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

__all__ = [
    'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32',
    'float64', 'complex64', 'complex128',

    'bfloat16', 'half', 'float', 'double', 'cfloat', 'cdouble', 'short', 'int', 'long', 'bool'
]

uint8 = jnp.uint8
uint16 = jnp.uint16
uint32 = jnp.uint32
uint64 = jnp.uint64
int8 = jnp.int8
int16 = jnp.int16
int32 = jnp.int32
int64 = jnp.int64
float16 = jnp.float16
float32 = jnp.float32
float64 = jnp.float64
complex64 = jnp.complex64
complex128 = jnp.complex128

# data types in PyTorch
bfloat16 = jnp.bfloat16
half = jnp.float16
float = jnp.float32
double = jnp.float64
cfloat = jnp.complex64
cdouble = jnp.complex128
short = jnp.int16
int = jnp.int32
long = jnp.int64
bool = jnp.bool_

# missing types in PyTorch #
# chalf = np.complex32
# quint8 = jnp.quint8
# qint8 = jnp.qint8
# qint32 = jnp.qint32
# quint4x2 = jnp.quint4x2
