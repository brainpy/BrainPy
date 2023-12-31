import jax.numpy as jnp
from jax import config

from brainpy._src.dependency_check import import_taichi
from .modes import NonBatchingMode
from .scales import IdScaling

__all__ = ['mode', 'membrane_scaling', 'dt', 'bool_', 'int_', 'ti_int', 'float_', 'ti_float', 'complex_']

ti = import_taichi()

# Default computation mode.
mode = NonBatchingMode()

# '''Default computation mode.'''
membrane_scaling = IdScaling()

# '''Default time step.'''
dt = 0.1

# '''Default bool data type.'''
bool_ = jnp.bool_

# '''Default integer data type.'''
int_ = jnp.int64 if config.read('jax_enable_x64') else jnp.int32

# '''Default integer data type in Taichi.'''
ti_int = ti.int64 if config.read('jax_enable_x64') else ti.int32

# '''Default float data type.'''
float_ = jnp.float64 if config.read('jax_enable_x64') else jnp.float32

# '''Default float data type in Taichi.'''
ti_float = ti.float64 if config.read('jax_enable_x64') else ti.float32

# '''Default complex data type.'''
complex_ = jnp.complex128 if config.read('jax_enable_x64') else jnp.complex64

