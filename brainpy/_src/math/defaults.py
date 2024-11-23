import jax.numpy as jnp
from jax import config

from .modes import NonBatchingMode
from .scales import IdScaling

__all__ = ['mode', 'membrane_scaling', 'dt', 'bool_', 'int_', 'float_', 'complex_']

# Default computation mode.
mode = NonBatchingMode()

# Default computation mode.
membrane_scaling = IdScaling()

# Default time step.
dt = 0.1

# Default bool data type.
bool_ = jnp.bool_

# Default integer data type.
int_ = jnp.int64 if config.read('jax_enable_x64') else jnp.int32

# Default float data type.
float_ = jnp.float64 if config.read('jax_enable_x64') else jnp.float32

# Default complex data type.
complex_ = jnp.complex128 if config.read('jax_enable_x64') else jnp.complex64

# register brainpy object as pytree
bp_object_as_pytree = False


# default return array type
numpy_func_return = 'bp_array'  # 'bp_array','jax_array'
