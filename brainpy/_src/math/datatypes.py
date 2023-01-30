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

