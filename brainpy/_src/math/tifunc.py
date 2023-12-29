from brainpy._src.dependency_check import import_taichi
from . import defaults

ti = import_taichi()

__all__ = [
  # taichi function for other utilities
  'warp_reduce_sum',

  # taichi functions for random number generator with LFSR88 algorithm
  'lfsr88_key', 'lfsr88_next_key', 'lfsr88_normal', 'lfsr88_randn',
  'lfsr88_random_integers', 'lfsr88_randint', 'lfsr88_uniform', 'lfsr88_rand',

  # taichi functions for random number generator with LFSR113 algorithm
  'lfsr113_key', 'lfsr113_next_key', 'lfsr113_normal', 'lfsr113_randn',
  'lfsr113_random_integers', 'lfsr113_randint', 'lfsr113_uniform', 'lfsr113_rand',
]


@ti.func
def _lcg_rand(state: ti.types.ndarray(ndim=1)):
  # LCG constants
  state[0] = ti.u32(1664525) * state[0] + ti.u32(1013904223)
  return state[0]


@ti.func
def taichi_lcg_rand(seed: ti.types.ndarray(ndim=1)):
  """
  Generate a random number using the Taichi LCG algorithm.

  Parameters:
    seed (ti.types.ndarray): The seed value for the random number generator.

  Returns:
    float: A random number between 0 and 1.
  """

  return float(_lcg_rand(seed)) / ti.u32(2 ** 32 - 1)


#############################################
# Random Number Generator: LFSR88 algorithm #
#############################################


@ti.func
def lfsr88_key(seed: ti.u32) -> ti.types.vector(4, ti.u32):
  """Initialize the random key of LFSR88 algorithm (Combined LFSR random number generator by L'Ecuyer).

  This key is used in LFSR88 based random number generator functions, like ``lfsr88_rand()``.

  Source:
  https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr88.c

  /**** VERY IMPORTANT **** :
    The initial seeds s1, s2, s3  MUST be larger than
    1, 7, and 15 respectively.
  */

  Args:
    seed: int. The seed value for the random number generator.

  Returns:
    ti.math.uvec4: The random key for the LFSR88 random number generator.
  """
  return ti.math.uvec4(ti.u32(seed + 1), ti.u32(seed + 7), ti.u32(seed + 15), ti.u32(0))


@ti.func
def lfsr88_next_key(key: ti.types.vector(4, ti.u32)) -> ti.types.vector(4, ti.u32):
  """Next random key of LFSR88 algorithm (Combined LFSR random number generator by L'Ecuyer).

  Args:
    key: The state value for the random number generator.

  Returns:
    ti.math.uvec4: The next random key.
  """
  b = ti.u32(((key[0] << 13) ^ key[0]) >> 19)
  s1 = ((key[0] & ti.u32(4294967294)) << 12) ^ b
  b = ((key[1] << 2) ^ key[1]) >> 25
  s2 = ((key[1] & ti.u32(4294967288)) << 4) ^ b
  b = ((key[2] << 3) ^ key[2]) >> 11
  s3 = ((key[2] & ti.u32(4294967280)) << 17) ^ b
  return ti.math.uvec4(s1, s2, s3, b)


@ti.func
def lfsr88_normal(key: ti.types.vector(4, ti.u32), mu, sigma, epsilon=1e-10):
  """
  Generate a random number of the normal distribution ``N(mu, sigma)`` using the LFSR88 algorithm.

  Args:
    key: The state value for the random number generator.
    mu: The mean of the normal distribution.
    sigma: The standard deviation of the normal distribution.
    epsilon: The epsilon value to avoid log(0).
  """

  key, r = lfsr88_randn(key, epsilon)
  return key, mu + sigma * r


@ti.func
def lfsr88_randn(key: ti.types.vector(4, ti.u32), epsilon=1e-10):
  """
  Generate a random number with the standard normal distribution using the LFSR88 algorithm.

  Args:
    key: The state value for the random number generator.
    epsilon: The epsilon value to avoid log(0).

  References:
    Box–Muller transform. https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    Marsaglia polar method. https://en.wikipedia.org/wiki/Marsaglia_polar_method

  """

  key, u1 = lfsr88_rand(key)
  key, u2 = lfsr88_rand(key)

  # Ensure state1 is not zero to avoid log(0)
  u1 = ti.cast(ti.max(u1, epsilon), defaults.ti_float)

  # Normalize the uniform samples
  mag = ti.cast(ti.sqrt(-2.0 * ti.log(u1)), defaults.ti_float)

  # Box-Muller transform
  # z1 = mag * ti.cos(2 * ti.math.pi * u2)
  z2 = ti.cast(mag * ti.sin(2 * ti.math.pi * u2), defaults.ti_float)

  return key, z2


@ti.func
def lfsr88_random_integers(key: ti.types.vector(4, ti.u32), low, high):
  """
  Generates a uniformly distributed random integer between `low` and `high` (inclusive) using the LFSR88 algorithm.

  Parameters:
    key: The state value used for random number generation.
    low: The lower bound of the range.
    high: The upper bound of the range.
  """
  key = lfsr88_next_key(key)
  return key, ti.cast((key[0] ^ key[1] ^ key[2]) % (high + 1 - low) + low, defaults.ti_int)


@ti.func
def lfsr88_randint(key: ti.types.vector(4, ti.u32), dtype=ti.u32):
  key = lfsr88_next_key(key)
  return key, dtype(key[0] ^ key[1] ^ key[2])


@ti.func
def lfsr88_uniform(key: ti.types.vector(4, ti.u32), low, high):
  """
  Generates a uniformly distributed random float between `low` and `high` (inclusive) using the LFSR88 algorithm.

  Args:
    key: The state value used for random number generation.
    low: The lower bound of the range.
    high: The upper bound of the range.
  """
  key = lfsr88_next_key(key)
  r = (key[0] ^ key[1] ^ key[2]) * ti.cast(2.3283064365386963e-10, defaults.ti_float)
  return key, ti.cast(r * (high - low) + low, defaults.ti_float)


@ti.func
def lfsr88_rand(key: ti.types.vector(4, ti.u32)):
  """
  Generates a uniformly distributed random float between 0 and 1 using the LFSR88 algorithm.

  Args:
    key: The state value used for random number generation.
  """
  key = lfsr88_next_key(key)
  return key, (key[0] ^ key[1] ^ key[2]) * ti.cast(2.3283064365386963e-10, defaults.ti_float)


##############################################
# Random Number Generator: LFSR113 algorithm #
##############################################


@ti.func
def lfsr113_key(seed: ti.u32) -> ti.types.vector(4, ti.u32):
  """Initialize the random key of LFSR113 algorithm (Combined LFSR random number generator by L'Ecuyer).

  This key is used in LFSR113 based random number generator functions, like ``lfsr113_rand()``.

  Source:
  https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr113.c

  /**** VERY IMPORTANT **** :
    The initial seeds s1, s2, s3, s4  MUST be larger than
    1, 7, 15, and 127 respectively.
  */

  Args:
    seed: int. The seed value for the random number generator.

  Returns:
    ti.math.uvec4: The random key for the LFSR113 random number generator.
  """
  return ti.math.uvec4(ti.u32(seed + 1), ti.u32(seed + 7), ti.u32(seed + 15), ti.u32(seed + 127))


@ti.func
def lfsr113_next_key(key: ti.types.vector(4, ti.u32)) -> ti.types.vector(4, ti.u32):
  """Next random key of LFSR113 algorithm (Combined LFSR random number generator by L'Ecuyer).

  Args:
    key: The state value for the random number generator.

  Returns:
    ti.math.uvec4: The next random key.
  """
  z1 = key[0]
  z2 = key[1]
  z3 = key[2]
  z4 = key[3]
  b = ((z1 << 6) ^ z1) >> 13
  z1 = ti.u32(((z1 & ti.u64(4294967294)) << 18) ^ b)
  b = ((z2 << 2) ^ z2) >> 27
  z2 = ti.u32(((z2 & ti.u64(4294967288)) << 2) ^ b)
  b = ((z3 << 13) ^ z3) >> 21
  z3 = ti.u32(((z3 & ti.u64(4294967280)) << 7) ^ b)
  b = ((z4 << 3) ^ z4) >> 12
  z4 = ti.u32(((z4 & ti.u64(4294967168)) << 13) ^ b)
  return ti.math.uvec4(z1, z2, z3, z4)


@ti.func
def lfsr113_normal(key: ti.types.vector(4, ti.u32), mu, sigma, epsilon=1e-10):
  """
  Generate a random number of the normal distribution ``N(mu, sigma)`` using the LFSR113 algorithm.

  Args:
    key: The state value for the random number generator.
    mu: The mean of the normal distribution.
    sigma: The standard deviation of the normal distribution.
    epsilon: The epsilon value to avoid log(0).
  """

  key, r = lfsr113_randn(key, epsilon)
  return key, ti.cast(mu + sigma * r, defaults.ti_float)


@ti.func
def lfsr113_randn(key: ti.types.vector(4, ti.u32), epsilon=1e-10):
  """
  Generate a random number with standard normal distribution using the LFSR113 algorithm.

  Args:
    key: The state value for the random number generator.
    epsilon: The epsilon value to avoid log(0).

  References:
    Box–Muller transform. https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    Marsaglia polar method. https://en.wikipedia.org/wiki/Marsaglia_polar_method

  """

  key, u1 = lfsr113_rand(key)
  key, u2 = lfsr113_rand(key)

  # Ensure state1 is not zero to avoid log(0)
  u1 = ti.cast(ti.max(u1, epsilon), defaults.ti_float)

  # Normalize the uniform samples
  mag = ti.cast(ti.sqrt(-2.0 * ti.log(u1)), defaults.ti_float)

  # Box-Muller transform
  # z1 = mag * ti.cos(2 * ti.math.pi * u2)
  z2 = ti.cast(mag * ti.sin(2 * ti.math.pi * u2), defaults.ti_float)

  return key, z2


@ti.func
def lfsr113_random_integers(key: ti.types.vector(4, ti.u32), low, high):
  """
  Generates a uniformly distributed random integer between `low` and `high` (inclusive) using the LFSR113 algorithm.

  Parameters:
    key: The state value used for random number generation.
    low: The lower bound of the range.
    high: The upper bound of the range.
  """
  key = lfsr113_next_key(key)
  return key, ti.cast((key[0] ^ key[1] ^ key[2] ^ key[3]) % (high + 1 - low) + low, defaults.ti_int)


@ti.func
def lfsr113_randint(key: ti.types.vector(4, ti.u32)):
  key = lfsr113_next_key(key)
  return key, ti.cast(key[0] ^ key[1] ^ key[2] ^ key[3], defaults.ti_int)


@ti.func
def lfsr113_uniform(key: ti.types.vector(4, ti.u32), low, high):
  """
  Generates a uniformly distributed random float between `low` and `high` (inclusive) using the LFSR113 algorithm.

  Args:
    key: The state value used for random number generation.
    low: The lower bound of the range.
    high: The upper bound of the range.
  """
  key = lfsr88_next_key(key)
  r = (key[0] ^ key[1] ^ key[2] ^ key[3]) * ti.cast(2.3283064365386963e-10, defaults.ti_float)
  return key, ti.cast(r * (high - low) + low, defaults.ti_float)


@ti.func
def lfsr113_rand(key: ti.types.vector(4, ti.u32)):
  """
  Generates a uniformly distributed random float between 0 and 1 using the LFSR113 algorithm.

  Args:
    key: The state value used for random number generation.
  """
  key = lfsr113_next_key(key)
  return key, (key[0] ^ key[1] ^ key[2] ^ key[3]) * ti.cast(2.3283064365386963e-10, defaults.ti_float)


###########################
# Reductions: warp reduce #
###########################


@ti.func
def warp_reduce_sum_all(val):
  """
  Warp reduce sum.

  Args:
    val (float): The value to be reduced.

  Returns:
    float: The reduced value.
  """
  for i in ti.static(range(1, 32)):
    val += ti.static(ti.simt.warp.shfl_xor(val, i))
  return val


@ti.func
def warp_reduce_sum(val):
  """
  Warp reduce sum.

  Args:
    val (float): The value to be reduced.

  Returns:
    float: The reduced value.
  """
  for offset in ti.static((16, 8, 4, 2, 1)):
    val += ti.simt.warp.shfl_down_f32(ti.u32(0xFFFFFFFF), val, offset)
  return val
