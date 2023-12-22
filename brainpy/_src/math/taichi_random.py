from brainpy._src.dependency_check import import_taichi

ti = import_taichi()

__all__ = [
  # taichi.func random generator implementation
  'taichi_lcg_rand', 'taichi_uniform_int_distribution',
  'taichi_uniform_real_distribution', 'taichi_normal_distribution',
  'taichi_lfsr88', 'taichi_lfsr88_init',
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


@ti.func
def taichi_lfsr88(seeds):
  """
  32-bits Random number generator U[0,1): lfsr88
  Author: Pierre L'Ecuyer,
  Source:
  https://github.com/cmcqueen/simplerandom/blob/main/c/lecuyer/lfsr88.c
  /**** VERY IMPORTANT **** :
    The initial seeds s1, s2, s3  MUST be larger than
    1, 7, and 15 respectively.
  */
  ```cpp
  double taus88_double ()
  {                   /* Generates numbers between 0 and 1. */
  b = (((s1 << 13) ^ s1) >> 19);
  s1 = (((s1 & 4294967294) << 12) ^ b);
  b = (((s2 << 2) ^ s2) >> 25);
  s2 = (((s2 & 4294967288) << 4) ^ b);
  b = (((s3 << 3) ^ s3) >> 11);
  s3 = (((s3 & 4294967280) << 17) ^ b);
  return ((s1 ^ s2 ^ s3) * 2.3283064365386963e-10);
  }
  ```
  """
  b = ti.cast((((seeds[0] << 13) ^ seeds[0]) >> 19), ti.u32);
  s1 = (((seeds[0] & ti.u32(4294967294)) << 12) ^ b)
  b = (((seeds[1] << 2) ^ seeds[1]) >> 25)
  s2 = (((seeds[1] & ti.u32(4294967288)) << 4) ^ b)
  b = (((seeds[2] << 3) ^ seeds[2]) >> 11)
  s3 = (((seeds[2] & ti.u32(4294967280)) << 17) ^ b)
  return ti.math.uvec4(s1, s2, s3, b), ((s1 ^ s2 ^ s3) * ti.f32(2.3283064365386963e-10))

@ti.func
def taichi_lfsr88_init(seed: ti.u32):
  """
  Initialize the seeds for the LFSR88 random number generator.

  Args:
    seed (int): The seed value for the random number generator.

  Returns:
    ti.math.uvec4: The seeds for the LFSR88 random number generator.
  """
  return ti.math.uvec4(seed + 1, seed + 7, seed + 15, ti.u32(0))

@ti.func
def taichi_uniform_int_distribution(state: ti.f32, low: ti.i32, high: ti.i32):
  """
  Generates a uniformly distributed random integer between `low` and `high` (inclusive).

  Parameters:
    state (ti.f32): The state value used for random number generation.
    low (ti.i32): The lower bound of the range.
    high (ti.i32): The upper bound of the range.

  Returns:
    ti.i32: A random integer between `low` and `high`.
  """
  return ti.cast(ti.floor(state * (high - low) + low), ti.i32)


@ti.func
def taichi_uniform_real_distribution(state: ti.f32, low: ti.f32, high: ti.f32):
  """
  Generate a random real number in the range [low, high).

  Args:
    state (float): The state value for the random number generator.
    low (float): The lower bound of the range.
    high (float): The upper bound of the range.

  Returns:
    float: A random real number in the range [low, high).
  """
  return state * (high - low) + low


@ti.func
def taichi_normal_distribution(state1: ti.f32, state2: ti.f32, mu: ti.f32, sigma: ti.f32):
  """
  Generate a random number with normal distribution.

  Args:
    state1 (float): The first state value for the random number generator, uniform in [0, 1).
    state2 (float): The second state value for the random number generator, uniform in [0, 1).
    mu (float): The mean of the normal distribution.
    sigma (float): The standard deviation of the normal distribution.

  Returns:
    float: A random number with normal distribution.
  """
  # Ensure state1 is not zero to avoid log(0)
  epsilon = 1e-10
  state1 = ti.max(state1, epsilon)

  # Box-Muller transform
  z = ti.sqrt(-2 * ti.log(state1)) * ti.sin(2 * ti.math.pi * state2)

  # Return the value, scaled by sigma and shifted by mu
  return mu + sigma * z
