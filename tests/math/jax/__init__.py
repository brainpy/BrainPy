# -*- coding: utf-8 -*-

from brainpy.math import jax as jnp


def try1():
  a = jnp.zeros(100)
  b = jnp.ones(100)
  c = a.__truediv__(b)
  print(c)

if __name__ == '__main__':
    try1()

