# -*- coding: utf-8 -*-

from importlib import import_module
from typing import Union, Tuple

ShapeOrScalar = Union[Tuple[int, ...], int]

jax = None
key = None


def _check():
    global jax
    if jax is None:
        jax = import_module('jax')


def _get_subkey():
    _check()
    global key
    if key is None:
        key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    return subkey


def seed(seed=0):
    _check()
    global key
    key = jax.random.PRNGKey(seed)


def uniform(low: float = 0.0, high: float = 1.0, size: ShapeOrScalar = 1):
    subkey = _get_subkey()
    return jax.random.uniform(subkey, size, minval=low, maxval=high)


def rand(*size):
    return uniform(low=0., high=1., size=size)


def randint(low, high=None, size=None):
    if high is None:
        low, high = 0, low
    subkey = _get_subkey()
    return jax.random.randint(subkey, size, minval=low, maxval=high)


def normal(mean: float = 0.0, stddev: float = 1.0, size: ShapeOrScalar = 1):
    subkey = _get_subkey()
    return jax.random.normal(subkey, size) * stddev + mean


def randn(*size):
    subkey = _get_subkey()
    return jax.random.normal(subkey, size)


def random(size=None):
    size = (1,) if size is None else size
    return uniform(low=0., high=1., size=size)
