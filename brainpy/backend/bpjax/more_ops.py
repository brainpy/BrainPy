# -*- coding: utf-8 -*-

from brainpy.backend import ops

import jax.numpy as jnp

__all__ = []

ops.set_buffer('jax',
               clip=jnp.clip,
               unsqueeze=jnp.expand_dims,
               squeeze=jnp.squeeze,
               )
