# -*- coding: utf-8 -*-
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
from brainstate.transform import unvmap

from .ndarray import Array

__all__ = [
    'remove_vmap'
]


def remove_vmap(x, op='any'):
    """Reduce ``x`` with ``any``/``all`` *across the vmap batch axis as well*.

    This is a thin backward-compatible alias for
    :func:`brainstate.transform.unvmap`, which is the actively maintained
    implementation. ``unvmap`` collapses the batch axis into a single
    **global** scalar: when called under :func:`jax.vmap`,
    ``remove_vmap(x, 'any')`` returns one ``bool`` summarising *all* batch
    elements together (``True`` if any element of any batch is truthy), rather
    than a per-batch vector of results.

    This is intentional: the primitive is used for global convergence / NaN-style
    checks where the batch dimension must not survive the reduction. Delegating to
    :func:`brainstate.transform.unvmap` keeps BrainPy compatible across JAX
    releases (jax ``>= 0.10`` removed ``jax.interpreters.batching.not_mapped``,
    which the previous in-tree primitive relied on).

    Parameters
    ----------
    x : Array or jax.Array
        The input array. ``brainpy.math.Array`` inputs are unwrapped.
    op : {'any', 'all'}
        The reduction to apply. ``'any'`` -> logical OR, ``'all'`` -> logical AND.

    Returns
    -------
    jax.Array
        A scalar boolean. Under :func:`jax.vmap` it is a single global scalar,
        not a per-batch result.

    Raises
    ------
    ValueError
        If ``op`` is not supported by :func:`brainstate.transform.unvmap`.

    See Also
    --------
    brainstate.transform.unvmap

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainpy.math.remove_vmap import remove_vmap
        >>> bool(remove_vmap(jnp.array([False, True])))
        True
        >>> bool(remove_vmap(jnp.array([True, False]), 'all'))
        False
    """
    if isinstance(x, Array):
        x = x.value
    return unvmap(x, op)
