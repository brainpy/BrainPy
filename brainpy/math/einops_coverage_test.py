# -*- coding: utf-8 -*-
"""Supplementary coverage tests for ``brainpy/math/einops.py``.

``einops_test.py`` and ``math_compat_fixes_test.py`` cover the happy paths.
This module targets the remaining uncovered error / edge branches:

* ``_reconstruct_from_shape_uncached`` shape-mismatch ``EinopsError``s
  (``len(unknown)==0`` exact-match and the ``length % known_product`` chunk
  check).
* ``_prepare_transformation_recipe`` validation branches:
  - ellipsis on the right but not the left (line 283);
  - non-unitary anonymous axes in ``rearrange`` (288);
  - unexpected identifiers on the left of ``repeat`` (295);
  - too few dims for an ellipsis pattern (312);
  - invalid kwarg axis name (373) / unused kwarg axis (375);
  - "could not infer sizes" with two unknowns in a group (384).
* ``_prepare_recipes_for_all_dims`` for both the fixed-rank and ellipsis cases.
* ``ein_reduce`` error-message formatting for list inputs (line 503).
* ``ein_shape`` ellipsis / non-ellipsis dimension-count ``RuntimeError``s and the
  composite-axis ``RuntimeError``.

The ``_apply_recipe`` unhashable-shape ``TypeError`` fallback (lines 216-219)
and the entire ``_apply_recipe_array_api`` helper (237-264) are not reachable
through the public ``ein_*`` API with concrete jax arrays (they target symbolic
shapes / the array-API path), so they are documented rather than forced.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from brainpy.math import einops as ein
from brainpy.math.einops import _prepare_recipes_for_all_dims
from brainpy.math.einops_parsing import EinopsError


# ---------------------------------------------------------------------------
# _reconstruct_from_shape_uncached shape checks
# ---------------------------------------------------------------------------

def test_reconstruct_exact_known_product_mismatch():
    # group has no unknown axis; the product of known sizes must equal the dim.
    # '(a b) -> a b' with a=2,b=2 expects 4 but the tensor axis is 6.
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.), '(a b) -> a b', a=2, b=2)


def test_reconstruct_indivisible_chunk_mismatch():
    # one unknown axis, but length not divisible by the known product.
    # axis length 6 cannot be split into chunks of a=4.
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.), '(a b) -> a b', a=4)


# ---------------------------------------------------------------------------
# _prepare_transformation_recipe validation branches
# ---------------------------------------------------------------------------

def test_ellipsis_on_right_only_raises():
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.).reshape(2, 3), 'a b -> a b ...')


def test_rearrange_non_unitary_anonymous_axis_raises():
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.).reshape(2, 3), 'a b -> a b 2')


def test_repeat_unexpected_left_identifier_raises():
    # 'c' appears on the left but not the right of a repeat -> error.
    with pytest.raises(EinopsError):
        ein.ein_repeat(jnp.arange(6.).reshape(2, 3), 'a b c -> a b', c=1)


def test_ellipsis_too_few_dims_raises():
    # pattern needs >= 2 explicit dims plus the ellipsis, tensor is only 1-D.
    with pytest.raises(EinopsError):
        ein.ein_reduce(jnp.arange(6.), 'a b ... -> a', 'sum')


def test_invalid_kwarg_axis_name_raises():
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.).reshape(2, 3), 'a b -> a b', **{'1bad': 2})


def test_unused_kwarg_axis_raises():
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.).reshape(2, 3), 'a b -> a b', z=2)


def test_cannot_infer_two_unknowns_raises():
    # both 'a' and 'b' are unknown inside one composite group -> not inferable.
    with pytest.raises(EinopsError):
        ein.ein_rearrange(jnp.arange(6.), '(a b) -> a b')


# ---------------------------------------------------------------------------
# _prepare_recipes_for_all_dims
# ---------------------------------------------------------------------------

def test_prepare_recipes_fixed_rank():
    recipes = _prepare_recipes_for_all_dims('a b -> b a', 'rearrange', ())
    # exactly one entry, keyed by the fixed number of left composite axes (2)
    assert list(recipes.keys()) == [2]


def test_prepare_recipes_with_ellipsis_precomputes_8():
    recipes = _prepare_recipes_for_all_dims('a ... -> ... a', 'rearrange', ())
    # ellipsis path pre-computes recipes for 0..7 extra ellipsis dims -> 8 entries
    assert len(recipes) == 8


# ---------------------------------------------------------------------------
# ein_reduce error-message formatting
# ---------------------------------------------------------------------------

def test_error_message_for_list_input_mentions_list():
    with pytest.raises(EinopsError) as exc:
        ein.ein_reduce([jnp.arange(6.), jnp.arange(6.)], 'a -> b', 'sum')
    assert 'Input is list' in str(exc.value)


def test_error_message_for_array_input_mentions_shape():
    with pytest.raises(EinopsError) as exc:
        ein.ein_rearrange(jnp.arange(6.), 'a b c -> a b c')  # wrong ndim
    assert 'Input tensor shape' in str(exc.value)


# ---------------------------------------------------------------------------
# ein_shape edge branches
# ---------------------------------------------------------------------------

def test_ein_shape_composite_axes_raises():
    with pytest.raises(RuntimeError):
        ein.ein_shape(jnp.zeros((6,)), '(a b)')


def test_ein_shape_wrong_ndim_no_ellipsis_raises():
    with pytest.raises(RuntimeError):
        ein.ein_shape(jnp.zeros((2, 3)), 'a b c')   # 2 dims vs 3 names


def test_ein_shape_ellipsis_too_few_dims_raises():
    with pytest.raises(RuntimeError):
        ein.ein_shape(jnp.zeros((2,)), 'a b ... c')   # needs >= 3 dims


def test_ein_shape_ellipsis_ok():
    out = ein.ein_shape(jnp.zeros((2, 3, 5, 7)), 'b ... w')
    assert out == {'b': 2, 'w': 7}
