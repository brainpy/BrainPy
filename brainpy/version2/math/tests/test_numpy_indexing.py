# -*- coding: utf-8 -*-

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest

pytest.skip("No need to test.", allow_module_level=True)

import enum
import itertools
import typing
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Optional, Tuple

import jax
import numpy as np
from absl.testing import parameterized
from jax import numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lax import lax as lax_internal
from jax.config import config
import brainpy.version2.math as bm

config.parse_flags_with_absl()

# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation


ARRAY_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[array\(seq\)\]"
TUPLE_MSG = r"Using a non-tuple sequence for multidimensional indexing is not allowed.*arr\[tuple\(seq\)\]"

float_dtypes = jtu.dtypes.floating
default_dtypes = float_dtypes + jtu.dtypes.integer
all_dtypes = default_dtypes + jtu.dtypes.boolean


class IndexSpec(typing.NamedTuple):
    shape: Tuple[int, ...]
    indexer: Any
    out_shape: Optional[Tuple[int, ...]] = None


def check_grads(f, args, order, atol=None, rtol=None, eps=None):
    # TODO(mattjj,dougalm): add higher-order check
    default_tol = 1e-6 if config.x64_enabled else 1e-2
    atol = atol or default_tol
    rtol = rtol or default_tol
    eps = eps or default_tol
    jtu.check_jvp(f, partial(jax.jvp, f), args, atol, rtol, eps)
    jtu.check_vjp(f, partial(jax.vjp, f), args, atol, rtol, eps)


STATIC_INDEXING_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1, out_shape=()),
        IndexSpec(shape=(3, 3), indexer=0, out_shape=(3,)),
        IndexSpec(shape=(3, 4, 5), indexer=2, out_shape=(4, 5)),
        IndexSpec(shape=(3,), indexer=-1, out_shape=()),
        IndexSpec(shape=(3,), indexer=-2, out_shape=()),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1), out_shape=()),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2), out_shape=(5,)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2), out_shape=(5,)),
    ]),
    ("ThreeIntIndices", [
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2, 3), out_shape=()),
    ]),
    ("OneSliceIndex", [
        IndexSpec(shape=(10,), indexer=slice(1, 3), out_shape=(2,)),
        IndexSpec(shape=(10,), indexer=slice(1, -1), out_shape=(8,)),
        IndexSpec(shape=(10,), indexer=slice(None, -1), out_shape=(9,)),
        IndexSpec(shape=(10,), indexer=slice(None, None, None), out_shape=(10,)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 3), out_shape=(2, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(1, None), out_shape=(9, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 3), out_shape=(3, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(-3, None), out_shape=(3, 8)),
    ]),
    ("OneSliceIndexNegativeStride", [
        IndexSpec(shape=(10,), indexer=slice(3, 1, -1), out_shape=(2,)),
        IndexSpec(shape=(10,), indexer=slice(1, 8, -1), out_shape=(0,)),
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2), out_shape=(4,)),
        IndexSpec(shape=(10,), indexer=slice(None, None, -1), out_shape=(10,)),
        IndexSpec(shape=(10, 8), indexer=slice(3, 1, -1), out_shape=(2, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(0, 8, -1), out_shape=(0, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -1), out_shape=(10, 8)),
    ]),
    ("OneSliceIndexNonUnitStride", [
        IndexSpec(shape=(10,), indexer=slice(0, 8, 2), out_shape=(4,)),
        IndexSpec(shape=(10,), indexer=slice(0, 8, 3), out_shape=(3,)),
        IndexSpec(shape=(10,), indexer=slice(1, 3, 2), out_shape=(1,)),
        IndexSpec(shape=(10,), indexer=slice(1, None, 2), out_shape=(5,)),
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2), out_shape=(4,)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 8, 3), out_shape=(3, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, 2), out_shape=(5, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 1, -2), out_shape=(4, 8)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -2), out_shape=(5, 8)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(10, 8), indexer=(slice(1, 3), slice(0, 2)),
                  out_shape=(2, 2)),
        IndexSpec(shape=(10, 8), indexer=(slice(1, None), slice(None, 2)),
                  out_shape=(9, 2)),
        IndexSpec(shape=(10, 8), indexer=(slice(None, None, -1), slice(None, 2)),
                  out_shape=(10, 2)),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, 2)),
                  out_shape=(2, 2, 3)),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, None)),
                  out_shape=(2, 8, 3)),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, None), slice(0, 2)),
                  out_shape=(9, 2, 3)),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None), out_shape=(3,)),
        IndexSpec(shape=(3, 4), indexer=slice(None), out_shape=(3, 4)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None)),
                  out_shape=(3, 4)),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None)),
                  out_shape=(3, 4, 5)),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(10, 4), indexer=(slice(None), slice(0, 2)),
                  out_shape=(10, 2)),
        IndexSpec(shape=(10, 4), indexer=(1, slice(None)),
                  out_shape=(4,)),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis, out_shape=(3,)),
        IndexSpec(shape=(3, 4), indexer=Ellipsis, out_shape=(3, 4)),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis), out_shape=(4, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3), out_shape=(3,)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None, out_shape=(1,)),
        IndexSpec(shape=(), indexer=(None, None), out_shape=(1, 1)),
        IndexSpec(shape=(), indexer=(Ellipsis, None), out_shape=(1,)),
        IndexSpec(shape=(3,), indexer=None, out_shape=(1, 3)),
        IndexSpec(shape=(3, 4), indexer=None, out_shape=(1, 3, 4)),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None), out_shape=(3, 4, 1)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis), out_shape=(1, 4)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis), out_shape=(1, 4, 5)),
    ]),
    ("EmptyIndex", [
        IndexSpec(shape=(), indexer=(), out_shape=()),
        IndexSpec(shape=(3,), indexer=(), out_shape=(3,)),
        IndexSpec(shape=(3, 4), indexer=(), out_shape=(3, 4)),
    ]),
    ("TupleOfIntAndSliceAndIntArray", [
        IndexSpec(shape=(3, 2, 3), indexer=(0, slice(None), np.arange(3)),
                  out_shape=(3, 2)),
        IndexSpec(shape=(3, 2, 3), indexer=(np.int32(1), slice(None), np.arange(3)),
                  out_shape=(3, 2)),
        IndexSpec(shape=(3, 2, 3), indexer=(np.array(2), slice(None), np.arange(3)),
                  out_shape=(3, 2)),
    ]),
]

STATIC_INDEXING_OUT_OF_BOUNDS_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=-4, out_shape=()),
        IndexSpec(shape=(3, 3), indexer=3, out_shape=(3,)),
        IndexSpec(shape=(3, 4, 5), indexer=4, out_shape=(4, 5)),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, -4), out_shape=()),
        IndexSpec(shape=(3, 4, 5), indexer=(3, 2), out_shape=()),
        IndexSpec(shape=(3, 4, 5), indexer=(-4, 4), out_shape=(5,)),
    ]),
]

ADVANCED_INDEXING_TESTS = [
    ("One1DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
        IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 1]), out_shape=(3, 3)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 0, 1]),
                  out_shape=(4, 4, 5)),
        IndexSpec(shape=(3,), indexer=np.array([-1, 1]), out_shape=(2,)),
        IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
        IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32),
                  out_shape=(0,)),
    ]),
    ("One2DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([[0, 0]]), out_shape=(1, 2)),
        IndexSpec(shape=(3, 3), indexer=np.array([[1, 2, 1], [0, 1, -1]]),
                  out_shape=(2, 3, 3)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([[0, 2, 0, 1], [-1, -2, 1, 0]]),
                  out_shape=(2, 4, 4, 5)),
    ]),
    ("Two1DIntArrayIndicesNoBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])),
                  out_shape=(2,)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2, 0, 1]), np.array([-1, 0, -1, 2])),
                  out_shape=(4, 5)),
    ]),
    ("Two1DIntArrayIndicesWithBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])),
                  out_shape=(1, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([[0, 2, 0, 1]]), np.array([-1, 0, -1, 2])),
                  out_shape=(1, 4, 5)),
    ]),
    ("ArrayOfInts", [
        IndexSpec(shape=(3,), indexer=np.array([0, 1, 0]), out_shape=(3,)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([0, -1]), out_shape=(2, 4, 5)),
    ]),
    ("TupleOfListsOfPythonInts", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0, 3]]),
                  out_shape=(2, 4, 5)),
    ]),
    ("TupleOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[2, 3, 0, 3]])),
                  out_shape=(1, 4)),
    ]),
    ("TupleOfListsOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
                  out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[2, 3, 0, 3]])),
                  out_shape=(2, 4, 5)),
    ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("One1DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
        IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 0]), out_shape=(3, 3)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 1]),
                  out_shape=(3, 4, 5)),
        IndexSpec(shape=(3,), indexer=np.array([-1, 1]), out_shape=(2,)),
        IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
        IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32), out_shape=(0,)),
    ]),
    ("One2DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([[0, 1]]), out_shape=(1, 2)),
        IndexSpec(shape=(6, 6), indexer=np.array([[1, 2, 0], [3, 4, -1]]),
                  out_shape=(2, 3, 6)),
    ]),
    ("Two1DIntArrayIndicesNoBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])), out_shape=(2,)),
        IndexSpec(shape=(4, 5, 6), indexer=(np.array([0, 2, 1, 3]), np.array([-1, 0, -2, 1])),
                  out_shape=(4, 6)),
    ]),
    ("Two1DIntArrayIndicesWithBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])), out_shape=(1, 2)),
        IndexSpec(shape=(4, 5, 6),
                  indexer=(np.array([[0, 2, -1, 1]]), np.array([-1, 0, -2, 2])), out_shape=(1, 4, 6)),
    ]),
    ("ArrayOfInts", [
        IndexSpec(shape=(3,), indexer=np.array([0, 2, 1]), out_shape=(3,)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([0, -1]), out_shape=(2, 4, 5)),
    ]),
    ("TupleOfListsOfPythonInts", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0]]), out_shape=(2, 3, 5)),
    ]),
    ("TupleOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[2, 3, 0]])), out_shape=(1, 3)),
    ]),
    ("TupleOfListsOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
                  out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[2, 3, 0]])),
                  out_shape=(2, 3, 5)),
    ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS_SORTED = [
    ("One1DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([0, 1]), out_shape=(2,)),
        IndexSpec(shape=(3, 3), indexer=np.array([0, 1, 2]), out_shape=(3, 3)),
        IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 1, 2]), out_shape=(3, 4, 5)),
        IndexSpec(shape=(3,), indexer=np.array([-1, 1]), out_shape=(2,)),
        IndexSpec(shape=(3,), indexer=np.array([-2, -1]), out_shape=(2,)),
        IndexSpec(shape=(0,), indexer=np.array([], dtype=np.int32), out_shape=(0,)),
    ]),
    ("One2DIntArrayIndex", [
        IndexSpec(shape=(3,), indexer=np.array([[0, 1]]), out_shape=(1, 2)),
        IndexSpec(shape=(6, 6), indexer=np.array([[-1, 0, 1],
                                                  [2, 3, 4]]), out_shape=(2, 3, 6)),
    ]),
    ("Two1DIntArrayIndicesNoBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]), np.array([1, 2])),
                  out_shape=(2,)),
        IndexSpec(shape=(4, 5, 6),
                  indexer=(np.array([0, 1, 2, 3]), np.array([-2, -1, 0, 1])),
                  out_shape=(4, 6)),
    ]),
    ("Two1DIntArrayIndicesWithBroadcasting", [
        IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]), np.array([1, 2])),
                  out_shape=(1, 2)),
        IndexSpec(shape=(4, 5, 6),
                  indexer=(np.array([[-1, 0, 1, 2]]), np.array([-2, -1, 0, 2])),
                  out_shape=(1, 4, 6)),
    ]),
    ("TupleOfListsOfPythonInts", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1],), out_shape=(2, 4, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[0, 2, 3]]),
                  out_shape=(2, 3, 5)),
    ]),
    ("TupleOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1])), out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=(0, 1, np.array([[0, 2, 3]])),
                  out_shape=(1, 3)),
    ]),
    ("TupleOfListsOfPythonIntsAndIntArrays", [
        IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0])),
                  out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], np.array([[0, 2, 3]])),
                  out_shape=(2, 3, 5)),
    ]),
]

MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("SlicesAndOneIntArrayIndex", [
        IndexSpec(shape=(2, 3), indexer=(np.array([0, 1]), slice(1, 2)),
                  out_shape=(2, 1)),
        IndexSpec(shape=(2, 3), indexer=(slice(0, 2), np.array([0, 2])),
                  out_shape=(2, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(Ellipsis, np.array([0, 2]), slice(None)),
                  out_shape=(3, 2, 5)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(Ellipsis, np.array([[0, 2], [1, 3]]), slice(None)),
                  out_shape=(3, 2, 2, 5)),
    ]),
    ("SlicesAndTwoIntArrayIndices", [
        IndexSpec(shape=(3, 4, 5),
                  indexer=(Ellipsis, np.array([0, 2]), np.array([-1, 2])),
                  out_shape=(3, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), Ellipsis, np.array([-1, 2])),
                  out_shape=(2, 4)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), np.array([-1, 2]), Ellipsis),
                  out_shape=(2, 5)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), np.array([-1, 2]), slice(1, 3)),
                  out_shape=(2, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), slice(1, 3), np.array([-1, 2])),
                  out_shape=(2, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2, -2]), slice(None, None, 2),
                           np.array([-1, 2, 1])),
                  out_shape=(3, 2)),
    ]),
    ("NonesAndIntArrayIndices", [
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), None, np.array([-1, 2])),
                  out_shape=(2, 1, 5)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2]), None, None, np.array([-1, 2])),
                  out_shape=(2, 1, 1, 5)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(Ellipsis, np.array([0, 2]), None, None,
                           np.array([-1, 2])),
                  out_shape=(2, 3, 1, 1)),
    ]),
    ("IntArrayWithInt32Type", [
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, np.array(1, dtype=np.int32)),
                  out_shape=(3,)),
    ]),
]

MIXED_ADVANCED_INDEXING_TESTS = MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS + [
    ("SlicesAndOneIntArrayIndex", [
        IndexSpec(shape=(3, 4, 5),
                  indexer=(Ellipsis, np.array([[0, 2], [1, 1]]), slice(None)),
                  out_shape=(3, 2, 2, 5)),
    ]),
    ("SlicesAndTwoIntArrayIndices", [
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([0, 2, -2]), slice(None, None, 2),
                           np.array([-1, 2, -1])),
                  out_shape=(3, 2)),
        IndexSpec(shape=(3, 4, 5),
                  indexer=(np.array([[0, 2], [2, 0]]), Ellipsis,
                           np.array([[1, 0], [1, 0]])),
                  out_shape=(2, 2, 4)),
    ]),
]

MODES = ["clip", "drop", "promise_in_bounds"]


@pytest.mark.skipif(True, reason="No longer need to test.")
class IndexingTest(jtu.JaxTestCase):
    """Tests for Numpy indexing translation rules."""

    @parameterized.named_parameters(
        jtu.cases_from_list(
            {"testcase_name": "{}_inshape={}_indexer={}".format(name, jtu.format_shape_dtype_string(shape, dtype),
                                                                indexer),
             "shape": shape,
             "dtype": dtype,
             "indexer": indexer}
            for name, index_specs in STATIC_INDEXING_TESTS
            for shape, indexer, _ in index_specs
            for dtype in all_dtypes)
    )
    def testStaticIndexing(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype)]
        np_fun = lambda x: np.asarray(x)[indexer]
        jnp_fun = lambda x: bm.asarray(x)[indexer]
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)
        # Tests x.at[...].get(...) as well.
        jnp_fun = lambda x: bm.asarray(x).at[indexer].get()
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)

    @parameterized.named_parameters(
        jtu.cases_from_list({"testcase_name": f"_{funcname}", "funcname": funcname}
                            for funcname in
                            ["negative", "sin", "cos", "square", "sqrt", "log", "exp"])
    )
    def testIndexApply(self, funcname, size=10, dtype='float32'):
        if not hasattr(jnp.zeros(1).at[0], 'apply'):
            self.skipTest('Has not apply() function')

        rng = jtu.rand_default(self.rng())
        idx_rng = jtu.rand_int(self.rng(), -size, size)
        np_func = getattr(np, funcname)
        jnp_func = getattr(jnp, funcname)

        @jtu.ignore_warning(category=RuntimeWarning)
        def np_op(x, idx):
            y = x.copy()
            np_func.at(y, idx)
            return y

        def jnp_op(x, idx):
            return bm.asarray(x).at[idx].apply(jnp_func)

        args_maker = lambda: [rng(size, dtype), idx_rng(size, int)]
        self._CheckAgainstNumpy(np_op, jnp_op, args_maker)
        self._CompileAndCheck(jnp_op, args_maker)

    @parameterized.named_parameters({
                                        "testcase_name":
                                            f"{jtu.format_shape_dtype_string(shape, dtype)}_inshape={name}"
                                            f"_indexer={indexer}_mode={mode}",
                                        "shape": shape, "dtype": dtype, "indexer": indexer, "mode": mode
                                    }
                                    for mode in MODES
                                    for name, index_specs in (
                                        STATIC_INDEXING_TESTS if mode == "promise_in_bounds" else
                                        STATIC_INDEXING_TESTS + STATIC_INDEXING_OUT_OF_BOUNDS_TESTS)
                                    for shape, indexer, _ in index_specs
                                    for dtype in float_dtypes)
    def testStaticIndexingGrads(self, shape, dtype, indexer, mode):
        rng = jtu.rand_default(self.rng())
        tol = 1e-2 if bm.finfo(dtype).bits == 32 else None
        arg = rng(shape, dtype)
        # Use an arbitrary finite fill_value, since NaNs won't work in a numerical
        # gradient test.
        fun = lambda x: bm.asarray(x).at[indexer].get(mode=mode, fill_value=7) ** 2
        check_grads(fun, (arg,), 2, tol, tol, tol)

    def _ReplaceSlicesWithTuples(self, idx):
        """Helper method to replace slices with tuples for dynamic indexing args."""
        if isinstance(idx, slice):
            triple = idx.start, idx.stop, idx.step
            isnone = [i for i, elt in enumerate(triple) if elt is None]
            zeros = itertools.repeat(0)
            nones = itertools.repeat(None)
            out = util.subvals(triple, zip(isnone, zeros))
            return out, lambda out: slice(*util.subvals(out, zip(isnone, nones)))
        elif isinstance(idx, (tuple, list)) and idx:
            t = type(idx)
            elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
            return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
        else:
            return idx, lambda x: x

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in [
            ("OneSliceIndex",
             [IndexSpec(shape=(5,), indexer=slice(1, 3)),
              IndexSpec(shape=(5, 4), indexer=slice(1, 3))]),
            ("TwoSliceIndices",
             [IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
              IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2)))]),
            ("NonUnitStrides", [
                IndexSpec(shape=(3,), indexer=slice(None, None, -1)),
                IndexSpec(shape=(3, 3), indexer=slice(0, 3, -2)),
                IndexSpec(shape=(3, 4, 5), indexer=slice(0, 4, 2))
            ]),
            ("OnlyStartOrStopDynamic", [
                IndexSpec(shape=(5, 4), indexer=(slice(None, 3), slice(0, 2))),
                IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None)))
            ]),
        ]
        for shape, indexer, _ in index_specs
        for dtype in all_dtypes)
    def testDynamicIndexingWithSlicesErrors(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

        @jax.jit
        def fun(x, unpacked_indexer):
            indexer = pack_indexer(unpacked_indexer)
            return x[indexer]

        args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
        self.assertRaises(IndexError, lambda: fun(*args_maker()))

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in [
            ("OneIntIndex",
             [IndexSpec(shape=(3,), indexer=1),
              IndexSpec(shape=(3, 3), indexer=0),
              IndexSpec(shape=(3, 4, 5), indexer=2),
              IndexSpec(shape=(3,), indexer=-1),
              IndexSpec(shape=(3,), indexer=-2)]),
            ("TwoIntIndices",
             [IndexSpec(shape=(3, 3), indexer=(2, 1)),
              IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
              IndexSpec(shape=(3, 4, 5), indexer=(-1, 2))]),
            ("ThreeIntIndices",
             [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
        ]
        for shape, indexer, _ in index_specs
        for dtype in all_dtypes)
    def testDynamicIndexingWithIntegers(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

        def np_fun(x, unpacked_indexer):
            indexer = pack_indexer(unpacked_indexer)
            return np.asarray(x)[indexer]

        def jnp_fun(x, unpacked_indexer):
            indexer = pack_indexer(unpacked_indexer)
            return bm.array(x)[indexer]

        args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in [
            ("OneIntIndex",
             [IndexSpec(shape=(3,), indexer=1),
              IndexSpec(shape=(3, 3), indexer=0),
              IndexSpec(shape=(3, 4, 5), indexer=2),
              IndexSpec(shape=(3,), indexer=-1),
              IndexSpec(shape=(3,), indexer=-2),
              ]),
            ("TwoIntIndices",
             [IndexSpec(shape=(3, 3), indexer=(2, 1)),
              IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
              IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
              ]),
            ("ThreeIntIndices",
             [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
        ]
        for shape, indexer, _ in index_specs
        for dtype in float_dtypes)
    def testDynamicIndexingWithIntegersGrads(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        tol = 1e-2 if bm.finfo(dtype).bits == 32 else None
        unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

        @jax.jit
        def fun(unpacked_indexer, x):
            indexer = pack_indexer(unpacked_indexer)
            return x[indexer]

        arr = rng(shape, dtype)
        check_grads(partial(fun, unpacked_indexer), (arr,), 2, tol, tol, tol)

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in ADVANCED_INDEXING_TESTS
        for shape, indexer, _ in index_specs
        for dtype in all_dtypes)
    def testAdvancedIntegerIndexing(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype), indexer]
        np_fun = lambda x, idx: np.asarray(x)[idx]
        jnp_fun = lambda x, idx: bm.asarray(x)[idx]
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in [
            ("One1DIntArrayIndex",
             [IndexSpec(shape=(3,), indexer=np.array([0, 1])),
              IndexSpec(shape=(3, 3), indexer=np.array([1, 2, 1])),
              IndexSpec(shape=(3, 4, 5), indexer=np.array([0, 2, 0, 1])),
              IndexSpec(shape=(3,), indexer=np.array([-1, 1])),
              IndexSpec(shape=(3,), indexer=np.array([-2, -1])),
              ]),
            ("One2DIntArrayIndex",
             [IndexSpec(shape=(3,), indexer=np.array([[0, 0]])),
              IndexSpec(shape=(3, 3), indexer=np.array([[1, 2, 1],
                                                        [0, 1, -1]])),
              IndexSpec(shape=(3, 4, 5), indexer=np.array([[0, 2, 0, 1],
                                                           [-1, -2, 1, 0]])),
              ]),
            ("Two1DIntArrayIndicesNoBroadcasting",
             [IndexSpec(shape=(3, 3), indexer=(np.array([0, 1]),
                                               np.array([1, 2]))),
              IndexSpec(shape=(3, 4, 5), indexer=(np.array([0, 2, 0, 1]),
                                                  np.array([-1, 0, -1, 2]))),
              ]),
            ("Two1DIntArrayIndicesWithBroadcasting",
             [IndexSpec(shape=(3, 3), indexer=(np.array([[0, 1]]),
                                               np.array([1, 2]))),
              IndexSpec(shape=(3, 4, 5), indexer=(np.array([[0, 2, 0, 1]]),
                                                  np.array([-1, 0, -1, 2]))),
              ]),
            ("TupleOfPythonIntsAndIntArrays",
             [IndexSpec(shape=(3, 4, 5), indexer=(0, np.array([0, 1]))),
              IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                                  np.array([[2, 3, 0, 3]]))),
              ]),
            ("TupleOfListsOfPythonIntsAndIntArrays",
             [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], np.array([0]))),
              IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                                  np.array([[2, 3, 0, 3]]))),
              ]),
        ]
        for shape, indexer, _ in index_specs
        for dtype in float_dtypes)
    def testAdvancedIntegerIndexingGrads(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        tol = 1e-2 if bm.finfo(dtype).bits == 32 else None
        arg = rng(shape, dtype)
        fun = lambda x: bm.asarray(x)[indexer]
        check_grads(fun, (arg,), 2, tol, tol, eps=1.)

    @parameterized.named_parameters(
        {"testcase_name": "{}_inshape={}_indexer={}"
        .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
         "shape": shape, "dtype": dtype, "indexer": indexer}
        for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS
        for shape, indexer, _ in index_specs
        for dtype in all_dtypes)
    def testMixedAdvancedIntegerIndexing(self, shape, dtype, indexer):
        rng = jtu.rand_default(self.rng())
        indexer_with_dummies = [e if isinstance(e, np.ndarray) else ()
                                for e in indexer]
        substitutes = [(i, e) for i, e in enumerate(indexer)
                       if not isinstance(e, np.ndarray)]
        args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

        def jnp_fun(x, indexer_with_dummies):
            idx = type(indexer)(util.subvals(indexer_with_dummies, substitutes))
            return bm.asarray(x)[idx]

        def np_fun(x, indexer_with_dummies):
            idx = type(indexer)(util.subvals(indexer_with_dummies, substitutes))
            return np.asarray(x)[idx]

        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)

    def testAdvancedIndexingManually(self):
        x = self.rng().randn(3, 4, 5)
        index_array = np.array([0, 2, -1, 0])

        op = lambda x, index_array: x[..., index_array, :]
        cop = jax.jit(op)

        a1 = op(x, index_array)
        a2 = cop(x, index_array)

        self.assertAllClose(a1, a2)

        op = lambda x, index_array: x[..., index_array, :, index_array, None]
        cop = jax.jit(op)

        a1 = op(x, index_array)
        a2 = cop(x, index_array)

        self.assertAllClose(a1, a2)

        op = lambda x, index_array: x[index_array, ..., index_array[:, None], None]
        cop = jax.jit(op)

        a1 = op(x, index_array)
        a2 = cop(x, index_array)

        self.assertAllClose(a1, a2)

    def testUnpacking(self):

        def foo(x):
            a, b, c = x
            return a + b + c

        cfoo = jax.jit(foo)

        a1 = foo(np.arange(3))
        a2 = cfoo(np.arange(3))

        self.assertAllClose(a1, a2)

    def testBooleanIndexingArray1D(self):
        idx = np.array([True, True, False])
        x = jax.device_put(np.arange(3))
        ans = x[idx]
        expected = np.arange(3)[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBooleanIndexingList1D(self):
        idx = [True, True, False]
        x = jax.device_put(np.arange(3))
        with self.assertRaisesRegex(TypeError, ARRAY_MSG):
            x[idx]

    def testBooleanIndexingArray2DBroadcast(self):
        idx = np.array([True, True, False, True])
        x = np.arange(8).reshape(4, 2)
        ans = jax.device_put(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBooleanIndexingList2DBroadcast(self):
        idx = [True, True, False, True]
        x = np.arange(8).reshape(4, 2)
        with self.assertRaisesRegex(TypeError, ARRAY_MSG):
            jax.device_put(x)[idx]

    def testBooleanIndexingArray2D(self):
        idx = np.array([[True, False],
                        [False, True],
                        [False, False],
                        [True, True]])
        x = np.arange(8).reshape(4, 2)
        ans = jax.device_put(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBoolean1DIndexingWithEllipsis(self):
        # Regression test for https://github.com/google/jax/issues/8412
        x = np.arange(24).reshape(4, 3, 2)
        idx = (..., np.array([True, False]))
        ans = bm.array(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBoolean1DIndexingWithEllipsis2(self):
        # Regression test for https://github.com/google/jax/issues/9050
        x = np.arange(3)
        idx = (..., np.array([True, False, True]))
        ans = bm.array(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBoolean1DIndexingWithEllipsis3(self):
        x = np.arange(6).reshape(2, 3)
        idx = (0, ..., np.array([True, False, True]))
        ans = bm.array(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBoolean2DIndexingWithEllipsis(self):
        x = np.arange(24).reshape(4, 3, 2)
        idx = (..., np.array([[True, False], [True, False], [False, False]]))
        ans = bm.array(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBoolean1DIndexingWithTrailingEllipsis(self):
        x = np.arange(24).reshape(4, 3, 2)
        idx = (np.array([True, False, True, False]), ...)
        ans = bm.array(x)[idx]
        expected = x[idx]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBooleanIndexingDynamicShapeError(self):
        x = np.zeros(3)
        i = np.array([True, True, False])
        self.assertRaises(IndexError, lambda: jax.jit(lambda x, i: x[i])(x, i))

    def testScalarBooleanIndexingNotImplemented(self):
        msg = "JAX arrays do not support boolean scalar indices"
        with self.assertRaisesRegex(TypeError, msg):
            bm.arange(4)[True]
        with self.assertRaisesRegex(TypeError, msg):
            bm.arange(4)[False]
        with self.assertRaisesRegex(TypeError, msg):
            bm.arange(4)[..., True]

    def testIssue187(self):
        x = bm.ones((5, 5))
        x[[0, 2, 4], [0, 2, 4]]  # doesn't crash

        x = np.arange(25).reshape((5, 5))
        ans = jax.jit(lambda x: x[[0, 2, 4], [0, 2, 4]])(x)
        expected = x[[0, 2, 4], [0, 2, 4]]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testJVPOfGradOfIndexing(self):
        # Should return a value, even though we didn't pass a symbolic zero as the
        # index tangent.
        x = bm.ones((3, 4), bm.float32)
        i = bm.ones((3,), bm.int32).value
        f = lambda x, i: bm.sum(x[i])
        primals, tangents = jax.jvp(jax.grad(f), (x, i),
                                    (x, np.zeros(i.shape, dtypes.float0)))
        expected = np.broadcast_to(
            np.array([0, 3, 0], dtype=np.float32)[:, None], (3, 4))
        self.assertAllClose(expected, primals)
        self.assertAllClose(np.zeros_like(x), tangents)

    def testIndexingEmptyDimension(self):
        # Issue 2671: XLA error when indexing into dimension of size 0
        x = bm.ones((2, 0))
        # The following work, even on axis 1 of size 0
        with jax.numpy_rank_promotion('allow'):
            _ = x[0, :] + x[0, None] + x[0, 1:] + x[0, 1:3:2]

        with self.assertRaisesRegex(IndexError,
                                    "index .* is out of bounds for axis .* with size 0"):
            _ = np.ones((2, 0))[0, 0]  # The numpy error
        with self.assertRaisesRegex(IndexError,
                                    "index is out of bounds for axis .* with size 0"):
            _ = x[0, 0]  # JAX indexing
        with self.assertRaisesRegex(IndexError,
                                    "index is out of bounds for axis .* with size 0"):
            jax.jit(lambda i: x[0, i])(0)  # JAX indexing under jit

    def testBooleanIndexingWithEmptyResult(self):
        # based on a TensorFlow Probability test that started failing after #1622
        x = bm.array([-1])
        mask = bm.array([False])
        ans = x[mask]  # doesn't crash

        expected = np.array([-1])[np.array([False])]
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testBooleanIndexingShapeMismatch(self):
        # Regression test for https://github.com/google/jax/issues/7329
        x = bm.arange(4)
        idx = bm.array([True, False])
        with self.assertRaisesRegex(IndexError, "boolean index did not match shape.*"):
            x[idx]

    def testNontrivialBooleanIndexing(self):
        # Test nontrivial corner case in boolean indexing shape validation
        rng = jtu.rand_default(self.rng())
        index = (rng((2, 3), np.bool_), rng((6,), np.bool_))

        args_maker = lambda: [rng((2, 3, 6), np.int32)]
        np_fun = lambda x: np.asarray(x)[index]
        jnp_fun = lambda x: bm.asarray(x)[index]

        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
        self._CompileAndCheck(jnp_fun, args_maker)

    def testFloatIndexingError(self):
        BAD_INDEX_TYPE_ERROR = "Indexer must have integer or boolean type, got indexer with type"
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            bm.zeros(2)[0.]
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            bm.zeros((2, 2))[(0, 0.)]
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            bm.zeros((2, 2))[(0, 0.)]
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            jax.jit(lambda idx: bm.zeros((2, 2))[idx])((0, 0.))
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            bm.zeros(2).at[0.].add(1.)
        with self.assertRaisesRegex(TypeError, BAD_INDEX_TYPE_ERROR):
            bm.zeros(2).at[0.].set(1.)

    def testIndexOutOfBounds(self):  # https://github.com/google/jax/issues/2245
        x = bm.arange(5, dtype=bm.int32) + 1
        self.assertAllClose(x, x[:10])

        idx = bm.array([-10, -6, -5, -4, 0, 3, 4, 5, 6, 100]).value
        self.assertArraysEqual(
            x.at[idx].get(mode="clip"),
            bm.array([1, 1, 1, 2, 1, 4, 5, 5, 5, 5], bm.int32))
        nan = np.nan
        self.assertArraysEqual(
            x.astype(bm.float32).at[idx].get(mode="fill"),
            bm.array([nan, nan, 1, 2, 1, 4, 5, nan, nan, nan], bm.float32))
        imin = np.iinfo(np.int32).min
        self.assertArraysEqual(
            x.at[idx].get(mode="fill"),
            bm.array([imin, imin, 1, 2, 1, 4, 5, imin, imin, imin], bm.int32))
        umax = np.iinfo(np.uint32).max
        self.assertArraysEqual(
            x.astype(np.uint32).at[idx].get(mode="fill"),
            bm.array([umax, umax, 1, 2, 1, 4, 5, umax, umax, umax], bm.uint32))
        self.assertArraysEqual(
            x.at[idx].get(mode="fill", fill_value=7),
            bm.array([7, 7, 1, 2, 1, 4, 5, 7, 7, 7], bm.int32))

    def testIndexingWeakTypes(self):
        x = lax_internal._convert_element_type(bm.arange(5), int, weak_type=True)

        a = x.at[0].set(1.0)
        self.assertEqual(a.dtype, x.dtype)
        self.assertTrue(dtypes.is_weakly_typed(a))

        b = x.at[0].add(1.0)
        self.assertEqual(b.dtype, x.dtype)
        self.assertTrue(dtypes.is_weakly_typed(b))

        c = x.at[0].mul(1.0)
        self.assertEqual(c.dtype, x.dtype)
        self.assertTrue(dtypes.is_weakly_typed(c))


def _broadcastable_shapes(shape):
    """Returns all shapes that broadcast to `shape`."""

    def f(rshape):
        yield []
        if rshape:
            for s in f(rshape[1:]):
                yield rshape[0:1] + s
            if rshape[0] != 1:
                for s in f(rshape[1:]):
                    yield [1] + s

    for x in f(list(reversed(shape))):
        yield list(reversed(x))


class UpdateOps(enum.Enum):
    UPDATE = 0
    ADD = 1
    MUL = 2
    DIV = 3
    POW = 4
    MIN = 5
    MAX = 6

    def np_fn(op, indexer, x, y):
        x = x.copy()
        x[indexer] = {
            UpdateOps.UPDATE: lambda: y,
            UpdateOps.ADD: lambda: x[indexer] + y,
            UpdateOps.MUL: lambda: x[indexer] * y,
            UpdateOps.DIV: jtu.ignore_warning(category=RuntimeWarning)(
                lambda: x[indexer] / y.astype(x.dtype)),
            UpdateOps.POW: jtu.ignore_warning(category=RuntimeWarning)(
                lambda: x[indexer] ** y.astype(x.dtype)),
            UpdateOps.MIN: lambda: np.minimum(x[indexer], y),
            UpdateOps.MAX: lambda: np.maximum(x[indexer], y),
        }[op]()
        return x

    def jax_fn(op, indexer, x, y, indices_are_sorted=False,
               unique_indices=False, mode=None):
        x = bm.array(x)
        return {
            UpdateOps.UPDATE: x.at[indexer].set,
            UpdateOps.ADD: x.at[indexer].add,
            UpdateOps.MUL: x.at[indexer].multiply,
            UpdateOps.DIV: x.at[indexer].divide,
            UpdateOps.POW: x.at[indexer].power,
            UpdateOps.MIN: x.at[indexer].min,
            UpdateOps.MAX: x.at[indexer].max,
        }[op](y, indices_are_sorted=indices_are_sorted,
              unique_indices=unique_indices, mode=mode)

    def dtypes(op):
        if op == UpdateOps.UPDATE:
            return all_dtypes
        elif op == UpdateOps.DIV or op == UpdateOps.POW:
            return jtu.dtypes.inexact
        else:
            return default_dtypes


def _update_tol(op):
    if op == UpdateOps.POW:
        tol = {np.complex64: 1e-4 if jtu.device_under_test() == "tpu" else 1e-5,
               np.complex128: 1e-14}
    else:
        tol = {np.complex128: 1e-14}
    return tol


@pytest.mark.skipif(True, reason="No longer need to test.")
@jtu.with_config(jax_numpy_dtype_promotion='standard')
class IndexedUpdateTest(jtu.JaxTestCase):

    @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
        "testcase_name":
            f"{name}_inshape={jtu.format_shape_dtype_string(shape, dtype)}"
            f"_indexer={indexer}"
            f"_update={jtu.format_shape_dtype_string(update_shape, update_dtype)}"
            f"_op={op.name}",
        "shape": shape, "dtype": dtype, "indexer": indexer,
        "update_shape": update_shape, "update_dtype": update_dtype,
        "op": op, "mode": mode,
    } for name, index_specs in s(STATIC_INDEXING_TESTS)
        for shape, indexer, update_shape in s(index_specs)
        for op in s(UpdateOps)
        for dtype in s(UpdateOps.dtypes(op))
        for update_shape in s(_broadcastable_shapes(update_shape))
        for update_dtype in s([dtype] if op == UpdateOps.ADD else all_dtypes)
        for mode in s(MODES))))
    def testStaticIndexing(self, shape, dtype, update_shape, update_dtype,
                           indexer, op, mode):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
        np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
        jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y, mode=mode)
        self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
        self._CompileAndCheck(jax_fn, args_maker)

    @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
        "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
            name, jtu.format_shape_dtype_string(shape, dtype), indexer,
            jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
        "shape": shape,
        "dtype": dtype,
        "indexer": indexer,
        "update_shape": update_shape,
        "update_dtype": update_dtype,
        "op": op
    } for name, index_specs in s(ADVANCED_INDEXING_TESTS_NO_REPEATS)
        for shape, indexer, update_shape in s(index_specs)
        for op in s(UpdateOps)
        for dtype in s(UpdateOps.dtypes(op))
        for update_shape in s(_broadcastable_shapes(update_shape))
        for update_dtype in s([dtype] if op == UpdateOps.ADD else all_dtypes))))
    def testAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                             indexer, op):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
        np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
        jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y,
                                               unique_indices=True)
        self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
        self._CompileAndCheck(jax_fn, args_maker)

    @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
        "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
            name, jtu.format_shape_dtype_string(shape, dtype), indexer,
            jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
        "shape": shape, "dtype": dtype, "indexer": indexer,
        "update_shape": update_shape, "update_dtype": update_dtype,
        "op": op
    } for name, index_specs in s(ADVANCED_INDEXING_TESTS_NO_REPEATS_SORTED)
        for shape, indexer, update_shape in s(index_specs)
        for op in s(UpdateOps)
        for dtype in s(UpdateOps.dtypes(op))
        for update_shape in s(_broadcastable_shapes(update_shape))
        for update_dtype in s([dtype] if op == UpdateOps.ADD else all_dtypes))))
    def testAdvancedIndexingSorted(self, shape, dtype, update_shape, update_dtype,
                                   indexer, op):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
        np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
        jax_fn = lambda x, y: UpdateOps.jax_fn(
            op, indexer, x, y, indices_are_sorted=True, unique_indices=True)
        self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, check_dtypes=True,
                                tol=_update_tol(op))
        self._CompileAndCheck(jax_fn, args_maker, check_dtypes=True)

    @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
        "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
            name, jtu.format_shape_dtype_string(shape, dtype), indexer,
            jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
        "shape": shape, "dtype": dtype, "indexer": indexer,
        "update_shape": update_shape, "update_dtype": update_dtype,
        "op": op
    } for name, index_specs in s(MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS)
        for shape, indexer, update_shape in s(index_specs)
        for op in s(UpdateOps)
        for dtype in s(UpdateOps.dtypes(op))
        for update_shape in s(_broadcastable_shapes(update_shape))
        for update_dtype in s([dtype] if op == UpdateOps.ADD else all_dtypes))))
    def testMixedAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                                  indexer, op):
        rng = jtu.rand_default(self.rng())
        args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
        np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
        jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y)
        self._CheckAgainstNumpy(np_fn, jax_fn, args_maker, tol=_update_tol(op))
        self._CompileAndCheck(jax_fn, args_maker)

    @parameterized.named_parameters(jtu.cases_from_list({
                                                            "testcase_name":
                                                                f"{name}_inshape={jtu.format_shape_dtype_string(shape, dtype)}"
                                                                f"_indexer={indexer}"
                                                                f"_update={jtu.format_shape_dtype_string(update_shape, update_dtype)}"
                                                                f"_op={op.name}_mode={mode}",
                                                            "shape": shape, "dtype": dtype, "indexer": indexer,
                                                            "update_shape": update_shape, "update_dtype": update_dtype,
                                                            "op": op, "mode": mode,
                                                        } for mode in [None] + MODES
                                                        for name, index_specs in (
                                                            STATIC_INDEXING_TESTS if mode == "promise_in_bounds" else
                                                            STATIC_INDEXING_TESTS + STATIC_INDEXING_OUT_OF_BOUNDS_TESTS)
                                                        for shape, indexer, update_shape in index_specs
                                                        for op in [UpdateOps.ADD, UpdateOps.MUL, UpdateOps.UPDATE]
                                                        for dtype in float_dtypes
                                                        for update_shape in _broadcastable_shapes(update_shape)
                                                        for update_dtype in
                                                        ([dtype] if op == UpdateOps.ADD else float_dtypes)))
    def testStaticIndexingGrads(self, shape, dtype, update_shape, update_dtype,
                                indexer, op, mode):
        rng = jtu.rand_default(self.rng())
        jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y, mode=mode,
                                               unique_indices=True)
        x = rng(shape, dtype)
        y = rng(update_shape, update_dtype)
        check_grads(jax_fn, (x, y), 2, rtol=1e-3, atol=1e-3, eps=1.)

    @parameterized.named_parameters(jtu.named_cases_from_sampler(lambda s: ({
        "testcase_name": "{}_inshape={}_indexer={}_update={}_op={}".format(
            name, jtu.format_shape_dtype_string(shape, dtype), indexer,
            jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
        "shape": shape, "dtype": dtype, "indexer": indexer,
        "update_shape": update_shape, "update_dtype": update_dtype,
        "op": op, "unique_indices": unique_indices,
    } for unique_indices in s([False, True])
        for name, index_specs in s(
        ADVANCED_INDEXING_TESTS_NO_REPEATS if unique_indices
        else ADVANCED_INDEXING_TESTS)
        for shape, indexer, update_shape in s(index_specs)
        for op in s(
        [UpdateOps.ADD, UpdateOps.MUL, UpdateOps.UPDATE] if unique_indices
        else [UpdateOps.ADD])
        for dtype in s(float_dtypes)
        for update_shape in s(_broadcastable_shapes(update_shape))
        for update_dtype in s([dtype] if op == UpdateOps.ADD else float_dtypes))))
    def testAdvancedIndexingGrads(self, shape, dtype, update_shape, update_dtype,
                                  indexer, op, unique_indices):
        rng = jtu.rand_default(self.rng())
        jax_fn = lambda x, y: UpdateOps.jax_fn(op, indexer, x, y,
                                               unique_indices=unique_indices)
        x = rng(shape, dtype)
        y = rng(update_shape, update_dtype)
        check_grads(jax_fn, (x, y), 2, rtol=1e-3, atol=1e-3, eps=1.)

    def testIndexMulGradFailsIfNotUnique(self):
        y = bm.ones((10,), bm.int32).value
        f = lambda x, z: x.at[y].mul(z.value)

        x = bm.ones((100,), bm.float32)
        z = bm.ones((10,), bm.float32)
        with self.assertRaises(NotImplementedError,
                               msg="scatter_mul gradients are only implemented if "
                                   "`unique_indices=True`"):
            jax.jvp(f, (x, z), (x, z))

    def testSegmentSumBehavior(self):
        # testAdvancedIndexing compares against NumPy, and as a result doesn't check
        # repeated indices. This test is just a simple manual check, based on
        # https://www.tensorflow.org/api_docs/python/tf/math/segment_sum
        data = np.array([5, 1, 7, 2, 3, 4, 1, 3])
        segment_ids = np.array([0, 0, 0, 1, 2, 2, 3, 3])

        ans = bm.zeros(np.max(segment_ids) + 1).at[segment_ids].add(data)
        expected = np.array([13, 2, 7, 4])
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testSegmentSum(self):
        data = bm.array([5, 1, 7, 2, 3, 4, 1, 3])
        segment_ids = bm.array([0, 0, 0, 1, 2, 2, 3, 3])

        # test with explicit num_segments
        ans = bm.segment_sum(data, segment_ids, num_segments=4)
        expected = bm.array([13, 2, 7, 4])
        self.assertAllClose(ans, expected, check_dtypes=False)

        # test with explicit num_segments larger than the higher index.
        ans = bm.segment_sum(data, segment_ids, num_segments=5)
        expected = bm.array([13, 2, 7, 4, 0])
        self.assertAllClose(ans, expected, check_dtypes=False)

        # test without explicit num_segments
        ans = bm.segment_sum(data, segment_ids)
        expected = bm.array([13, 2, 7, 4])
        self.assertAllClose(ans, expected, check_dtypes=False)

        # test with negative segment ids and segment ids larger than num_segments,
        # that will be wrapped with the `mod`.
        segment_ids = bm.array([0, 4, 8, 1, 2, -6, -1, 3])
        ans = bm.segment_sum(data, segment_ids, num_segments=4)
        expected = bm.array([5, 2, 3, 3])
        self.assertAllClose(ans, expected, check_dtypes=False)

        # test with negative segment ids and without without explicit num_segments
        # such as num_segments is defined by the smaller index.
        segment_ids = bm.array([3, 3, 3, 4, 5, 5, -7, -6])
        ans = bm.segment_sum(data, segment_ids)
        expected = bm.array([0, 0, 0, 13, 2, 7])
        self.assertAllClose(ans, expected, check_dtypes=False)

    def testSegmentSumOutOfBounds(self):
        def fn(data, segment_ids):
            return bm.segment_sum(data, segment_ids, num_segments).sum()

        data = np.array([0, 0], dtype=np.float32)
        num_segments = 2
        segment_ids = np.array([2, 3])
        val, grad = jax.value_and_grad(fn)(data, segment_ids)
        self.assertAllClose(val, np.array(0., np.float32))
        self.assertAllClose(grad, np.array([0., 0.], np.float32))

    def testIndexDtypeError(self):
        # https://github.com/google/jax/issues/2795
        bm.array(1)  # get rid of startup warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            bm.zeros(5).at[::2].set(1)
            self.assertLen(w, 0)

    @contextmanager
    def assertNoWarnings(self):
        with warnings.catch_warnings(record=True) as caught_warnings:
            yield
        self.assertEmpty(caught_warnings)

    @parameterized.named_parameters(jtu.cases_from_list({
                                                            "testcase_name": "idx={}".format(idx), "idx": idx,
                                                            "idx_type": idx_type}
                                                        for idx, idx_type in [
                                                            ([0], "array"),
                                                            ([0, 0], "array"),
                                                            ([[0, 0]], "tuple"),
                                                            ([0, [0, 1]], "tuple"),
                                                            ([0, np.arange(2)], "tuple"),
                                                            ([0, None], "tuple"),
                                                            ([0, slice(None)], "tuple"),
                                                        ]))
    def testIndexSequenceDeprecation(self, idx, idx_type):
        normalize = {"array": np.array, "tuple": tuple}[idx_type]
        msg = {"array": ARRAY_MSG, "tuple": TUPLE_MSG}[idx_type]
        x = bm.arange(6).reshape(3, 2)

        with self.assertRaisesRegex(TypeError, msg):
            x[idx]
        with self.assertNoWarnings():
            x[normalize(idx)]

        with self.assertRaisesRegex(TypeError, msg):
            x.at[idx].set(0)
        with self.assertNoWarnings():
            x.at[normalize(idx)].set(0)

    def testIndexedUpdateAliasingBug(self):
        # https://github.com/google/jax/issues/7461
        fn = lambda x: x.at[1:].set(1 + x[:-1])
        y = bm.zeros(8)
        self.assertArraysEqual(fn(y), jax.jit(fn)(y))
