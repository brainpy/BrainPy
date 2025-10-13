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
import numpy
import pytest

import brainpy.math as bm
from brainpy.math.einops import ein_rearrange, ein_reduce, ein_repeat, _enumerate_directions
from brainpy.math.einops_parsing import EinopsError

REDUCTIONS = ("min", "max", "sum", "mean", "prod")

identity_patterns = [
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
]

equivalent_rearrange_patterns = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]

equivalent_reduction_patterns = [
    ("a b c d e -> ", " ... ->  "),
    ("a b c d e -> (e a)", "a ... e -> (e a)"),
    ("a b c d e -> d (a e)", " a b c d e ... -> d (a e) "),
    ("a b c d e -> (a b)", " ... c d e  -> (...) "),
]


def test_collapsed_ellipsis_errors_out():
    x = numpy.zeros([1, 1, 1, 1, 1])
    ein_rearrange(x, "a b c d ... ->  a b c ... d")
    with pytest.raises(EinopsError):
        ein_rearrange(x, "a b c d (...) ->  a b c ... d")

    ein_rearrange(x, "... ->  (...)")
    with pytest.raises(EinopsError):
        ein_rearrange(x, "(...) -> (...)")


def test_ellipsis_ops_numpy():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in identity_patterns:
        assert numpy.array_equal(x, ein_rearrange(x, pattern)), pattern

    for pattern1, pattern2 in equivalent_rearrange_patterns:
        assert numpy.array_equal(ein_rearrange(x, pattern1), ein_rearrange(x, pattern2))

    for reduction in ["min", "max", "sum"]:
        for pattern1, pattern2 in equivalent_reduction_patterns:
            assert numpy.array_equal(ein_reduce(x, pattern1, reduction=reduction),
                                     ein_reduce(x, pattern2, reduction=reduction))

    # now just check coincidence with numpy
    all_rearrange_patterns = [*identity_patterns]
    for pattern_pairs in equivalent_rearrange_patterns:
        all_rearrange_patterns.extend(pattern_pairs)


def test_rearrange_consistency_numpy():
    shape = [1, 2, 3, 5, 7, 11]
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    for pattern in [
        "a b c d e f -> a b c d e f",
        "b a c d e f -> a b d e f c",
        "a b c d e f -> f e d c b a",
        "a b c d e f -> (f e) d (c b a)",
        "a b c d e f -> (f e d c b a)",
    ]:
        result = ein_rearrange(x, pattern)
        assert len(numpy.setdiff1d(x, result)) == 0

    result = ein_rearrange(x, "a b c d e f -> a (b) (c d e) f")
    assert numpy.array_equal(x.flatten(), result.flatten())

    result = ein_rearrange(x, "a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11")
    assert numpy.array_equal(x, result)

    result1 = ein_rearrange(x, "a b c d e f -> f e d c b a")
    result2 = ein_rearrange(x, "f e d c b a -> a b c d e f")
    assert numpy.array_equal(result1, result2)

    result = ein_rearrange(ein_rearrange(x, "a b c d e f -> (f d) c (e b) a"), "(f d) c (e b) a -> a b c d e f", b=2,
                           d=5)
    assert numpy.array_equal(x, result)

    sizes = dict(zip("abcdef", shape))
    temp = ein_rearrange(x, "a b c d e f -> (f d) c (e b) a", **sizes)
    result = ein_rearrange(temp, "(f d) c (e b) a -> a b c d e f", **sizes)
    assert numpy.array_equal(x, result)

    x2 = numpy.arange(2 * 3 * 4).reshape([2, 3, 4])
    result = ein_rearrange(x2, "a b c -> b c a")
    assert x2[1, 2, 3] == result[2, 3, 1]
    assert x2[0, 1, 2] == result[1, 2, 0]


def test_rearrange_permutations_numpy():
    # tests random permutation of axes against two independent numpy ways
    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
        right_expression = " ".join("i" + str(axis) for axis in permutation)
        expression = left_expression + " -> " + right_expression
        result = ein_rearrange(input, expression)

        for pick in numpy.random.randint(0, 2, [10, n_axes]):
            assert input[tuple(pick)] == result[tuple(pick[permutation])]

    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = " ".join("i" + str(axis) for axis in range(n_axes)[::-1])
        right_expression = " ".join("i" + str(axis) for axis in permutation[::-1])
        expression = left_expression + " -> " + right_expression
        result = ein_rearrange(input, expression)
        assert result.shape == input.shape
        expected_result = numpy.zeros_like(input)
        for original_axis, result_axis in enumerate(permutation):
            expected_result |= ((input >> original_axis) & 1) << result_axis

        assert numpy.array_equal(result, expected_result)


def test_reduction_imperatives():
    for reduction in REDUCTIONS:
        # slight redundancy for simpler order - numpy version is evaluated multiple times
        input = numpy.arange(2 * 3 * 4 * 5 * 6, dtype="int64").reshape([2, 3, 4, 5, 6])
        if reduction in ["mean", "prod"]:
            input = input / input.astype("float64").mean()
        test_cases = [
            ["a b c d e -> ", {}, getattr(input, reduction)()],
            ["a ... -> ", {}, getattr(input, reduction)()],
            ["(a1 a2) ... (e1 e2) -> ", dict(a1=1, e2=2), getattr(input, reduction)()],
            [
                "a b c d e -> (e c) a",
                {},
                getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
            ],
            [
                "a ... c d e -> (e c) a",
                {},
                getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
            ],
            [
                "a b c d e ... -> (e c) a",
                {},
                getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
            ],
            ["a b c d e -> (e c a)", {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1])],
            ["(a a2) ... -> (a2 a) ...", dict(a2=1), input],
        ]
        for pattern, axes_lengths, expected_result in test_cases:
            result = ein_reduce(bm.from_numpy(input.copy()), pattern, reduction=reduction, **axes_lengths)
            result = bm.as_numpy(result)
            print(reduction, pattern, expected_result, result)
            assert numpy.allclose(result, expected_result), f"Failed at {pattern}"


def test_enumerating_directions():
    for shape in [[], [1], [1, 1, 1], [2, 3, 5, 7]]:
        x = numpy.arange(numpy.prod(shape)).reshape(shape)
        axes1 = _enumerate_directions(x)
        axes2 = _enumerate_directions(bm.from_numpy(x))
        assert len(axes1) == len(axes2) == len(shape)
        for ax1, ax2 in zip(axes1, axes2):
            ax2 = bm.as_numpy(ax2)
            assert ax1.shape == ax2.shape
            assert numpy.allclose(ax1, ax2)


def test_concatenations_and_stacking():
    for n_arrays in [1, 2, 5]:
        shapes = [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]
        for shape in shapes:
            arrays1 = [numpy.arange(i, i + numpy.prod(shape)).reshape(shape) for i in range(n_arrays)]
            arrays2 = [bm.from_numpy(array) for array in arrays1]
            result0 = numpy.asarray(arrays1)
            result1 = ein_rearrange(arrays1, "...->...")
            result2 = ein_rearrange(arrays2, "...->...")
            assert numpy.array_equal(result0, result1)
            assert numpy.array_equal(result1, bm.as_numpy(result2))

            result1 = ein_rearrange(arrays1, "b ... -> ... b")
            result2 = ein_rearrange(arrays2, "b ... -> ... b")
            assert numpy.array_equal(result1, bm.as_numpy(result2))


def test_gradients_imperatives():
    # lazy - just checking reductions
    for reduction in REDUCTIONS:
        if reduction in ("any", "all"):
            continue  # non-differentiable ops
        x = numpy.arange(1, 1 + 2 * 3 * 4).reshape([2, 3, 4]).astype("float32")
        y0 = bm.from_numpy(x)
        if not hasattr(y0, "grad"):
            continue

        y1 = ein_reduce(y0, "a b c -> c a", reduction=reduction)
        y2 = ein_reduce(y1, "c a -> a c", reduction=reduction)
        y3 = ein_reduce(y2, "a (c1 c2) -> a", reduction=reduction, c1=2)
        y4 = ein_reduce(y3, "... -> ", reduction=reduction)

        y4.backward()
        grad = bm.as_numpy(y0.grad)


def test_tiling_imperatives():
    input = numpy.arange(2 * 3 * 5, dtype="int64").reshape([2, 1, 3, 1, 5])
    test_cases = [
        (1, 1, 1, 1, 1),
        (1, 2, 1, 3, 1),
        (3, 1, 1, 4, 1),
    ]
    for repeats in test_cases:
        expected = numpy.tile(input, repeats)
        converted = bm.from_numpy(input)
        repeated = bm.tile(converted, repeats)
        result = bm.as_numpy(repeated)
        assert numpy.array_equal(result, expected)


repeat_test_cases = [
    # all assume that input has shape [2, 3, 5]
    ("a b c -> c a b", dict()),
    ("a b c -> (c copy a b)", dict(copy=2, a=2, b=3, c=5)),
    ("a b c -> (a copy) b c ", dict(copy=1)),
    ("a b c -> (c a) (copy1 b copy2)", dict(a=2, copy1=1, copy2=2)),
    ("a ...  -> a ... copy", dict(copy=4)),
    ("... c -> ... (copy1 c copy2)", dict(copy1=1, copy2=2)),
    ("...  -> ... ", dict()),
    (" ...  -> copy1 ... copy2 ", dict(copy1=2, copy2=3)),
    ("a b c  -> copy1 a copy2 b c () ", dict(copy1=2, copy2=1)),
]


def check_reversion(x, repeat_pattern, **sizes):
    """Checks repeat pattern by running reduction"""
    left, right = repeat_pattern.split("->")
    reduce_pattern = right + "->" + left
    repeated = ein_repeat(x, repeat_pattern, **sizes)
    reduced_min = ein_reduce(repeated, reduce_pattern, reduction="min", **sizes)
    reduced_max = ein_reduce(repeated, reduce_pattern, reduction="max", **sizes)
    assert numpy.array_equal(x, reduced_min)
    assert numpy.array_equal(x, reduced_max)


def test_repeat_numpy():
    # check repeat vs reduce. Repeat works ok if reverse reduction with min and max work well
    x = numpy.arange(2 * 3 * 5).reshape([2, 3, 5])
    x1 = ein_repeat(x, "a b c -> copy a b c ", copy=1)
    assert numpy.array_equal(x[None], x1)
    for pattern, axis_dimensions in repeat_test_cases:
        check_reversion(x, pattern, **axis_dimensions)


test_cases_repeat_anonymous = [
    # all assume that input has shape [1, 2, 4, 6]
    ("a b c d -> c a d b", dict()),
    ("a b c d -> (c 2 d a b)", dict(a=1, c=4, d=6)),
    ("1 b c d -> (d copy 1) 3 b c ", dict(copy=3)),
    ("1 ...  -> 3 ... ", dict()),
    ("() ... d -> 1 (copy1 d copy2) ... ", dict(copy1=2, copy2=3)),
    ("1 b c d -> (1 1) (1 b) 2 c 3 d (1 1)", dict()),
]


def test_anonymous_axes():
    x = numpy.arange(1 * 2 * 4 * 6).reshape([1, 2, 4, 6])
    for pattern, axis_dimensions in test_cases_repeat_anonymous:
        check_reversion(x, pattern, **axis_dimensions)


def test_list_inputs():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])

    assert numpy.array_equal(
        ein_rearrange(list(x), "... -> (...)"),
        ein_rearrange(x, "... -> (...)"),
    )
    assert numpy.array_equal(
        ein_reduce(list(x), "a ... e -> (...)", "min"),
        ein_reduce(x, "a ... e -> (...)", "min"),
    )
    assert numpy.array_equal(
        ein_repeat(list(x), "...  -> b (...)", b=3),
        ein_repeat(x, "...  -> b (...)", b=3),
    )


def bit_count(x):
    return sum((x >> i) & 1 for i in range(20))


def test_reduction_imperatives_booleans():
    """Checks that any/all reduction works in all frameworks"""
    x_np = numpy.asarray([(bit_count(x) % 2) == 0 for x in range(2 ** 6)]).reshape([2] * 6)

    for axis in range(6):
        expected_result_any = numpy.any(x_np, axis=axis, keepdims=True)
        expected_result_all = numpy.all(x_np, axis=axis, keepdims=True)
        assert not numpy.array_equal(expected_result_any, expected_result_all)

        axes = list("abcdef")
        axes_in = list(axes)
        axes_out = list(axes)
        axes_out[axis] = "1"
        pattern = (" ".join(axes_in)) + " -> " + (" ".join(axes_out))

        res_any = ein_reduce(bm.from_numpy(x_np), pattern, reduction="any")
        res_all = ein_reduce(bm.from_numpy(x_np), pattern, reduction="all")

        assert numpy.array_equal(expected_result_any, bm.as_numpy(res_any))
        assert numpy.array_equal(expected_result_all, bm.as_numpy(res_all))

    # expected result: any/all
    expected_result_any = numpy.any(x_np, axis=(0, 1), keepdims=True)
    expected_result_all = numpy.all(x_np, axis=(0, 1), keepdims=True)
    pattern = "a b ... -> 1 1 ..."
    res_any = ein_reduce(bm.from_numpy(x_np), pattern, reduction="any")
    res_all = ein_reduce(bm.from_numpy(x_np), pattern, reduction="all")
    assert numpy.array_equal(expected_result_any, bm.as_numpy(res_any))
    assert numpy.array_equal(expected_result_all, bm.as_numpy(res_all))
