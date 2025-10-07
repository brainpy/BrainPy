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
import functools
import itertools
from collections import OrderedDict
from typing import Set, Tuple, List, Dict, Union, Callable, Optional, cast

import jax
import numpy as np

from . import compat_numpy as bnp
from . import others as bnp2
from .einops_parsing import ParsedExpression, _ellipsis, AnonymousAxis, EinopsError
from .ndarray import Array

__all__ = [
    'ein_reduce', 'ein_rearrange', 'ein_repeat', 'ein_shape',
]

Tensor = Union[Array, jax.Array]
ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]
Reduction = Union[str, ReductionCallable]

_reductions = ("min", "max", "sum", "mean", "prod", "any", "all")

# magic integers are required to stay within
# traceable subset of language
_unknown_axis_length = -999999
_expected_axis_length = -99999


def _product(sequence: List[int]) -> int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    result = 1
    for element in sequence:
        result *= element
    return result


def _reduce_axes(tensor, reduction_type: Reduction, reduced_axes: List[int]):
    if callable(reduction_type):
        # custom callable
        return reduction_type(tensor, tuple(reduced_axes))
    else:
        # one of built-in operations
        assert reduction_type in _reductions
        if reduction_type == "mean":
            if not bnp2.is_float_type(tensor):
                raise NotImplementedError("reduce_mean is not available for non-floating tensors")
        return __reduce(tensor, reduction_type, tuple(reduced_axes))


def __reduce(x: Union[Array, jax.Array], operation: str, reduced_axes):
    if operation == "min":
        return x.min(axis=reduced_axes)
    elif operation == "max":
        return x.max(axis=reduced_axes)
    elif operation == "sum":
        return x.sum(axis=reduced_axes)
    elif operation == "mean":
        return x.mean(axis=reduced_axes)
    elif operation == "prod":
        return x.prod(axis=reduced_axes)
    elif operation == "any":
        return x.any(axis=reduced_axes)
    elif operation == "all":
        return x.all(axis=reduced_axes)
    else:
        raise NotImplementedError("Unknown reduction ", operation)


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes):
    # 'collapses' neighboring axes if those participate in the result pattern in the same order
    # TODO add support for added_axes
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    # joining consecutive axes that will be reduced
    # possibly we can skip this if all backends can optimize this (not sure)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[: i + 1] + tuple(axis - 1 for axis in reduced_axes[i + 2:])

    # removing axes that are moved together during reshape
    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.values())
                init_to_final[axis] = list(axes_reordering).index(after_reduction)
        return init_to_final

    init_axis_to_final_axis = build_mapping()

    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in range(removed_axis))

            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            old_reordering = axes_reordering
            axes_reordering = []
            for axis in old_reordering:
                if axis == removed_axis_after_reduction:
                    pass
                elif axis < removed_axis_after_reduction:
                    axes_reordering.append(axis)
                else:
                    axes_reordering.append(axis - 1)
            init_axis_to_final_axis = build_mapping()

    return init_shapes, reduced_axes, axes_reordering, final_shapes


CookedRecipe = Tuple[Optional[List[int]], Optional[List[int]], List[int], Dict[int, int], Optional[List[int]], int]

# Actual type is tuple[tuple[str, int], ...]
# However torch.jit.script does not "understand" the correct type,
# and torch_specific will use list version.
HashableAxesLengths = Tuple[Tuple[str, int], ...]
FakeHashableAxesLengths = List[Tuple[str, int]]


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    # structure is non-mutable. In future, this can be non-mutable dataclass (python 3.7+)
    # update: pytorch 2.0 torch.jit.script seems to have problems with dataclasses unless they were explicitly provided

    def __init__(
        self,
        # list of sizes (or just sizes) for elementary axes as they appear in left expression.
        # this is what (after computing unknown parts) will be a shape after first transposition.
        # This does not include any ellipsis dimensions.
        elementary_axes_lengths: List[int],
        # if additional axes are provided, they should be set in prev array
        # This shows mapping from name to position
        axis_name2elementary_axis: Dict[str, int],
        # each dimension in input can help to reconstruct length of one elementary axis
        # or verify one of dimensions. Each element points to element of elementary_axes_lengths.
        input_composition_known_unknown: List[Tuple[List[int], List[int]]],
        # permutation applied to elementary axes, if ellipsis is absent
        axes_permutation: List[int],
        # permutation puts reduced axes in the end, we only need to know the first position.
        first_reduced_axis: int,
        # at which positions which of elementary axes should appear. Axis position -> axis index.
        added_axes: Dict[int, int],
        # ids of axes as they appear in result, again pointers to elementary_axes_lengths,
        # only used to infer result dimensions
        output_composite_axes: List[List[int]],
    ):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.axis_name2elementary_axis: Dict[str, int] = axis_name2elementary_axis
        self.input_composition_known_unknown: List[Tuple[List[int], List[int]]] = input_composition_known_unknown
        self.axes_permutation: List[int] = axes_permutation

        self.first_reduced_axis: int = first_reduced_axis
        self.added_axes: Dict[int, int] = added_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes


def _reconstruct_from_shape_uncached(
    self: TransformRecipe, shape: List[int], axes_dims: FakeHashableAxesLengths
) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    # magic number
    need_init_reshape = False

    # last axis is allocated for collapsed ellipsis
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim

    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composition_known_unknown):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            # shortcut for the most common case
            axes_lengths[unknown_axes[0]] = length
            continue

        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]

        if len(unknown_axes) == 0:
            if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                raise EinopsError(f"Shape mismatch, {length} != {known_product}")
        else:
            # assert len(unknown_axes) == 1, 'this is enforced when recipe is created, so commented out'
            if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                raise EinopsError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}")

            unknown_axis = unknown_axes[0]
            inferred_length: int = length // known_product
            axes_lengths[unknown_axis] = inferred_length

        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # elementary axes are ordered as they appear in input, then all added axes
    init_shapes: Optional[List[int]] = axes_lengths[: len(self.axes_permutation)] if need_init_reshape else None

    need_final_reshape = False
    final_shapes: List[int] = []
    for grouping in self.output_composite_axes:
        lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
        final_shapes.append(_product(lengths))
        if len(lengths) != 1:
            need_final_reshape = True

    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()
    }

    # this list can be empty
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))

    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)

    axes_reordering: Optional[List[int]] = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None

    _final_shapes = final_shapes if need_final_reshape else None
    return init_shapes, axes_reordering, reduced_axes, added_axes, _final_shapes, n_axes_after_adding_axes


_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)


def _apply_recipe(
    recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths
) -> Tensor:
    # this method implements actual work for all backends for 3 operations
    try:
        init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = (
            _reconstruct_from_shape(recipe, bnp.shape(tensor), axes_lengths))
    except TypeError:
        # shape or one of passed axes lengths is not hashable (i.e. they are symbols)
        _result = _reconstruct_from_shape_uncached(recipe, bnp.shape(tensor), axes_lengths)
        (init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added) = _result
    if init_shapes is not None:
        tensor = bnp.reshape(bnp.as_jax(tensor), init_shapes)
    if axes_reordering is not None:
        tensor = bnp.transpose(bnp.as_jax(tensor), axes_reordering)
    if len(reduced_axes) > 0:
        tensor = _reduce_axes(bnp.as_jax(tensor), reduction_type=reduction_type, reduced_axes=reduced_axes)
    if len(added_axes) > 0:
        tensor = bnp2.add_axes(tensor, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        tensor = bnp.reshape(bnp.as_jax(tensor), final_shapes)
    return tensor


def _apply_recipe_array_api(
    xp, recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction, axes_lengths: HashableAxesLengths
) -> Tensor:
    # completely-inline implementation
    init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = _reconstruct_from_shape(
        recipe, tensor.shape, axes_lengths
    )
    if init_shapes is not None:
        tensor = xp.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = xp.permute_dims(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        if callable(reduction_type):
            # custom callable
            tensor = reduction_type(tensor, tuple(reduced_axes))
        else:
            # one of built-in operations
            assert reduction_type in _reductions
            tensor = getattr(xp, reduction_type)(tensor, axis=tuple(reduced_axes))
    if len(added_axes) > 0:
        # we use broadcasting
        for axis_position, axis_length in added_axes.items():
            tensor = xp.expand_dims(tensor, axis=axis_position)

        final_shape = list(tensor.shape)
        for axis_position, axis_length in added_axes.items():
            final_shape[axis_position] = axis_length

        tensor = xp.broadcast_to(tensor, final_shape)
    if final_shapes is not None:
        tensor = xp.reshape(tensor, final_shapes)
    return tensor


@functools.lru_cache(256)
def _prepare_transformation_recipe(
    pattern: str,
    operation: Reduction,
    axes_names: Tuple[str, ...],
    ndim: int,
) -> TransformRecipe:
    """Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # checking that axes are in agreement - new axes appear only in repeat, while disappear only in reduction
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError("Ellipsis found in right side, but not left side of a pattern {}".format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError("Ellipsis inside parenthesis in the left side is not allowed: {}".format(pattern))
    if operation == "rearrange":
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Identifiers only on one side of expression (should be on both): {}".format(difference))
    elif operation == "repeat":
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the left side of repeat: {}".format(difference))
        axes_without_size = set.difference(
            {ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)},
            {*left.identifiers, *axes_names},
        )
        if len(axes_without_size) > 0:
            raise EinopsError("Specify sizes for new axes in repeat: {}".format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError("Unexpected identifiers on the right side of reduce {}: {}".format(operation, difference))
    else:
        raise EinopsError("Unknown reduction {}. Expect one of {}.".format(operation, _reductions))

    if left.has_ellipsis:
        n_other_dims = len(left.composition) - 1
        if ndim < n_other_dims:
            raise EinopsError(f"Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.")
        ellipsis_ndim = ndim - n_other_dims
        ell_axes = [_ellipsis + str(i) for i in range(ellipsis_ndim)]
        left_composition = []
        for composite_axis in left.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    left_composition.append([axis])
            else:
                left_composition.append(composite_axis)

        rght_composition = []
        for composite_axis in rght.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    rght_composition.append([axis])
            else:
                group = []
                for axis in composite_axis:
                    if axis == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(axis)
                rght_composition.append(group)

        left.identifiers.update(ell_axes)
        left.identifiers.remove(_ellipsis)
        if rght.has_ellipsis:
            rght.identifiers.update(ell_axes)
            rght.identifiers.remove(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.")
        left_composition = left.composition
        rght_composition = rght.composition

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left_composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length

    # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)

    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}

    # axes provided as kwargs
    for elementary_axis in axes_names:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError("Invalid name for an axis", elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError("Axis {} is not used in transform".format(elementary_axis))
        axis_name2known_length[elementary_axis] = _expected_axis_length

    input_axes_known_unknown = []
    # some shapes are inferred later - all information is prepared for faster inference
    for i, composite_axis in enumerate(left_composition):
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError("Could not infer sizes for {}".format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(
            ([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown])
        )

    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left_composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

    result_axes_grouping: List[List[int]] = [
        [axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(rght_composition)
    ]

    ordered_axis_left = list(itertools.chain(*left_composition))
    ordered_axis_rght = list(itertools.chain(*rght_composition))
    reduced_axes = [axis for axis in ordered_axis_left if axis not in rght.identifiers]
    order_after_transposition = [axis for axis in ordered_axis_rght if axis in left.identifiers] + reduced_axes
    axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
    added_axes = {
        i: axis_name2position[axis_name]
        for i, axis_name in enumerate(ordered_axis_rght)
        if axis_name not in left.identifiers
    }

    first_reduced_axis = len(order_after_transposition) - len(reduced_axes)

    return TransformRecipe(
        elementary_axes_lengths=list(axis_name2known_length.values()),
        axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names},
        input_composition_known_unknown=input_axes_known_unknown,
        axes_permutation=axes_permutation,
        first_reduced_axis=first_reduced_axis,
        added_axes=added_axes,
        output_composite_axes=result_axes_grouping,
    )


def _prepare_recipes_for_all_dims(
    pattern: str, operation: Reduction, axes_names: Tuple[str, ...]
) -> Dict[int, TransformRecipe]:
    """
    Internal function, used in layers.
    Layer makes all recipe creation when it is initialized, thus to keep recipes simple we pre-compute for all dims
    """
    left_str, rght_str = pattern.split("->")
    left = ParsedExpression(left_str)
    dims = [len(left.composition)]
    if left.has_ellipsis:
        dims = [len(left.composition) - 1 + ellipsis_dims for ellipsis_dims in range(8)]
    return {ndim: _prepare_transformation_recipe(pattern, operation, axes_names, ndim=ndim) for ndim in dims}


def ein_reduce(tensor: Union[Tensor, List[Tensor]], pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    """
    ``ein_reduce`` provides combination of reordering and reduction using reader-friendly notation.

    Examples for reduce operation:

    ```python
    >>> x = np.random.randn(100, 32, 64)

    # perform max-reduction on the first axis
    >>> y = ein_reduce(x, 't b c -> b c', 'max')

    # same as previous, but with clearer axes meaning
    >>> y = ein_reduce(x, 'time batch channel -> batch channel', 'max')

    >>> x = np.random.randn(10, 20, 30, 40)

    # 2d max-pooling with kernel size = 2 * 2 for image processing
    >>> y1 = ein_reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)

    # if one wants to go back to the original height and width, depth-to-space trick can be applied
    >>> y2 = ein_rearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
    >>> assert ein_shape(x, 'b _ h w') == ein_shape(y2, 'b _ h w')

    # Adaptive 2d max-pooling to 3 * 4 grid
    >>> ein_reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=3, w1=4).shape
    (10, 20, 3, 4)

    # Global average pooling
    >>> ein_reduce(x, 'b c h w -> b c', 'mean').shape
    (10, 20)

    # Subtracting mean over batch for each channel
    >>> y = x - ein_reduce(x, 'b c h w -> () c () ()', 'mean')

    # Subtracting per-image mean for each channel
    >>> y = x - ein_reduce(x, 'b c h w -> b c () ()', 'mean')

    ```

    Parameters:
        tensor: tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, reduction pattern
        reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
            alternatively, a callable f(tensor, reduced_axes) -> tensor can be provided.
            This allows using various reductions, examples: np.max, tf.reduce_logsumexp, torch.var, etc.
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input
    """
    try:
        hashable_axes_lengths = tuple(axes_lengths.items())
        shape = bnp.shape(tensor)
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=len(shape))
        return _apply_recipe(recipe,
                             cast(Tensor, tensor),
                             reduction_type=reduction,
                             axes_lengths=hashable_axes_lengths)
    except EinopsError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(tensor, list):
            message += "\n Input tensor shape: {}. ".format(shape)
        else:
            message += "\n Input is list. "
        message += "Additional info: {}.".format(axes_lengths)
        raise EinopsError(message + "\n {}".format(e))


def ein_rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    ``ein_rearrange`` is a reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    Examples for rearrange operation:

    ```python
    # suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> images = [np.random.randn(30, 40, 3) for _ in range(32)]

    # stack along first (batch) axis, output is a single array
    >>> ein_rearrange(images, 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    # concatenate images along height (vertical axis), 960 = 32 * 30
    >>> ein_rearrange(images, 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    # concatenated images along horizontal axis, 1280 = 32 * 40
    >>> ein_rearrange(images, 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    # reordered axes to "b c h w" format for deep learning
    >>> ein_rearrange(images, 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    # flattened each image into a vector, 3600 = 30 * 40 * 3
    >>> ein_rearrange(images, 'b h w c -> b (c h w)').shape
    (32, 3600)

    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
    >>> ein_rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    (128, 15, 20, 3)

    # space-to-depth operation
    >>> ein_rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    (32, 15, 20, 12)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
                list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return ein_reduce(tensor, pattern, reduction="rearrange", **axes_lengths)


def ein_repeat(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    ``ein_repeat`` allows reordering elements and repeating them in arbitrary combinations.
    This operation includes functionality of repeat, tile, broadcast functions.

    Examples for repeat operation:

    ```python
    # a grayscale image (of shape height x width)
    >>> image = np.random.randn(30, 40)

    # change it to RGB format by repeating in each channel
    >>> ein_repeat(image, 'h w -> h w c', c=3).shape
    (30, 40, 3)

    # repeat image 2 times along height (vertical axis)
    >>> ein_repeat(image, 'h w -> (repeat h) w', repeat=2).shape
    (60, 40)

    # repeat image 2 time along height and 3 times along width
    >>> ein_repeat(image, 'h w -> (h2 h) (w3 w)', h2=2, w3=3).shape
    (60, 120)

    # convert each pixel to a small square 2x2. Upsample image by 2x
    >>> ein_repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (60, 80)

    # pixelate image first by downsampling by 2x, then upsampling
    >>> downsampled = ein_reduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
    >>> ein_repeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (30, 40)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        Tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return ein_reduce(tensor, pattern, reduction="repeat", **axes_lengths)


def ein_shape(x, pattern: str) -> dict:
    """
    Parse a tensor shape to dictionary mapping axes names to their lengths.

    ```python
    # Use underscore to skip the dimension in parsing.
    >>> x = np.zeros([2, 3, 5, 7])
    >>> ein_shape(x, 'batch _ h w')
    {'batch': 2, 'h': 5, 'w': 7}

    # `parse_shape` output can be used to specify axes_lengths for other operations:
    >>> y = np.zeros([700])
    >>> ein_rearrange(y, '(b c h w) -> b c h w', **ein_shape(x, 'b _ h w')).shape
    (2, 10, 5, 7)

    ```

    For symbolic frameworks may return symbols, not integers.

    Parameters:
        x: tensor of any supported framework
        pattern: str, space separated names for axes, underscore means skip axis

    Returns:
        dict, maps axes names to their lengths
    """
    exp = ParsedExpression(pattern, allow_underscore=True)
    shape = bnp.shape(x)
    if exp.has_composed_axes():
        raise RuntimeError(f"Can't parse shape with composite axes: {pattern} {shape}")
    if len(shape) != len(exp.composition):
        if exp.has_ellipsis:
            if len(shape) < len(exp.composition) - 1:
                raise RuntimeError(f"Can't parse shape with this number of dimensions: {pattern} {shape}")
        else:
            raise RuntimeError(f"Can't parse shape with different number of dimensions: {pattern} {shape}")
    if exp.has_ellipsis:
        ellipsis_idx = exp.composition.index(_ellipsis)
        composition = (
            exp.composition[:ellipsis_idx]
            + ["_"] * (len(shape) - len(exp.composition) + 1)
            + exp.composition[ellipsis_idx + 1:]
        )
    else:
        composition = exp.composition
    result = {}
    for (axis_name,), axis_length in zip(composition, shape):  # type: ignore
        if axis_name != "_":
            result[axis_name] = axis_length
    return result


# _enumerate_directions is not exposed in the public API
def _enumerate_directions(x):
    """
    For an n-dimensional tensor, returns tensors to enumerate each axis.
    ```python
    x = np.zeros([2, 3, 4]) # or any other tensor
    i, j, k = _enumerate_directions(x)
    result = i + 2*j + 3*k
    ```

    `result[i, j, k] = i + 2j + 3k`, and also has the same shape as result
    Works very similarly to numpy.ogrid (open indexing grid)
    """
    shape = bnp.shape(x)
    result = []
    for axis_id, axis_length in enumerate(shape):
        shape = [1] * len(shape)
        shape[axis_id] = axis_length
        result.append(bnp.reshape(bnp.arange(0, axis_length), shape))
    return result
