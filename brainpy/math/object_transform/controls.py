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
import numbers
from typing import Union, Sequence, Any, Callable, Optional

import brainstate
import jax
import jax.numpy as jnp

from brainpy.math.ndarray import Array
from ._utils import warp_to_no_state_input_output

__all__ = [
    'cond',
    'ifelse',
    'for_loop',
    'scan',
    'while_loop',
]


def _convert_progress_bar_to_pbar(
    progress_bar: Union[bool, brainstate.transform.ProgressBar, int, None]
) -> Optional[brainstate.transform.ProgressBar]:
    """Convert progress_bar parameter to brainstate pbar format.

    Parameters
    ----------
    progress_bar : bool, ProgressBar, int, None
      The progress_bar parameter value.

    Returns
    -------
    pbar : ProgressBar or None
      The converted ProgressBar instance or None.

    Raises
    ------
    TypeError
      If progress_bar is not a valid type.
    """
    if progress_bar is False or progress_bar is None:
        return None
    elif progress_bar is True:
        return brainstate.transform.ProgressBar()
    elif isinstance(progress_bar, int):
        # Support brainstate convention: int means freq parameter
        return brainstate.transform.ProgressBar(freq=progress_bar)
    elif isinstance(progress_bar, brainstate.transform.ProgressBar):
        return progress_bar
    else:
        raise TypeError(
            f"progress_bar must be bool, int, or ProgressBar instance, "
            f"got {type(progress_bar).__name__}"
        )


def cond(
    pred: bool,
    true_fun: Union[Callable, jnp.ndarray, Array, numbers.Number],
    false_fun: Union[Callable, jnp.ndarray, Array, numbers.Number],
    operands: Any = (),
):
    """Simple conditional statement (if-else) with instance of :py:class:`~.Variable`.

    >>> import brainpy.math as bm
    >>> a = bm.Variable(bm.zeros(2))
    >>> b = bm.Variable(bm.ones(2))
    >>> def true_f():  a.value += 1
    >>> def false_f(): b.value -= 1
    >>>
    >>> bm.cond(True, true_f, false_f)
    >>> a, b
    Variable([1., 1.], dtype=float32), Variable([1., 1.], dtype=float32)
    >>>
    >>> bm.cond(False, true_f, false_f)
    >>> a, b
    Variable([1., 1.], dtype=float32), Variable([0., 0.], dtype=float32)

    Parameters::
    
    pred: bool
      Boolean scalar type, indicating which branch function to apply.
    true_fun: callable, ArrayType, float, int, bool
      Function to be applied if ``pred`` is True.
      This function must receive one arguement for ``operands``.
    false_fun: callable, ArrayType, float, int, bool
      Function to be applied if ``pred`` is False.
      This function must receive one arguement for ``operands``.
    operands: Any
      Operands (A) input to branching function depending on ``pred``. The type
      can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof.

    dyn_vars: optional, Variable, sequence of Variable, dict
      The dynamically changed variables.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.
    child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
      The children objects used in the target function.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.

    Returns::
    
    res: Any
      The conditional results.
    """
    if not isinstance(operands, (tuple, list)):
        operands = (operands,)
    return brainstate.transform.cond(
        pred,
        warp_to_no_state_input_output(true_fun),
        warp_to_no_state_input_output(false_fun),
        *operands
    )


def ifelse(
    conditions: Union[bool, Sequence[bool]],
    branches: Sequence[Any],
    operands: Any = None,
):
    """``If-else`` control flows looks like native Pythonic programming.

    Examples::
    
    >>> import brainpy.math as bm
    >>> def f(a):
    >>>    return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
    >>>                     branches=[lambda: 1,
    >>>                               lambda: 2,
    >>>                               lambda: 3,
    >>>                               lambda: 4,
    >>>                               lambda: 5])
    >>> f(1)
    4
    >>> # or, it can be expressed as:
    >>> def f(a):
    >>>   return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
    >>>                    branches=[1, 2, 3, 4, 5])
    >>> f(3)
    3

    Parameters::
    
    conditions: bool, sequence of bool
      The boolean conditions.
    branches: Any
      The branches, at least has two elements. Elements can be functions,
      arrays, or numbers. The number of ``branches`` and ``conditions`` has
      the relationship of `len(branches) == len(conditions) + 1`.
      Each branch should receive one arguement for ``operands``.
    operands: optional, Any
      The operands for each branch.
    show_code: bool
      Whether show the formatted code.
    dyn_vars: Variable, sequence of Variable, dict
      The dynamically changed variables.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.
    child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
      The children objects used in the target function.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.

    Returns::
    
    res: Any
      The results of the control flow.
    """
    if operands is None:
        operands = ()
    elif not isinstance(operands, (tuple, list)):
        operands = (operands,)

    # Convert non-callable branches to callables
    def make_callable(branch):
        if callable(branch):
            return warp_to_no_state_input_output(branch)
        else:
            return warp_to_no_state_input_output(lambda *args: branch)

    branches = [make_callable(branch) for branch in branches]

    # Convert if-elif-else chain to mutually exclusive conditions
    if isinstance(conditions, (list, tuple)) and len(conditions) > 0:
        conditions = list(conditions)
        # Convert to mutually exclusive conditions for brainstate
        exclusive_conditions = []
        for i, cond in enumerate(conditions):
            if i == 0:
                exclusive_conditions.append(cond)
            else:
                # This condition is true AND all previous conditions are false
                prev_conds_false = jnp.logical_not(conditions[0])
                for j in range(1, i):
                    prev_conds_false = prev_conds_false & jnp.logical_not(conditions[j])
                exclusive_conditions.append(cond & prev_conds_false)

        # If we have equal number of branches and conditions, the last branch is the default case
        if len(branches) == len(conditions):
            # Replace the last condition with "all previous conditions are false"
            all_false = jnp.logical_not(conditions[0])
            for cond in conditions[1:-1]:  # exclude the last condition
                all_false = all_false & jnp.logical_not(cond)
            exclusive_conditions[-1] = all_false
        elif len(branches) > len(conditions):
            # Add the default case (all conditions false)
            all_false = jnp.logical_not(conditions[0])
            for cond in conditions[1:]:
                all_false = all_false & jnp.logical_not(cond)
            exclusive_conditions.append(all_false)

        conditions = exclusive_conditions

    return brainstate.transform.ifelse(conditions, branches, *operands)


def for_loop(
    body_fun: Callable,
    operands: Any,
    reverse: bool = False,
    unroll: int = 1,
    jit: Optional[bool] = None,
    progress_bar: Union[bool, brainstate.transform.ProgressBar, int] = False,
):
    """``for-loop`` control flow with :py:class:`~.Variable`.

    .. versionadded:: 2.1.11

    .. versionchanged:: 2.3.0
       ``dyn_vars`` has been changed into a default argument.
       Please change your call from ``for_loop(fun, dyn_vars, operands)``
       to ``for_loop(fun, operands, dyn_vars)``.

    All returns in body function will be gathered
    as the return of the whole loop.

    >>> import brainpy.math as bm
    >>> a = bm.Variable(bm.zeros(1))
    >>> b = bm.Variable(bm.ones(1))
    >>> # first example
    >>> def body(x):
    >>>    a.value += x
    >>>    b.value *= x
    >>>    return a.value
    >>> a_hist = bm.for_loop(body, operands=bm.arange(1, 5))
    >>> a_hist
    DeviceArray([[ 1.],
                 [ 3.],
                 [ 6.],
                 [10.]], dtype=float32)
    >>> a
    Variable([10.], dtype=float32)
    >>> b
    Variable([24.], dtype=float32)
    >>>
    >>> # another example
    >>> def body(x, y):
    >>>   a.value += x
    >>>   b.value *= y
    >>>   return a.value
    >>> a_hist = bm.for_loop(body, operands=(bm.arange(1, 5), bm.arange(2, 6)))
    >>> a_hist
    [[11.]
     [13.]
     [16.]
     [20.]]

    Parameters::
    
    body_fun: callable
      A Python function to be scanned. This function accepts one argument and returns one output.
      The argument denotes a slice of ``operands`` along its leading axis, and that
      output represents a slice of the return value.
    operands: Any
      The value over which to scan along the leading axis,
      where ``operands`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
      If body function `body_func` receives multiple arguments,
      `operands` should be a tuple/list whose length is equal to the
      number of arguments.
    reverse: bool
      Optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: int
      Optional positive int specifying, in the underlying operation of the
      scan primitive, how many scan iterations to unroll within a single
      iteration of a loop.
    jit: bool
      Whether to just-in-time compile the function. Set to ``False`` to disable JIT compilation.
    progress_bar: bool, ProgressBar, int
      Whether and how to display a progress bar during execution:

      - ``False`` (default): No progress bar
      - ``True``: Display progress bar with default settings
      - ``ProgressBar`` instance: Display progress bar with custom settings
      - ``int``: Display progress bar updating every N iterations (treated as freq parameter)

      For advanced customization, create a :py:class:`brainpy.math.ProgressBar` instance:

      >>> import brainpy.math as bm
      >>> # Custom update frequency
      >>> pbar = bm.ProgressBar(freq=10)
      >>> result = bm.for_loop(body_fun, operands, progress_bar=pbar)
      >>>
      >>> # Custom description
      >>> pbar = bm.ProgressBar(desc="Processing data")
      >>> result = bm.for_loop(body_fun, operands, progress_bar=pbar)
      >>>
      >>> # Update exactly 20 times during execution
      >>> pbar = bm.ProgressBar(count=20)
      >>> result = bm.for_loop(body_fun, operands, progress_bar=pbar)
      >>>
      >>> # Integer shorthand (equivalent to ProgressBar(freq=10))
      >>> result = bm.for_loop(body_fun, operands, progress_bar=10)

      .. versionadded:: 2.4.2
      .. versionchanged:: 2.7.3
         Now accepts ProgressBar instances and integers for advanced customization.
    dyn_vars: Variable, sequence of Variable, dict
      The instances of :py:class:`~.Variable`.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.
    child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
      The children objects used in the target function.

      .. versionadded:: 2.3.1

      .. deprecated:: 2.4.0
         No longer need to provide ``child_objs``. This function is capable of automatically
         collecting the children objects used in the target ``func``.

    Returns::
    
    outs: Any
      The stacked outputs of ``body_fun`` when scanned over the leading axis of the inputs.
    """
    if not isinstance(operands, (tuple, list)):
        operands = (operands,)

    # Convert progress_bar to pbar format
    pbar = _convert_progress_bar_to_pbar(progress_bar)

    # Handle jit parameter
    # Note: JAX's scan doesn't support zero-length inputs in disable_jit mode.
    # For zero-length inputs, we need to use JIT mode even when jit=False.
    should_disable_jit = False
    if jit is False:
        # Check if any operand has zero length
        first_operand = operands[0]
        is_zero_length = False
        if hasattr(first_operand, 'shape') and len(first_operand.shape) > 0:
            is_zero_length = (first_operand.shape[0] == 0)

        if is_zero_length:
            # Use JIT mode for zero-length inputs to avoid JAX limitation
            import warnings
            warnings.warn(
                "for_loop with jit=False and zero-length input detected. "
                "Using JIT mode to avoid JAX's disable_jit limitation with zero-length scans.",
                UserWarning
            )
        else:
            should_disable_jit = True

    if should_disable_jit:
        with jax.disable_jit():
            return brainstate.transform.for_loop(
                warp_to_no_state_input_output(body_fun),
                *operands, reverse=reverse, unroll=unroll,
                pbar=pbar,
            )
    else:
        return brainstate.transform.for_loop(
            warp_to_no_state_input_output(body_fun),
            *operands, reverse=reverse, unroll=unroll,
            pbar=pbar,
        )


def scan(
    body_fun: Callable,
    init: Any,
    operands: Any,
    reverse: bool = False,
    unroll: int = 1,
    remat: bool = False,
    progress_bar: Union[bool, brainstate.transform.ProgressBar, int] = False,
):
    """``scan`` control flow with :py:class:`~.Variable`.

    Similar to ``jax.lax.scan``.

    .. versionadded:: 2.4.7

    All returns in body function will be gathered
    as the return of the whole loop.

    Parameters::
    
    body_fun: callable
      A Python function to be scanned. This function accepts one argument and returns one output.
      The argument denotes a slice of ``operands`` along its leading axis, and that
      output represents a slice of the return value.
    init: Any
      An initial loop carry value of type ``c``, which can be a scalar, array, or any pytree
      (nested Python tuple/list/dict) thereof, representing the initial loop carry value.
      This value must have the same structure as the first element of the pair returned
      by ``body_fun``.
    operands: Any
      The value over which to scan along the leading axis,
      where ``operands`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.
      If body function `body_func` receives multiple arguments,
      `operands` should be a tuple/list whose length is equal to the
      number of arguments.
    remat: bool
      Make ``fun`` recompute internal linearization points when differentiated.
    reverse: bool
      Optional boolean specifying whether to run the scan iteration
      forward (the default) or in reverse, equivalent to reversing the leading
      axes of the arrays in both ``xs`` and in ``ys``.
    unroll: int
      Optional positive int specifying, in the underlying operation of the
      scan primitive, how many scan iterations to unroll within a single
      iteration of a loop.
    progress_bar: bool, ProgressBar, int
      Whether and how to display a progress bar during execution:

      - ``False`` (default): No progress bar
      - ``True``: Display progress bar with default settings
      - ``ProgressBar`` instance: Display progress bar with custom settings
      - ``int``: Display progress bar updating every N iterations (treated as freq parameter)

      See :py:func:`for_loop` for detailed examples of ProgressBar usage.

      .. versionadded:: 2.4.2
      .. versionchanged:: 2.7.3
         Now accepts ProgressBar instances and integers for advanced customization.

    Returns::
    
    outs: Any
      The stacked outputs of ``body_fun`` when scanned over the leading axis of the inputs.
    """
    # Convert progress_bar to pbar format
    pbar = _convert_progress_bar_to_pbar(progress_bar)

    return brainstate.transform.scan(
        warp_to_no_state_input_output(body_fun),
        init=init,
        xs=operands,
        reverse=reverse,
        unroll=unroll,
        pbar=pbar,
    )


def while_loop(
    body_fun: Callable,
    cond_fun: Callable,
    operands: Any,
):
    """``while-loop`` control flow with :py:class:`~.Variable`.

    .. versionchanged:: 2.3.0
       ``dyn_vars`` has been changed into a default argument.
       Please change your call from ``while_loop(f1, f2, dyn_vars, operands)``
       to ``while_loop(f1, f2, operands, dyn_vars)``.

    Note the diference between ``for_loop`` and ``while_loop``:

    1. ``while_loop`` does not support accumulating history values.
    2. The returns on the body function of ``for_loop`` represent the values to stack at one moment.
       However, the functional returns of body function in ``while_loop`` represent the operands'
       values at the next moment, meaning that the body function of ``while_loop`` defines the
       updating rule of how the operands are updated.

    >>> import brainpy.math as bm
    >>>
    >>> a = bm.Variable(bm.zeros(1))
    >>> b = bm.Variable(bm.ones(1))
    >>>
    >>> def cond(x, y):
    >>>    return x < 6.
    >>>
    >>> def body(x, y):
    >>>    a.value += x
    >>>    b.value *= y
    >>>    return x + b[0], y + 1.
    >>>
    >>> res = bm.while_loop(body, cond, operands=(1., 1.))
    >>> res
    (10.0, 4.0)

    .. versionadded:: 2.1.11

    Parameters::
    
    body_fun: callable
      A function which define the updating logic. It receives one argument for ``operands``, without returns.
    cond_fun: callable
      A function which define the stop condition. It receives one argument for ``operands``,
      with one boolean value return.
    operands: Any
      The operands for ``body_fun`` and ``cond_fun`` functions.
    dyn_vars: Variable, sequence of Variable, dict
      The dynamically changed variables.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.
    child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
      The children objects used in the target function.

      .. deprecated:: 2.4.0
         No longer need to provide ``child_objs``. This function is capable of automatically
         collecting the children objects used in the target ``func``.

    """
    if not isinstance(operands, (tuple, list)):
        operands = (operands,)
    operands = tuple(operands)

    def body(x):
        r = body_fun(*x)
        if r is None:
            return x
        else:
            return r

    return brainstate.transform.while_loop(
        warp_to_no_state_input_output(lambda x: cond_fun(*x)),
        warp_to_no_state_input_output(body),
        operands
    )
