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
from typing import Union, Callable, Dict, Sequence, Optional

import brainstate.transform

from ._utils import warp_to_no_state_input_output
from .variables import Variable

__all__ = [
    'grad',  # gradient of scalar function
    'vector_grad',  # gradient of vector/matrix/...
    'functional_vector_grad',
    'jacobian', 'jacrev', 'jacfwd',  # gradient of jacobian
    'hessian',  # gradient of hessian
]


def grad(
    func: Optional[Callable] = None,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    holomorphic: Optional[bool] = False,
    allow_int: Optional[bool] = False,
    has_aux: Optional[bool] = None,
    return_value: Optional[bool] = False,
) -> Union[Callable, Callable[..., Callable]]:
    """Automatic gradient computation for functions or class objects.

    This gradient function only support scalar return. It creates a function
    which evaluates the gradient of ``func``.

    It's worthy to note that the returns are different for different argument settings (where ``arg_grads`` refers
    to the gradients of "argnums", and ``var_grads`` refers to the gradients of "grad_vars").

    1. When "grad_vars" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_vars" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_vars" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

    Let's see some examples below.

    Before start, let's figure out what should be provided as ``grad_vars``?
    And, what should be labeled in ``argnums``?
    Take the following codes as example:

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class Example(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>     super(Example, self).__init__()
    >>>     self.x = bm.TrainVar(bm.zeros(1))
    >>>     self.y = bm.random.rand(10)
    >>>   def __call__(self, z, v):
    >>>     t1 = self.x * self.y.sum()
    >>>     t2 = bm.tanh(z * v + t1)
    >>>     return t2.mean()
    >>>
    >>> # This code is equivalent to the following function:
    >>>
    >>> x = bm.TrainVar(bm.zeros(1))
    >>> y = bm.random.rand(10)
    >>> def f(z, v):
    >>>   t1 = x * y.sum()
    >>>   t2 = bm.tanh(z * v + t1)
    >>>   return t2.mean()

    Generally speaking, all gradient variables which not provided in arguments should be
    labeled as ``grad_vars``, while all gradient variables provided in the function arguments
    should be declared in ``argnums``.
    In above codes, we try to take gradients of ``self.x`` and arguments ``z`` and ``v``, we should
    call ``brainpy.math.grad`` as:

    >>> f = Example()
    >>> f_grad = bm.grad(f, grad_vars=f.x, argnums=(0, 1))


    Examples::
    
    Grad for a pure function:

    >>> import brainpy as bp
    >>> grad_tanh = grad(bp.math.tanh)
    >>> print(grad_tanh(0.2))
    0.961043

    Parameters::
    
    func : callable, function, BrainPyObject
      Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers.
      Argument arrays in the positions specified by ``argnums`` must be of
      inexact (i.e., floating-point or complex) type. It should return a scalar
      (which includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
    grad_vars : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    argnums : optional, integer or sequence of integers
      Specifies which positional argument(s) to differentiate with respect to (default 0).
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    holomorphic: optional, bool
      Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: optional, bool
      Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

    Returns::
    
    func : GradientTransform
      A function with the same arguments as ``fun``, that evaluates the gradient
      of ``fun``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If
      argnums is a tuple of integers, the gradient is a tuple of values with the
      same shapes and types as the corresponding arguments. If ``has_aux`` is True
      then a pair of (gradient, auxiliary_data) is returned.
    """
    if func is None:
        return lambda f: grad(f,
                              grad_vars=grad_vars,
                              argnums=argnums,
                              holomorphic=holomorphic,
                              allow_int=allow_int,
                              has_aux=has_aux,
                              return_value=return_value)
    else:
        return brainstate.transform.grad(
            warp_to_no_state_input_output(func),
            grad_states=grad_vars,
            argnums=argnums,
            holomorphic=holomorphic,
            allow_int=allow_int,
            has_aux=has_aux,
            return_value=return_value,
            check_states=False,
        )


def jacrev(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
):
    """Extending automatic Jacobian (reverse-mode) of ``func`` to classes.

    This function extends the JAX official ``jacrev`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
    different for different argument settings in ``brainpy.math.jacrev``.

    1. When "grad_vars" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_vars" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_vars" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.


    Parameters::
    
    func: Function whose Jacobian is to be computed.
    grad_vars : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers.
      Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool.
      Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool.
      Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

    Returns::
    
    fun: GradientTransform
      The transformed object.
    """

    return brainstate.transform.jacrev(
        warp_to_no_state_input_output(func),
        grad_states=grad_vars,
        argnums=argnums,
        holomorphic=holomorphic,
        allow_int=allow_int,
        has_aux=has_aux,
        return_value=return_value,
        check_states=False,
    )


jacobian = jacrev


def jacfwd(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    return_value: bool = False,
    holomorphic: bool = False,
):
    """Extending automatic Jacobian (forward-mode) of ``func`` to classes.

    This function extends the JAX official ``jacfwd`` to make automatic jacobian
    computation on functions and class functions. Moreover, it supports returning
    value ("return_value") and returning auxiliary data ("has_aux").

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_, the returns are
    different for different argument settings in ``brainpy.math.jacfwd``.

    1. When "grad_vars" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_vars" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_vars" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.

    Parameters::
    
    func: Function whose Jacobian is to be computed.
    grad_vars : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

    Returns::
    
    obj: GradientTransform
      The transformed object.
    """
    return brainstate.transform.jacfwd(
        warp_to_no_state_input_output(func),
        grad_states=grad_vars,
        argnums=argnums,
        holomorphic=holomorphic,
        has_aux=has_aux,
        return_value=return_value,
        check_states=False,
    )


def hessian(
    func: Callable,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    has_aux: Optional[bool] = None,
    holomorphic=False,
):
    """Hessian of ``func`` as a dense array.

    Parameters::
    
    func : callable, function
      Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    grad_vars : optional, ArrayCollector, sequence of ArrayType
      The variables required to compute their gradients.
    argnums: Optional, integer or sequence of integers
      Specifies which positional argument(s) to differentiate with respect to (default ``0``).
    holomorphic : bool
      Indicates whether ``fun`` is promised to be holomorphic. Default False.
    has_aux : bool, optional
      Indicates whether ``fun`` returns a pair where the first element is
      considered the output of the mathematical function to be differentiated
      and the second element is auxiliary data. Default False.

    Returns::
    
    obj: ObjectTransform
      The transformed object.
    """

    return brainstate.transform.hessian(
        warp_to_no_state_input_output(func),
        grad_states=grad_vars,
        argnums=argnums,
        holomorphic=holomorphic,
        has_aux=has_aux,
        check_states=False,
    )


def vector_grad(
    func: Optional[Callable] = None,
    grad_vars: Optional[Union[Variable, Sequence[Variable], Dict[str, Variable]]] = None,
    argnums: Optional[Union[int, Sequence[int]]] = None,
    return_value: bool = False,
    has_aux: Optional[bool] = None,
) -> Callable:
    """Take vector-valued gradients for function ``func``.

    Same as `brainpy.math.grad <./brainpy.math.autograd.grad.html>`_,
    `brainpy.math.jacrev <./brainpy.math.autograd.jacrev.html>`_ and
    `brainpy.math.jacfwd <./brainpy.math.autograd.jacfwd.html>`_,
    the returns in this function are different for different argument settings.

    1. When "grad_vars" is None
      - "has_aux=False" + "return_value=False" => ``arg_grads``.
      - "has_aux=True" + "return_value=False" => ``(arg_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(arg_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(arg_grads, loss_value, aux_data)``.
    2. When "grad_vars" is not None and "argnums" is None
      - "has_aux=False" + "return_value=False" => ``var_grads``.
      - "has_aux=True" + "return_value=False" => ``(var_grads, aux_data)``.
      - "has_aux=False" + "return_value=True" => ``(var_grads, loss_value)``.
      - "has_aux=True" + "return_value=True" => ``(var_grads, loss_value, aux_data)``.
    3. When "grad_vars" is not None and "argnums" is not None
      - "has_aux=False" + "return_value=False" => ``(var_grads, arg_grads)``.
      - "has_aux=True" + "return_value=False" => ``((var_grads, arg_grads), aux_data)``.
      - "has_aux=False" + "return_value=True" => ``((var_grads, arg_grads), loss_value)``.
      - "has_aux=True" + "return_value=True" => ``((var_grads, arg_grads), loss_value, aux_data)``.


    Parameters::
    
    func: Callable
      Function whose gradient is to be computed.
    grad_vars : optional, ArrayType, sequence of ArrayType, dict
      The variables in ``func`` to take their gradients.
    has_aux: optional, bool
      Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    return_value : bool
      Whether return the loss value.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).

    Returns::
    
    func : GradientTransform
      The vector gradient function.
    """
    return brainstate.transform.vector_grad(
        warp_to_no_state_input_output(func),
        grad_states=grad_vars,
        argnums=argnums,
        return_value=return_value,
        has_aux=has_aux,
        check_states=False,
    )


functional_vector_grad = vector_grad
