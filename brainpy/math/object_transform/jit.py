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
"""
The JIT compilation tools for JAX backend.

1. Just-In-Time compilation is implemented by the 'jit()' function

"""

from typing import Callable, Union, Sequence, Iterable

import brainstate.transform
import jax.tree
from brainstate.typing import Missing

from ._utils import warp_to_no_state_input_output

__all__ = [
    'jit',
    'cls_jit',
]

_jit_par = '''
  func : BrainPyObject, function, callable
    The instance of Base or a function.
  static_argnums: optional, int, sequence of int
    An optional int or collection of ints that specify which
    positional arguments to treat as static (compile-time constant).
    Operations that only depend on static arguments will be constant-folded in
    Python (during tracing), and so the corresponding argument values can be
    any Python object.
  static_argnames : optional, str, list, tuple, dict
    An optional string or collection of strings specifying which named arguments to treat
    as static (compile-time constant). See the comment on ``static_argnums`` for details.
    If not provided but ``static_argnums`` is set, the default is based on calling
    ``inspect.signature(fun)`` to find corresponding named arguments.
  donate_argnums: int, sequence of int
    Specify which positional argument buffers are "donated" to
    the computation. It is safe to donate argument buffers if you no longer
    need them once the computation has finished. In some cases XLA can make
    use of donated buffers to reduce the amount of memory needed to perform a
    computation, for example recycling one of your input buffers to store a
    result. You should not reuse buffers that you donate to a computation, JAX
    will raise an error if you try to. By default, no argument buffers are
    donated. Note that donate_argnums only work for positional arguments, and keyword
    arguments will not be donated.
  device: optional, Any
    This is an experimental feature and the API is likely to change.
    Optional, the Device the jitted function will run on. (Available devices
    can be retrieved via :py:func:`jax.devices`.) The default is inherited
    from XLA's DeviceAssignment logic and is usually to use
    ``jax.devices()[0]``.
  keep_unused: bool
    If `False` (the default), arguments that JAX determines to be
    unused by `fun` *may* be dropped from resulting compiled XLA executables.
    Such arguments will not be transferred to the device nor provided to the
    underlying executable. If `True`, unused arguments will not be pruned.
  backend: optional, str
    This is an experimental feature and the API is likely to change.
    Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
    ``'tpu'``.
  inline: bool
    Specify whether this function should be inlined into enclosing
    jaxprs (rather than being represented as an application of the xla_call
    primitive with its own subjaxpr). Default False.
'''


def jit(
    func: Callable = Missing(),

    # original jax.jit parameters
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    donate_argnums: Union[int, Sequence[int]] = (),
    inline: bool = False,
    keep_unused: bool = False,
    # others
    **kwargs,
) -> Union[Callable, Callable[..., Callable]]:
    """
    JIT (Just-In-Time) compilation for BrainPy computation.

    This function has the same ability to just-in-time compile a pure function,
    but it can also JIT compile a :py:class:`brainpy.DynamicalSystem`, or a
    :py:class:`brainpy.BrainPyObject` object.

    Examples::
    
    You can JIT any object in which all dynamical variables are defined as :py:class:`~.Variable`.

    >>> import brainpy as bp
    >>> class Hello(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>     super(Hello, self).__init__()
    >>>     self.a = bp.math.Variable(bp.math.array(10.))
    >>>     self.b = bp.math.Variable(bp.math.array(2.))
    >>>   def transform(self):
    >>>     self.a *= self.b
    >>>
    >>> test = Hello()
    >>> bp.math.jit(test.transform)

    Further, you can JIT a normal function, just used like in JAX.

    >>> @bp.math.jit
    >>> def selu(x, alpha=1.67, lmbda=1.05):
    >>>   return lmbda * bp.math.where(x > 0, x, alpha * bp.math.exp(x) - alpha)


    Parameters::
    
    {jit_par}
    dyn_vars : optional, dict, sequence of Variable, Variable
      These variables will be changed in the function, or needed in the computation.

      .. deprecated:: 2.4.0
         No longer need to provide ``dyn_vars``. This function is capable of automatically
         collecting the dynamical variables used in the target ``func``.
    child_objs: optional, dict, sequence of BrainPyObject, BrainPyObject
      The children objects used in the target function.

      .. deprecated:: 2.4.0
         No longer need to provide ``child_objs``. This function is capable of automatically
         collecting the children objects used in the target ``func``.

    Returns::
    
    func : JITTransform
      A callable jitted function, set up for just-in-time compilation.
    """
    return brainstate.transform.jit(
        warp_to_no_state_input_output(func),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        static_argnames=static_argnames,
        inline=inline,
        keep_unused=keep_unused,
        **kwargs
    )


jit.__doc__ = jit.__doc__.format(jit_par=_jit_par.strip())


def cls_jit(
    func: Callable = Missing(),
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
    inline: bool = False,
    keep_unused: bool = False,
    **kwargs
) -> Callable:
    """Just-in-time compile a function and then the jitted function as the bound method for a class.

    Examples::
    
    This transformation can be put on any class function. For example,

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> class SomeProgram(bp.BrainPyObject):
    >>>   def __init__(self):
    >>>      super(SomeProgram, self).__init__()
    >>>      self.a = bm.zeros(2)
    >>>      self.b = bm.Variable(bm.ones(2))
    >>>
    >>>   @bm.cls_jit(inline=True)
    >>>   def __call__(self, *args, **kwargs):
    >>>      a = bm.random.uniform(size=2)
    >>>      a = a.at[0].set(1.)
    >>>      self.b += a
    >>>
    >>> program = SomeProgram()
    >>> program()

    Parameters::
    
    {jit_pars}

    Returns::
    
    func : JITTransform
      A callable jitted function, set up for just-in-time compilation.
    """
    if static_argnums is None:
        static_argnums = (0,)
    elif isinstance(static_argnums, int):
        static_argnums = (0, static_argnums + 1,)
    elif isinstance(static_argnums, (tuple, list)):
        static_argnums = (0,) + tuple(jax.tree.map(lambda x: x + 1, static_argnums))
    else:
        raise ValueError('static_argnums is not supported yet.')

    return jit(
        func=func,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        inline=inline,
        keep_unused=keep_unused,
        **kwargs
    )


cls_jit.__doc__ = cls_jit.__doc__.format(jit_pars=_jit_par)
