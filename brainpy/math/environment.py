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
import functools
import gc
import inspect
import os
import re
import sys
import warnings
from typing import Any, Callable, TypeVar, cast

import brainstate.environ
import jax
from jax import config, numpy as jnp, devices
from jax.lib import xla_bridge

from . import modes
from . import scales
from .defaults import defaults
from .object_transform import naming

__all__ = [
    # context manage for environment setting
    'environment',
    'batching_environment',
    'training_environment',
    'set_environment',
    'set',

    # default data types
    'set_float', 'get_float',
    'set_int', 'get_int',
    'set_bool', 'get_bool',
    'set_complex', 'get_complex',

    # default numerical integration step
    'set_dt', 'get_dt',

    # default computation modes
    'set_mode', 'get_mode',

    # default membrane_scaling
    'set_membrane_scaling', 'get_membrane_scaling',

    # set jax environments
    'enable_x64', 'disable_x64',
    'set_platform', 'get_platform',
    'set_host_device_count',

    # device memory
    'clear_buffer_memory',
    'enable_gpu_memory_preallocation',
    'disable_gpu_memory_preallocation',

    # deprecated
    'ditype',
    'dftype',

]

# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, func: F) -> F:
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.clone():
                return func(*args, **kwargs)

        return cast(F, decorate_context)

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""

        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # Generators are suspended and unsuspended at `yield`, hence we
            # make sure the grad modes is properly set every time the execution
            # flow returns into the wrapped generator and restored when it
            # returns through our `yield` to our caller (see PR #49017).
            try:
                # Issuing `None` to a generator fires it up
                with self.clone():
                    response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        request = yield response
                    except GeneratorExit:
                        # Inform the still active generator about its imminent closure
                        with self.clone():
                            gen.close()
                        raise
                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        with self.clone():
                            response = gen.throw(*sys.exc_info())
                    else:
                        # Pass the last request to the generator and get its response
                        with self.clone():
                            response = gen.send(request)

            # We let the exceptions raised above by the generator's `.throw` or
            # `.send` methods bubble up to our caller, except for StopIteration
            except StopIteration as e:
                # The generator informed us that it is done: take whatever its
                # returned value (if any) was and indicate that we're done too
                # by returning it (see docs for python's return-statement).
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class environment(_DecoratorContextManager):
    r"""Context-manager that sets a computing environment for brain dynamics computation.

    In BrainPy, there are several basic computation settings when constructing models,
    including ``mode`` for controlling model computing behavior, ``dt`` for numerical
    integration, ``int_`` for integer precision, and ``float_`` for floating precision.
    :py:class:`~.environment`` provides a context for model construction and
    computation. In this temporal environment, models are constructed with the given
    ``mode``, ``dt``, ``int_``, etc., environment settings.

    For instance::

      >>> import brainpy as bp
      >>> import brainpy.math as bm
      >>>
      >>> with bm.environment(mode=bm.training_mode, dt=0.1):
      >>>   lif1 = bp.neurons.LIF(1)
      >>>
      >>> with bm.environment(mode=bm.nonbatching_mode, dt=0.05, float_=bm.float64):
      >>>   lif2 = bp.neurons.LIF(1)

    """

    def __init__(
        self,
        mode: modes.Mode = None,
        membrane_scaling: scales.Scaling = None,
        dt: float = None,
        x64: bool = None,
        complex_: type = None,
        float_: type = None,
        int_: type = None,
        bool_: type = None,
        bp_object_as_pytree: bool = None,
        numpy_func_return: str = None,
    ) -> None:
        super().__init__()

        if dt is not None:
            assert isinstance(dt, float), '"dt" must a float.'
            self.old_dt = get_dt()

        if mode is not None:
            assert isinstance(mode, modes.Mode), f'"mode" must a {modes.Mode}.'
            self.old_mode = get_mode()

        if membrane_scaling is not None:
            assert isinstance(membrane_scaling, scales.Scaling), f'"membrane_scaling" must a {scales.Scaling}.'
            self.old_membrane_scaling = get_membrane_scaling()

        if x64 is not None:
            assert isinstance(x64, bool), f'"x64" must be a bool.'
            self.old_x64 = config.read("jax_enable_x64")

        if float_ is not None:
            assert isinstance(float_, type), '"float_" must a float.'
            self.old_float = get_float()

        if int_ is not None:
            assert isinstance(int_, type), '"int_" must a type.'
            self.old_int = get_int()

        if bool_ is not None:
            assert isinstance(bool_, type), '"bool_" must a type.'
            self.old_bool = get_bool()

        if complex_ is not None:
            assert isinstance(complex_, type), '"complex_" must a type.'
            self.old_complex = get_complex()

        if bp_object_as_pytree is not None:
            assert isinstance(bp_object_as_pytree, bool), '"bp_object_as_pytree" must be a bool.'
            self.old_bp_object_as_pytree = defaults.bp_object_as_pytree

        if numpy_func_return is not None:
            assert isinstance(numpy_func_return, str), '"numpy_func_return" must be a string.'
            assert numpy_func_return in ['bp_array', 'jax_array'], \
                f'"numpy_func_return" must be "bp_array" or "jax_array". Got {numpy_func_return}.'
            self.old_numpy_func_return = defaults.numpy_func_return

        self.dt = dt
        self.mode = mode
        self.membrane_scaling = membrane_scaling
        self.x64 = x64
        self.complex_ = complex_
        self.float_ = float_
        self.int_ = int_
        self.bool_ = bool_
        self.bp_object_as_pytree = bp_object_as_pytree
        self.numpy_func_return = numpy_func_return

    def __enter__(self) -> 'environment':
        if self.dt is not None: set_dt(self.dt)
        if self.mode is not None: set_mode(self.mode)
        if self.membrane_scaling is not None: set_membrane_scaling(self.membrane_scaling)
        if self.x64 is not None: set_x64(self.x64)
        if self.float_ is not None: set_float(self.float_)
        if self.int_ is not None: set_int(self.int_)
        if self.complex_ is not None: set_complex(self.complex_)
        if self.bool_ is not None: set_bool(self.bool_)
        if self.bp_object_as_pytree is not None: defaults.bp_object_as_pytree = self.bp_object_as_pytree
        if self.numpy_func_return is not None: defaults.numpy_func_return = self.numpy_func_return
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.dt is not None: set_dt(self.old_dt)
        if self.mode is not None: set_mode(self.old_mode)
        if self.membrane_scaling is not None: set_membrane_scaling(self.old_membrane_scaling)
        if self.x64 is not None: set_x64(self.old_x64)
        if self.int_ is not None: set_int(self.old_int)
        if self.float_ is not None:  set_float(self.old_float)
        if self.complex_ is not None:  set_complex(self.old_complex)
        if self.bool_ is not None:  set_bool(self.old_bool)
        if self.bp_object_as_pytree is not None: defaults.bp_object_as_pytree = self.old_bp_object_as_pytree
        if self.numpy_func_return is not None: defaults.numpy_func_return = self.old_numpy_func_return

    def clone(self):
        return self.__class__(dt=self.dt,
                              mode=self.mode,
                              membrane_scaling=self.membrane_scaling,
                              x64=self.x64,
                              bool_=self.bool_,
                              complex_=self.complex_,
                              float_=self.float_,
                              int_=self.int_,
                              bp_object_as_pytree=self.bp_object_as_pytree,
                              numpy_func_return=self.numpy_func_return)

    def __eq__(self, other):
        return id(self) == id(other)


class training_environment(environment):
    """Environment with the training mode.

    This is a short-cut context setting for an environment with the training mode.
    It is equivalent to::

      >>> import brainpy.math as bm
      >>> with bm.environment(mode=bm.training_mode):
      >>>   pass

    """

    def __init__(
        self,
        dt: float = None,
        x64: bool = None,
        complex_: type = None,
        float_: type = None,
        int_: type = None,
        bool_: type = None,
        batch_size: int = 1,
        membrane_scaling: scales.Scaling = None,
        bp_object_as_pytree: bool = None,
        numpy_func_return: str = None,
    ):
        super().__init__(dt=dt,
                         x64=x64,
                         complex_=complex_,
                         float_=float_,
                         int_=int_,
                         bool_=bool_,
                         membrane_scaling=membrane_scaling,
                         mode=modes.TrainingMode(batch_size),
                         bp_object_as_pytree=bp_object_as_pytree,
                         numpy_func_return=numpy_func_return)


class batching_environment(environment):
    """Environment with the batching mode.

    This is a short-cut context setting for an environment with the batching mode.
    It is equivalent to::

      >>> import brainpy.math as bm
      >>> with bm.environment(mode=bm.batching_mode):
      >>>   pass


    """

    def __init__(
        self,
        dt: float = None,
        x64: bool = None,
        complex_: type = None,
        float_: type = None,
        int_: type = None,
        bool_: type = None,
        batch_size: int = 1,
        membrane_scaling: scales.Scaling = None,
        bp_object_as_pytree: bool = None,
        numpy_func_return: str = None,
    ):
        super().__init__(dt=dt,
                         x64=x64,
                         complex_=complex_,
                         float_=float_,
                         int_=int_,
                         bool_=bool_,
                         mode=modes.BatchingMode(batch_size),
                         membrane_scaling=membrane_scaling,
                         bp_object_as_pytree=bp_object_as_pytree,
                         numpy_func_return=numpy_func_return)


def set(
    mode: modes.Mode = None,
    membrane_scaling: scales.Scaling = None,
    dt: float = None,
    x64: bool = None,
    complex_: type = None,
    float_: type = None,
    int_: type = None,
    bool_: type = None,
    bp_object_as_pytree: bool = None,
    numpy_func_return: str = None,
):
    """Set the default computation environment.

    Parameters::

    mode: Mode
      The computing mode.
    membrane_scaling: Scaling
      The numerical membrane_scaling.
    dt: float
      The numerical integration precision.
    x64: bool
      Enable x64 computation.
    complex_: type
      The complex data type.
    float_
      The floating data type.
    int_
      The integer data type.
    bool_
      The bool data type.
    bp_object_as_pytree: bool
      Whether to register brainpy object as pytree.
    numpy_func_return: str
      The array to return in all numpy functions. Support 'bp_array' and 'jax_array'.
    """
    if dt is not None:
        assert isinstance(dt, float), '"dt" must a float.'
        set_dt(dt)

    if mode is not None:
        assert isinstance(mode, modes.Mode), f'"mode" must a {modes.Mode}.'
        set_mode(mode)

    if membrane_scaling is not None:
        assert isinstance(membrane_scaling, scales.Scaling), f'"membrane_scaling" must a {scales.Scaling}.'
        set_membrane_scaling(membrane_scaling)

    if x64 is not None:
        assert isinstance(x64, bool), f'"x64" must be a bool.'
        set_x64(x64)

    if float_ is not None:
        assert isinstance(float_, type), '"float_" must a float.'
        set_float(float_)

    if int_ is not None:
        assert isinstance(int_, type), '"int_" must a type.'
        set_int(int_)

    if bool_ is not None:
        assert isinstance(bool_, type), '"bool_" must a type.'
        set_bool(bool_)

    if complex_ is not None:
        assert isinstance(complex_, type), '"complex_" must a type.'
        set_complex(complex_)

    if bp_object_as_pytree is not None:
        defaults.bp_object_as_pytree = bp_object_as_pytree

    if numpy_func_return is not None:
        assert numpy_func_return in ['bp_array', 'jax_array'], f'"numpy_func_return" must be "bp_array" or "jax_array".'
        defaults.numpy_func_return = numpy_func_return


set_environment = set


# default dtype
# --------------------------


def ditype():
    """Default int type.

    .. deprecated:: 2.3.1
       Use `brainpy.math.int_` instead.
    """
    # raise errors.NoLongerSupportError('\nGet default integer data type through `ditype()` has been deprecated. \n'
    #                                   'Use `brainpy.math.int_` instead.')
    return defaults.int_


def dftype():
    """Default float type.

    .. deprecated:: 2.3.1
       Use `brainpy.math.float_` instead.
    """

    # raise errors.NoLongerSupportError('\nGet default floating data type through `dftype()` has been deprecated. \n'
    #                                   'Use `brainpy.math.float_` instead.')
    return defaults.float_


def set_float(dtype: type):
    """Set global default float type.

    Parameters::

    dtype: type
      The float type.
    """
    defaults.float_ = dtype


def get_float():
    """Get the default float data type.

    Returns::

    dftype: type
      The default float data type.
    """
    return defaults.float_


def set_int(dtype: type):
    """Set global default integer type.

    Parameters::

    dtype: type
      The integer type.
    """
    defaults.int_ = dtype


def get_int():
    """Get the default int data type.

    Returns::

    dftype: type
      The default int data type.
    """
    return defaults.int_


def set_bool(dtype: type):
    """Set global default boolean type.

    Parameters::

    dtype: type
      The bool type.
    """
    defaults.bool_ = dtype


def get_bool():
    """Get the default boolean data type.

    Returns::

    dftype: type
      The default bool data type.
    """
    return defaults.bool_


def set_complex(dtype: type):
    """Set global default complex type.

    Parameters::

    dtype: type
      The complex type.
    """
    defaults.complex_ = dtype


def get_complex():
    """Get the default complex data type.

    Returns::

    dftype: type
      The default complex data type.
    """
    return defaults.complex_


# numerical precision
# --------------------------

def set_dt(dt):
    """Set the default numerical integrator precision.

    Parameters::

    dt : float
        Numerical integration precision.
    """
    assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
    defaults.dt = dt


def get_dt():
    """Get the numerical integrator precision.

    Returns::

    dt : float
        Numerical integration precision.
    """
    return defaults.dt


def set_mode(mode: modes.Mode):
    """Set the default computing mode.

    Parameters::

    mode: Mode
      The instance of :py:class:`~.Mode`.
    """
    if not isinstance(mode, modes.Mode):
        raise TypeError(f'Must be instance of brainpy.math.Mode. '
                        f'But we got {type(mode)}: {mode}')
    defaults.mode = mode


def get_mode() -> modes.Mode:
    """Get the default computing mode.

    References::

    mode: Mode
      The default computing mode.
    """
    return defaults.mode


def set_membrane_scaling(membrane_scaling: scales.Scaling):
    """Set the default computing membrane_scaling.

    Parameters::

    scaling: Scaling
      The instance of :py:class:`~.Scaling`.
    """
    if not isinstance(membrane_scaling, scales.Scaling):
        raise TypeError(f'Must be instance of brainpy.math.Scaling. '
                        f'But we got {type(membrane_scaling)}: {membrane_scaling}')
    defaults.membrane_scaling = membrane_scaling


def get_membrane_scaling() -> scales.Scaling:
    """Get the default computing membrane_scaling.

    Returns::

    membrane_scaling: Scaling
      The default computing membrane_scaling.
    """
    return defaults.membrane_scaling


def enable_x64(x64=None):
    if x64 is None:
        x64 = True
    else:
        warnings.warn(
            '\n'
            'Instead of "brainpy.math.enable_x64(True)", use "brainpy.math.enable_x64()". \n'
            'Instead of "brainpy.math.enable_x64(False)", use "brainpy.math.disable_x64()". \n',
            DeprecationWarning
        )
    if x64:
        brainstate.environ.set(precision=64)
        set_int(jnp.int64)
        set_float(jnp.float64)
        set_complex(jnp.complex128)
    else:
        brainstate.environ.set(precision=32)
        disable_x64()


def disable_x64():
    config.update("jax_enable_x64", False)
    set_int(jnp.int32)
    set_float(jnp.float32)
    set_complex(jnp.complex64)


def set_x64(enable: bool):
    assert isinstance(enable, bool)
    if enable:
        enable_x64()
    else:
        disable_x64()


def set_platform(platform: str):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.
    """
    assert platform in ['cpu', 'gpu', 'tpu']
    config.update("jax_platform_name", platform)


def get_platform() -> str:
    """Get the computing platform.

    Returns::

    platform: str
      Either 'cpu', 'gpu' or 'tpu'.
    """
    return devices()[0].platform


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.

    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.

    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.

    :param int n: number of devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join(["--xla_force_host_platform_device_count={}".format(n)] + xla_flags)


def clear_buffer_memory(
    platform: str = None,
    array: bool = True,
    transform: bool = True,
    compilation: bool = False,
    object_name: bool = False,
):
    """Clear all on-device buffers.

    This function will be very useful when you call models in a Python loop,
    because it can clear all cached arrays, and clear device memory.

    .. warning::

       This operation may cause errors when you use a deleted buffer.
       Therefore, regenerate data always.

    Parameters::

    platform: str
      The device to clear its memory.
    array: bool
      Clear all buffer array. Default is True.
    compilation: bool
      Clear compilation cache. Default is False.
    transform: bool
      Clear transform cache. Default is True.
    object_name: bool
      Clear name cache. Default is True.

    """
    if array:
        for buf in xla_bridge.get_backend(platform).live_buffers():
            buf.delete()
    if compilation:
        jax.clear_caches()
    if transform:
        naming.clear_stack_cache()
    if object_name:
        naming.clear_name_cache()
    gc.collect()


def disable_gpu_memory_preallocation(release_memory: bool = True):
    """Disable pre-allocating the GPU memory.

    This disables the preallocation behavior. JAX will instead allocate GPU memory as needed,
    potentially decreasing the overall memory usage. However, this behavior is more prone to
    GPU memory fragmentation, meaning a JAX program that uses most of the available GPU memory
    may OOM with preallocation disabled.

    Args:
      release_memory: bool. Whether we release memory during the computation.
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if release_memory:
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


def enable_gpu_memory_preallocation():
    """Disable pre-allocating the GPU memory."""
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    os.environ.pop('XLA_PYTHON_CLIENT_ALLOCATOR', None)


def gpu_memory_preallocation(percent: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert 0. <= percent < 1., f'GPU memory preallocation must be in [0., 1.]. But we got {percent}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(percent)
