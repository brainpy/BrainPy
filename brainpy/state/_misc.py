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


from typing import TypeVar
from collections.abc import Callable

T = TypeVar('T', bound=Callable | type)


def set_module_as(module: str) -> Callable[[T], T]:
    """Decorator overriding the ``__module__`` of a function or class.

    Reassigning ``__module__`` makes an object appear to belong to its public
    module (e.g. ``brainpy.state``) in documentation, ``repr`` and ``help``
    output, even though it is implemented in a private submodule. Both
    functions and classes are supported; the decorated object is returned
    unchanged apart from its ``__module__`` attribute.

    Parameters
    ----------
    module : str
        Public module path to assign, e.g. ``'brainpy.state'``.

    Returns
    -------
    callable
        A decorator that sets ``__module__`` on the function or class it wraps
        and returns it unchanged.
    """

    def wrapper(obj: T) -> T:
        obj.__module__ = module
        return obj

    return wrapper
