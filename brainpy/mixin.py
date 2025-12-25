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
import warnings
from dataclasses import dataclass
from typing import Union, Dict, Callable, Sequence, Optional, Any

import brainstate
import jax

bm, delay_identifier, init_delay_by_return, DynamicalSystem = None, None, None, None

__all__ = [
    'MixIn',
    'ParamDesc',
    'ParamDescriber',
    'AlignPost',
    'Container',
    'TreeNode',
    'BindCondData',
    'JointType',
    'SupportSTDP',
    'SupportAutoDelay',
    'SupportInputProj',
    'SupportOnline',
    'SupportOffline',
]

MixIn = brainstate.mixin.Mixin
ParamDesc = brainstate.mixin.ParamDesc
ParamDescriber = brainstate.mixin.ParamDescriber
JointType = brainstate.mixin.JointTypes


def _get_bm():
    global bm
    if bm is None:
        from brainpy import math
        bm = math
    return bm


class AlignPost(brainstate.mixin.Mixin):
    """
    Mixin for aligning post-synaptic inputs.

    This mixin provides an interface for components that need to receive and
    process post-synaptic inputs, such as synaptic connections or neural
    populations. The ``align_post_input_add`` method should be implemented
    to handle the accumulation of external currents or inputs.

    Notes
    -----
    Classes that inherit from this mixin must implement the
    ``align_post_input_add`` method.

    Examples
    --------
    Implementing a synapse with post-synaptic alignment:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class Synapse(brainstate.mixin.AlignPost):
        ...     def __init__(self, weight):
        ...         self.weight = weight
        ...         self.post_current = brainstate.State(0.0)
        ...
        ...     def align_post_input_add(self, current):
        ...         # Accumulate the weighted current into post-synaptic target
        ...         self.post_current.value += current * self.weight
        >>>
        >>> # Usage
        >>> synapse = Synapse(weight=0.5)
        >>> synapse.align_post_input_add(10.0)
        >>> print(synapse.post_current.value)  # Output: 5.0

    Using with neural populations:

    .. code-block:: python

        >>> class NeuronGroup(brainstate.mixin.AlignPost):
        ...     def __init__(self, size):
        ...         self.size = size
        ...         self.input_current = brainstate.State(jnp.zeros(size))
        ...
        ...     def align_post_input_add(self, current):
        ...         # Add external current to neurons
        ...         self.input_current.value = self.input_current.value + current
        >>>
        >>> neurons = NeuronGroup(100)
        >>> external_input = jnp.ones(100) * 0.5
        >>> neurons.align_post_input_add(external_input)
    """

    def align_post_input_add(self, *args, **kwargs):
        """
        Add external inputs to the post-synaptic component.

        Parameters
        ----------
        *args
            Positional arguments for the input.
        **kwargs
            Keyword arguments for the input.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class BindCondData(brainstate.mixin.Mixin):
    """
    Mixin for binding temporary conductance data.

    This mixin provides an interface for temporarily storing conductance data,
    which is useful in synaptic models where conductance values need to be
    passed between computation steps without being part of the permanent state.

    Attributes
    ----------
    _conductance : Any, optional
        Temporarily bound conductance data.

    Examples
    --------
    Using conductance binding in a synapse:

    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> class ConductanceBasedSynapse(brainstate.mixin.BindCondData):
        ...     def __init__(self):
        ...         self._conductance = None
        ...
        ...     def compute(self, pre_spike):
        ...         if pre_spike:
        ...             # Bind conductance data temporarily
        ...             self.bind_cond(0.5)
        ...
        ...         # Use conductance if available
        ...         if self._conductance is not None:
        ...             current = self._conductance * (0.0 - (-70.0))
        ...             # Clear after use
        ...             self.unbind_cond()
        ...             return current
        ...         return 0.0
        >>>
        >>> synapse = ConductanceBasedSynapse()
        >>> current = synapse.compute(pre_spike=True)

    Managing conductance in a network:

    .. code-block:: python

        >>> class SynapticConnection(brainstate.mixin.BindCondData):
        ...     def __init__(self, g_max):
        ...         self.g_max = g_max
        ...         self._conductance = None
        ...
        ...     def prepare_conductance(self, activation):
        ...         # Bind conductance based on activation
        ...         g = self.g_max * activation
        ...         self.bind_cond(g)
        ...
        ...     def apply_conductance(self, voltage):
        ...         if self._conductance is not None:
        ...             current = self._conductance * voltage
        ...             self.unbind_cond()
        ...             return current
        ...         return 0.0
    """
    # Attribute to store temporary conductance data
    _conductance: Optional

    def bind_cond(self, conductance):
        """
        Bind conductance data temporarily.

        Parameters
        ----------
        conductance : Any
            The conductance data to bind.
        """
        self._conductance = conductance

    def unbind_cond(self):
        """
        Unbind (clear) the conductance data.
        """
        self._conductance = None


def _get_delay_tool():
    global delay_identifier, init_delay_by_return
    if init_delay_by_return is None: from brainpy.delay import init_delay_by_return
    if delay_identifier is None: from brainpy.delay import delay_identifier
    return delay_identifier, init_delay_by_return


@dataclass
class ReturnInfo:
    size: Sequence[int]
    axis_names: Optional[Sequence[str]] = None
    batch_or_mode: Optional[Union[int, brainstate.mixin.Mode]] = None
    data: Union[Callable, jax.Array] = jax.numpy.zeros

    def get_data(self):
        bm = _get_bm()
        if isinstance(self.data, Callable):
            if isinstance(self.batch_or_mode, int):
                size = (self.batch_or_mode,) + tuple(self.size)
            elif isinstance(self.batch_or_mode, bm.NonBatchingMode):
                size = tuple(self.size)
            elif isinstance(self.batch_or_mode, bm.BatchingMode):
                size = (self.batch_or_mode.batch_size,) + tuple(self.size)
            else:
                size = tuple(self.size)
            init = self.data(size)
        elif isinstance(self.data, (bm.Array, jax.Array)):
            init = self.data
        else:
            raise ValueError
        return init


class Container(MixIn):
    """Container :py:class:`~.MixIn` which wrap a group of objects.
    """
    children: dict()

    def __getitem__(self, item):
        """Overwrite the slice access (`self['']`). """
        if item in self.children:
            return self.children[item]
        else:
            raise ValueError(f'Unknown item {item}, we only found {list(self.children.keys())}')

    def __getattr__(self, item):
        """Overwrite the dot access (`self.`). """
        if item == 'children':
            return super().__getattribute__('children')
        else:
            children = super().__getattribute__('children')
            if item in children:
                return children[item]
            else:
                return super().__getattribute__(item)

    def __repr__(self):
        from brainpy import tools
        cls_name = self.__class__.__name__
        indent = ' ' * len(cls_name)
        child_str = [tools.repr_context(repr(val), indent) for val in self.children.values()]
        string = ", \n".join(child_str)
        return f'{cls_name}({string})'

    def __get_elem_name(self, elem):
        bm = _get_bm()
        if isinstance(elem, bm.BrainPyObject):
            return elem.name
        else:
            from brainpy.math.object_transform.base import get_unique_name
            return get_unique_name('ContainerElem')

    def format_elements(self, child_type: type, *children_as_tuple, **children_as_dict):
        res = dict()

        # add tuple-typed components
        for module in children_as_tuple:
            if isinstance(module, child_type):
                res[self.__get_elem_name(module)] = module
            elif isinstance(module, (list, tuple)):
                for m in module:
                    if not isinstance(m, child_type):
                        raise ValueError(f'Should be instance of {child_type.__name__}. '
                                         f'But we got {type(m)}')
                    res[self.__get_elem_name(m)] = m
            elif isinstance(module, dict):
                for k, v in module.items():
                    if not isinstance(v, child_type):
                        raise ValueError(f'Should be instance of {child_type.__name__}. '
                                         f'But we got {type(v)}')
                    res[k] = v
            else:
                raise ValueError(f'Cannot parse sub-systems. They should be {child_type.__name__} '
                                 f'or a list/tuple/dict of {child_type.__name__}.')
        # add dict-typed components
        for k, v in children_as_dict.items():
            if not isinstance(v, child_type):
                raise ValueError(f'Should be instance of {child_type.__name__}. '
                                 f'But we got {type(v)}')
            res[k] = v
        return res

    def add_elem(self, *elems, **elements):
        """Add new elements.

        >>> obj = Container()
        >>> obj.add_elem(a=1.)

        Args:
          elements: children objects.
        """
        self.children.update(self.format_elements(object, *elems, **elements))


class TreeNode(MixIn):
    """Tree node. """

    master_type: type

    def check_hierarchies(self, root, *leaves, **named_leaves):
        global DynamicalSystem
        if DynamicalSystem is None:
            from brainpy.dynsys import DynamicalSystem

        for leaf in leaves:
            if isinstance(leaf, DynamicalSystem):
                self.check_hierarchy(root, leaf)
            elif isinstance(leaf, (list, tuple)):
                self.check_hierarchies(root, *leaf)
            elif isinstance(leaf, dict):
                self.check_hierarchies(root, **leaf)
            else:
                raise ValueError(f'Do not support {type(leaf)}.')
        for leaf in named_leaves.values():
            if not isinstance(leaf, DynamicalSystem):
                raise ValueError(f'Do not support {type(leaf)}. Must be instance of {DynamicalSystem.__name__}')
            self.check_hierarchy(root, leaf)

    def check_hierarchy(self, root, leaf):
        if hasattr(leaf, 'master_type'):
            master_type = leaf.master_type
        else:
            raise ValueError('Child class should define "master_type" to '
                             'specify the type of the root node. '
                             f'But we did not found it in {leaf}')
        if not issubclass(root, master_type):
            raise TypeError(f'Type does not match. {leaf} requires a master with type '
                            f'of {leaf.master_type}, but the master now is {root}.')


class SupportInputProj(MixIn):
    """The :py:class:`~.MixIn` that receives the input projections.

    Note that the subclass should define a ``cur_inputs`` attribute. Otherwise,
    the input function utilities cannot be used.

    """
    current_inputs: dict
    delta_inputs: dict

    def add_inp_fun(self, key: str, fun: Callable, label: Optional[str] = None, category: str = 'current'):
        """Add an input function.

        Args:
          key: str. The dict key.
          fun: Callable. The function to generate inputs.
          label: str. The input label.
          category: str. The input category, should be ``current`` (the current) or
             ``delta`` (the delta synapse, indicating the delta function).
        """
        if not callable(fun):
            raise TypeError('Must be a function.')

        key = self._input_label_repr(key, label)
        if category == 'current':
            if key in self.current_inputs:
                raise ValueError(f'Key "{key}" has been defined and used.')
            self.current_inputs[key] = fun
        elif category == 'delta':
            if key in self.delta_inputs:
                raise ValueError(f'Key "{key}" has been defined and used.')
            self.delta_inputs[key] = fun
        else:
            raise NotImplementedError(f'Unknown category: {category}. Only support "current" and "delta".')

    def get_inp_fun(self, key: str):
        """Get the input function.

        Args:
          key: str. The key.

        Returns:
          The input function which generates currents.
        """
        if key in self.current_inputs:
            return self.current_inputs[key]
        elif key in self.delta_inputs:
            return self.delta_inputs[key]
        else:
            raise ValueError(f'Unknown key: {key}')

    def sum_current_inputs(self, *args, init: Any = 0., label: Optional[str] = None, **kwargs):
        """Summarize all current inputs by the defined input functions ``.current_inputs``.

        Args:
          *args: The arguments for input functions.
          init: The initial input data.
          label: str. The input label.
          **kwargs: The arguments for input functions.

        Returns:
          The total currents.
        """
        if label is None:
            for key, out in self.current_inputs.items():
                init = init + out(*args, **kwargs)
        else:
            label_repr = self._input_label_start(label)
            for key, out in self.current_inputs.items():
                if key.startswith(label_repr):
                    init = init + out(*args, **kwargs)
        return init

    def sum_delta_inputs(self, *args, init: Any = 0., label: Optional[str] = None, **kwargs):
        """Summarize all delta inputs by the defined input functions ``.delta_inputs``.

        Args:
          *args: The arguments for input functions.
          init: The initial input data.
          label: str. The input label.
          **kwargs: The arguments for input functions.

        Returns:
          The total currents.
        """
        if label is None:
            for key, out in self.delta_inputs.items():
                init = init + out(*args, **kwargs)
        else:
            label_repr = self._input_label_start(label)
            for key, out in self.delta_inputs.items():
                if key.startswith(label_repr):
                    init = init + out(*args, **kwargs)
        return init

    @classmethod
    def _input_label_start(cls, label: str):
        # unify the input label repr.
        return f'{label} // '

    @classmethod
    def _input_label_repr(cls, name: str, label: Optional[str] = None):
        # unify the input label repr.
        return name if label is None else (cls._input_label_start(label) + str(name))

    # deprecated #
    # ---------- #

    @property
    def cur_inputs(self):
        return self.current_inputs

    def sum_inputs(self, *args, **kwargs):
        warnings.warn('Please use ".sum_current_inputs()" instead. ".sum_inputs()" will be removed.', UserWarning)
        return self.sum_current_inputs(*args, **kwargs)


class SupportReturnInfo(MixIn):
    """``MixIn`` to support the automatic delay in synaptic projection :py:class:`~.SynProj`."""

    def return_info(self):
        raise NotImplementedError('Must implement the "return_info()" function.')


class SupportAutoDelay(SupportReturnInfo):
    pass


class SupportOnline(MixIn):
    """:py:class:`~.MixIn` to support the online training methods.

    .. versionadded:: 2.4.5
    """

    online_fit_by: Optional  # methods for online fitting

    def online_init(self, *args, **kwargs):
        raise NotImplementedError

    def online_fit(self, target, fit_record: Dict):
        raise NotImplementedError


class SupportOffline(MixIn):
    """:py:class:`~.MixIn` to support the offline training methods.

    .. versionadded:: 2.4.5
    """

    offline_fit_by: Optional  # methods for offline fitting

    def offline_init(self, *args, **kwargs):
        pass

    def offline_fit(self, target, fit_record: Dict):
        raise NotImplementedError


class SupportSTDP(MixIn):
    """Support synaptic plasticity by modifying the weights.
    """

    def stdp_update(self, *args, on_pre=None, onn_post=None, **kwargs):
        raise NotImplementedError
