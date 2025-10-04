# -*- coding: utf-8 -*-

import numbers
import sys
import warnings
from dataclasses import dataclass
from typing import Union, Dict, Callable, Sequence, Optional, Any
from typing import (_type_check, _remove_dups_flatten)

import brainstate.mixin
import jax

from brainpy import math as bm, tools
from brainpy._src.math.object_transform.naming import get_unique_name
from brainpy.types import ArrayType

if sys.version_info.minor > 8:
    from typing import (_UnionGenericAlias)
else:
    from typing import (_GenericAlias, _tp_cache)

DynamicalSystem = None
delay_identifier, init_delay_by_return = None, None

__all__ = [
    'MixIn',
    'ParamDesc',
    'ParamDescriber',
    'DelayRegister',
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


def _get_delay_tool():
    global delay_identifier, init_delay_by_return
    if init_delay_by_return is None: from brainpy._src.delay import init_delay_by_return
    if delay_identifier is None: from brainpy._src.delay import delay_identifier
    return delay_identifier, init_delay_by_return


def _get_dynsys():
    global DynamicalSystem
    if DynamicalSystem is None: from brainpy._src.dynsys import DynamicalSystem
    return DynamicalSystem


MixIn = brainstate.mixin.Mixin
ParamDesc = brainstate.mixin.ParamDesc
ParamDescriber = brainstate.mixin.ParamDescriber
AlignPost = brainstate.mixin.AlignPost
BindCondData = brainstate.mixin.BindCondData
JointType = brainstate.mixin.JointTypes


@dataclass
class ReturnInfo:
    size: Sequence[int]
    axis_names: Optional[Sequence[str]] = None
    batch_or_mode: Optional[Union[int, bm.Mode]] = None
    data: Union[Callable, bm.Array, jax.Array] = bm.zeros

    def get_data(self):
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
        elif isinstance(self.data, (bm.BaseArray, jax.Array)):
            init = self.data
        else:
            raise ValueError
        return init


class Container(MixIn):
    """Container :py:class:`~.MixIn` which wrap a group of objects.
    """
    children: bm.node_dict

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
        cls_name = self.__class__.__name__
        indent = ' ' * len(cls_name)
        child_str = [tools.repr_context(repr(val), indent) for val in self.children.values()]
        string = ", \n".join(child_str)
        return f'{cls_name}({string})'

    def __get_elem_name(self, elem):
        if isinstance(elem, bm.BrainPyObject):
            return elem.name
        else:
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
            from brainpy._src.dynsys import DynamicalSystem

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


class DelayRegister(MixIn):

    def register_delay(
        self,
        identifier: str,
        delay_step: Optional[Union[int, ArrayType, Callable]],
        delay_target: bm.Variable,
        initial_delay_data: Union[Callable, ArrayType, numbers.Number] = None,
    ):
        """Register delay variable.

        Args:
          identifier: str. The delay access name.
          delay_target: The target variable for delay.
          delay_step: The delay time step.
          initial_delay_data: The initializer for the delay data.

        Returns:
          delay_pos: The position of the delay.
        """
        _delay_identifier, _init_delay_by_return = _get_delay_tool()
        DynamicalSystem = _get_dynsys()
        assert isinstance(self, DynamicalSystem), f'self must be an instance of {DynamicalSystem.__name__}'
        _delay_identifier = _delay_identifier + identifier
        if not self.has_aft_update(_delay_identifier):
            self.add_aft_update(_delay_identifier, _init_delay_by_return(delay_target, initial_delay_data))
        delay_cls = self.get_aft_update(_delay_identifier)
        name = get_unique_name('delay')
        delay_cls.register_entry(name, delay_step)
        return name

    def get_delay_data(
        self,
        identifier: str,
        delay_pos: str,
        *indices: Union[int, slice, bm.Array, jax.Array],
    ):
        """Get delay data according to the provided delay steps.

        Parameters::

        identifier: str
          The delay variable name.
        delay_pos: str
          The delay length.
        indices: optional, int, slice, ArrayType
          The indices of the delay.

        Returns::

        delay_data: ArrayType
          The delay data at the given time.
        """
        _delay_identifier, _init_delay_by_return = _get_delay_tool()
        _delay_identifier = _delay_identifier + identifier
        delay_cls = self.get_aft_update(_delay_identifier)
        return delay_cls.at(delay_pos, *indices)

    def update_local_delays(self, nodes: Union[Sequence, Dict] = None):
        """Update local delay variables.

        This function should be called after updating neuron groups or delay sources.
        For example, in a network model,


        Parameters::

        nodes: sequence, dict
          The nodes to update their delay variables.
        """
        warnings.warn('.update_local_delays() has been removed since brainpy>=2.4.6',
                      DeprecationWarning)

    def reset_local_delays(self, nodes: Union[Sequence, Dict] = None):
        """Reset local delay variables.

        Parameters::

        nodes: sequence, dict
          The nodes to Reset their delay variables.
        """
        warnings.warn('.reset_local_delays() has been removed since brainpy>=2.4.6',
                      DeprecationWarning)

    def get_delay_var(self, name):
        _delay_identifier, _init_delay_by_return = _get_delay_tool()
        _delay_identifier = _delay_identifier + name
        delay_cls = self.get_aft_update(_delay_identifier)
        return delay_cls


class SupportInputProj(MixIn):
    """The :py:class:`~.MixIn` that receives the input projections.

    Note that the subclass should define a ``cur_inputs`` attribute. Otherwise,
    the input function utilities cannot be used.

    """
    current_inputs: bm.node_dict
    delta_inputs: bm.node_dict

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

    def return_info(self) -> Union[bm.Variable, ReturnInfo]:
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

    def online_fit(self, target: ArrayType, fit_record: Dict[str, ArrayType]):
        raise NotImplementedError


class SupportOffline(MixIn):
    """:py:class:`~.MixIn` to support the offline training methods.

    .. versionadded:: 2.4.5
    """

    offline_fit_by: Optional  # methods for offline fitting

    def offline_init(self, *args, **kwargs):
        pass

    def offline_fit(self, target: ArrayType, fit_record: Dict[str, ArrayType]):
        raise NotImplementedError


class SupportSTDP(MixIn):
    """Support synaptic plasticity by modifying the weights.
    """

    def stdp_update(self, *args, on_pre=None, onn_post=None, **kwargs):
        raise NotImplementedError
