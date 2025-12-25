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
import collections
import inspect
import numbers
import warnings
from typing import Union, Dict, Callable, Sequence, Optional, Any

import jax
import numpy as np

from brainpy import tools, math as bm
from brainpy._errors import NoImplementationError, UnsupportedError
from brainpy.context import share
from brainpy.deprecations import _update_deprecate_msg
from brainpy.initialize import parameter, variable_
from brainpy.math.object_transform.naming import get_unique_name
from brainpy.mixin import SupportAutoDelay, Container, SupportInputProj, _get_delay_tool, MixIn
from brainpy.types import ArrayType, Shape

__all__ = [
    # general
    'DynamicalSystem',

    # containers
    'DynSysGroup', 'Network', 'Sequential',

    # category
    'Dynamic', 'Projection',
]

IonChaDyn = None
SLICE_VARS = 'slice_vars'
the_top_layer_reset_state = True
clear_input = None
reset_state = None


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


def not_implemented(fun):
    def new_fun(*args, **kwargs):
        return fun(*args, **kwargs)

    new_fun._not_implemented = True
    return new_fun


class DynamicalSystem(bm.BrainPyObject, DelayRegister, SupportInputProj):
    """Base Dynamical System class.

    .. note::
       In general, every instance of :py:class:`~.DynamicalSystem` implemented in
       BrainPy only defines the evolving function at each time step :math:`t`.

       If users want to define the logic of running models across multiple steps,
       we recommend users to use :py:func:`~.for_loop`, :py:class:`~.LoopOverTime`,
       :py:class:`~.DSRunner`, or :py:class:`~.DSTrainer`.

       To be compatible with previous APIs, :py:class:`~.DynamicalSystem` inherits
       from the :py:class:`~.DelayRegister`. It's worthy to note that the methods of
       :py:class:`~.DelayRegister` will be removed in the future, including:

       - ``.register_delay()``
       - ``.get_delay_data()``
       - ``.update_local_delays()``
       - ``.reset_local_delays()``

    Parameters::

    name : optional, str
      The name of the dynamical system.
    mode: optional, Mode
      The model computation mode. It should be an instance of :py:class:`~.Mode`.
    """

    supported_modes: Optional[Sequence[bm.Mode]] = None
    '''Supported computing modes.'''

    def __init__(
        self,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        # mode setting
        mode = bm.get_mode() if mode is None else mode
        if not isinstance(mode, bm.Mode):
            raise ValueError(f'Should be instance of {bm.Mode.__name__}, '
                             f'but we got {type(mode)}: {mode}')
        self._mode = mode

        if self.supported_modes is not None:
            if not self.mode.is_parent_of(*self.supported_modes):
                raise UnsupportedError(f'The mode only supports computing modes '
                                       f'which are parents of {self.supported_modes}, '
                                       f'but we got {self.mode}.')

        # Attribute for "SupportInputProj"
        # each instance of "SupportInputProj" should have a "cur_inputs" attribute
        self._current_inputs: Optional[Dict[str, Callable]] = None
        self._delta_inputs: Optional[Dict[str, Callable]] = None

        # the before- / after-updates used for computing
        # added after the version of 2.4.3
        self._before_updates: Optional[Dict[str, Callable]] = None
        self._after_updates: Optional[Dict[str, Callable]] = None

        # super initialization
        super().__init__(name=name)

    @property
    def current_inputs(self):
        if self._current_inputs is None:
            self._current_inputs = bm.node_dict()
        return self._current_inputs

    @property
    def delta_inputs(self):
        if self._delta_inputs is None:
            self._delta_inputs = bm.node_dict()
        return self._delta_inputs

    @property
    def before_updates(self):
        if self._before_updates is None:
            self._before_updates = bm.node_dict()
        return self._before_updates

    @property
    def after_updates(self):
        if self._after_updates is None:
            self._after_updates = bm.node_dict()
        return self._after_updates

    def add_bef_update(self, key: Any, fun: Callable):
        """Add the before update into this node"""
        if key in self.before_updates:
            raise KeyError(f'{key} has been registered in before_updates of {self}')
        self.before_updates[key] = fun

    def add_aft_update(self, key: Any, fun: Callable):
        """Add the after update into this node"""
        if key in self.after_updates:
            raise KeyError(f'{key} has been registered in after_updates of {self}')
        self.after_updates[key] = fun

    def get_bef_update(self, key: Any):
        """Get the before update of this node by the given ``key``."""
        if key not in self.before_updates:
            raise KeyError(f'{key} is not registered in before_updates of {self}')
        return self.before_updates.get(key)

    def get_aft_update(self, key: Any):
        """Get the after update of this node by the given ``key``."""
        if key not in self.after_updates:
            raise KeyError(f'{key} is not registered in after_updates of {self}')
        return self.after_updates.get(key)

    def has_bef_update(self, key: Any):
        """Whether this node has the before update of the given ``key``."""
        return key in self.before_updates

    def has_aft_update(self, key: Any):
        """Whether this node has the after update of the given ``key``."""
        return key in self.after_updates

    def update(self, *args, **kwargs):
        """The function to specify the updating rule.
        """
        raise NotImplementedError('Must implement "update" function by subclass self.')

    def reset(self, *args, **kwargs):
        """Reset function which reset the whole variables in the model (including its children models).

        ``reset()`` function is a collective behavior which resets all states in this model.

        See https://brainpy.readthedocs.io/en/latest/tutorial_toolbox/state_resetting.html for details.
        """
        global reset_state
        if reset_state is None:
            from brainpy.helpers import reset_state
        reset_state(self, *args, **kwargs)

    @not_implemented
    def reset_state(self, *args, **kwargs):
        """Reset function which resets local states in this model.

        Simply speaking, this function should implement the logic of resetting of
        local variables in this node.

        See https://brainpy.readthedocs.io/en/latest/tutorial_toolbox/state_resetting.html for details.
        """
        pass

    def clear_input(self, *args, **kwargs):
        """Clear the input at the current time step."""
        pass

    def step_run(self, i, *args, **kwargs):
        """The step run function.

        This function can be directly applied to run the dynamical system.
        Particularly, ``i`` denotes the running index.

        Args:
          i: The current running index.
          *args: The arguments of ``update()`` function.
          **kwargs: The arguments of ``update()`` function.

        Returns:
          out: The update function returns.
        """
        global clear_input
        if clear_input is None:
            from brainpy.helpers import clear_input
        share.save(i=i, t=i * bm.dt)
        out = self.update(*args, **kwargs)
        clear_input(self)
        return out

    @bm.cls_jit(inline=True)
    def jit_step_run(self, i, *args, **kwargs):
        """The jitted step function for running.

        Args:
          i: The current running index.
          *args: The arguments of ``update()`` function.
          **kwargs: The arguments of ``update()`` function.

        Returns:
          out: The update function returns.
        """
        return self.step_run(i, *args, **kwargs)

    @property
    def mode(self) -> bm.Mode:
        """Mode of the model, which is useful to control the multiple behaviors of the model."""
        return self._mode

    @mode.setter
    def mode(self, value):
        if not isinstance(value, bm.Mode):
            raise ValueError(f'Must be instance of {bm.Mode.__name__}, '
                             f'but we got {type(value)}: {value}')
        self._mode = value

    def register_local_delay(
        self,
        var_name: str,
        delay_name: str,
        delay_time: Union[numbers.Number, ArrayType] = None,
        delay_step: Union[numbers.Number, ArrayType] = None,
    ):
        """Register local relay at the given delay time.

        Args:
          var_name: str. The name of the delay target variable.
          delay_name: str. The name of the current delay data.
          delay_time: The delay time. Float.
          delay_step: The delay step. Int. ``delay_step`` and ``delay_time`` are exclusive. ``delay_step = delay_time / dt``.
        """
        delay_identifier, init_delay_by_return = _get_delay_tool()
        delay_identifier = delay_identifier + var_name
        # check whether the "var_name" has been registered
        try:
            target = getattr(self, var_name)
        except AttributeError:
            raise AttributeError(f'This node {self} does not has attribute of "{var_name}".')
        if not self.has_aft_update(delay_identifier):
            # add a model to receive the return of the target model
            # moreover, the model should not receive the return of the update function
            model = not_receive_update_output(init_delay_by_return(target))
            # register the model
            self.add_aft_update(delay_identifier, model)
        delay_cls = self.get_aft_update(delay_identifier)
        delay_cls.register_entry(delay_name, delay_time=delay_time, delay_step=delay_step)

    def get_local_delay(self, var_name, delay_name):
        """Get the delay at the given identifier (`name`).

        Args:
          var_name: The name of the target delay variable.
          delay_name: The identifier of the delay.

        Returns:
          The delayed data at the given delay position.
        """
        delay_identifier, init_delay_by_return = _get_delay_tool()
        delay_identifier = delay_identifier + var_name
        return self.get_aft_update(delay_identifier).at(delay_name)

    def _compatible_update(self, *args, **kwargs):
        update_fun = super().__getattribute__('update')
        update_args = tuple(inspect.signature(update_fun).parameters.values())

        if len(update_args) and update_args[0].name in ['tdi', 'sh', 'sha']:
            # define the update function with:
            #     update(tdi, *args, **kwargs)
            #
            if len(args) > 0:
                if isinstance(args[0], dict) and all([bm.ndim(v) == 0 for v in args[0].values()]):
                    # define:
                    #    update(tdi, *args, **kwargs)
                    # call:
                    #    update(tdi, *args, **kwargs)
                    ret = update_fun(*args, **kwargs)
                    warnings.warn(_update_deprecate_msg, UserWarning)
                else:
                    # define:
                    #    update(tdi, *args, **kwargs)
                    # call:
                    #    update(*args, **kwargs)
                    ret = update_fun(share.get_shargs(), *args, **kwargs)
                    warnings.warn(_update_deprecate_msg, UserWarning)
            else:
                if update_args[0].name in kwargs:
                    # define:
                    #    update(tdi, *args, **kwargs)
                    # call:
                    #    update(tdi=??, **kwargs)
                    ret = update_fun(**kwargs)
                    warnings.warn(_update_deprecate_msg, UserWarning)
                else:
                    # define:
                    #    update(tdi, *args, **kwargs)
                    # call:
                    #    update(**kwargs)
                    ret = update_fun(share.get_shargs(), *args, **kwargs)
                    warnings.warn(_update_deprecate_msg, UserWarning)
            return ret

        try:
            ba = inspect.signature(update_fun).bind(*args, **kwargs)
        except TypeError:
            if len(args) and isinstance(args[0], dict):
                # user define ``update()`` function which does not receive the shared argument,
                # but do provide these shared arguments when calling ``update()`` function
                # -----
                # change
                #    update(tdi, *args, **kwargs)
                # as
                #    update(*args, **kwargs)
                share.save(**args[0])
                ret = update_fun(*args[1:], **kwargs)
                warnings.warn(_update_deprecate_msg, UserWarning)
                return ret
            else:
                # user define ``update()`` function which receives the shared argument,
                # but not provide these shared arguments when calling ``update()`` function
                # -----
                # change
                #    update(*args, **kwargs)
                # as
                #    update(tdi, *args, **kwargs)
                ret = update_fun(share.get_shargs(), *args, **kwargs)
                warnings.warn(_update_deprecate_msg, UserWarning)
                return ret
        else:
            if len(args) and isinstance(args[0], dict) and all([bm.ndim(v) == 0 for v in args[0].values()]):
                try:
                    ba = inspect.signature(update_fun).bind(*args[1:], **kwargs)
                except TypeError:
                    pass
                else:
                    # -----
                    # define as:
                    #    update(x=None)
                    # call as
                    #    update(tdi)
                    share.save(**args[0])
                    ret = update_fun(*args[1:], **kwargs)
                    warnings.warn(_update_deprecate_msg, UserWarning)
                    return ret
            return update_fun(*args, **kwargs)

    def _compatible_reset_state(self, *args, **kwargs):
        global the_top_layer_reset_state
        the_top_layer_reset_state = False
        try:
            if hasattr(self.reset_state, '_not_implemented'):
                self.reset(*args, **kwargs)
                warnings.warn(
                    '''
                From version >= 2.4.6, the policy of ``.reset_state()`` has been changed. See https://brainpy.readthedocs.io/en/latest/tutorial_toolbox/state_saving_and_loading.html for details.
          
                1. If you are resetting all states in a network by calling "net.reset_state(*args, **kwargs)", please use
                   "bp.reset_state(net, *args, **kwargs)" function, or "net.reset(*args, **kwargs)". 
                   ".reset_state()" only defines the resetting of local states in a local node (excluded its children nodes).
          
                2. If you does not customize "reset_state()" function for a local node, please implement it in your subclass.
          
                    ''',
                    DeprecationWarning
                )
            else:
                self.reset_state(*args, **kwargs)
        finally:
            the_top_layer_reset_state = True

    def _get_update_fun(self):
        return object.__getattribute__(self, 'update')

    def __getattribute__(self, item):
        if item == 'update':
            return self._compatible_update  # update function compatible with previous ``update()`` function
        if item == 'reset_state':
            if the_top_layer_reset_state:
                return self._compatible_reset_state  # reset_state function compatible with previous ``reset_state()`` function
        return super().__getattribute__(item)

    def __repr__(self):
        return f'{self.name}(mode={self.mode})'

    def __call__(self, *args, **kwargs):
        """The shortcut to call ``update`` methods."""

        # ``before_updates``
        for model in self.before_updates.values():
            if hasattr(model, '_receive_update_input'):
                model(*args, **kwargs)
            else:
                model()

        # update the model self
        ret = self.update(*args, **kwargs)

        # ``after_updates``
        for model in self.after_updates.values():
            if hasattr(model, '_not_receive_update_output'):
                model()
            else:
                model(ret)
        return ret

    def __rrshift__(self, other):
        """Support using right shift operator to call modules.

        Examples::

        >>> import brainpy as bp
        >>> x = bp.math.random.rand((10, 10))
        >>> l = bp.layers.Activation(bm.tanh)
        >>> y = x >> l
        """
        return self.__call__(other)


class DynSysGroup(DynamicalSystem, Container):
    """A group of :py:class:`~.DynamicalSystem`s in which the updating order does not matter.

    Args:
      children_as_tuple: The children objects.
      children_as_dict: The children objects.
      name: The object name.
      mode: The mode which controls the model computation.
      child_type: The type of the children object. Default is :py:class:`DynamicalSystem`.
    """

    def __init__(
        self,
        *children_as_tuple,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        child_type: type = DynamicalSystem,
        **children_as_dict
    ):
        super().__init__(name=name, mode=mode)

        # Attribute of "Container"
        self.children = bm.node_dict(self.format_elements(child_type, *children_as_tuple, **children_as_dict))

    def update(self, *args, **kwargs):
        """Step function of a network.

        In this update function, the update functions in children systems are
        iteratively called.
        """
        nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().not_subset(DynView)

        # update nodes of projections
        for node in nodes.subset(Projection).values():
            node()

        # update nodes of dynamics
        for node in nodes.subset(Dynamic).values():
            node()

        # update nodes with other types, including delays, ...
        for node in nodes.not_subset(Dynamic).not_subset(Projection).values():
            node()


class Network(DynSysGroup):
    """A group of :py:class:`~.DynamicalSystem`s which defines the nodes and edges in a network.
    """
    pass


class Sequential(DynamicalSystem, SupportAutoDelay, Container):
    """A sequential `input-output` module.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``dict`` of modules can be
    passed in. The ``update()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :py:class:`Container`? A ``Container`` is exactly what it
    sounds like--a container to store :py:class:`DynamicalSystem` s!
    On the other hand, the layers in a ``Sequential`` are connected
    in a cascading way.

    Examples::

    >>> import brainpy as bp
    >>> import brainpy.math as bm
    >>>
    >>> # composing ANN models
    >>> l = bp.Sequential(bp.layers.Dense(100, 10),
    >>>                   bm.relu,
    >>>                   bp.layers.Dense(10, 2))
    >>> l(bm.random.random((256, 100)))
    >>>
    >>> # Using Sequential with Dict. This is functionally the
    >>> # same as the above code
    >>> l = bp.Sequential(l1=bp.layers.Dense(100, 10),
    >>>                   l2=bm.relu,
    >>>                   l3=bp.layers.Dense(10, 2))
    >>> l(bm.random.random((256, 100)))


    Args:
      modules_as_tuple: The children modules.
      modules_as_dict: The children modules.
      name: The object name.
      mode: The object computing context/mode. Default is ``None``.
    """

    def __init__(
        self,
        *modules_as_tuple,
        name: str = None,
        mode: bm.Mode = None,
        **modules_as_dict
    ):
        super().__init__(name=name, mode=mode)

        # Attribute of "Container"
        self.children = bm.node_dict(self.format_elements(object, *modules_as_tuple, **modules_as_dict))

    def update(self, x):
        """Update function of a sequential model.
        """
        for m in self.children.values():
            x = m(x)
        return x

    def return_info(self):
        last = self[-1]
        if not isinstance(last, SupportAutoDelay):
            raise UnsupportedError(f'Does not support "return_info()" because the last node is '
                                   f'not instance of {SupportAutoDelay.__name__}')
        return last.return_info()

    def __getitem__(self, key: Union[int, slice, str]):
        if isinstance(key, str):
            if key in self.children:
                return self.children[key]
            else:
                raise KeyError(f'Does not find a component named {key} in\n {str(self)}')
        elif isinstance(key, slice):
            return Sequential(**dict(tuple(self.children.items())[key]))
        elif isinstance(key, int):
            return tuple(self.children.values())[key]
        elif isinstance(key, (tuple, list)):
            _all_nodes = tuple(self.children.items())
            return Sequential(**dict(_all_nodes[k] for k in key))
        else:
            raise KeyError(f'Unknown type of key: {type(key)}')

    def __repr__(self):
        nodes = self.children.values()
        entries = '\n'.join(f'  [{i}] {tools.repr_object(x)}' for i, x in enumerate(nodes))
        return f'{self.__class__.__name__}(\n{entries}\n)'


class Projection(DynamicalSystem):
    """Base class to model synaptic projections.

    Args:
      name: The name of the dynamic system.
      mode: The computing mode. It should be an instance of :py:class:`~.Mode`.
    """

    def update(self, *args, **kwargs):
        nodes = tuple(self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().values())
        if len(nodes):
            for node in nodes:
                node.update(*args, **kwargs)
        else:
            raise ValueError('Do not implement the update() function.')

    def clear_input(self, *args, **kwargs):
        """Empty function of clearing inputs."""
        pass

    def reset_state(self, *args, **kwargs):
        pass


class Dynamic(DynamicalSystem):
    """Base class to model dynamics.

    There are several essential attributes:

    - ``size``: the geometry of the neuron group. For example, `(10, )` denotes a line of
      neurons, `(10, 10)` denotes a neuron group aligned in a 2D space, `(10, 15, 4)` denotes
      a 3-dimensional neuron group.
    - ``num``: the flattened number of neurons in the group. For example, `size=(10, )` => \
      `num=10`, `size=(10, 10)` => `num=100`, `size=(10, 15, 4)` => `num=600`.

    Args:
      size: The neuron group geometry.
      name: The name of the dynamic system.
      keep_size: Whether keep the geometry information.
      mode: The computing mode.
    """

    def __init__(
        self,
        size: Shape,
        keep_size: bool = False,
        sharding: Optional[Any] = None,
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
        method: str = 'exp_auto'
    ):
        # size
        if isinstance(size, (list, tuple)):
            if len(size) <= 0:
                raise ValueError(f'size must be int, or a tuple/list of int. '
                                 f'But we got {type(size)}')
            if not isinstance(size[0], (int, np.integer)):
                raise ValueError('size must be int, or a tuple/list of int.'
                                 f'But we got {type(size)}')
            size = tuple(size)
        elif isinstance(size, (int, np.integer)):
            size = (size,)
        else:
            raise ValueError('size must be int, or a tuple/list of int.'
                             f'But we got {type(size)}')
        self.size = size
        self.keep_size = keep_size

        # number of neurons
        self.num = tools.size2num(size)

        # axis names for parallelization
        self.sharding = sharding

        # integration method
        self.method = method

        # initialize
        super().__init__(name=name, mode=mode)

    @property
    def varshape(self):
        """The shape of variables in the neuron group."""
        return self.size if self.keep_size else (self.num,)

    def get_batch_shape(self, batch_size=None):
        if batch_size is None:
            return self.varshape
        else:
            return (batch_size,) + self.varshape

    def update(self, *args, **kwargs):
        """The function to specify the updating rule.
        """
        raise NotImplementedError(f'Subclass of {self.__class__.__name__} must '
                                  f'implement "update" function.')

    def init_param(self, param, shape=None, sharding=None):
        """Initialize parameters.

        If ``sharding`` is provided and ``param`` is array, this function will
        partition the parameter across the default device mesh.

        See :py:func:`~.brainpy.math.sharding.device_mesh` for the mesh setting.
        """
        shape = self.varshape if shape is None else shape
        sharding = self.sharding if sharding is None else sharding
        return parameter(param,
                         sizes=shape,
                         allow_none=False,
                         sharding=sharding)

    def init_variable(self, var_data, batch_or_mode, shape=None, sharding=None):
        """Initialize variables.

        If ``sharding`` is provided and ``var_data`` is array, this function will
        partition the variable across the default device mesh.

        See :py:func:`~.brainpy.math.sharding.device_mesh` for the mesh setting.
        """
        shape = self.varshape if shape is None else shape
        sharding = self.sharding if sharding is None else sharding
        return variable_(var_data,
                         sizes=shape,
                         batch_or_mode=batch_or_mode,
                         axis_names=sharding,
                         batch_axis_name=bm.sharding.BATCH_AXIS)

    def __repr__(self):
        return f'{self.name}(mode={self.mode}, size={self.size})'

    def __getitem__(self, item):
        return DynView(target=self, index=item)

    def clear_input(self, *args, **kwargs):
        """Empty function of clearing inputs."""
        pass


class DynView(Dynamic):
    """DSView, an object used to get a view of a dynamical system instance.

    It can get a subset view of variables in a dynamical system instance.
    For instance,

    >>> import brainpy as bp
    >>> hh = bp.neurons.HH(10)
    >>> DynView(hh, slice(5, 10, None))
    >>> # or, simply
    >>> hh[5:]
    """

    def __init__(
        self,
        target: Dynamic,
        index: Union[slice, Sequence, ArrayType],
        name: Optional[str] = None,
    ):
        # check target
        if not isinstance(target, Dynamic):
            raise TypeError(f'Should be instance of {Dynamic.__name__}, but we got {type(target)}.')
        self.target = target  # the target object to slice

        # check slicing
        if isinstance(index, (int, slice)):
            index = (index,)
        self.index = index  # the slice
        if len(self.index) > len(target.varshape):
            raise ValueError(f"Length of the index should be less than "
                             f"that of the target's varshape. But we "
                             f"got {len(self.index)} > {len(target.varshape)}")

        # get all variables for slicing
        if hasattr(self.target, SLICE_VARS):
            all_vars = {}
            for var_str in getattr(self.target, SLICE_VARS):
                v = eval(f'target.{var_str}')
                all_vars[var_str] = v
        else:
            all_vars = target.vars(level=1, include_self=True, method='relative')
            all_vars = {k: v for k, v in all_vars.items()}  # TODO
            # all_vars = {k: v for k, v in all_vars.items() if v.nobatch_shape == varshape}

        # slice variables
        self.slice_vars = dict()
        for k, v in all_vars.items():
            if v.batch_axis is not None:
                index = (
                    (self.index[:v.batch_axis] + (slice(None, None, None),) + self.index[v.batch_axis:])
                    if (len(self.index) > v.batch_axis) else
                    (self.index + tuple([slice(None, None, None) for _ in range(v.batch_axis - len(self.index) + 1)]))
                )
            else:
                index = self.index
            self.slice_vars[k] = bm.VariableView(v, index)

        # sub-nodes
        # nodes = target.nodes(method='relative', level=0, include_self=True).subset(DynamicalSystem)
        # for k, node in nodes.items():
        #   if isinstance(node, Dynamic):
        #     node = DynView(node, self.index)
        #   else:
        #     node = DynView(node, self.index)
        #   setattr(self, k, node)

        # initialization
        # get size
        size = []
        for i, idx in enumerate(self.index):
            if isinstance(idx, int):
                size.append(1)
            elif isinstance(idx, slice):
                size.append(_slice_to_num(idx, target.varshape[i]))
            else:
                # should be a list/tuple/array of int
                # do not check again
                if not isinstance(idx, collections.abc.Iterable):
                    raise TypeError('Should be an iterable object of int.')
                size.append(len(idx))
        size += list(target.varshape[len(self.index):])

        super().__init__(size, keep_size=target.keep_size, name=name, mode=target.mode)

    def __repr__(self):
        return f'{self.name}(target={self.target}, index={self.index})'

    def __getattribute__(self, item):
        try:
            slice_vars = object.__getattribute__(self, 'slice_vars')
            if item in slice_vars:
                value = slice_vars[item]
                return value
            return object.__getattribute__(self, item)
        except AttributeError:
            return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if hasattr(self, 'slice_vars'):
            slice_vars = super().__getattribute__('slice_vars')
            if key in slice_vars:
                v = slice_vars[key]
                v.value = value
                return
        super(DynView, self).__setattr__(key, value)

    def update(self, *args, **kwargs):
        raise NoImplementationError(f'{DynView.__name__} {self} cannot be updated. '
                                    f'Please update its parent {self.target}')

    def reset_state(self, batch_size=None):
        pass


@tools.numba_jit
def _slice_to_num(slice_: slice, length: int):
    # start
    start = slice_.start
    if start is None:
        start = 0
    if start < 0:
        start = length + start
    start = max(start, 0)
    # stop
    stop = slice_.stop
    if stop is None:
        stop = length
    if stop < 0:
        stop = length + stop
    stop = min(stop, length)
    # step
    step = slice_.step
    if step is None:
        step = 1
    # number
    num = 0
    while start < stop:
        start += step
        num += 1
    return num


def receive_update_output(cls: object):
    """
    The decorator to mark the object (as the after updates) to receive the output of the update function.

    That is, the `aft_update` will receive the return of the update function::

      ret = model.update(*args, **kwargs)
      for fun in model.aft_updates:
        fun(ret)

    """
    # assert isinstance(cls, DynamicalSystem), 'The input class should be instance of DynamicalSystem.'
    if hasattr(cls, '_not_receive_update_output'):
        delattr(cls, '_not_receive_update_output')
    return cls


def not_receive_update_output(cls: object):
    """
    The decorator to mark the object (as the after updates) to not receive the output of the update function.

    That is, the `aft_update` will not receive the return of the update function::

      ret = model.update(*args, **kwargs)
      for fun in model.aft_updates:
        fun()

    """
    # assert isinstance(cls, DynamicalSystem), 'The input class should be instance of DynamicalSystem.'
    cls._not_receive_update_output = True
    return cls


def receive_update_input(cls: object):
    """
    The decorator to mark the object (as the before updates) to receive the input of the update function.

    That is, the `bef_update` will receive the input of the update function::


      for fun in model.bef_updates:
        fun(*args, **kwargs)
      model.update(*args, **kwargs)

    """
    # assert isinstance(cls, DynamicalSystem), 'The input class should be instance of DynamicalSystem.'
    cls._receive_update_input = True
    return cls


def not_receive_update_input(cls: object):
    """
    The decorator to mark the object (as the before updates) to not receive the input of the update function.

    That is, the `bef_update` will not receive the input of the update function::

        for fun in model.bef_updates:
          fun()
        model.update()

    """
    # assert isinstance(cls, DynamicalSystem), 'The input class should be instance of DynamicalSystem.'
    if hasattr(cls, '_receive_update_input'):
        delattr(cls, '_receive_update_input')
    return cls
