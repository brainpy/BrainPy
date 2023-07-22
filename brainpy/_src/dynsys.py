# -*- coding: utf-8 -*-

import collections
import gc
import inspect
import warnings
from typing import Union, Dict, Callable, Sequence, Optional, Any

import numpy as np

from brainpy import tools, math as bm
from brainpy._src.initialize import parameter, variable_
from brainpy._src.mixin import AutoDelaySupp, Container, DelayRegister, global_delay_data
from brainpy.errors import NoImplementationError, UnsupportedError
from brainpy.types import ArrayType, Shape
from brainpy._src.deprecations import _update_deprecate_msg

share = None

__all__ = [
  # general
  'DynamicalSystem',

  # containers
  'DynSysGroup', 'Network', 'Sequential',

  # category
  'Dynamic', 'Projection',
]

SLICE_VARS = 'slice_vars'


def not_pass_shared(func: Callable):
  """Label the update function as the one without passing shared arguments.

  The original update function explicitly requires shared arguments at the first place::

    class TheModel(DynamicalSystem):
        def update(self, s, x):
            # s is the shared arguments, like `t`, `dt`, etc.
            pass

  So, each time we call the model we should provide shared arguments into the model::

    TheModel()(shared, inputs)

  When we label the update function as ``do_not_pass_sha_args``, this time there is no
  need to call the dynamical system with shared arguments::

    class NewModel(DynamicalSystem):
       @no_shared
       def update(self, x):
         pass

    NewModel()(inputs)

  .. versionadded:: 2.3.5

  Parameters
  ----------
  func: Callable
    The function in the :py:class:`~.DynamicalSystem`.

  Returns
  -------
  func: Callable
    The wrapped function for the class.
  """
  func._new_style = True
  return func


class DynamicalSystem(bm.BrainPyObject, DelayRegister):
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

  Parameters
  ----------
  name : optional, str
    The name of the dynamical system.
  mode: optional, Mode
    The model computation mode. It should be instance of :py:class:`~.Mode`.
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

    # local delay variables:
    # Compatible for ``DelayRegister``
    # TODO: will be deprecated in the future
    self.local_delay_vars: Dict = bm.node_dict()

    # the before- / after-updates used for computing
    # added after the version of 2.4.3
    self.before_updates: Dict[str, Callable] = bm.node_dict()
    self.after_updates: Dict[str, Callable] = bm.node_dict()

    # super initialization
    super().__init__(name=name)

  def update(self, *args, **kwargs):
    """The function to specify the updating rule.

    Assume any dynamical system depends on the shared variables (`sha`),
    like time variable ``t``, the step precision ``dt``, and the time step `i`.
    """
    raise NotImplementedError('Must implement "update" function by subclass self.')

  def reset(self, *args, **kwargs):
    """Reset function which reset the whole variables in the model.
    """
    self.reset_state(*args, **kwargs)

  def reset_state(self, *args, **kwargs):
    """Reset function which reset the states in the model.
    """
    child_nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique()
    if len(child_nodes) > 0:
      for node in child_nodes.values():
        node.reset_state(*args, **kwargs)
      self.reset_local_delays(child_nodes)
    else:
      raise NotImplementedError('Must implement "reset_state" function by subclass self. '
                                f'Error of {self.name}')

  def clear_input(self):
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
    global share
    if share is None:
      from brainpy._src.context import share
    share.save(i=i, t=i * bm.dt)
    return self.update(*args, **kwargs)

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

  def _compatible_update(self, *args, **kwargs):
    global share
    if share is None:
      from brainpy._src.context import share
    update_fun = super().__getattribute__('update')
    update_args = tuple(inspect.signature(update_fun).parameters.values())

    if len(update_args) and update_args[0].name in ['tdi', 'sh', 'sha']:
      # define the update function with:
      #     update(tdi, *args, **kwargs)
      #
      if len(args) > 0:
        if isinstance(args[0], dict) and all([bm.isscalar(v) for v in args[0].values()]):
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
      return update_fun(*args, **kwargs)

  def __getattribute__(self, item):
    if item == 'update':
      return self._compatible_update  # update function compatible with previous ``update()`` function
    else:
      return super().__getattribute__(item)

  def _get_update_fun(self):
    return object.__getattribute__(self, 'update')

  def __repr__(self):
    return f'{self.name}(mode={self.mode})'

  def __call__(self, *args, **kwargs):
    """The shortcut to call ``update`` methods."""

    # ``before_updates``
    for model in self.before_updates.values():
      model()

    # update the model self
    ret = self.update(*args, **kwargs)

    # ``after_updates``
    for model in self.after_updates.values():
      model(ret)
    return ret

  def __del__(self):
    """Function for handling `del` behavior.

    This function is used to pop out the variables which registered in global delay data.
    """
    try:
      if hasattr(self, 'local_delay_vars'):
        for key in tuple(self.local_delay_vars.keys()):
          val = global_delay_data.pop(key)
          del val
          val = self.local_delay_vars.pop(key)
          del val
      if hasattr(self, 'implicit_nodes'):
        for key in tuple(self.implicit_nodes.keys()):
          del self.implicit_nodes[key]
      if hasattr(self, 'implicit_vars'):
        for key in tuple(self.implicit_vars.keys()):
          del self.implicit_vars[key]
      for key in tuple(self.__dict__.keys()):
        del self.__dict__[key]
    finally:
      gc.collect()

  def __rrshift__(self, other):
    """Support using right shift operator to call modules.

    Examples
    --------

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

    # update delays
    # TODO: Will be deprecated in the future
    self.update_local_delays(nodes)

  def reset_state(self, batch_size=None):
    nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().not_subset(DynView)

    # reset projections
    for node in nodes.subset(Projection).values():
      node.reset_state(batch_size)

    # reset dynamics
    for node in nodes.subset(Dynamic).values():
      node.reset_state(batch_size)

    # reset other types of nodes, including delays, ...
    for node in nodes.not_subset(Dynamic).not_subset(Projection).values():
      node.reset_state(batch_size)

    # reset delays
    # TODO: will be removed in the future
    self.reset_local_delays(nodes)

  def clear_input(self):
    """Clear inputs in the children classes."""
    nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique().not_subset(DynView)
    for node in nodes.values():
      node.clear_input()


class Network(DynSysGroup):
  """A group of :py:class:`~.DynamicalSystem`s which defines the nodes and edges in a network.
  """
  pass


class Sequential(DynamicalSystem, AutoDelaySupp, Container):
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

  Examples
  --------

  >>> import brainpy as bp
  >>> import brainpy.math as bm
  >>>
  >>> # composing ANN models
  >>> l = bp.Sequential(bp.layers.Dense(100, 10),
  >>>                   bm.relu,
  >>>                   bp.layers.Dense(10, 2))
  >>> l({}, bm.random.random((256, 100)))
  >>>
  >>> # Using Sequential with Dict. This is functionally the
  >>> # same as the above code
  >>> l = bp.Sequential(l1=bp.layers.Dense(100, 10),
  >>>                   l2=bm.relu,
  >>>                   l3=bp.layers.Dense(10, 2))
  >>> l({}, bm.random.random((256, 100)))


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
    self.children = bm.node_dict(self.format_elements(object, *modules_as_tuple, **modules_as_dict))

  def update(self, x):
    """Update function of a sequential model.
    """
    for m in self.children.values():
      x = m(x)
    return x

  def return_info(self):
    last = self[-1]
    if not isinstance(last, AutoDelaySupp):
      raise UnsupportedError(f'Does not support "return_info()" because the last node is '
                             f'not instance of {AutoDelaySupp.__name__}')
    return last.return_info()

  def __format_key(self, i):
    return f'l-{i}'

  def __all_nodes(self):
    nodes = []
    for i in range(self._num):
      key = self.__format_key(i)
      if key not in self._dyn_modules:
        nodes.append(self._static_modules[key])
      else:
        nodes.append(self._dyn_modules[key])
    return nodes

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
    nodes = self.__all_nodes()
    entries = '\n'.join(f'  [{i}] {tools.repr_object(x)}' for i, x in enumerate(nodes))
    return f'{self.__class__.__name__}(\n{entries}\n)'


class Projection(DynamicalSystem):
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

    # inputs
    self.cur_inputs: Dict = bm.node_dict()

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
    nodes = target.nodes(method='relative', level=1, include_self=False).subset(DynamicalSystem)
    for k, node in nodes.items():
      if isinstance(node, Dynamic):
        node = DynView(node, self.index)
      else:
        node = DynView(node, self.index)
      setattr(self, k, node)

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
        if not isinstance(idx, collections.Iterable):
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
