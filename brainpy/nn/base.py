# -*- coding: utf-8 -*-


"""
This module provide basic Node class for whole ``brainpy.nn`` system.

- ``brainpy.nn.Node``: The fundamental class representing the node or the element.
- ``brainpy.nn.RecurrentNode``: The recurrent node which has a self-connection.
- ``brainpy.nn.Network``: The network model which is composed of multiple node elements.
  Once the Network instance receives a node operation, the wrapped elements, the new
  elements, and their connection edges will be formed as another Network instance.
  This means ``brainpy.nn.Network`` is only used to pack element nodes. It will be
  never be an element node.
- ``brainpy.nn.FrozenNetwork``: The whole network which can be represented as a basic
  elementary node when composing a larger network (TODO).
"""

from copy import copy, deepcopy
from typing import (Dict, Sequence, Tuple, Union, Optional, Any, Callable)

import jax.numpy as jnp

from brainpy import tools, math as bm
from brainpy.base import Base, Collector
from brainpy.errors import (UnsupportedError,
                            PackageMissingError,
                            ModelBuildError,
                            MathError)
from brainpy.nn.algorithms.offline import OfflineAlgorithm
from brainpy.nn.algorithms.online import OnlineAlgorithm
from brainpy.nn.datatypes import (DataType, SingleData, MultipleData)
from brainpy.nn.graph_flow import (find_senders_and_receivers,
                                   find_entries_and_exits,
                                   detect_cycle,
                                   detect_path)
from brainpy.tools.checking import (check_dict_data,
                                    check_shape_except_batch,
                                    check_integer)
from brainpy.types import Tensor

operations = None

__all__ = [
  'Node', 'Network',
  'RecurrentNode',  # a marker for recurrent node
  'FrozenNetwork',  # a marker for frozen network
]

NODE_STATES = ['inputs', 'feedbacks', 'state', 'output']

SUPPORTED_LAYOUTS = ['shell_layout',
                     'multipartite_layout',
                     'spring_layout',
                     'spiral_layout',
                     'spectral_layout',
                     'random_layout',
                     'planar_layout',
                     'kamada_kawai_layout',
                     'circular_layout']


def not_implemented(fun: Callable) -> Callable:
  """Marks the given module method is not implemented.

  Methods wrapped in @not_implemented can define submodules directly within the method.

  For instance::

    @not_implemented
    init_fb(self):
      ...

    @not_implemented
    def feedback(self):
      ...
  """
  fun.not_implemented = True
  return fun


class Node(Base):
  """Basic Node class for neural network building in BrainPy."""

  '''Support multiple types of data pass, including "PassOnlyOne" (by default), 
  "PassSequence", "PassNameDict", etc. and user-customized type which inherits 
  from basic "SingleData" or "MultipleData". 
  
  This setting will change the feedforward/feedback input data which pass into 
  the "call()" function and the sizes of the feedforward/feedback input data.'''
  data_pass = SingleData()

  '''Offline fitting method.'''
  offline_fit_by: Union[Callable, OfflineAlgorithm]

  '''Online fitting method.'''
  online_fit_by: OnlineAlgorithm

  def __init__(
      self,
      name: Optional[str] = None,
      input_shape: Optional[Union[Sequence[int], int]] = None,
      trainable: bool = True
  ):

    # initialize parameters
    self._feedforward_shapes = None  # input shapes
    self._output_shape = None  # output size
    self._feedback_shapes = None  # feedback shapes
    self._is_ff_initialized = False
    self._is_fb_initialized = False
    self._is_state_initialized = False
    self._is_fb_state_initialized = False
    self._trainable = trainable
    self._state = None  # the state of the current node
    self._fb_output = None  # the feedback output of the current node
    # data pass
    if not isinstance(self.data_pass, DataType):
      raise ValueError(f'Unsupported data pass type {type(self.data_pass)}. '
                       f'Only support {DataType.__class__}')

    # super initialization
    super(Node, self).__init__(name=name)

    # parameters
    if input_shape is not None:
      self._feedforward_shapes = {self.name: (None,) + tools.to_size(input_shape)}

  def __repr__(self):
    return (f"{type(self).__name__}(name={self.name}, "
            f"forwards={self.feedforward_shapes}, "
            f"feedbacks={self.feedback_shapes}, "
            f"output={self.output_shape})")

  def __call__(self, *args, **kwargs) -> Tensor:
    """The main computation function of a Node.

    Parameters
    ----------
    ff: dict, sequence, JaxArray, ndarray
      The feedforward inputs.
    fb: optional, dict, sequence, JaxArray, ndarray
      The feedback inputs.
    forced_states: optional, dict
      The fixed state for the nodes in the network.
    forced_feedbacks: optional, dict
      The fixed feedback for the nodes in the network.
    monitors: optional, sequence
      Can be used to monitor the state or the attribute of a node in the network.
    **kwargs
      Other parameters which will be parsed into every node.

    Returns
    -------
    Tensor
      A output tensor value, or a dict of output tensors.
    """
    return self._call(*args, **kwargs)

  def __rshift__(self, other):  # "self >> other"
    global operations
    if operations is None: from . import operations
    return operations.ff_connect(self, other)

  def __rrshift__(self, other):  # "other >> self"
    global operations
    if operations is None: from . import operations
    return operations.ff_connect(other, self)

  def __irshift__(self, other):  # "self >>= other"
    raise ValueError('Only Network objects support inplace feedforward connection.')

  def __lshift__(self, other):  # "self << other"
    global operations
    if operations is None: from . import operations
    return operations.fb_connect(other, self)

  def __rlshift__(self, other):  # "other << self"
    global operations
    if operations is None: from . import operations
    return operations.fb_connect(self, other)

  def __ilshift__(self, other):  # "self <<= other"
    raise ValueError('Only Network objects support inplace feedback connection.')

  def __and__(self, other):  # "self & other"
    global operations
    if operations is None: from . import operations
    return operations.merge(self, other)

  def __rand__(self, other):  # "other & self"
    global operations
    if operations is None: from . import operations
    return operations.merge(other, self)

  def __iand__(self, other):
    raise ValueError('Only Network objects support inplace merging.')

  def __getitem__(self, item):  # like "[:10]"
    if isinstance(item, str):
      raise ValueError('Node only supports slice, not retrieve by the name.')
    else:
      global operations
      if operations is None: from . import operations
      return operations.select(self, item)

  @property
  def state(self) -> Optional[Tensor]:
    """Node current internal state."""
    if self._is_ff_initialized:
      return self._state
    return None

  @state.setter
  def state(self, value: Tensor):
    raise NotImplementedError('Please use "set_state()" to reset the node state, '
                              'or use "self.state.value" to change the state content.')

  def set_state(self, state):
    """
    Safely set the state of the node.

    This method allows the maximum flexibility to change the
    node state. It can set a new data (same shape, same dtype)
    to the state. It can also set a new data with the different
    shape. We highly recommend the user to use this function.
    instead of using ``self.state.value``.
    """
    if self.state is None:
      if self.output_shape is not None:
        check_shape_except_batch(self.output_shape, state.shape)
      self._state = bm.Variable(state) if not isinstance(state, bm.Variable) else state
    else:
      check_shape_except_batch(self.state.shape, state.shape)
      if self.state.dtype != state.dtype:
        raise MathError('Cannot set the state, because the dtype is not consistent: '
                        f'{self.state.dtype} != {state.dtype}')
      self.state._value = bm.as_device_array(state)

  @property
  def fb_output(self) -> Optional[Tensor]:
    return self._fb_output

  @fb_output.setter
  def fb_output(self, value: Tensor):
    raise NotImplementedError('Please use "set_fb_output()" to reset the node feedback state, '
                              'or use "self.fb_output.value" to change the state content.')

  def set_fb_output(self, state: Tensor):
    """
    Safely set the feedback state of the node.

    This method allows the maximum flexibility to change the
    node state. It can set a new data (same shape, same dtype)
    to the state. It can also set a new data with the different
    shape. We highly recommend the user to use this function.
    instead of using ``self.fb_output.value``.
    """
    if self.fb_output is None:
      if self.output_shape is not None:
        check_shape_except_batch(self.output_shape, state.shape)
      self._fb_output = bm.Variable(state) if not isinstance(state, bm.Variable) else state
    else:
      check_shape_except_batch(self.fb_output.shape, state.shape)
      if self.fb_output.dtype != state.dtype:
        raise MathError('Cannot set the feedback state, because the dtype is '
                        f'not consistent: {self.fb_output.dtype} != {state.dtype}')
      self.fb_output._value = bm.as_device_array(state)

  @property
  def trainable(self) -> bool:
    """Returns if the Node can be trained."""
    return self._trainable

  @property
  def is_initialized(self) -> bool:
    if self._is_ff_initialized and self._is_state_initialized:
      if self.feedback_shapes is not None:
        if self._is_fb_initialized and self._is_fb_state_initialized:
          return True
        else:
          return False
      else:
        return True
    else:
      return False

  @trainable.setter
  def trainable(self, value: bool):
    """Freeze or unfreeze the Node. If set to False,
    learning is stopped."""
    assert isinstance(value, bool), 'Must be a boolean.'
    self._trainable = value

  @property
  def feedforward_shapes(self):
    """Input data size."""
    return self.data_pass.filter(self._feedforward_shapes)

  @feedforward_shapes.setter
  def feedforward_shapes(self, size):
    self.set_feedforward_shapes(size)

  def set_feedforward_shapes(self, feedforward_shapes: Dict):
    if not self._is_ff_initialized:
      check_dict_data(feedforward_shapes,
                      key_type=(Node, str),
                      val_type=(list, tuple),
                      name='feedforward_shapes')
      self._feedforward_shapes = feedforward_shapes
    else:
      if self.feedforward_shapes is not None:
        sizes1 = sorted(list(self._feedforward_shapes.values()))
        sizes2 = sorted(list(feedforward_shapes.values()))
        if sizes1 != sizes2:
          raise ValueError(f"Impossible to reset the input shapes of {self.name}. "
                           f"Because this Node has the input shapes {sizes1}. "
                           f"While we got input shapes of {sizes2}")
        self._feedforward_shapes = feedforward_shapes

  @property
  def feedback_shapes(self):
    """Output data size."""
    return self.data_pass.filter(self._feedback_shapes)

  @feedback_shapes.setter
  def feedback_shapes(self, size):
    self.set_feedback_shapes(size)

  def set_feedback_shapes(self, fb_shapes: Dict):
    if not self._is_fb_initialized:
      check_dict_data(fb_shapes,
                      key_type=(Node, str),
                      val_type=(tuple, list),
                      name='fb_shapes')
      self._feedback_shapes = fb_shapes
    else:
      if self.feedback_shapes is not None:
        sizes1 = sorted(list(self._feedback_shapes.values()))
        sizes2 = sorted(list(fb_shapes.values()))
        if sizes1 != sizes2:
          raise ValueError(f"Impossible to reset the feedback shapes of {self.name}. "
                           f"Because this Node has the feedback shapes {sizes1}. "
                           f"While we got feedback shapes of {sizes2}")
        self._feedback_shapes = fb_shapes

  @property
  def output_shape(self) -> Optional[Tuple[int]]:
    """Output data size."""
    return self._output_shape

  @output_shape.setter
  def output_shape(self, size):
    self.set_output_shape(size)

  @property
  def is_feedback_input_supported(self):
    if hasattr(self.init_fb_conn, 'not_implemented'):
      if self.init_fb_conn.not_implemented:
        return False
    return True

  @property
  def is_feedback_supported(self):
    if self.fb_output is None:
      return False
    else:
      return True

  def set_output_shape(self, shape: Sequence[int]):
    if not self._is_ff_initialized:
      if not isinstance(shape, (tuple, list)):
        raise ValueError(f'Must be a sequence of int, but got {shape}')
      self._output_shape = tuple(shape)
    else:
      check_shape_except_batch(shape, self.output_shape)

  def nodes(self, method='absolute', level=1, include_self=True):
    return super(Node, self).nodes(method=method, level=level, include_self=include_self)

  def vars(self, method='absolute', level=1, include_self=True):
    return super(Node, self).vars(method=method, level=level, include_self=include_self)

  def train_vars(self, method='absolute', level=1, include_self=True):
    return super(Node, self).train_vars(method=method, level=level, include_self=include_self)

  def copy(self,
           name: str = None,
           shallow: bool = False):
    """Returns a copy of the Node.

    Parameters
    ----------
    name : str
        Name of the Node copy.
    shallow : bool, default to False
        If False, performs a deep copy of the Node.

    Returns
    -------
    Node
        A copy of the Node.
    """
    if shallow:
      new_obj = copy(self)
    else:
      new_obj = deepcopy(self)
    new_obj.name = self.unique_name(name or (self.name + '_copy'))
    return new_obj

  def _init_ff_conn(self):
    if not self._is_ff_initialized:
      self.init_ff_conn()
      if self.output_shape is None:
        raise ValueError(f'Please set the output shape when implementing '
                         f'"init_ff_conn()" of the node {self.name}')
      self._is_ff_initialized = True

  def _init_fb_conn(self):
    if not self._is_fb_initialized:
      try:
        self.init_fb_conn()
      except Exception as e:
        raise ModelBuildError(f"{self.name} initialization failed.") from e
      self._is_fb_initialized = True

  @not_implemented
  def init_fb_conn(self):
    """Initialize the feedback connections.
    This function will be called only once."""
    raise ValueError(f'This node \n\n{self} \n\ndoes not support feedback connection.')

  def init_ff_conn(self):
    """Initialize the feedforward connections.
    This function will be called only once."""
    raise NotImplementedError('Please implement the feedforward initialization.')

  def _init_state(self, num_batch=1):
    state = self.init_state(num_batch)
    if state is not None:
      self.set_state(state)
    self._is_state_initialized = True

  def _init_fb_output(self, num_batch=1):
    output = self.init_fb_output(num_batch)
    if output is not None:
      self.set_fb_output(output)
    self._is_fb_state_initialized = True

  def init_state(self, num_batch=1) -> Optional[Tensor]:
    """Set the initial node state.

    This function can be called multiple times."""
    pass

  def init_fb_output(self, num_batch=1) -> Optional[Tensor]:
    """Set the initial node feedback state.

    This function can be called multiple times. However,
    it is only triggered when the node has feedback connections.
    """
    return bm.zeros((num_batch,) + self.output_shape[1:], dtype=bm.float_)

  def initialize(self, num_batch: int = 1):
    """
    Initialize the node. This function must be called before applying JIT.

    This function is useful, because it is independent of the __call__ function.
    We can use this function before we apply JIT to __call__ function.
    """

    # feedforward initialization
    if self.feedforward_shapes is None:
      raise ValueError('Cannot initialize this node, because we detect '
                       'both "feedforward_shapes" is None. '
                       'Two ways can solve this problem:\n\n'
                       '1. Connecting an instance of "brainpy.nn.Input()" to this node. \n'
                       '2. Providing the "input_shape" when initialize the node.')
    check_integer(num_batch, 'num_batch', min_bound=0, allow_none=False)
    self._init_ff_conn()

    # initialize state
    self._init_state(num_batch)

    if self.feedback_shapes is not None:
      # feedback initialization
      self._init_fb_conn()
      # initialize feedback state
      self._init_fb_output(num_batch)

  def _check_inputs(self, ff, fb=None):
    # check feedforward inputs
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      ff = {self.name: ff}
    if not isinstance(ff, dict):
      raise ValueError(f'"ff" must be a dict or a tensor, got {type(ff)}: {ff}')
    if self.name not in ff:
      raise ValueError(f'Cannot find input for this node {self} when given "ff" {ff}')
    for k, size in self._feedforward_shapes.items():
      if k not in ff:
        raise ValueError(f"The required key {k} is not provided in feedforward inputs.")
      check_shape_except_batch(size, ff[k].shape)
    if self.state is not None:
      for inp in ff.values():
        if self.state.shape[0] != inp.shape[0]:
          raise ValueError(f'The batch size of the input data {inp.shape[0]} is not '
                           f'equal to the batch size of the node state {self.state.shape[0]}. '
                           f'Maybe you need to reinitialize the data with the desired '
                           f'batch size by ".initialize(num_batch)", or change the data '
                           f'consistent with the data batch size {self.state.shape[0]}.')

    # check feedback inputs
    if fb is not None:
      if not isinstance(fb, dict):
        raise ValueError(f'"fb" must be a dict, got {type(fb)}: {fb}')
      # check feedback consistency
      for k, size in self._feedback_shapes.items():
        if k not in fb:
          raise ValueError(f"The required key {k} is not provided in feedback inputs.")
        check_shape_except_batch(size, fb[k].shape)
      if self.state is not None:
        for inp in fb.values():
          if self.state.shape[0] != inp.shape[0]:
            raise ValueError(f'The batch size of the feedback data {inp.shape[0]} is not '
                             f'equal to the batch size of the node state {self.state.shape[0]}. '
                             f'Maybe you need to reinitialize the data with the desired '
                             f'batch size by ".initialize(num_batch)", or change the data '
                             f'consistent with the data batch size {self.state.shape[0]}.')
    # data
    ff = self.data_pass.filter(ff)
    fb = self.data_pass.filter(fb)
    return ff, fb

  def _call(self,
            ff: Union[Tensor, Dict[Any, Tensor]],
            fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
            forced_states: Dict[str, Tensor] = None,
            forced_feedbacks: Dict[str, Tensor] = None,
            monitors=None,
            **kwargs) -> Union[Tensor, Tuple[Tensor, Dict]]:
    if not self.is_initialized:
      raise ValueError('Please initialize the Node first by calling "initialize()" function.')

    # initialize the forced data
    if forced_states is None:
      forced_states = dict()
    if isinstance(forced_states, (bm.ndarray, jnp.ndarray)):
      forced_states = {self.name: forced_states}
    check_dict_data(forced_states, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    if forced_feedbacks is not None:
      if len(forced_feedbacks) != 0:
        raise ValueError('Single instance of brainpy.nn.Node do '
                         'not support "forced_feedbacks"')
    # monitors
    need_return_monitor = True
    if monitors is None:
      monitors = tuple()
      need_return_monitor = False
    attr_monitors: Dict[str, Tensor] = {}
    state_monitors: Dict[str, Tensor] = {}
    for key in monitors:
      splits = key.split('.')
      if len(splits) != 2:
        raise ValueError(f'Every term in "monitors" must be (node.item), '
                         f'while we got {key}')
      if splits[0] not in self.implicit_nodes:
        raise ValueError(f'Cannot found the node {splits[0]}, this network '
                         f'only has {list(self.implicit_nodes.keys())}.')

      if splits[1] not in NODE_STATES:  # attribute monitor
        if not hasattr(self.implicit_nodes[splits[0]], splits[1]):
          raise UnsupportedError(f'Each node can monitor its states (including {NODE_STATES}), '
                                 f'or its attribute. While {splits[1]} is neither the state nor '
                                 f'the attribute of node {splits[0]}.')
        else:
          attr_monitors[key] = getattr(self.implicit_nodes[splits[0]], splits[1])
      else:  # state monitor
        if splits[1] == 'state':
          assert self.implicit_nodes[splits[0]].state is not None, (f'{splits[0]} has no state, while '
                                                                    f'the user try to monitor it.')
        state_monitors[key] = None

      if not isinstance(key, str):
        raise ValueError(f'"extra_returns" must be a sequence of string, '
                         f'while we got {type(key)}')
      splits = key.split('.')
      if len(splits) != 2:
        raise ValueError(f'Every term in "monitors" must be (node.item), '
                         f'while we got {key}')
      if splits[0] != self.name:
        raise ValueError(f"Cannot found the node {splits[0]}, this name of "
                         f"this node is {self.name}.")
      if splits[1] not in NODE_STATES:  # monitor attributes
        if not hasattr(self, key):
          raise UnsupportedError(f'Each node can monitor its states (including {NODE_STATES}), '
                                 f'or its attribute. While {key} is neither the state nor '
                                 f'the attribute of node \n\n{self}.')
        else:
          attr_monitors[key] = getattr(self, key)
      else:  # monitor states
        if splits[1] == 'state':
          if self.state is None:
            raise ValueError(f'{self} \n\nhas no state, while '
                             f'the user try to monitor its state.')
        state_monitors[key] = None

    # checking
    ff, fb = self._check_inputs(ff, fb=fb)

    # monitoring
    if f'{self.name}.inputs' in state_monitors:
      state_monitors[f'{self.name}.inputs'] = ff
    if f'{self.name}.feedbacks' in state_monitors:
      state_monitors[f'{self.name}.feedbacks'] = fb

    # forward pass
    output = self.forward(ff, fb, **kwargs)

    # monitoring
    if f'{self.name}.output' in state_monitors:
      state_monitors[f'{self.name}.output'] = output
    if f'{self.name}.state' in state_monitors:
      state_monitors[f'{self.name}.state'] = self.state
    attr_monitors.update(state_monitors)

    # outputs
    if need_return_monitor:
      return output, attr_monitors
    else:
      return output

  def forward(self, ff, fb=None, **shared_kwargs):
    """The feedforward computation function of a node.

    Parameters
    ----------
    ff: tensor, dict, sequence
      The feedforward inputs.
    fb: optional, tensor, dict, sequence
      The feedback inputs.
    **shared_kwargs
      Other parameters.

    Returns
    -------
    Tensor
      A output tensor value.
    """
    raise NotImplementedError

  def feedback(self, ff_output, **shared_kwargs):
    """The feedback computation function of a node.

    Parameters
    ----------
    ff_output: JaxArray
      The feedforward output when calling ``forward()`` function.
    **shared_kwargs
      Other global parameters.

    Returns
    -------
    Tensor
      A feedback output tensor value.
    """
    return ff_output

  @not_implemented
  def offline_fit(self, targets, ffs, fbs=None):
    """Offline training interface."""
    raise ValueError(f'This node \n\n{self} \n\ndoes not support offline training.')

  @not_implemented
  def online_init(self):
    """Online training initialization interface."""
    raise ValueError(f'This node \n\n{self} \n\ndoes not support online training.')

  @not_implemented
  def online_fit(self, target, ff, fb=None):
    """Online training fitting interface."""
    raise ValueError(f'This node \n\n{self} \n\ndoes not support online training.')


class RecurrentNode(Node):
  """
  Basic class for recurrent node.

  The supports for the recurrent node are:

  - Self-connection when using ``plot_node_graph()`` function
  - Set trainable state with ``state_trainable=True``.
  """

  def __init__(self,
               name: Optional[str] = None,
               input_shape: Optional[Union[Sequence[int], int]] = None,
               trainable: bool = False,
               state_trainable: bool = False):
    self._state_trainable = state_trainable
    self._train_state = None
    super(RecurrentNode, self).__init__(name=name,
                                        input_shape=input_shape,
                                        trainable=trainable)

  @property
  def state_trainable(self) -> bool:
    """Returns if the Node can be trained."""
    return self._state_trainable

  @property
  def train_state(self):
    return self._train_state

  def set_state(self, state):
    """Safely set the state of the node.

    This method allows the maximum flexibility to change the
    node state. It can set a new data (same shape, same dtype)
    to the state. It can also set the data with another batch size.

    We highly recommend the user to use this function.
    """
    if self.state is None:
      if self.output_shape is not None:
        check_shape_except_batch(self.output_shape, state.shape)
      self._state = bm.Variable(state) if not isinstance(state, bm.Variable) else state
      if self.state_trainable:
        self._train_state = bm.TrainVar(self._state[0])  # get the first elements as the initial state
        self._state[:] = self._train_state  # set all batch states the same
    else:
      check_shape_except_batch(self.state.shape, state.shape)
      if self.state.dtype != state.dtype:
        raise MathError('Cannot set the state, because the dtype is not consistent: '
                        f'{self.state.dtype} != {state.dtype}')
      if self.state_trainable:
        # get the batch size information
        state = bm.repeat(bm.expand_dims(self.train_state, axis=0), state.shape[0], axis=0)
        # set the state
        self.state._value = bm.as_device_array(state)
      else:
        self.state._value = bm.as_device_array(state)


class Network(Node):
  """Basic Network class for neural network building in BrainPy."""

  data_pass = MultipleData('sequence')

  def __init__(self,
               nodes: Optional[Sequence[Node]] = None,
               ff_edges: Optional[Sequence[Tuple[Node]]] = None,
               fb_edges: Optional[Sequence[Tuple[Node]]] = None,
               **kwargs):
    super(Network, self).__init__(**kwargs)
    # nodes (with tuple/list format)
    if nodes is None:
      self._nodes = tuple()
    else:
      self._nodes = tuple(nodes)
    # feedforward edges
    if ff_edges is None:
      self._ff_edges = tuple()
    else:
      self._ff_edges = tuple(ff_edges)
    # feedback edges
    if fb_edges is None:
      self._fb_edges = tuple()
    else:
      self._fb_edges = tuple(fb_edges)
    # initialize network
    self._network_init()

  def _network_init(self):
    # detect input and output nodes
    self._entry_nodes, self._exit_nodes = find_entries_and_exits(self._nodes, self._ff_edges)
    # build feedforward connection graph
    self._ff_senders, self._ff_receivers = find_senders_and_receivers(self._ff_edges)
    # build feedback connection graph
    self._fb_senders, self._fb_receivers = find_senders_and_receivers(self._fb_edges)
    # register nodes for brainpy.Base object
    self.implicit_nodes = Collector({n.name: n for n in self._nodes})
    # set initialization states
    self._is_initialized = False
    self._is_fb_initialized = False

  def __repr__(self):
    return f"{type(self).__name__}({', '.join([n.name for n in self._nodes])})"

  def __irshift__(self, other):  # "self >>= other"
    global operations
    if operations is None: from . import operations
    return operations.ff_connect(self, other, inplace=True)

  def __ilshift__(self, other):  # "self <<= other"
    global operations
    if operations is None: from . import operations
    return operations.fb_connect(self, other, inplace=True)

  def __iand__(self, other):
    global operations
    if operations is None: from . import operations
    return operations.merge(self, other, inplace=True)

  def __getitem__(self, item):
    if isinstance(item, str):
      return self.get_node(item)
    else:
      global operations
      if operations is None: from . import operations
      return operations.select(self, item)

  def get_node(self, name):
    if name in self.implicit_nodes:
      return self.implicit_nodes[name]
    else:
      raise KeyError(f"No node named '{name}' found in model {self.name}.")

  def nodes(self, method='absolute', level=1, include_self=False):
    return super(Node, self).nodes(method=method, level=level, include_self=include_self)

  @property
  def trainable(self) -> bool:
    """Returns True if at least one Node in the Model is trainable."""
    return any([n.trainable for n in self.lnodes])

  @trainable.setter
  def trainable(self, value: bool):
    """Freeze or unfreeze trainable Nodes in the Model."""
    for node in [n for n in self.lnodes]:
      node.trainable = value

  @property
  def lnodes(self) -> Tuple[Node]:
    return self._nodes

  @property
  def ff_edges(self) -> Sequence[Tuple[Node]]:
    return self._ff_edges

  @property
  def fb_edges(self) -> Sequence[Tuple[Node]]:
    return self._fb_edges

  @property
  def entry_nodes(self) -> Sequence[Node]:
    """First Nodes in the graph held by the Model."""
    return self._entry_nodes

  @property
  def exit_nodes(self) -> Sequence[Node]:
    """Last Nodes in the graph held by the Model."""
    return self._exit_nodes

  @property
  def feedback_nodes(self) -> Sequence[Node]:
    """Nodes which project feedback connections."""
    return tuple(self._fb_receivers.keys())

  @property
  def nodes_has_feedback(self) -> Sequence[Node]:
    """Nodes which receive feedback connections."""
    return tuple(self._fb_senders.keys())

  @property
  def ff_senders(self) -> Dict:
    """Nodes which project feedforward connections."""
    return self._ff_senders

  @property
  def ff_receivers(self) -> Dict:
    """Nodes which receive feedforward connections."""
    return self._ff_receivers

  @property
  def fb_senders(self) -> Dict:
    """Nodes which project feedback connections."""
    return self._fb_senders

  @property
  def fb_receivers(self) -> Dict:
    """Nodes which receive feedback connections."""
    return self._fb_receivers

  def update_graph(self,
                   new_nodes: Sequence[Node],
                   new_ff_edges: Sequence[Tuple[Node, Node]],
                   new_fb_edges: Sequence[Tuple[Node, Node]] = None) -> "Network":
    """Update current Model's with new nodes and edges, inplace (a copy
    is not performed).

    Parameters
    ----------
    new_nodes : list of Node
        New nodes.
    new_ff_edges : list of (Node, Node)
        New feedforward edges between nodes.
    new_fb_edges : list of (Node, Node)
        New feedback edges between nodes.

    Returns
    -------
    Network
        The updated network.
    """
    if new_fb_edges is None: new_fb_edges = tuple()
    self._nodes = tuple(set(new_nodes) | set(self.lnodes))
    self._ff_edges = tuple(set(new_ff_edges) | set(self.ff_edges))
    self._fb_edges = tuple(set(new_fb_edges) | set(self.fb_edges))
    # detect cycles in the graph flow
    if detect_cycle(self._nodes, self._ff_edges):
      raise ValueError('We detect cycles in feedforward connections. '
                       'Maybe you should replace some connection with '
                       'as feedback ones.')
    if detect_cycle(self._nodes, self._fb_edges):
      raise ValueError('We detect cycles in feedback connections. ')
    self._network_init()
    return self

  def replace_graph(self,
                    nodes: Sequence[Node],
                    ff_edges: Sequence[Tuple[Node, ...]],
                    fb_edges: Sequence[Tuple[Node, ...]] = None) -> "Network":
    if fb_edges is None: fb_edges = tuple()

    # assign nodes and edges
    self._nodes = tuple(nodes)
    self._ff_edges = tuple(ff_edges)
    self._fb_edges = tuple(fb_edges)
    self._network_init()
    return self

  def set_output_shape(self, shape: Dict[str, Sequence[int]]):
    # check shape
    if not isinstance(shape, dict):
      raise ValueError(f'Must be a dict of <node name, shape>, but got {type(shape)}: {shape}')
    for key, val in shape.items():
      if not isinstance(val, (tuple, list)):
        raise ValueError(f'Must be a sequence of int, but got {val} for key "{key}"')
      # for s in val:
      #   if not (isinstance(s, int) or (s is None)):
      #     raise ValueError(f'Must be a sequence of int, but got {val}')

    if not self._is_ff_initialized:
      if len(self.exit_nodes) == 1:
        self._output_shape = tuple(shape.values())[0]
      else:
        self._output_shape = shape
    else:
      for val in shape.values():
        check_shape_except_batch(val, self.output_shape)

  def init_ff_conn(self):
    """Initialize the feedforward connections of the network.
    This function will be called only once."""
    # input shapes of entry nodes
    for node in self.entry_nodes:
      # set ff shapes
      if node.feedforward_shapes is None:
        if self.feedforward_shapes is None:
          raise ValueError('Cannot find the input size. '
                           'Cannot initialize the network.')
        else:
          node.set_feedforward_shapes({node.name: self._feedforward_shapes[node.name]})
      # set fb shapes
      if node in self.fb_senders:
        fb_shapes = {node: node.output_shape for node in self.fb_senders.get(node, [])}
        if None not in fb_shapes.values():
          node.set_feedback_shapes(fb_shapes)
      # init ff conn
      node._init_ff_conn()

    # initialize the data
    children_queue = []
    ff_senders, _ = find_senders_and_receivers(self.ff_edges)

    # init shapes of other nodes
    for node in self._entry_nodes:
      for child in self.ff_receivers.get(node, []):
        ff_senders[child].remove(node)
        if len(ff_senders.get(child, [])) == 0:
          children_queue.append(child)
    while len(children_queue):
      node = children_queue.pop(0)
      # set ff shapes
      parent_sizes = {p: p.output_shape for p in self.ff_senders.get(node, [])}
      node.set_feedforward_shapes(parent_sizes)
      if node in self.fb_senders:
        # set fb shapes
        fb_shapes = {node: node.output_shape for node in self.fb_senders.get(node, [])}
        if None not in fb_shapes.values():
          node.set_feedback_shapes(fb_shapes)
      # init ff conn
      node._init_ff_conn()
      # append children
      for child in self.ff_receivers.get(node, []):
        ff_senders[child].remove(node)
        if len(ff_senders.get(child, [])) == 0:
          children_queue.append(child)

    # set output shape
    out_sizes = {node: node.output_shape for node in self.exit_nodes}
    self.set_output_shape(out_sizes)

  def init_fb_conn(self):
    """Initialize the feedback connections of the network.
    This function will be called only once."""
    for receiver, senders in self.fb_senders.items():
      fb_sizes = {node: node.output_shape for node in senders}
      if None in fb_sizes.values():
        none_size_nodes = [repr(n) for n, v in fb_sizes.items() if v is None]
        none_size_nodes = "\n".join(none_size_nodes)
        raise ValueError(f'Output shapes of nodes \n\n'
                         f'{none_size_nodes}\n\n'
                         f'have not been initialized, '
                         f'leading us cannot initialize the '
                         f'feedback connection of node \n\n'
                         f'{receiver}')
      receiver.set_feedback_shapes(fb_sizes)
      receiver._init_fb_conn()

  def _init_state(self, num_batch=1):
    """Initialize the states of all children nodes.
    This function can be called multiple times."""
    for node in self.lnodes:
      node._init_state(num_batch)
    self._is_state_initialized = True

  def _init_fb_output(self, num_batch=1):
    """Initialize the node feedback state.

    This function can be called multiple times. However,
    it is only triggered when the node has feedback connections.
    """
    for node in self.feedback_nodes:
      node._init_fb_output(num_batch)
    self._is_fb_state_initialized = True

  def initialize(self, num_batch: int = 1):
    """
    Initialize the whole network. This function must be called before applying JIT.

    This function is useful, because it is independent of the __call__ function.
    We can use this function before we apply JIT to __call__ function.
    """

    # set feedforward shapes
    if not self._is_ff_initialized:
      # check input and output nodes
      if len(self.entry_nodes) <= 0:
        raise ValueError(f"We found this network \n\n"
                         f"{self} "
                         f"\n\nhas no input nodes.")
      if len(self.exit_nodes) <= 0:
        raise ValueError(f"We found this network \n\n"
                         f"{self} "
                         f"\n\nhas no output nodes.")

      # check whether it has a feedforward path for each feedback pair
      ff_edges = [(a.name, b.name) for a, b in self.ff_edges]
      for node, receiver in self.fb_edges:
        if not detect_path(receiver.name, node.name, ff_edges):
          raise ValueError(f'Cannot build a feedback connection from '
                           f'\n\n{node} \n\n'
                           f'to '
                           f'\n\n{receiver} \n\n'
                           f'because there is no feedforward path between them. \n'
                           f'Maybe you should use "ff_connect" first to establish a '
                           f'feedforward connection between them. ')

      # feedforward checking
      in_sizes = dict()
      for node in self.entry_nodes:
        if node.feedforward_shapes is None:
          raise ValueError('Cannot initialize this node, because we detect '
                           '"feedforward_shapes" is None. '
                           'Maybe you need a brainpy.nn.Input instance '
                           'to instruct the input size.')
        in_sizes.update(node._feedforward_shapes)
      self.set_feedforward_shapes(in_sizes)

    # feedforward initialization
    if self.feedforward_shapes is None:
      raise ValueError('Cannot initialize this node, because we detect '
                       'both "feedforward_shapes" is None. ')
    check_integer(num_batch, 'num_batch', min_bound=1, allow_none=False)
    self._init_ff_conn()

    # initialize state
    self._init_state(num_batch)

    # set feedback shapes
    if not self._is_fb_initialized:
      if len(self.fb_senders) > 0:
        fb_sizes = dict()
        for sender in self.fb_senders.keys():
          fb_sizes[sender] = sender.output_shape
        self.set_feedback_shapes(fb_sizes)

    # feedback initialization
    if self.feedback_shapes is not None:
      self._init_fb_conn()

    # initialize feedback state
    self._init_fb_output(num_batch)

  def _check_inputs(self, ff, fb=None):
    # feedforward inputs
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      ff = {self.entry_nodes[0].name: ff}
    if not isinstance(ff, dict):
      raise ValueError(f'ff must be a dict or a tensor, got {type(ff)}: {ff}')
    if len(self.entry_nodes) != len(ff):
      raise ValueError(f'This network has {len(self.entry_nodes)} '
                       f'entry nodes. While only {len(ff)} input '
                       f'data are given.')
    for n in self.entry_nodes:
      if n.name not in ff:
        raise ValueError(f'Cannot find the input of the node: \n{n}')
    for k, size in self._feedforward_shapes.items():
      if k not in ff:
        raise ValueError(f"The required key {k} is not provided in feedforward inputs.")
      if not check_shape_except_batch(size, ff[k].shape, mode='bool'):
        raise ValueError(f'Input size {ff[k].shape} is not consistent with '
                         f'the input size {size}')

    # feedback inputs
    if fb is not None:
      if isinstance(fb, (bm.ndarray, jnp.ndarray)):
        fb = {self.feedback_nodes[0]: fb}
      if not isinstance(fb, dict):
        raise ValueError(f'fb must be a dict or a tensor, '
                         f'got {type(fb)}: {fb}')
      if len(self.feedback_nodes) != len(fb):
        raise ValueError(f'This network has {len(self.feedback_nodes)} '
                         f'feedback nodes. While only {len(ff)} '
                         f'feedback data are given.')
      for n in self.feedback_nodes:
        if n.name not in fb:
          raise ValueError(f'Cannot find the feedback data from the node {n}')
      # check feedback consistency
      for k, size in self._feedback_shapes.items():
        if k not in fb:
          raise ValueError(f"The required key {k} is not provided in feedback inputs.")
        check_shape_except_batch(size, fb[k].shape)

    # data transformation
    ff = self.data_pass.filter(ff)
    fb = self.data_pass.filter(fb)
    return ff, fb

  def _call(self,
            ff: Union[Tensor, Dict[Any, Tensor]],
            fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
            forced_states: Optional[Dict[str, Tensor]] = None,
            forced_feedbacks: Optional[Dict[str, Tensor]] = None,
            monitors: Optional[Sequence[str]] = None,
            **kwargs):
    # initialization
    if not self.is_initialized:
      raise ValueError('Please initialize the Network first by calling "initialize()" function.')

    # initialize the forced data
    if forced_feedbacks is None: forced_feedbacks = dict()
    check_dict_data(forced_feedbacks, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    if forced_states is None: forced_states = dict()
    check_dict_data(forced_states, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    # initialize the monitors
    need_return_monitor = True
    if monitors is None:
      monitors = tuple()
      need_return_monitor = False
    attr_monitors: Dict[str, Tensor] = {}
    state_monitors: Dict[str, Tensor] = {}
    for key in monitors:
      if not isinstance(key, str):
        raise ValueError(f'"extra_returns" must be a sequence of string, '
                         f'while we got {type(key)}')
      splits = key.split('.')
      if len(splits) != 2:
        raise ValueError(f'Every term in "extra_returns" must be (node.item), '
                         f'while we got {key}')
      if splits[0] not in self.implicit_nodes:
        raise ValueError(f'Cannot found the node {splits[0]}, this network '
                         f'only has {list(self.implicit_nodes.keys())}.')

      if splits[1] not in NODE_STATES:  # attribute monitor
        if not hasattr(self.implicit_nodes[splits[0]], splits[1]):
          raise UnsupportedError(f'Each node can monitor its states (including {NODE_STATES}), '
                                 f'or its attribute. While {splits[1]} is neither the state nor '
                                 f'the attribute of node {splits[0]}.')
        else:
          attr_monitors[key] = getattr(self.implicit_nodes[splits[0]], splits[1])
      else:  # state monitor
        if splits[1] == 'state':
          assert self.implicit_nodes[splits[0]].state is not None, (f'{splits[0]} has no state, while '
                                                                    f'the user try to monitor it.')
        state_monitors[key] = None
    # calling the computation core
    ff, fb = self._check_inputs(ff, fb=fb)
    output, state_monitors = self.forward(ff, fb, forced_states, forced_feedbacks, state_monitors, **kwargs)
    if need_return_monitor:
      attr_monitors.update(state_monitors)
      return output, attr_monitors
    else:
      return output

  def _call_a_node(self, node, ff, fb, monitors, forced_states,
                   parent_outputs, children_queue, ff_senders,
                   **shared_kwargs):
    ff = node.data_pass.filter(ff)
    if f'{node.name}.inputs' in monitors:
      monitors[f'{node.name}.inputs'] = ff
    # get the output results
    if len(fb):
      fb = node.data_pass.filter(fb)
      if f'{node.name}.feedbacks' in monitors:
        monitors[f'{node.name}.feedbacks'] = fb
      parent_outputs[node] = node.forward(ff, fb, **shared_kwargs)
    else:
      parent_outputs[node] = node.forward(ff, **shared_kwargs)
    # get the feedback state
    if node in self.fb_receivers:
      node.set_fb_output(node.feedback(parent_outputs[node], **shared_kwargs))
    # forced state
    if node.name in forced_states:
      node.state.value = forced_states[node.name]
    # monitor the values
    if f'{node.name}.state' in monitors:
      monitors[f'{node.name}.state'] = node.state.value
    if f'{node.name}.output' in monitors:
      monitors[f'{node.name}.output'] = parent_outputs[node]
    # append children nodes
    for child in self.ff_receivers.get(node, []):
      ff_senders[child].remove(node)
      if len(ff_senders.get(child, [])) == 0:
        children_queue.append(child)

  def forward(self,
              ff,
              fb=None,
              forced_states: Dict[str, Tensor] = None,
              forced_feedbacks: Dict[str, Tensor] = None,
              monitors: Dict = None,
              **shared_kwargs):
    """The main computation function of a network.

    Parameters
    ----------
    ff: dict, sequence
      The feedforward inputs.
    fb: optional, dict, sequence
      The feedback inputs.
    forced_states: optional, dict
      The fixed state for the nodes in the network.
    forced_feedbacks: optional, dict
      The fixed feedback for the nodes in the network.
    monitors: optional, sequence
      Can be used to monitor the state or the attribute of a node in the network.
    **shared_kwargs
      Other parameters which will be parsed into every node.

    Returns
    -------
    Tensor
      A output tensor value, or a dict of output tensors.
    """
    all_nodes = set([n.name for n in self.lnodes])
    runned_nodes = set()
    output_nodes = set([n.name for n in self.exit_nodes])

    # initialize the feedback
    if forced_feedbacks is None: forced_feedbacks = dict()
    if monitors is None: monitors = dict()

    # initialize the data
    children_queue = []
    ff_senders, _ = find_senders_and_receivers(self.ff_edges)

    # initialize the parent output data
    parent_outputs = {}
    for i, node in enumerate(self._entry_nodes):
      ff_ = {node.name: ff[i]}
      fb_ = {p: (forced_feedbacks[p.name] if (p.name in forced_feedbacks) else p.fb_output)
             for p in self.fb_senders.get(node, [])}
      self._call_a_node(node, ff_, fb_, monitors, forced_states,
                        parent_outputs, children_queue, ff_senders,
                        **shared_kwargs)
      runned_nodes.add(node.name)

    # run the model
    while len(children_queue):
      node = children_queue.pop(0)
      # get feedforward and feedback inputs
      ff = {p: parent_outputs[p] for p in self.ff_senders.get(node, [])}
      fb = {p: (forced_feedbacks[p.name] if (p.name in forced_feedbacks) else p.fb_output)
            for p in self.fb_senders.get(node, [])}
      # call the node
      self._call_a_node(node, ff, fb, monitors, forced_states,
                        parent_outputs, children_queue, ff_senders,
                        **shared_kwargs)

      # - remove unnecessary parent outputs - #
      needed_parents = []
      runned_nodes.add(node.name)
      for child in (all_nodes - runned_nodes):
        for parent in self.ff_senders[self.implicit_nodes[child]]:
          needed_parents.append(parent.name)
      for parent in list(parent_outputs.keys()):
        _name = parent.name
        if _name not in needed_parents and _name not in output_nodes:
          parent_outputs.pop(parent)

    # returns
    if len(self.exit_nodes) > 1:
      state = {n.name: parent_outputs[n] for n in self.exit_nodes}
    else:
      state = parent_outputs[self.exit_nodes[0]]
    return state, monitors

  def plot_node_graph(self,
                      fig_size: tuple = (10, 10),
                      node_size: int = 2000,
                      arrow_size: int = 20,
                      layout='shell_layout'):
    """Plot the node graph based on NetworkX package

    Parameters
    ----------
    fig_size: tuple, default to (10, 10)
      The size of the figure
    node_size: int, default to 2000
      The size of the node
    arrow_size:int, default to 20
      The size of the arrow
    layout: str
      The graph layout. The supported layouts are:

      - "shell_layout"
      - "multipartite_layout"
      - "spring_layout"
      - "spiral_layout"
      - "spectral_layout"
      - "random_layout"
      - "planar_layout"
      - "kamada_kawai_layout"
      - "circular_layout"
    """
    try:
      import networkx as nx
    except (ModuleNotFoundError, ImportError):
      raise PackageMissingError('The node graph plotting currently need package "networkx". '
                                'But it can not be imported. ')
    try:
      import matplotlib.pyplot as plt
      from matplotlib.lines import Line2D
    except (ModuleNotFoundError, ImportError):
      raise PackageMissingError('The node graph plotting currently need package "matplotlib". '
                                'But it can not be imported. ')

    nodes_trainable = []
    nodes_untrainable = []
    for node in self.lnodes:
      if node.trainable:
        nodes_trainable.append(node.name)
      else:
        nodes_untrainable.append(node.name)

    ff_edges = []
    fb_edges = []
    rec_edges = []
    for edge in self.ff_edges:
      ff_edges.append((edge[0].name, edge[1].name))
    for edge in self.fb_edges:
      fb_edges.append((edge[0].name, edge[1].name))
    for node in self.lnodes:
      if isinstance(node, RecurrentNode):
        rec_edges.append((node.name, node.name))

    trainable_color = 'orange'
    untrainable_color = 'skyblue'
    ff_color = 'green'
    fb_color = 'red'
    rec_color = 'purple'
    G = nx.DiGraph()
    mid_nodes = list(set(self.lnodes) - set(self.entry_nodes) - set(self.exit_nodes))
    mid_nodes.sort(key=lambda x: x.name)
    index = 0
    for node in list(self.entry_nodes) + mid_nodes + list(self.exit_nodes):
      index = index + 1
      G.add_node(node.name, subset=index)
    G.add_edges_from(ff_edges)
    G.add_edges_from(fb_edges)
    G.add_edges_from(rec_edges)

    if layout not in SUPPORTED_LAYOUTS:
      raise UnsupportedError(f'Only support layouts: {SUPPORTED_LAYOUTS}')
    layout = getattr(nx, layout)(G)

    plt.figure(figsize=fig_size)
    nx.draw_networkx_nodes(G, pos=layout,
                           nodelist=nodes_trainable,
                           node_color=trainable_color,
                           node_size=node_size)
    nx.draw_networkx_nodes(G, pos=layout,
                           nodelist=nodes_untrainable,
                           node_color=untrainable_color,
                           node_size=node_size)

    ff_conn_style = "arc3,rad=0."
    nx.draw_networkx_edges(G, pos=layout,
                           edgelist=ff_edges,
                           edge_color=ff_color,
                           connectionstyle=ff_conn_style,
                           arrowsize=arrow_size,
                           node_size=node_size)
    fb_conn_style = "arc3,rad=0.3"
    nx.draw_networkx_edges(G, pos=layout,
                           edgelist=fb_edges,
                           edge_color=fb_color,
                           connectionstyle=fb_conn_style,
                           arrowsize=arrow_size,
                           node_size=node_size)
    rec_conn_style = "arc3,rad=-0.3"
    nx.draw_networkx_edges(G, pos=layout,
                           edgelist=rec_edges,
                           edge_color=rec_color,
                           arrowsize=arrow_size,
                           connectionstyle=rec_conn_style,
                           node_size=node_size,
                           node_shape='s')

    nx.draw_networkx_labels(G, pos=layout)
    proxie = []
    labels = []
    if len(nodes_trainable):
      proxie.append(Line2D([], [], color='white', marker='o',
                           markerfacecolor=trainable_color))
      labels.append('Trainable')
    if len(nodes_untrainable):
      proxie.append(Line2D([], [], color='white', marker='o',
                           markerfacecolor=untrainable_color))
      labels.append('Nontrainable')
    if len(ff_edges):
      proxie.append(Line2D([], [], color=ff_color, linewidth=2))
      labels.append('Feedforward')
    if len(fb_edges):
      proxie.append(Line2D([], [], color=fb_color, linewidth=2))
      labels.append('Feedback')
    if len(rec_edges):
      proxie.append(Line2D([], [], color=rec_color, linewidth=2))
      labels.append('Recurrent')

    plt.legend(proxie, labels, scatterpoints=1, markerscale=2, loc='best')
    plt.tight_layout()
    plt.show()


class FrozenNetwork(Network):
  """A FrozenNetwork is a Network that can not be linked to other nodes or networks."""

  def update_graph(self, new_nodes, new_ff_edges, new_fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")

  def replace_graph(self, nodes, ff_edges, fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")


class Sequential(Network):
  pass

# def _process_params(G, center, dim):
#     # Some boilerplate code.
#     import numpy as np
#
#     if not isinstance(G, nx.Graph):
#         empty_graph = nx.Graph()
#         empty_graph.add_nodes_from(G)
#         G = empty_graph
#
#     if center is None:
#         center = np.zeros(dim)
#     else:
#         center = np.asarray(center)
#
#     if len(center) != dim:
#         msg = "length of center coordinates must match dimension of layout"
#         raise ValueError(msg)
#
#     return G, center
#
#
# def multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None):
#     import numpy as np
#
#     if align not in ("vertical", "horizontal"):
#       msg = "align must be either vertical or horizontal."
#       raise ValueError(msg)
#
#     G, center = _process_params(G, center=center, dim=2)
#     if len(G) == 0:
#       return {}
#
#     layers = {}
#     for v, data in G.nodes(data=True):
#       try:
#         layer = data[subset_key]
#       except KeyError:
#         msg = "all nodes must have subset_key (default='subset') as data"
#         raise ValueError(msg)
#       layers[layer] = [v] + layers.get(layer, [])
#
#     pos = None
#     nodes = []
#
#     width = len(layers)
#     for i, layer in layers.items():
#       height = len(layer)
#       xs = np.repeat(i, height)
#       ys = np.arange(0, height, dtype=float)
#       offset = ((width - 1) / 2, (height - 1) / 2)
#       layer_pos = np.column_stack([xs, ys]) - offset
#       if pos is None:
#         pos = layer_pos
#       else:
#         pos = np.concatenate([pos, layer_pos])
#       nodes.extend(layer)
#     pos = rescale_layout(pos, scale=scale) + center
#     if align == "horizontal":
#       pos = np.flip(pos, 1)
#     pos = dict(zip(nodes, pos))
#     return pos
#
#
# def rescale_layout(pos, scale=1):
#     """Returns scaled position array to (-scale, scale) in all axes.
#
#     The function acts on NumPy arrays which hold position information.
#     Each position is one row of the array. The dimension of the space
#     equals the number of columns. Each coordinate in one column.
#
#     To rescale, the mean (center) is subtracted from each axis separately.
#     Then all values are scaled so that the largest magnitude value
#     from all axes equals `scale` (thus, the aspect ratio is preserved).
#     The resulting NumPy Array is returned (order of rows unchanged).
#
#     Parameters
#     ----------
#     pos : numpy array
#         positions to be scaled. Each row is a position.
#
#     scale : number (default: 1)
#         The size of the resulting extent in all directions.
#
#     Returns
#     -------
#     pos : numpy array
#         scaled positions. Each row is a position.
#
#     See Also
#     --------
#     rescale_layout_dict
#     """
#     # Find max length over all dimensions
#     lim = 0  # max coordinate for all axes
#     for i in range(pos.shape[1]):
#         pos[:, i] -= pos[:, i].mean()
#         lim = max(abs(pos[:, i]).max(), lim)
#     # rescale to (-scale, scale) in all directions, preserves aspect
#     if lim > 0:
#         for i in range(pos.shape[1]):
#             pos[:, i] *= scale / lim
#     return pos
