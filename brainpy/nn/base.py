# -*- coding: utf-8 -*-

from copy import copy, deepcopy
from typing import Dict, Sequence, Tuple, Union, Optional, Any

import jax.numpy as jnp

from brainpy import tools, math as bm
from brainpy.base import Base, Collector
from brainpy.errors import UnsupportedError
from brainpy.nn.constants import (PASS_SEQUENCE,
                                  DATA_PASS_FUNC,
                                  DATA_PASS_TYPES)
from brainpy.nn.graph_flow import (find_senders_and_receivers,
                                   find_entries_and_exits,
                                   detect_cycle,
                                   detect_path)
from brainpy.tools.checking import check_dict_data
from brainpy.types import Tensor

operations = None

__all__ = [
  'Node', 'Network', 'FrozenNetwork',
]

NODE_STATES = ['inputs', 'feedbacks', 'state', 'output']


class Node(Base):
  """Basic Node class for neural network building in BrainPy."""

  """Support multiple types of data pass, including "PASS_SEQUENCE" (by default), 
  "PASS_NAME_DICT", "PASS_NODE_DICT" and user-customized type which registered 
  by ``brainpy.nn.register_data_pass_type()`` function. 
  
  This setting will change the feedforward/feedback input data which pass into 
  the "call()" function and the sizes of the feedforward/feedback input data.
  """
  data_pass_type = PASS_SEQUENCE

  def __init__(self,
               name: Optional[str] = None,
               input_shape: Optional[Union[Sequence[int], int]] = None,
               trainable: bool = False):  # initialize parameters
    super(Node, self).__init__(name=name)

    self._input_shapes = None  # input shapes
    self._output_shape = None  # output size
    self._feedback_shapes = None  # feedback shapes
    if input_shape is not None:
      self._input_shapes = {self.name: tools.to_size(input_shape)}

    self._is_initialized = False
    self._is_fb_initialized = False
    self._fb_states = {}  # used to store the states of feedback nodes
    self._trainable = trainable
    self._state = None  # the state of the current node

    # data pass function
    if self.data_pass_type not in DATA_PASS_FUNC:
      raise ValueError(f'Unsupported data pass type {self.data_pass_type}. '
                       f'Only support {DATA_PASS_TYPES}')
    self.data_pass_func = DATA_PASS_FUNC[self.data_pass_type]

    self.check_shape_consistency = True

  def __repr__(self):
    name = type(self).__name__
    return (f"{name}(name={self.name}, output={self.output_shape}, \n"
            f"{' ' * (len(name) + 1)}inputs={self.input_shapes}, feedbacks={self.feedback_shapes}, \n"
            f"{' ' * (len(name) + 1)}state={self._state})")

  def __call__(self, *args, **kwargs) -> Tensor:
    """The main computation function of a Node.

    Parameters
    ----------
    ff: dict, sequence, tensor
      The feedforward inputs.
    fb: optional, dict, sequence, tensor
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

  def __getitem__(self, item):  # "[:10]"
    if isinstance(item, str):
      raise ValueError('Node only supports slice, not retrieve by the name.')
    else:
      global operations
      if operations is None: from . import operations
      return operations.select(self, item)

  @property
  def state(self) -> Optional[Tensor]:
    """Node current internal state."""
    if self.is_initialized:
      return self._state
    return None

  @state.setter
  def state(self, value: Tensor):
    if self.state is None:
      if self.output_shape is not None:
        assert self.output_shape == value.shape
      self._state = bm.Variable(value) if not isinstance(value, bm.Variable) else value
    else:
      self._state.value = value

  @property
  def feedback_states(self) -> Dict[str, Tensor]:
    return self._fb_states

  @property
  def trainable(self) -> bool:
    """Returns if the Node can be trained."""
    return self._trainable

  @property
  def is_initialized(self) -> bool:
    return self._is_initialized

  @is_initialized.setter
  def is_initialized(self, value: bool):
    self._is_initialized = value

  @property
  def is_fb_initialized(self) -> bool:
    return self._is_fb_initialized

  @is_fb_initialized.setter
  def is_fb_initialized(self, value: bool):
    self._is_fb_initialized = value

  @trainable.setter
  def trainable(self, value: bool):
    """Freeze or unfreeze the Node. If set to False,
    learning is stopped."""
    assert isinstance(value, bool), 'Must be a boolean.'
    self._trainable = value

  @property
  def input_shapes(self):
    """Input data size."""
    return self.data_pass_func(self._input_shapes)

  @input_shapes.setter
  def input_shapes(self, size):
    self.set_input_shapes(size)

  @property
  def output_shape(self) -> Optional[Tuple[int]]:
    """Output data size."""
    return self._output_shape

  @output_shape.setter
  def output_shape(self, size):
    self.set_output_shape(size)

  @property
  def feedback_shapes(self):
    """Output data size."""
    return self.data_pass_func(self._feedback_shapes)

  @feedback_shapes.setter
  def feedback_shapes(self, size):
    self.set_feedback_shapes(size)

  def set_input_shapes(self, input_shapes: Dict):
    if not self.is_initialized:
      if self.input_shapes is not None:
        for key, size in self._input_shapes.items():
          if key not in input_shapes:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While we do not find it in the given input_shapes")
          if size != input_shapes[key]:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While the give shape is {input_shapes[key]}")
      self._input_shapes = input_shapes
    else:
      raise TypeError(f"Input dimension of {self.name} is immutable after initialization.")

  def set_output_shape(self, shape: Sequence[int]):
    if not self.is_initialized:
      self._output_shape = tuple(shape)
    else:
      raise TypeError(f"Output dimension of {self.name} is immutable after initialization.")

  def set_feedback_shapes(self, fb_shapes: Dict):
    if not self.is_fb_initialized:
      if self.feedback_shapes is not None:
        for key, size in self._input_shapes.items():
          if key not in fb_shapes:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While we do not find it in {fb_shapes}")
          if size != fb_shapes[key]:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While the give shape is {fb_shapes[key]}")
      self._feedback_shapes = fb_shapes
    else:
      raise TypeError(f"Feedback dimension of {self.name} is immutable after initialization.")

  def nodes(self, method='absolute', level=1, include_self=True):
    return super(Node, self).nodes(method=method, level=level, include_self=include_self)

  def vars(self, method='absolute', level=1, include_self=True):
    return super(Node, self).vars(method=method, level=level, include_self=include_self)

  def train_vars(self, method='absolute', level=1, include_self=True):
    return super(Node, self).train_vars(method=method, level=level, include_self=include_self)

  def copy(self, name: str = None, shallow: bool = False):
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

  def reset_state(self, to_state: Tensor = None):
    """A null state vector."""
    if to_state is None:
      if self._state is not None:
        self._state.value = bm.zeros_like(self._state)
    else:
      self._state.value = to_state

  def _ff_init(self):
    if not self.is_initialized:
      self.ff_init()
      self._is_initialized = True

  def _fb_init(self):
    if not self.is_fb_initialized:
      self.fb_init()
      self._is_fb_initialized = True

  def fb_init(self):
    raise ValueError(f'This node \n\n{self} \n\ndoes not support feedback connection.')

  def ff_init(self):
    raise NotImplementedError('Please implement the feedforward initialization.')

  def initialize(self,
                 ff: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
                 fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None):

    # feedforward initialization
    if not self.is_initialized:
      # feedforward data
      if ff is None:
        if self._input_shapes is None:
          raise ValueError('Cannot initialize this node, because we detect '
                           'both "input_shapes"and "ff" inputs are None. ')
        in_sizes = self._input_shapes
      else:
        if isinstance(ff, (bm.ndarray, jnp.ndarray)):
          ff = {self.name: ff}
        assert isinstance(ff, dict), f'"ff" must be a dict or a tensor, got {type(ff)}: {ff}'
        assert self.name in ff, f'Cannot find input for this node {self} when given "ff" {ff}'
        in_sizes = {k: v.shape for k, v in ff.items()}

      # initialize feedforward
      self.set_input_shapes(in_sizes)
      self._ff_init()

    # feedback initialization
    if fb is not None:
      if not self.is_fb_initialized:  # initialize feedback
        assert isinstance(fb, dict), f'"fb" must be a dict, got {type(fb)}'
        fb_sizes = {k: v.shape for k, v in fb.items()}
        self.set_feedback_shapes(fb_sizes)
        self._fb_init()

  def _check_inputs(self, ff, fb=None):
    # feedforward inputs
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      ff = {self.name: ff}
    assert isinstance(ff, dict), f'"ff" must be a dict or a tensor, got {type(ff)}: {ff}'
    assert self.name in ff, f'Cannot find input for this node {self} when given "ff" {ff}'

    if self.check_shape_consistency:
      # check feedforward consistency
      for k, size in self._input_shapes.items():
        assert k in ff, f"The required key {k} is not provided in feedforward inputs."
        assert size == ff[k].shape, (f'Shape does not match the required '
                                     f'input size {size} != {ff[k].shape}')
    # feedback inputs
    if fb is not None:
      assert isinstance(fb, dict), f'"fb" must be a dict, got {type(fb)}: {fb}'
      if self.check_shape_consistency:
        # check feedback consistency
        for k, size in self._feedback_shapes.items():
          assert k in fb, f"The required key {k} is not provided in feedback inputs."
          assert size == fb[k].shape, (f'Shape does not match the required '
                                       f'feedback size {size} != {fb[k].shape}')
    ff = self.data_pass_func(ff)
    fb = self.data_pass_func(fb)
    return ff, fb

  def _call(self,
            ff: Union[Tensor, Dict[Any, Tensor]],
            fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
            forced_states: Dict[str, Tensor] = None,
            forced_feedbacks: Dict[str, Tensor] = None,
            monitors=None,
            **kwargs) -> Union[Tensor, Tuple[Tensor, Dict]]:
    # initialization
    self.initialize(ff, fb)
    # initialize the forced data
    if forced_states is None: forced_states = dict()
    if isinstance(forced_states, (bm.ndarray, jnp.ndarray)):
      forced_states = {self.name: forced_states}
    check_dict_data(forced_states, key_type=str, val_type=(bm.ndarray, jnp.ndarray))
    assert forced_feedbacks is None, ('Single instance of brainpy.nn.Node do '
                                      'not support "forced_feedbacks"')
    # monitors
    need_return_monitor = True
    if monitors is None:
      monitors = tuple()
      need_return_monitor = False
    attr_monitors: Dict[str, Tensor] = {}
    state_monitors: Dict[str, Tensor] = {}
    for key in monitors:
      assert isinstance(key, str), (f'"extra_returns" must be a sequence of string, '
                                    f'while we got {type(key)}')
      if key not in NODE_STATES:
        if not hasattr(self, key):
          raise UnsupportedError(f'Each node can monitor its states (including {NODE_STATES}), '
                                 f'or its attribute. While {key} is neither the state nor '
                                 f'the attribute of node {self}.')
        else:
          attr_monitors[key] = getattr(self, key)
      else:
        if key == 'state':
          assert self.state is not None, (f'{self} has no state, while '
                                          f'the user try to monitor its state.')
        state_monitors[key] = None
    # calling
    ff, fb = self._check_inputs(ff, fb=fb)
    if 'inputs' in state_monitors:
      state_monitors['inputs'] = ff
    if 'feedbacks' in state_monitors:
      state_monitors['feedbacks'] = fb
    output = self.call(ff, fb, **kwargs)
    if 'output' in state_monitors:
      state_monitors['output'] = output
    if 'state' in state_monitors:
      state_monitors['state'] = self.state
    attr_monitors.update(state_monitors)
    if need_return_monitor:
      return output, attr_monitors
    else:
      return output

  def call(self, ff, fb=None, **kwargs):
    """The main computation function of a node.

    Parameters
    ----------
    ff: tensor, dict, sequence
      The feedforward inputs.
    fb: optional, tensor, dict, sequence
      The feedback inputs.
    **kwargs
      Other parameters.

    Returns
    -------
    Tensor
      A output tensor value.
    """
    raise NotImplementedError


class Network(Node):
  """Basic Network class for neural network building in BrainPy."""

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

  def reset_state(self, to_state: Dict[str, Tensor] = None):
    """Reset the last state saved to zero for all nodes in the network or to other state values.

    Parameters
    ----------
    to_state : dict, optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new Tensor state vectors.
    """
    if to_state is None:
      for node in self.lnodes:
        node.reset_state()
    else:
      assert isinstance(to_state, dict)
      names = list(to_state.keys())
      for name, node in self.implicit_nodes.items():
        if name in to_state:
          self.get_node(name).reset(to_state=to_state[name])
          names.pop(name)
        else:
          self.get_node(name).reset()
      if len(names) > 0:
        raise ValueError(f'Unknown nodes: {names}')

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
                    ff_edges: Sequence[Tuple[Node, Node]],
                    fb_edges: Sequence[Tuple[Node, Node]] = None) -> "Network":
    if fb_edges is None: fb_edges = tuple()

    # assign nodes and edges
    self._nodes = tuple(nodes)
    self._ff_edges = tuple(ff_edges)
    self._fb_edges = tuple(fb_edges)
    self._network_init()
    return self

  def ff_init(self):
    # input shapes of entry nodes
    for node in self._entry_nodes:
      if node.input_shapes is None:
        if self.input_shapes is None:
          raise ValueError('Cannot find the input size. Cannot initialize the network.')
        else:
          node.set_input_shapes({node.name: self._input_shapes[node.name]})
      node._ff_init()
    # initialize the queue
    children_queue = []
    for node in self._entry_nodes:
      for child in self.ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)
    ff_states = {n: n.output_shape for n in self._entry_nodes}
    while len(children_queue):
      node = children_queue.pop(0)
      # initialize input and output sizes
      parent_sizes = {p: p.output_shape for p in self.ff_senders.get(node, [])}
      node.set_input_shapes(parent_sizes)
      node._ff_init()
      # append parent size
      ff_states[node] = node.output_shape
      # append children
      for child in self.ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)

  def fb_init(self):
    for receiver, senders in self.fb_senders.items():
      fb_sizes = {node: node.output_shape for node in senders}
      receiver.set_feedback_shapes(fb_sizes)
      receiver._fb_init()

  def initialize(self, ff: Optional[Union[Tensor, Dict[Any, Tensor]]] = None, fb=None):
    """Initialize the whole network.

    This function is useful, because it is independent from the __call__ function.
    We can use this function before when we apply JIT to __call__ function."""

    # feedforward initialization
    if not self.is_initialized:
      # check input and output nodes
      assert len(self.entry_nodes) > 0, f"We found this network {self} has no input nodes."
      assert len(self.exit_nodes) > 0, f"We found this network {self} has no output nodes."

      # check whether has a feedforward path for each feedback pair
      for node, receiver in self.fb_edges:
        if not detect_path(receiver, node, self.ff_edges):
          raise ValueError(f'Cannot build a feedback connection from \n\n{node} \n\n'
                           f'to \n\n{receiver} \n\n'
                           f'because there is no feedforward path between them. \n'
                           f'Maybe you should use "ff_connect" first to establish a '
                           f'feedforward connection between them. ')

      # feedforward checking
      if ff is None:
        in_sizes = dict()
        for node in self.entry_nodes:
          if node._input_shapes is None:
            raise ValueError('Cannot initialize this node, because we detect '
                             'both "input_shapes"and "ff" inputs are None. ')
          in_sizes.update(node._input_shapes)
      else:
        if isinstance(ff, (bm.ndarray, jnp.ndarray)):
          ff = {self.entry_nodes[0].name: ff}
        assert isinstance(ff, dict), f'ff must be a dict or a tensor, got {type(ff)}: {ff}'
        for n in self.entry_nodes:
          if n.name not in ff:
            raise ValueError(f'Cannot find the input of the node {n}')
        in_sizes = {k: v.shape for k, v in ff.items()}

      # initialize feedforward
      self.set_input_shapes(in_sizes)
      self._ff_init()

    # feedback initialization
    if len(self.fb_senders):
      # initialize feedback
      if not self.is_fb_initialized:
        self._fb_init()
        # initialize feedback states
        for node in self.feedback_nodes:
          if node.state is None:
            fb_state = bm.Variable(bm.zeros(node.output_shape))
          else:
            fb_state = bm.Variable(node.state)
          self._fb_states[node.name] = fb_state
          self.implicit_vars[node.name] = fb_state  # import for var() register

  def _check_inputs(self, ff, fb=None):
    # feedforward inputs
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      ff = {self.entry_nodes[0].name: ff}
    assert isinstance(ff, dict), f'ff must be a dict or a tensor, got {type(ff)}: {ff}'
    assert len(self.entry_nodes) == len(ff), (f'This network has {len(self.entry_nodes)} '
                                              f'entry nodes. While only {len(ff)} input '
                                              f'data are given.')
    for n in self.entry_nodes:
      if n.name not in ff: raise ValueError(f'Cannot find the input of the node {n}')
    # check feedforward consistency
    for k, size in self._input_shapes.items():
      assert k in ff, f"The required key {k} is not provided in feedforward inputs."
      assert size == ff[k].shape, (f'Shape does not match the required '
                                   f'input size {size} != {ff[k].shape}')
    # feedback inputs
    if fb is not None:
      if isinstance(fb, (bm.ndarray, jnp.ndarray)):
        fb = {self.feedback_nodes[0]: fb}
      assert isinstance(fb, dict), (f'fb must be a dict or a tensor, '
                                    f'got {type(fb)}: {fb}')
      assert len(self.feedback_nodes) == len(fb), (f'This network has {len(self.feedback_nodes)} '
                                                   f'feedback nodes. While only {len(ff)} '
                                                   f'feedback data are given.')
      for n in self.feedback_nodes:
        if n.name not in fb:
          raise ValueError(f'Cannot find the feedback data from the node {n}')
      # check feedback consistency
      for k, size in self._feedback_shapes.items():
        assert k in fb, f"The required key {k} is not provided in feedback inputs."
        assert size == fb[k].shape, (f'Shape does not match the required '
                                     f'feedback size {size} != {fb[k].shape}')
    # data transformation
    ff = self.data_pass_func(ff)
    fb = self.data_pass_func(fb)
    return ff, fb

  def _call(self,
            ff: Union[Tensor, Dict[Any, Tensor]],
            fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
            forced_states: Optional[Dict[str, Tensor]] = None,
            forced_feedbacks: Optional[Dict[str, Tensor]] = None,
            monitors: Optional[Sequence[str]] = None,
            **kwargs):
    # initialization
    self.initialize(ff, fb)
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
      assert isinstance(key, str), (f'"extra_returns" must be a sequence of string, '
                                    f'while we got {type(key)}')
      splits = key.split('.')
      assert len(splits) == 2, (f'Every term in "extra_returns" must be (node.item), '
                                f'while we got {key}')
      assert splits[0] in self.implicit_nodes, (f'Cannot found the node {splits[0]}, this network '
                                                f'only has {list(self.implicit_nodes.keys())}.')

      if splits[1] not in NODE_STATES:
        if not hasattr(self.implicit_nodes[splits[0]], splits[1]):
          raise UnsupportedError(f'Each node can monitor its states (including {NODE_STATES}), '
                                 f'or its attribute. While {splits[1]} is neither the state nor '
                                 f'the attribute of node {splits[0]}.')
        else:
          attr_monitors[key] = getattr(self.implicit_nodes[splits[0]], splits[1])
      else:
        if splits[1] == 'state':
          assert self.implicit_nodes[splits[0]].state is not None, (f'{splits[0]} has no state, while '
                                                                    f'the user try to monitor it.')
        state_monitors[key] = None
    # calling the computation core
    ff, fb = self._check_inputs(ff, fb=fb)
    output, state_monitors = self.call(ff, fb, forced_states, forced_feedbacks, state_monitors, **kwargs)
    if need_return_monitor:
      attr_monitors.update(state_monitors)
      return output, attr_monitors
    else:
      return output

  def call(self,
           ff, fb=None,
           forced_states: Dict[str, Tensor] = None,
           forced_feedbacks: Dict[str, Tensor] = None,
           monitors: Dict = None,
           **kwargs):
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
    **kwargs
      Other parameters which will be parsed into every node.

    Returns
    -------
    Tensor
      A output tensor value, or a dict of output tensors.
    """
    # initialize the feedback
    if forced_feedbacks is None: forced_feedbacks = dict()
    for key, tensor in forced_feedbacks.items():
      self.feedback_states[key].value = tensor
    if monitors is None: monitors = dict()

    # initialize the children queue
    children_queue = []

    # initialize the parent output data
    parent_outputs = {}
    for i, node in enumerate(self._entry_nodes):
      ff = {node.name: ff[i]}
      fb = {p: self._fb_states[p.name] for p in self.fb_senders.get(node, [])}
      self._call_a_node(node, ff, fb, monitors, forced_states, forced_feedbacks,
                        parent_outputs, children_queue, **kwargs)

    # run the model
    while len(children_queue):
      node = children_queue.pop(0)
      # get feedforward and feedback inputs
      ff = {p: parent_outputs[p] for p in self.ff_senders.get(node, [])}
      fb = {p: self._fb_states[p.name] for p in self.fb_senders.get(node, [])}
      # call the node
      self._call_a_node(node, ff, fb, monitors, forced_states, forced_feedbacks,
                        parent_outputs, children_queue, **kwargs)
      # remove unnecessary parent outputs
      needed_parents = [] + list(self.exit_nodes)
      for child in children_queue:
        needed_parents += self.ff_senders[child]
      for parent in list(parent_outputs.keys()):
        if parent not in needed_parents:
          parent_outputs.pop(parent)

    # returns
    if len(self.exit_nodes) > 1:
      state = {n.name: parent_outputs[n] for n in self.exit_nodes}
    else:
      state = parent_outputs[self.exit_nodes[0]]
    return state, monitors

  def _call_a_node(self, node, ff, fb, monitors, forced_states, forced_feedbacks,
                   parent_outputs, children_queue, **kwargs):
    ff = node.data_pass_func(ff)
    if f'{node.name}.inputs' in monitors:
      monitors[f'{node.name}.inputs'] = ff
    # get the output results
    if len(fb):
      fb = node.data_pass_func(fb)
      if f'{node.name}.feedbacks' in monitors:
        monitors[f'{node.name}.feedbacks'] = fb
      parent_outputs[node] = node.call(ff, fb, **kwargs)
    else:
      parent_outputs[node] = node.call(ff, **kwargs)
    if node.name in forced_states:  # forced state
      node.state.value = forced_states[node.name]
      parent_outputs[node] = forced_states[node.name]
    if f'{node.name}.state' in monitors:
      monitors[f'{node.name}.state'] = node.state
    if f'{node.name}.output' in monitors:
      monitors[f'{node.name}.output'] = parent_outputs[node]
    # update feedback with node's output
    if (node.name in self._fb_states) and (node.name not in forced_feedbacks):
      self._fb_states[node.name].value = parent_outputs[node]
    # append children nodes
    for child in self.ff_receivers.get(node, []):
      if child not in children_queue:
        children_queue.append(child)

  def visualize(self):
    pass


class FrozenNetwork(Network):
  """A FrozenNetwork is a Network that can not be linked to other nodes or networks."""

  def update_graph(self, new_nodes, new_ff_edges, new_fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")

  def replace_graph(self, nodes, ff_edges, fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")
