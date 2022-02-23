# -*- coding: utf-8 -*-

from copy import copy, deepcopy
from typing import Dict, Sequence, Tuple, Union, Optional, Any

import jax.numpy as jnp

from brainpy import tools, math as bm
from brainpy.base import Base, Collector
from brainpy.rnn import graph_flow
from brainpy.types import Tensor

operations = None

__all__ = [
  'Node', 'Network', 'FrozenNetwork',
]


class Node(Base):
  """Basic module class."""

  def __init__(self, name=None, in_size=None, trainable=False):  # initialize parameters
    super(Node, self).__init__(name=name)

    self._in_size = None  # input size
    self._out_size = None  # output size
    self._fb_size = None  # feedback size
    if in_size is not None:
      self._in_size = {self.name: tools.to_size(in_size)}

    self._is_initialized = False
    self._is_fb_initialized = False
    self._fb_states = {}  # used to store the states of feedback nodes
    self._trainable = trainable
    self._state = None  # the state of the current node

  def __repr__(self):
    name = type(self).__name__
    return f"{name}(name={self.name}, out={self.out_size}, \n" \
           f"{' ' * len(name)}, state={self._state})"

  def __call__(self, *args, **kwargs) -> Tensor:
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
    return operations.fb_connect(self, other)

  def __rlshift__(self, other):  # "other << self"
    global operations
    if operations is None: from . import operations
    return operations.fb_connect(other, self)

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
      if self.out_size is not None:
        assert self.out_size == value.shape
      self._state = bm.Variable(value) if not isinstance(value, bm.Variable) else value
    else:
      self._state.value = value

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
  def in_size(self) -> Optional[Dict['Node', Tuple[int]]]:
    """Input data size."""
    return self._in_size

  @in_size.setter
  def in_size(self, size):
    self.set_in_size(size)

  @property
  def out_size(self) -> Optional[Tuple[int]]:
    """Output data size."""
    return self._out_size

  @out_size.setter
  def out_size(self, size):
    self.set_out_size(size)

  @property
  def fb_size(self) -> Optional[Dict['Node', Tuple[int]]]:
    """Output data size."""
    return self._fb_size

  @fb_size.setter
  def fb_size(self, size):
    self.set_fb_size(size)

  def set_in_size(self, in_sizes):
    if not self.is_initialized:
      if self.in_size is not None:
        for key, size in self.in_size.items():
          if key not in in_sizes:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While we do not find it in {in_sizes}")
          if size != in_sizes[key]:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While the give shape is {in_sizes[key]}")
      self._in_size = in_sizes
    else:
      raise TypeError(f"Input dimension of {self.name} is immutable after initialization.")

  def set_out_size(self, size):
    if not self.is_initialized:
      self._out_size = size
    else:
      raise TypeError(f"Output dimension of {self.name} is immutable after initialization.")

  def set_fb_size(self, fb_sizes):
    if not self.is_initialized:
      if self.in_size is not None:
        for key, size in self.in_size.items():
          if key not in fb_sizes:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While we do not find it in {fb_sizes}")
          if size != fb_sizes[key]:
            raise ValueError(f"Impossible to reset the input data of {self.name}. "
                             f"Because this Node has the input dimension {size} from {key}. "
                             f"While the give shape is {fb_sizes[key]}")
      self._fb_size = fb_sizes
    else:
      raise TypeError(f"Feedback dimension of {self.name} is immutable after initialization.")

  def nodes(self, method='absolute', level=1, include_self=True):
    return super(Node, self).nodes(method=method, level=level, include_self=include_self)

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

  def reset(self, to_state: Tensor = None):
    """A null state vector."""
    if to_state is None:
      if self.out_size is not None:
        self._state.value = bm.zeros_like(self._state)
    else:
      self._state.value = to_state

  def _ff_init(self):
    """Call the Node initializers on some data points.
    Initializers are functions called at first run of the Node,
    defining the dimensions and values of its parameters based on the
    dimension of some input data and its hyperparameters.

    Data point `x` is used to infer the input dimension of the Node.
    Data point `y` is used to infer the output dimension of the Node.
    """
    if not self.is_initialized:
      self.ff_init()
      self._is_initialized = True

  def _fb_init(self):
    if not self.is_fb_initialized:
      self.fb_init()
      self._is_fb_initialized = True

  def fb_init(self):
    raise ValueError('This node does not support feedback connection.')

  def ff_init(self):
    raise NotImplementedError('Please implement the feedforward initialization.')

  def initialize(self,
                 ff: Union[Tensor, Dict[Any, Tensor]],
                 fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None):
    # feedforward initialization
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      ff = {self.name: ff}
    assert isinstance(ff, dict), f'"ff" must be a dict or a tensor, got {type(ff)}: {ff}'
    assert self.name in ff, f'Cannot find input for this node {self} when given "ff" {ff}'
    if not self.is_initialized:  # initialize feedforward
      self.set_in_size({k: v.shape for k, v in ff.items()})
      self._ff_init()
    else:  # check consistency
      for k, size in self.in_size.items():
        assert k in ff, f"The required key {k} is not provided in feedforward inputs."
        assert size == ff[k].shape, f'Shape does not match the required input size {size} != {ff[k].shape}'

    # feedback initialization
    if fb is not None:
      assert isinstance(fb, dict), f'"fb" must be a dict, got {type(fb)}: {fb}'
      if not self.is_fb_initialized:  # initialize feedback
        self.set_fb_size({k: v.shape for k, v in fb.items()})
        self._fb_init()
      else:  # check consistency
        for k, size in self.fb_size.items():
          assert k in fb, f"The required key {k} is not provided in feedback inputs."
          assert size == fb[k].shape, f'Shape does not match the required feedback size {size} != {fb[k].shape}'
    return ff, fb

  def _call(self,
            ff: Union[Tensor, Dict[Any, Tensor]],
            fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
            **kwargs) -> Tensor:
    # initialization
    ff, fb = self.initialize(ff, fb)
    # calling
    return self.call(ff, fb, **kwargs)

  def call(self,
           ff: Union[Dict['Node', Tensor], Dict[str, Tensor]],
           fb: Optional[Union[Dict['Node', Tensor], Dict[str, Tensor]]] = None):
    raise NotImplementedError


class Network(Node):
  def __init__(self, nodes=None, ff_edges=None, fb_edges=None, name=None, in_size=None):
    super(Network, self).__init__(name=name, in_size=in_size)
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
    self._entry_nodes, self._exit_nodes = graph_flow.find_entries_and_exits(self._nodes, self._ff_edges)
    # build feedforward graph
    self._ff_senders, self._ff_receivers = graph_flow.find_senders_and_receivers(self._ff_edges)
    # build feedback graph
    self._fb_senders, self._fb_receivers = graph_flow.find_senders_and_receivers(self._fb_edges)
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
  def ff_edges(self) -> Tuple[Tuple[Node, Node]]:
    return self._ff_edges

  @property
  def fb_edges(self) -> Tuple[Tuple[Node, Node]]:
    return self._ff_edges

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
    return tuple(self._fb_receivers.keys())

  @property
  def nodes_has_feedback(self) -> Sequence[Node]:
    return tuple(self._fb_senders.keys())

  def reset(self, to_state: Dict[str, Tensor] = None):
    """Reset the last state saved to zero for all nodes in the network or to other state values.

    Parameters
    ----------
    to_state : dict, optional
        Pairs of keys and values, where keys are Model nodes names and
        value are new Tensor state vectors.
    """
    if to_state is None:
      for node in self.lnodes:
        node.reset()
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
    if graph_flow.detect_cycle(self._nodes, self._ff_edges):
      raise ValueError('We detect cycles in feedforward connections. '
                       'Maybe you should replace some connection with '
                       'as feedback ones.')
    if graph_flow.detect_cycle(self._nodes, self._fb_edges):
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
    # input size of entry nodes
    for node in self._entry_nodes:
      if node.in_size is None:
        if self.in_size is None:
          raise ValueError('Cannot find the input size. Cannot initialize the network.')
        else:
          node.set_in_size({node.name: self.in_size[node.name]})
      node._ff_init()
    # initialize the queue
    children_queue = []
    for node in self._entry_nodes:
      for child in self._ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)
    ff_states = {n: n.out_size for n in self._entry_nodes}
    while len(children_queue):
      node = children_queue.pop(0)
      # initialize input and output sizes
      parent_sizes = {p: p.out_size for p in self._ff_senders.get(node, [])}
      node.set_in_size(parent_sizes)
      node._ff_init()
      # append parent size
      ff_states[node] = node.out_size
      # append children
      for child in self._ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)

  def fb_init(self):
    for receiver, senders in self._fb_senders.items():
      fb_sizes = {node: node.out_size for node in senders}
      receiver.set_fb_size(fb_sizes)
      receiver._fb_init()

  def initialize(self,
                 ff: Union[Tensor, Dict[Any, Tensor]],
                 fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None):
    """Initialize the whole network.

    This function is useful, because it is independent from the __call__ function.
    We can use this function before when we apply JIT to __call__ function."""
    # feedforward checking
    if isinstance(ff, (bm.ndarray, jnp.ndarray)):
      if len(self.entry_nodes) == 1:
        ff = {self.entry_nodes[0].name: ff}
      else:
        raise ValueError(f'This network has {len(self.entry_nodes)} '
                         f'entry nodes. While only one input '
                         f'data is given.')
    assert isinstance(ff, dict), f'ff must be a dict or a tensor, got {type(ff)}: {ff}'
    assert len(self.entry_nodes) == len(ff), (f'This network has {len(self.entry_nodes)} '
                                              f'entry nodes. While only {len(ff)} input '
                                              f'data are given.')
    for n in self.entry_nodes:
      if n.name not in ff: raise ValueError(f'Cannot find the input of the node {n}')
    # feedforward initialization
    if not self.is_initialized:  # initialize feedforward
      self.set_in_size({k: v.shape for k, v in ff.items()})
      self._ff_init()
    else:  # check consistency
      for k, size in self.in_size.items():
        assert k in ff, f"The required key {k} is not provided in feedforward inputs."
        assert size == ff[k].shape, f'Shape does not match the required input size {size} != {ff[k].shape}'

    # feedback initialization
    if fb is not None:
      # checking
      if isinstance(fb, (bm.ndarray, jnp.ndarray)):
        fb = {self.feedback_nodes[0]: fb}
      assert isinstance(fb, dict), (f'fb must be a dict or a tensor, '
                                    f'got {type(fb)}: {fb}')
      assert len(self.feedback_nodes) == len(fb), (f'This network has {len(self.feedback_nodes)} '
                                                   f'feedback nodes. While only {len(ff)} feedback '
                                                   f'data are given.')
      for n in self.feedback_nodes:
        if n.name not in fb:
          raise ValueError(f'Cannot find the feedback data from the node {n}')
      # initialize feedback
      if not self.is_fb_initialized:
        self.set_fb_size({k: v.shape for k, v in fb.items()})
        self._fb_init()
        for sender in self._fb_senders.keys():
          if sender.state is None:
            fb_state = bm.Variable(bm.zeros(sender.out_size))
          else:
            fb_state = bm.Variable(sender.state)
          self._fb_states[sender] = fb_state
          self.implicit_vars[sender.name] = fb_state  # import for var() register
      # check consistency
      else:
        for k, size in self.fb_size.items():
          assert k in fb, f"The required key {k} is not provided in feedback inputs."
          assert size == fb[k].shape, (f'Shape does not match the required '
                                       f'feedback size {size} != {fb[k].shape}')

    return ff, fb

  def __call__(self,
               ff: Union[Tensor, Dict[Any, Tensor]],
               fb: Optional[Union[Tensor, Dict[Any, Tensor]]] = None,
               **kwargs):
    ff, fb = self.initialize(ff, fb=fb)
    return self.call(ff, fb, **kwargs)

  def call(
      self,
      ff: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
      fb: Optional[Union[Tensor, Sequence[Tensor], Dict[str, Tensor]]] = None,
      **kwargs
  ):
    # initialize the data
    parent_outputs = {n: n({n.name: ff[n.name]}) for n in self._entry_nodes}
    children_queue = []
    for node in self._entry_nodes:
      for child in self._ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)
    # run the model
    while len(children_queue):
      node = children_queue.pop(0)
      # initialize feedforward and feedback inputs
      ff = {p: parent_outputs[p] for p in self._ff_senders.get(node, [])}
      fb = {p: self._fb_states[p] for p in self._fb_senders.get(node, [])}
      # get the output results
      if len(fb):
        parent_outputs[node] = node(ff, fb)
      else:
        parent_outputs[node] = node(ff)
      # update the feedback state
      if node in self._fb_states:
        self._fb_states[node].value = parent_outputs[node]
      # append children nodes
      for child in self._ff_receivers.get(node, []):
        if child not in children_queue:
          children_queue.append(child)
    # returns
    if len(self.exit_nodes) > 1:
      state = {n.name: parent_outputs[n] for n in self.exit_nodes}
    else:
      state = parent_outputs[self.exit_nodes[0]]
    return state


# FeedbackProxy:
# 1. remember previous states,
# 2. calculate outputs of nodes without states,
# 3. replaced state


class FrozenNetwork(Network):
  """A FrozenModel is a Model that can not be linked to other nodes or models."""

  def update_graph(self, new_nodes, new_ff_edges, new_fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")

  def replace_graph(self, nodes, ff_edges, fb_edges=None):
    raise TypeError(f"Cannot update FrozenModel {self}: "
                    f"model is frozen and cannot be modified.")
