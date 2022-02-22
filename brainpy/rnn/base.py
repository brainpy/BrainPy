# -*- coding: utf-8 -*-

from copy import copy, deepcopy
from typing import Dict, Sequence, Tuple, Union, Optional

import jax.numpy as jnp

from brainpy import tools, math as bm
from brainpy.base import Base, Collector
from brainpy.types import Tensor
from . import graph_flow

operations = dispatcher = None

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
      self._in_size = tools.to_size(in_size)

    self._is_initialized = False
    self._is_fb_initialized = False
    self._trainable = trainable
    self._state = None

  def __repr__(self):
    all_params = [f"name={self.name}", f"in={self.in_size}",
                  f"out={self.out_size}\n\t", f"state={self._state}"]
    return f"{type(self).__name__}({', '.join(all_params)})"

  def __call__(self, *args, **kwargs) -> Tensor:
    return self._forward(*args, **kwargs)

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
  def in_size(self) -> Optional[Union[Tuple[int], Dict[str, Tuple[int]]]]:
    """Input data size."""
    return self._in_size

  @in_size.setter
  def in_size(self, size):
    self.set_in_size(size)

  def set_in_size(self, size):
    if not self.is_initialized:
      size = tools.to_size(size)
      if self.in_size is not None and size != self.in_size:
        raise ValueError(f"Impossible to use {self.name} with input "
                         f"data of dimension {size}. Node has input "
                         f"dimension {self.in_size}.")
      self._in_size = size
    else:
      raise TypeError(f"Input dimension of {self.name} is immutable after initialization.")

  @property
  def out_size(self) -> Optional[Union[Tuple[int], Dict[str, Tuple[int]]]]:
    """Output data size."""
    if self._state is None:
      return self._out_size
    else:
      return self._state.shape

  @out_size.setter
  def out_size(self, size):
    self.set_out_size(size)

  def set_out_size(self, size):
    if not self.is_initialized:
      size = tools.to_size(size)
      if self.out_size is not None and size != self.out_size:
        raise ValueError(f"Impossible to use {self.name} with target "
                         f"data of dimension {size}. Node has output "
                         f"dimension {self.out_size}.")
      self._out_size = size
    else:
      raise TypeError(f"Output dimension of {self.name} is immutable after initialization.")

  @property
  def fb_size(self) -> Optional[Union[Tuple[int], Dict[str, Tuple[int]]]]:
    """Output data size."""
    return self._fb_size

  @fb_size.setter
  def fb_size(self, size):
    self.set_fb_size(size)

  def set_fb_size(self, size):
    if not self.is_initialized:
      size = tools.to_size(size)
      if self.fb_size is not None and size != self.fb_size:
        raise ValueError(f"Impossible to use {self.name} with target "
                         f"data of dimension {size}. Node has feedback "
                         f"dimension {self.fb_size}.")
      self._fb_size = size
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
        self._state.value = bm.zeros(self.out_size, dtype=bm.float_)
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
    raise ValueError('This node does not support feedback connection. '
                     'Please implement fb_init() function if you want '
                     'to support.')

  def ff_init(self):
    raise NotImplementedError

  def _forward(self, ff: Optional[Tensor], fb: Optional[Tensor] = None):
    if not self.is_initialized:
      self.set_in_size(ff.shape)
      self._ff_init()
    if not self.is_fb_initialized:
      self.set_fb_size(None if fb is None else fb.shape)
      self._fb_init()
    return self.forward(ff, fb)

  def forward(self, ff, fb=None):
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
    global dispatcher
    if dispatcher is None:
      from . import dispatcher
    self._inputs, self._outputs = graph_flow.find_entries_and_exits(self._nodes, self._ff_edges)
    self._nodes = graph_flow.topological_sort(self._nodes, self._ff_edges, self._inputs)
    self._dispatcher = dispatcher.Dispatcher(model=self)
    self.implicit_nodes = Collector({n.name: n for n in self._nodes})  # brainpy.Base object
    self._is_initialized = False
    self._is_fb_initialized = False
    # self._fitted = all([n.fitted for n in self.nodes])

  def __repr__(self):
    nodes = [n.name for n in self._nodes]
    return f"{type(self).__name__}({', '.join(nodes)})"

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
    return self._inputs

  @property
  def exit_nodes(self) -> Sequence[Node]:
    """Last Nodes in the graph held by the Model."""
    return self._outputs

  @property
  def nodes_has_feedback(self) -> Sequence[Node]:
    """Returns all Nodes has a feedback connection in the Model."""
    return [n for n in self.lnodes if n.has_feedback]

  @property
  def dispatcher(self):
    return self._dispatcher

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
    self._network_init()
    return self

  def _ff_init(self):
    """Call the Model initializers on some data points.
    Model will be virtually run to infer shapes of all nodes given
    inputs and targets vectors.
    """
    if not self.is_initialized:
      self.ff_init()
      self._is_initialized = True

  def ff_init(self):
    self._dispatcher
    for node in self.lnodes:
      node._ff_init()

  def _fb_init(self):
    """Call the Model initializers on some data points.
    Model will be virtually run to infer shapes of all nodes given
    inputs and targets vectors.
    """
    if self.is_fb_initialized:
      self.fb_init()
      self._is_fb_initialized = True

  def fb_init(self):
    self._dispatcher
    for node in self.lnodes:
      node._ff_init()

  def _forward(self,
               ff: Optional[Union[Tensor, Dict[str, Tensor]]],
               fb: Optional[Union[Tensor, Dict[str, Tensor]]] = None):
    if not self.is_initialized:
      self.set_in_size({k: v.shape for k, v in ff.items()} if isinstance(ff, dict) else ff.shape)
      self._ff_init()
    if not self.is_fb_initialized:
      if fb is None:
        fb_size = None
      elif isinstance(fb, dict):
        fb_size = {k: v.shape for k, v in fb.items()}
      elif isinstance(fb, (bm.JaxArray, jnp.ndarray)):
        fb_size = fb.shape
      else:
        raise ValueError(f'Unknown type of feedback {type(fb)}: {fb}')
      self.set_fb_size(fb_size)
      self._fb_init()
    return self.forward(ff, fb)

  def forward(
      self,
      x: Union[Tensor, Sequence[Tensor], Dict[str, Tensor]],
      y: Optional[Union[Tensor, Sequence[Tensor], Dict[str, Tensor]]] = None,
  ):
    data = self.dispatcher.load(x, y)
    for node in self.lnodes:
      node(*data[node].x)
    if len(self.exit_nodes) > 1:
      state = dict()
      for node in self.exit_nodes:
        state[node.name] = node.state
    else:
      state = self.exit_nodes[0].state
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
