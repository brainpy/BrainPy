# -*- coding: utf-8 -*-

"""This module provides basic operations for constructing node graphs.

It supports the following operations:

1. feedforward connection: ">>", ">>="
2. feedback connection: "<<", "<<="
3. merge two nodes: "&", "&="
4. select subsets of one node: "[:]"
5. concatenate a sequence of nodes: "[node1, node2, ...]", "(node1, node2, ...)"
6. wrap a set of nodes: "{node1, node2, ...}"

However, all operations should satisfy the following assumptions:

1. Feedback connection of `(node1, node2)` should have a feedforward path from `node2` to `node1`.
2. Feedforward or feedback connections cannot generate a cycle.
3. Cannot concatenate multiple receiver nodes, e.g., `a >> [b, c]` is forbidden, but `a >> {b, c}`
   is allowed.

"""

from itertools import product
from typing import Union, Sequence, Set

from brainpy.compat.nn import graph_flow
from brainpy.compat.nn.base import Node, Network, FrozenNetwork
from brainpy.compat.nn.datatypes import SingleData
from brainpy.compat.nn.nodes.base import Select, Concat
from brainpy.types import Tensor

__all__ = [
  'ff_connect', 'fb_connect', 'merge', 'select', 'concatenate',
]


def _retrieve_nodes_and_edges(senders: Union[Node, Sequence[Node]],
                              receivers: Union[Node, Sequence[Node]]):
  # check senders
  if isinstance(senders, (tuple, list)):
    senders = [concatenate(senders)]
  elif isinstance(senders, set):
    senders = list(senders)
  elif isinstance(senders, Node):
    senders = [senders]
  else:
    raise TypeError(f"Impossible to send connection from {senders}: it is not "
                    f"a Node or a Network instance.")

  # check receivers
  if isinstance(receivers, (tuple, list)):
    raise TypeError('Cannot concatenate a list/tuple of receivers. '
                    'Please use set to wrap multiple receivers instead.')
  elif isinstance(receivers, set):
    receivers = list(receivers)
  elif isinstance(receivers, Node):
    receivers = [receivers]
  else:
    raise TypeError(f"Impossible to send connection to {receivers}: it is not "
                    f"a Node or a Network instance.")

  # fetch all nodes in two subgraphs
  all_nodes = set()
  for node in senders + receivers:
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_nodes.update(set(node.lnodes))
    elif isinstance(node, Node):
      all_nodes.add(node)
    else:
      raise TypeError(f"Impossible to link nodes: object {node} is neither a "
                      f"'brainpy.rnn.Node' nor a 'brainpy.rnn.Network'.")

  # fetch all feedforward edges in two subgraphs
  all_ff_edges = set()
  for node in senders + receivers:
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_ff_edges.update(set(node.ff_edges))

  # fetch all feedback edges in two subgraphs
  all_fb_edges = set()
  for node in senders + receivers:
    if isinstance(node, FrozenNetwork):
      raise TypeError(f"Cannot connect {FrozenNetwork.__name__} to other Nodes.")
    if isinstance(node, Network):
      all_fb_edges.update(set(node.fb_edges))

  # create edges between output nodes of the
  # subgraph 1 and input nodes of the subgraph 2.
  all_senders = set()
  for node in senders:
    if isinstance(node, Network) and not isinstance(node, FrozenNetwork):
      all_senders.update(node.exit_nodes)
    else:
      all_senders.add(node)
  all_receivers = set()
  for node in receivers:
    if isinstance(node, Network) and not isinstance(node, FrozenNetwork):
      all_receivers.update(node.entry_nodes)
    else:
      all_receivers.add(node)

  return all_nodes, all_ff_edges, all_fb_edges, all_senders, all_receivers


def _reorganize_many2one(ff_edges, fb_edges):
  """Reorganize the many-to-one connections.

  If some node whose "data_type" is :py:class:`brainpy.nn.datatypes.SingleData` receives
  multiple feedforward or feedback connections, we should concatenate all feedforward
  inputs (or feedback inputs) into one instance of :py:class:`brainpy.nn.Concat`, then
  the new Concat instance feeds into this node.

  """
  from brainpy.compat.nn.nodes.base import Concat

  new_nodes = []

  # find parents according to the child
  ff_senders = dict()
  for edge in ff_edges:
    sender, receiver = edge
    if receiver not in ff_senders:
      ff_senders[receiver] = [sender]
    else:
      ff_senders[receiver].append(sender)
  for receiver, senders in ff_senders.items():
    if isinstance(receiver.data_pass, SingleData):
      if len(senders) > 1:
        concat_nodes = [node for node in senders if isinstance(node, Concat)]
        if len(concat_nodes) == 1:
          concat = concat_nodes[0]
          for sender in senders:
            if sender != concat:
              ff_edges.remove((sender, receiver))
              ff_edges.add((sender, concat))
        else:
          concat = Concat()
          for sender in senders:
            ff_edges.remove((sender, receiver))
            ff_edges.add((sender, concat))
          ff_edges.add((concat, receiver))
          new_nodes.append(concat)

  # find parents according to the child
  fb_senders = dict()
  for edge in fb_edges:
    sender, receiver = edge
    if receiver not in fb_senders:
      fb_senders[receiver] = [sender]
    else:
      fb_senders[receiver].append(sender)
  for receiver, senders in fb_senders.items():
    if isinstance(receiver.data_pass, SingleData):
      if len(senders) > 1:
        concat_nodes = [node for node in senders if isinstance(node, Concat)]
        if len(concat_nodes) == 1:
          concat = concat_nodes[0]
          for sender in senders:
            if sender != concat:
              fb_edges.remove((sender, receiver))
              ff_edges.add((sender, concat))
        else:
          concat = Concat()
          for sender in senders:
            fb_edges.remove((sender, receiver))
            ff_edges.add((sender, concat))
          fb_edges.add((concat, receiver))
          new_nodes.append(concat)

  return new_nodes, ff_edges, fb_edges


def merge(
    node: Node,
    *other_nodes: Node,
    inplace: bool = False,
    name: str = None,
    need_detect_cycle=True
) -> Network:
  """Merge different :py:class:`~.Node` or :py:class:`brainpy.nn.base.Network`
  instances into a single :py:class:`brainpy.nn.base.Network` instance.

  :py:class:`~.Node` instances contained in the network to merge will be
  gathered in a single network, along with all previously defined connections
  between them, if they exists.

  You can also perform this operation using the ``&`` operator::

      network = (node1 >> node2) & (node1 >> node3))

  This is equivalent to::

      network = merge((node1 >> node2), (node1 >> node3))

  The inplace operator can also be used::

      network &= other_network

  Parameters
  ----------
  node: Network, Node
      First node or network to merge.
  *other_nodes : Network, Node
      All nodes to merge.
  inplace: bool, default to False
      If `True`, then will update `node` inplace. If `node` is not a Network
      instance, this parameter will causes the function to raise an error.
  name: str, optional
      Name of the resulting Network.
  need_detect_cycle: bool
      Whether need to detect the cycle defined in the graph.

  Returns
  -------
  Network
      A new :py:class:`brainpy.nn.base.Network` instance.
  """
  # checking
  for n in other_nodes + (node,):
    if not isinstance(n, Node):
      raise TypeError(f"Impossible to merge nodes: object {type(n)} is not a Node instance.")

  # get all node and edges
  all_nodes = set()
  all_ff_edges = set()
  all_fb_edges = set()
  for n in other_nodes + (node,):
    if isinstance(n, FrozenNetwork):
      raise TypeError(f'{FrozenNetwork.__name__} cannot merge with other nodes.')
    # fuse models nodes and edges (right side argument)
    if isinstance(n, Network):
      all_nodes |= set(n.lnodes)
      all_ff_edges |= set(n.ff_edges)
      all_fb_edges |= set(n.fb_edges)
    elif isinstance(n, Node):
      all_nodes.add(n)

  # reorganize
  new_nodes, all_ff_edges, all_fb_edges = _reorganize_many2one(all_ff_edges, all_fb_edges)
  all_nodes.update(new_nodes)

  # detect cycles in the graph flow
  all_nodes = tuple(all_nodes)
  all_ff_edges = tuple(all_ff_edges)
  all_fb_edges = tuple(all_fb_edges)
  if need_detect_cycle:
    if graph_flow.detect_cycle(all_nodes, all_ff_edges):
      raise ValueError('We detect cycles in feedforward connections. '
                       'Maybe you should replace some connection with '
                       'as feedback ones.')
    if graph_flow.detect_cycle(all_nodes, all_fb_edges):
      raise ValueError('We detect cycles in feedback connections. ')

  if inplace:
    if not isinstance(node, Network) or isinstance(node, FrozenNetwork):
      raise ValueError(f"Impossible to merge nodes inplace: "
                       f"{node} is not a {Network.__name__} instance.")
    return node.replace_graph(nodes=all_nodes,
                              ff_edges=all_ff_edges,
                              fb_edges=all_fb_edges)

  else:
    return Network(nodes=all_nodes,
                   ff_edges=all_ff_edges,
                   fb_edges=all_fb_edges,
                   name=name)


def ff_connect(
    senders: Union[Node, Sequence[Node], Set[Node]],
    receivers: Union[Node, Set[Node]],
    inplace: bool = False,
    name: str = None,
    need_detect_cycle=True
) -> Network:
  """Connect two sequences of :py:class:`~.Node` instances to form
  a :py:class:`brainpy.nn.base.Network` instance. `senders` output will be used as
  input for `receivers` in the created network. This is similar to a
  function composition operation:

  .. math::

      network(x) = (sender \\circ receiver)(x) = receiver(sender(x))

  You can also perform this operation using the ``>>`` operator::

      network = sender >> receiver

  Or using this function::

      network = ff_connect(sender, receiver)

  - `sender` and `receiver` can also be :py:class:`brainpy.nn.base.Network` instances. In this
    case, the new :py:class:`brainpy.nn.base.Network` created will contain all nodes previously
    contained in all the networks, and link all `node1` outputs to all `node2`
    inputs. This allows to chain the  ``>>`` operator::

      step1 = node0 >> node1  # this is a network
      step2 = step1 >> node2  # this is another

  - `node1` can finally be lists or tuples of nodes. In this
    case, all `node1` outputs will be linked to a :py:class:`~.Concat` node to
    concatenate them, and the :py:class:`~.Concat` node will be linked to all
    `node2` inputs::

      # many-concat-to-one
      network = [node1, node2, ..., node] >> node_out

  - If you do not want to concatenate all input nodes, you can use `set` to
    wrap all input nodes at once. Then, `node2` will receive multiple inputs
    defined in `node1`::

      # many-to-one
      network = {node1, node2, ..., node_N} >> node_out

  - In the case of "one-to-many" feedforward connection, `node2` only support
    a set of node. Using list or tuple to wrap multiple receivers will concatenate
    all nodes in the receiver end. This will cause errors::

      # wrong operation of one-to-many
      network = node_in >> {node1, node2, ..., node_N}

      # correct operation of one-to-many
      network = node_in >> {node1, node2, ..., node_N}

  - "many-to-many" connection is also allowed.

    You can still use the ``>>`` operator in this situation,
    except for many-to-many nodes connections::

      # many-to-many
      {node1, node2, ..., node} >> {node1, node2, ..., node}

  Parameters
  ----------
  senders, receivers : Node, sequence of Node
    Nodes or sequence of nodes to connect feedforward connections.
  inplace: bool
    Whether inplace update the node.
  name: str, optional
    Name for the chaining Network.
  need_detect_cycle: bool
    Whether we need to detect cycles exit in the final network.

  Returns
  -------
  Network
      A :py:class:`brainpy.nn.base.Network` instance chaining the nodes.

  Notes
  -----

  Be careful to how you link the different nodes: `reservoirpy` does not
  allow to have circular dependencies between them::

      network = node1 >> node2  # fine
      network = node1 >> node2 >> node1  # raises! data would flow in
                                       # circles forever...
  """

  all_nodes, all_ff_edges, all_fb_edges, ff_senders, ff_receivers = _retrieve_nodes_and_edges(senders, receivers)
  new_ff_edges = set(product(ff_senders, ff_receivers))

  # all outputs from subgraph 1 are connected to
  # all inputs from subgraph 2.
  all_ff_edges |= new_ff_edges

  # reorganize
  new_nodes, all_ff_edges, all_fb_edges = _reorganize_many2one(all_ff_edges, all_fb_edges)
  all_nodes.update(new_nodes)

  # detect cycles in the graph flow
  all_nodes = tuple(all_nodes)
  all_ff_edges = tuple(all_ff_edges)
  all_fb_edges = tuple(all_fb_edges)
  if need_detect_cycle:
    if graph_flow.detect_cycle(all_nodes, all_ff_edges):
      raise ValueError('We detect cycles in feedforward connections. '
                       'Maybe you should replace some connection with '
                       'as feedback ones.')
    if graph_flow.detect_cycle(all_nodes, all_fb_edges):
      raise ValueError('We detect cycles in feedback connections. ')

  # feedforward
  if inplace:
    if not isinstance(receivers, Network):
      raise TypeError(f'Cannot inplace update the feedback connection of a Node instance: {receivers}')
    if name is not None:
      raise ValueError('Cannot set name when inplace=True.')
    receivers.replace_graph(nodes=all_nodes,
                            ff_edges=all_ff_edges,
                            fb_edges=all_fb_edges)
    return receivers
  else:
    return Network(nodes=all_nodes,
                   ff_edges=all_ff_edges,
                   fb_edges=all_fb_edges,
                   name=name)


def fb_connect(
    senders: Union[Node, Sequence[Node], Set[Node]],
    receivers: Union[Node, Set[Node]],
    inplace: bool = False,
    name: str = None,
    need_detect_cycle=True
) -> Node:
  """Create a feedback connection from ``sender`` node to ``receiver`` node.
  Feedbacks nodes will be called at runtime using data from the previous call.

  You can also perform this operation using the ``<<`` operator.

  Which means that a feedback connection is now created between `node1` and
  `node2`. In other words, the forward function of `node1` depends on the
  previous output of `node2`:

  .. math::
      \\mathrm{node1}(x_t) = \\mathrm{node1}(x_t, \\mathrm{node2}(x_{t - 1}))

  You can also use this function to define feedback::

      node1 = fb_connect(node1, node2)
      # without copy (node1 is the same object throughout)
      node1 = fb_connect(node1, node2, inplace=True, name="n1_copy")

  Parameters
  ----------
  receivers : Node
      Node receiving feedback.
  senders : GenericNode
      Node or Network sending feedback
  inplace : bool, defaults to False
      If `True`, then the function returns a copy of `node`.
  name : str, optional
      Name of the copy of `node` if `inplace` is `True`.
  need_detect_cycle: bool
      Whether we need to detect cycles in the defined network.

  Returns
  -------
  Network
      A network with feedback connections.
  """

  all_nodes, all_ff_edges, all_fb_edges, fb_senders, fb_receivers = _retrieve_nodes_and_edges(senders, receivers)

  # detect whether the node implement its own "init_fb_conn()" function
  for node in fb_receivers:
    if not node.is_feedback_input_supported:
      raise ValueError(f'Establish a feedback connection to \n'
                       f'{node}\n'
                       f'is not allowed. Because this node does not '
                       f'support feedback connections.')

  # detect feedforward cycle
  if need_detect_cycle:
    all_nodes1 = list(all_nodes)
    all_ff_edges1 = tuple(all_ff_edges)
    if graph_flow.detect_cycle(all_nodes1, all_ff_edges1):
      raise ValueError('We detect cycles in feedforward connections. '
                       'Maybe you should replace some connection with '
                       'as feedback ones.')
  # establish feedback connections
  new_fb_edges = set(product(fb_senders, fb_receivers))

  # all outputs from subgraph 1 are connected to
  # all inputs from subgraph 2.
  all_fb_edges |= new_fb_edges

  # reorganize
  new_nodes, all_ff_edges, all_fb_edges = _reorganize_many2one(all_ff_edges, all_fb_edges)
  all_nodes.update(new_nodes)

  # detect cycles in the graph flow
  all_nodes = tuple(all_nodes)
  all_ff_edges = tuple(all_ff_edges)
  all_fb_edges = tuple(all_fb_edges)
  if need_detect_cycle:
    if graph_flow.detect_cycle(all_nodes, all_fb_edges):
      raise ValueError('We detect cycles in feedback connections. ')

  # feedback
  if inplace:
    if not isinstance(receivers, Network):
      raise TypeError(f'Cannot inplace update the feedback connection of a Node instance: {receivers}')
    if name is not None:
      raise ValueError('Cannot set name when inplace=True.')
    receivers.replace_graph(nodes=all_nodes,
                            ff_edges=all_ff_edges,
                            fb_edges=all_fb_edges)
    return receivers
  else:
    return Network(nodes=all_nodes,
                   ff_edges=all_ff_edges,
                   fb_edges=all_fb_edges,
                   name=name)


def select(
    node: Node,
    index: Union[int, Sequence[int], Tensor, slice],
    name: str = None
):
  if isinstance(node, Network) and len(node.exit_nodes) != 1:
    raise ValueError(f'Cannot select subsets of states when Network instance '
                     f'"{node}" has multiple output nodes.')
  return ff_connect(node, Select(index=index), name=name, need_detect_cycle=False)


def concatenate(nodes: Sequence[Node], axis=-1, name=None):
  right = Concat(axis=axis)
  model = Network(name=name)
  for node in nodes:
    if isinstance(node, FrozenNetwork):
      raise ValueError('Cannot concat a Frozen network.')
    if isinstance(node, Network) and len(node.exit_nodes) > 1:
      raise ValueError(f'Cannot concatenate network which has {len(node.exit_nodes)} '
                       f'output nodes with other nodes.')
    model = merge(model,
                  ff_connect(node, right, need_detect_cycle=False),
                  inplace=True,
                  need_detect_cycle=False)
  return model
